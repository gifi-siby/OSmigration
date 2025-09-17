import os
import re
import sys
import time
import json
import math
import logging
import argparse
from typing import Any, Dict, List, Optional, Tuple

from pymilvus import MilvusClient, DataType
from opensearchpy import OpenSearch, exceptions as os_exceptions

# -----------------------------
# Logging
# -----------------------------
logger = logging.getLogger("validate_migration")


# -----------------------------
# OpenSearch client and readiness
# -----------------------------

def create_opensearch_client(cfg: Dict[str, Any]) -> OpenSearch:
    auth = None
    if cfg.get("http_auth_username") and cfg.get("http_auth_password"):
        auth = (cfg.get("http_auth_username"), cfg.get("http_auth_password"))

    params = {
        "hosts": [{"host": cfg.get("host", "localhost"), "port": int(cfg.get("port", 9200))}],
        "use_ssl": bool(cfg.get("use_ssl", False)),
        "verify_certs": bool(cfg.get("verify_certs", False)),
        "http_compress": True,
        "timeout": int(cfg.get("timeout", 60)),
        "retry_on_timeout": True,
        "max_retries": int(cfg.get("max_retries", 3)),
    }
    if auth:
        params["http_auth"] = auth
    if cfg.get("ca_certs"):
        params["ca_certs"] = cfg.get("ca_certs")
        params["verify_certs"] = True

    client = OpenSearch(**params)
    logger.info("OpenSearch client initialized")
    return client


def wait_for_opensearch(client: OpenSearch, host: str, port: int, retries: int, backoff_sec: int) -> None:
    last_err: Optional[Exception] = None
    for attempt in range(1, int(retries) + 1):
        try:
            if client.ping():
                return
            last_err = RuntimeError("OpenSearch ping returned False")
        except Exception as e:  # noqa: BLE001
            last_err = e
        logger.warning(
            "OpenSearch connection attempt %d/%d to %s:%s failed: %s; retrying in %ds",
            attempt,
            retries,
            host,
            port,
            last_err,
            backoff_sec,
        )
        time.sleep(int(backoff_sec))
    raise RuntimeError(f"Failed to connect to OpenSearch at {host}:{port} after {retries} attempts: {last_err}")


# -----------------------------
# Helpers
# -----------------------------

def sanitize_index_name(name: str) -> str:
    name = name.lower()
    name = re.sub(r"[^a-z0-9_+\-]", "-", name)
    if not name or name in {".", ".."} or name[0] in {"-", "_", "+"}:
        name = f"idx-{name.lstrip('-_+').strip('.') or 'default'}"
    return name[:255]


def detect_vector_fields(fields: List[Dict[str, Any]]) -> List[Tuple[str, Optional[int], DataType]]:
    vectors: List[Tuple[str, Optional[int], DataType]] = []
    for f in fields:
        dtype = f.get("type")
        if dtype in (DataType.FLOAT_VECTOR, DataType.BINARY_VECTOR):
            params = f.get("params", {}) or {}
            dim = params.get("dim") or params.get("dimension")
            try:
                dim = int(dim) if dim is not None else None
            except Exception:
                dim = None
            vectors.append((f.get("name"), dim, dtype))
    return vectors


def detect_primary_key_field(fields: List[Dict[str, Any]]) -> Optional[str]:
    for f in fields:
        if f.get("is_primary") or f.get("is_primary_key"):
            return f.get("name")
    return None


def l2_normalize(vec: List[float]) -> List[float]:
    norm = math.sqrt(sum(float(x) * float(x) for x in vec))
    if norm == 0.0:
        return vec
    return [float(x) / norm for x in vec]


def _binary_bytes_to_bit_floats(value: Any, expected_dim: Optional[int]) -> Optional[List[float]]:
    b: Optional[bytes] = None
    if isinstance(value, (bytes, bytearray)):
        b = bytes(value)
    elif hasattr(value, "tobytes"):
        try:
            b = value.tobytes()
        except Exception:
            b = None
    elif isinstance(value, (list, tuple)):
        try:
            b = bytes(value)
        except Exception:
            b = None
    if b is None:
        return None

    bits: List[float] = []
    for by in b:
        for i in range(7, -1, -1):
            bits.append(float((by >> i) & 1))
    if expected_dim is not None:
        d = int(expected_dim)
        if len(bits) < d:
            bits.extend([0.0] * (d - len(bits)))
        elif len(bits) > d:
            bits = bits[:d]
    return bits


# -----------------------------
# Validation: Count + Search
# -----------------------------

def get_opensearch_doc_by_id(os_client: OpenSearch, index: str, _id: str) -> Optional[Dict[str, Any]]:
    try:
        res = os_client.get(index=index, id=_id, ignore=[404])
        if res and res.get("found"):
            return res.get("_source", {})
        return None
    except os_exceptions.NotFoundError:
        return None
    except Exception as e:  # noqa: BLE001
        logger.error("[%s] Error fetching document by _id %s: %s", index, _id, e)
        return None


def run_milvus_search(milvus_client: MilvusClient, coll_name: str, vector_field: str, query_vec: List[float], k: int,
                       pk_field: Optional[str], metric: str, milvus_search_params: Optional[Dict[str, Any]]) -> List[str]:
    params = dict(milvus_search_params or {})
    if "metric_type" not in (params or {}):
        params["metric_type"] = metric.upper()
    try:
        res = milvus_client.search(
            collection_name=coll_name,
            data=[query_vec],
            limit=int(k),
            output_fields=[pk_field] if pk_field else [],
            search_params=params,
        )
        hits: List[Dict[str, Any]] = []
        if isinstance(res, list) and res:
            first = res[0]
            if isinstance(first, list):
                hits = first
            elif isinstance(first, dict):
                hits = res
        ids: List[str] = []
        for h in hits:
            if isinstance(h, dict):
                if pk_field and pk_field in h:
                    ids.append(str(h[pk_field]))
                elif "id" in h:
                    ids.append(str(h["id"]))
        return ids
    except Exception as e:
        logger.error("[%s] Milvus search error: %s", coll_name, e)
        return []


def run_opensearch_knn(os_client: OpenSearch, index: str, vector_field: str, query_vec: List[float], k: int, num_candidates: int) -> List[str]:
    vec = [float(x) for x in query_vec]
    bodies: List[Dict[str, Any]] = [
        {"size": int(k), "query": {"knn": {vector_field: {"vector": vec, "k": int(k), "num_candidates": int(num_candidates)}}}},
        {"size": int(k), "query": {"knn": {vector_field: {"vector": vec, "k": int(k)}}}},
        {"size": int(k), "query": {"knn": {"field": vector_field, "query_vector": vec, "k": int(k), "num_candidates": int(num_candidates)}}},
        {"size": int(k), "query": {"knn": {"field": vector_field, "query_vector": vec, "k": int(k)}}},
    ]
    last_err: Optional[Exception] = None
    for body in bodies:
        try:
            res = os_client.search(index=index, body=body)
            hits = res.get("hits", {}).get("hits", [])
            return [str(h.get("_id")) for h in hits]
        except Exception as e:
            last_err = e
            continue
    logger.error("[%s] OpenSearch knn search error after trying multiple query formats: %s", index, last_err)
    return []


def compare_rankings(milvus_ids: List[str], os_ids: List[str], k: int) -> Tuple[float, float]:
    if not milvus_ids or not os_ids:
        return 0.0, 0.0
    set_os = set(os_ids[:k])
    inter = sum(1 for x in milvus_ids[:k] if x in set_os)
    recall_at_k = inter / float(k)
    top1 = 1.0 if (milvus_ids[0] == os_ids[0] if milvus_ids and os_ids else False) else 0.0
    return recall_at_k, top1


def validate_collection(milvus_client: MilvusClient, os_client: OpenSearch,
                        coll_name: str, os_index: str,
                        sample_size: int = 10,
                        do_search_validation: bool = False,
                        search_queries: int = 50,
                        search_k: int = 10,
                        os_num_candidates: int = 100,
                        metric: str = "L2",
                        normalize_cosine: bool = False,
                        milvus_search_params: Optional[Dict[str, Any]] = None) -> bool:
    logger = logging.getLogger("validate_migration")

    # Schema detection
    try:
        desc = milvus_client.describe_collection(collection_name=coll_name)
        fields = desc.get("fields", [])
        vector_fields = detect_vector_fields(fields)
        pk_field = detect_primary_key_field(fields)
        logger.info("[%s] vector_fields=%s pk=%s", coll_name, [(n, d, int(dt)) for (n, d, dt) in vector_fields], pk_field)
    except Exception as e:
        logger.error("[%s] Failed to describe collection: %s", coll_name, e)
        return False

    # Milvus count
    try:
        milvus_count = int(milvus_client.get_collection_stats(coll_name).get("row_count", 0))
    except Exception as e:
        logger.error("[%s] Failed to get Milvus row count: %s", coll_name, e)
        return False

    # Ensure OpenSearch index exists and force refresh for accurate count
    try:
        if not os_client.indices.exists(index=os_index):
            logger.error("[%s] OpenSearch index '%s' does not exist", coll_name, os_index)
            return False
        os_client.indices.refresh(index=os_index)
        os_count = int(os_client.count(index=os_index)["count"])
    except Exception as e:
        logger.error("[%s] Failed to count/refresh OpenSearch index: %s", coll_name, e)
        return False

    logger.info("[%s] Milvus count: %d, OpenSearch count: %d", coll_name, milvus_count, os_count)

    if milvus_count != os_count:
        logger.warning("[%s] Document count mismatch!", coll_name)

    # Sample-by-id validation
    try:
        milvus_docs = milvus_client.query(
            collection_name=coll_name,
            output_fields=["*"],
            limit=max(sample_size, search_queries if do_search_validation else sample_size),
        )
    except Exception as e:
        logger.error("[%s] Failed to query sample from Milvus: %s", coll_name, e)
        return False

    if not milvus_docs:
        logger.info("[%s] No documents returned for sampling; treating as pass if counts are zero.", coll_name)
        by_id_ok = (milvus_count == os_count)
    else:
        by_id_ok = True
        for doc in milvus_docs[:sample_size]:
            if pk_field and pk_field in doc:
                os_doc = get_opensearch_doc_by_id(os_client, os_index, str(doc[pk_field]))
                if not os_doc:
                    logger.warning("[%s] Document with _id=%s not found in OpenSearch", coll_name, doc[pk_field])
                    by_id_ok = False
                    break
        if by_id_ok:
            logger.info("[%s] Sample documents matched successfully by _id.", coll_name)

    # Optional: search validation per vector field
    search_ok = True
    if do_search_validation and milvus_docs and os_count > 0 and milvus_count > 0 and vector_fields:
        for (field_name, vec_dim, vec_dtype) in vector_fields:
            if vec_dim is None:
                logger.warning("[%s] Skipping field '%s' due to unknown dimension", coll_name, field_name)
                continue
            field_metric = "HAMMING" if vec_dtype == DataType.BINARY_VECTOR else metric.upper()
            queries_used = 0
            recalls: List[float] = []
            top1s: List[float] = []
            for doc in milvus_docs:
                if queries_used >= search_queries:
                    break
                if field_name not in doc:
                    continue
                qv = doc[field_name]
                # Prepare query vector
                if vec_dtype == DataType.FLOAT_VECTOR:
                    if hasattr(qv, "tolist"):
                        try:
                            qv = qv.tolist()
                        except Exception:
                            continue
                    try:
                        qv = [float(x.item() if hasattr(x, "item") else x) for x in (qv or [])]
                    except Exception:
                        continue
                    if len(qv) != int(vec_dim):
                        continue
                    if normalize_cosine and field_metric == "COSINE":
                        qv = l2_normalize(qv)
                else:
                    # binary
                    qv = _binary_bytes_to_bit_floats(qv, vec_dim)
                    if qv is None or len(qv) != int(vec_dim):
                        continue

                # Milvus search
                milvus_ids = run_milvus_search(milvus_client, coll_name, field_name, qv, search_k, pk_field, field_metric, milvus_search_params)
                # OpenSearch knn
                os_ids = run_opensearch_knn(os_client, os_index, field_name, qv, search_k, os_num_candidates)

                if milvus_ids and os_ids:
                    recall_k, top1 = compare_rankings(milvus_ids, os_ids, search_k)
                    recalls.append(recall_k)
                    top1s.append(top1)
                    queries_used += 1

            if queries_used == 0:
                logger.warning("[%s] Field '%s': No valid query vectors available for search validation.", coll_name, field_name)
                search_ok = False
            else:
                avg_recall = sum(recalls) / len(recalls) if recalls else 0.0
                avg_top1 = sum(top1s) / len(top1s) if top1s else 0.0
                logger.info("[%s] Field '%s' (%s) search validation over %d queries: avg recall@%d=%.3f, top1=%.3f",
                            coll_name, field_name, ("BINARY" if vec_dtype == DataType.BINARY_VECTOR else "FLOAT"),
                            queries_used, search_k, avg_recall, avg_top1)
                if avg_recall < 0.9:
                    logger.warning("[%s] Field '%s': Low recall@%d: %.3f", coll_name, field_name, search_k, avg_recall)
                    search_ok = False

    return by_id_ok and (not do_search_validation or search_ok) and (milvus_count == os_count)


# -----------------------------
# Orchestration
# -----------------------------

def validate_all(milvus_cfg: Dict[str, Any], os_cfg: Dict[str, Any], os_index_prefix: str,
                 collections_filter: Optional[List[str]] = None,
                 do_search_validation: bool = False,
                 search_queries: int = 50,
                 search_k: int = 10,
                 os_num_candidates: int = 100,
                 metric: str = "L2",
                 normalize_cosine: bool = False,
                 milvus_search_params: Optional[Dict[str, Any]] = None) -> None:
    logger.info("Connecting to Milvus and OpenSearch for validation...")

    # Milvus connect with retries
    try:
        milvus_client = MilvusClient(
            uri=milvus_cfg["uri"],
            user=milvus_cfg.get("user"),
            password=milvus_cfg.get("password"),
            db_name=milvus_cfg.get("db", "default"),
            secure=bool(milvus_cfg.get("secure", False)),
            server_pem_path=milvus_cfg.get("pem_path"),
            server_name=milvus_cfg.get("server"),
        )
        retries = int(milvus_cfg.get("connect_retries", 5))
        backoff = int(milvus_cfg.get("connect_backoff_sec", 3))
        for attempt in range(1, retries + 1):
            try:
                collections = milvus_client.list_collections()
                logger.info("Milvus connected; found %d collections", len(collections))
                break
            except Exception as e:  # noqa: BLE001
                if attempt >= retries:
                    raise
                logger.warning("Milvus connection attempt %d/%d failed: %s; retrying in %ds", attempt, retries, e, backoff)
                time.sleep(backoff)
    except Exception as e:
        logger.error("Failed to connect to Milvus after retries: %s", e)
        sys.exit(1)

    os_client = create_opensearch_client(os_cfg)
    try:
        wait_for_opensearch(os_client, str(os_cfg.get("host")), int(os_cfg.get("port", 9200)), int(os_cfg.get("connect_retries", 5)), int(os_cfg.get("connect_backoff_sec", 3)))
    except Exception as e:
        logger.error("Failed to connect to OpenSearch: %s", e)
        sys.exit(1)

    # Optionally filter collections
    if collections_filter:
        collections = [c for c in collections if c in collections_filter]
        logger.info("Filtered collections: %s", collections)

    all_passed = True
    for coll in collections:
        os_index = sanitize_index_name(f"{os_index_prefix}_{coll}")
        logger.info("Validating collection: %s -> index: %s", coll, os_index)
        passed = validate_collection(
            milvus_client, os_client, coll, os_index,
            sample_size=int(os.getenv("SAMPLE_SIZE", "10")),
            do_search_validation=do_search_validation,
            search_queries=search_queries,
            search_k=search_k,
            os_num_candidates=os_num_candidates,
            metric=metric,
            normalize_cosine=normalize_cosine,
            milvus_search_params=milvus_search_params,
        )
        if not passed:
            all_passed = False

    if all_passed:
        logger.info("All collections validated successfully.")
    else:
        logger.warning("Some collections failed validation.")


# -----------------------------
# CLI
# -----------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Validate migration from Milvus to OpenSearch")

    # Milvus
    p.add_argument("--milvus-uri", default=os.getenv("MILVUS_URI", "http://localhost:19530"))
    p.add_argument("--milvus-user", default=os.getenv("MILVUS_USER"))
    p.add_argument("--milvus-password", default=os.getenv("MILVUS_PASSWORD"))
    p.add_argument("--milvus-db", default=os.getenv("MILVUS_DB", "default"))
    p.add_argument("--milvus-secure", action="store_true", default=os.getenv("MILVUS_SECURE", "false").lower() == "true")
    p.add_argument("--milvus-pem-path", default=os.getenv("MILVUS_PEM_PATH"))
    p.add_argument("--milvus-server-name", default=os.getenv("MILVUS_SERVER_NAME"))

    # OpenSearch
    p.add_argument("--os-host", default=os.getenv("OS_HOST", "localhost"))
    p.add_argument("--os-port", type=int, default=int(os.getenv("OS_PORT", "9200")))
    p.add_argument("--os-user", default=os.getenv("OS_USER"))
    p.add_argument("--os-password", default=os.getenv("OS_PASSWORD"))
    p.add_argument("--os-use-ssl", action="store_true", default=os.getenv("OS_USE_SSL", "false").lower() == "true")
    p.add_argument("--os-verify-certs", action="store_true", default=os.getenv("OS_VERIFY_CERTS", "false").lower() == "true")
    p.add_argument("--os-ca-certs", default=os.getenv("OS_CA_CERTS"))
    p.add_argument("--os-timeout", type=int, default=int(os.getenv("OS_TIMEOUT", "60")))
    p.add_argument("--os-max-retries", type=int, default=int(os.getenv("OS_MAX_RETRIES", "3")))
    p.add_argument("--os-connect-retries", type=int, default=int(os.getenv("OS_CONNECT_RETRIES", "5")))
    p.add_argument("--os-connect-backoff", type=int, default=int(os.getenv("OS_CONNECT_BACKOFF", "3")))

    # Validation options
    p.add_argument("--index-prefix", default=os.getenv("OS_INDEX_PREFIX", "milvus"))
    p.add_argument("--collections", default=os.getenv("COLLECTIONS"), help="Comma-separated list of collections to validate")
    p.add_argument("--sample-size", type=int, default=int(os.getenv("SAMPLE_SIZE", "10")))
    p.add_argument("--log-level", default=os.getenv("LOG_LEVEL", "INFO"))

    # Search validation options
    p.add_argument("--validate-search", action="store_true", default=os.getenv("VALIDATE_SEARCH", "false").lower() == "true")
    p.add_argument("--search-queries", type=int, default=int(os.getenv("SEARCH_QUERIES", "50")))
    p.add_argument("--search-k", type=int, default=int(os.getenv("SEARCH_K", "10")))
    p.add_argument("--os-num-candidates", type=int, default=int(os.getenv("OS_NUM_CANDIDATES", "100")))
    p.add_argument("--metric", default=os.getenv("METRIC", "L2"))
    p.add_argument("--normalize-cosine", action="store_true", default=os.getenv("NORMALIZE_COSINE", "false").lower() == "true")
    p.add_argument("--milvus-search-params", default=os.getenv("MILVUS_SEARCH_PARAMS"), help="JSON string of Milvus search params, e.g., '{\"metric_type\": \"L2\", \"params\": {\"ef\": 128}}'")

    return p.parse_args()


def main() -> None:
    args = parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO),
                        format="%(asctime)s %(levelname)s %(name)s:%(message)s")

    milvus_cfg = {
        "uri": args.milvus_uri,
        "user": args.milvus_user,
        "password": args.milvus_password,
        "db": args.milvus_db,
        "secure": args.milvus_secure,
        "pem_path": args.milvus_pem_path,
        "server": args.milvus_server_name,
        "connect_retries": int(os.getenv("MILVUS_CONNECT_RETRIES", "5")),
        "connect_backoff_sec": int(os.getenv("MILVUS_CONNECT_BACKOFF", "3")),
    }

    os_cfg = {
        "host": args.os_host,
        "port": args.os_port,
        "http_auth_username": args.os_user,
        "http_auth_password": args.os_password,
        "use_ssl": args.os_use_ssl,
        "verify_certs": args.os_verify_certs,
        # "ca_certs": args.os_ca_certs,
        "timeout": args.os_timeout,
        "max_retries": args.os_max_retries,
        "connect_retries": args.os_connect_retries,
        "connect_backoff_sec": args.os_connect_backoff,
    }

    collections_filter = [c.strip() for c in args.collections.split(',')] if args.collections else None

    milvus_search_params = None
    if args.milvus_search_params:
        try:
            milvus_search_params = json.loads(args.milvus_search_params)
        except Exception as e:
            logger.warning("Failed to parse --milvus-search-params JSON: %s", e)

    validate_all(
        milvus_cfg=milvus_cfg,
        os_cfg=os_cfg,
        os_index_prefix=args.index_prefix,
        collections_filter=collections_filter,
        do_search_validation=args.validate_search,
        search_queries=args.search_queries,
        search_k=args.search_k,
        os_num_candidates=args.os_num_candidates,
        metric=args.metric,
        normalize_cosine=args.normalize_cosine,
        milvus_search_params=milvus_search_params,
    )


if __name__ == "__main__":
    main()
