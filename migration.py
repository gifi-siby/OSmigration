import os
import re
import sys
import json
import time
import logging
import argparse
from typing import Any, Dict, List, Optional, Tuple

from pymilvus import MilvusClient, DataType
from opensearchpy import OpenSearch, helpers, exceptions as os_exceptions
from tqdm import tqdm


# -----------------------------
# Logging
# -----------------------------
logger = logging.getLogger("migration")


# -----------------------------
# OpenSearch client
# -----------------------------

def create_opensearch_client(cfg: Dict[str, Any]) -> OpenSearch:
    """Create an OpenSearch client with sensible defaults and retries."""
    auth = None
    if cfg.get("http_auth_username") and cfg.get("http_auth_password"):
        auth = (cfg.get("http_auth_username"), cfg.get("http_auth_password"))

    params = {
        "hosts": [{"host": cfg.get("host", "localhost"), "port": int(cfg.get("port", 9200))}],
        "use_ssl": bool(cfg.get("use_ssl", False)),
        "verify_certs": bool(cfg.get("verify_certs", False)),
        "http_compress": True,
        "timeout": int(cfg.get("timeout", 120)),
        "retry_on_timeout": True,
        "max_retries": int(cfg.get("max_retries", 3)),
    }
    if auth:
        params["http_auth"] = auth

    # Optional CA certs path for TLS verification
    if cfg.get("ca_certs"):
        params["ca_certs"] = cfg.get("ca_certs")
        params["verify_certs"] = True

    client = OpenSearch(**params)
    logger.info("OpenSearch client initialized")
    return client


# -----------------------------
# Helpers
# -----------------------------

def sanitize_index_name(name: str) -> str:
    """Sanitize to a valid OpenSearch index name."""
    name = name.lower()
    name = re.sub(r"[^a-z0-9_+\-]", "-", name)
    # Cannot start with '-', '_', '+' and cannot be '.' or '..'
    if not name or name in {".", ".."} or name[0] in {"-", "_", "+"}:
        name = f"idx-{name.lstrip('-_+').strip('.') or 'default'}"
    return name[:255]


def detect_vector_fields(fields: List[Dict[str, Any]]) -> List[Tuple[str, Optional[int], DataType]]:
    """Detect all vector fields: list of (name, dim, dtype)."""
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
    """Detect the primary key field name from Milvus collection fields."""
    for f in fields:
        # Milvus schema may use 'is_primary' or 'is_primary_key'
        if f.get("is_primary") or f.get("is_primary_key"):
            return f.get("name")
    return None


def build_index_mapping(vector_fields: List[Tuple[str, Optional[int], DataType]],
                        fields: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Build an OpenSearch index mapping including all vector fields.

    - FLOAT_VECTOR and BINARY_VECTOR fields are mapped as knn_vector with their bit/float dimension.
    - BINARY vectors will be ingested as 0/1 float vectors to preserve Hamming ranking via L2.
    - Scalars are mapped explicitly where known; strings via dynamic templates.
    """
    mapping: Dict[str, Any] = {
        "settings": {
            "index": {
                "knn": True,
                # Ingestion-time speedups; will be overridden/restored later via put_settings.
                "number_of_replicas": 0,
                "refresh_interval": "-1",
            }
        },
        "mappings": {
            # Dynamic templates: strings as text + keyword
            "dynamic_templates": [
                {
                    "strings_as_text_and_keyword": {
                        "match_mapping_type": "string",
                        "mapping": {
                            "type": "text",
                            "fields": {"keyword": {"type": "keyword", "ignore_above": 256}},
                        },
                    }
                }
            ],
            "properties": {},
        },
    }

    vector_names = set()
    for name, dim, dtype in vector_fields:
        if not name:
            continue
        if dim is None or int(dim) <= 0:
            raise ValueError(f"Missing/invalid dimension for vector field '{name}'")
        vector_names.add(name)
        mapping["mappings"]["properties"][name] = {
            "type": "knn_vector",
            "dimension": int(dim),
        }

    # Explicitly map non-string numeric/boolean types when known.
    milvus_to_os_type = {
        DataType.BOOL: "boolean",
        DataType.INT8: "integer",
        DataType.INT16: "integer",
        DataType.INT32: "integer",
        DataType.INT64: "long",
        DataType.FLOAT: "float",
        DataType.DOUBLE: "double",
        # Strings handled by dynamic_templates (VARCHAR/STRING)
    }

    for f in fields:
        name = f.get("name")
        dtype = f.get("type")
        if not name or name in vector_names:
            continue
        os_type = milvus_to_os_type.get(dtype)
        if os_type:
            mapping["mappings"]["properties"][name] = {"type": os_type}

    return mapping


def make_serializable(value: Any) -> Any:
    """Recursively convert values to JSON-serializable forms without stringifying arrays of numbers."""
    if value is None:
        return None
    if isinstance(value, dict):
        return {k: make_serializable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [make_serializable(v) for v in value]
    try:
        json.dumps(value)
        return value
    except TypeError:
        if hasattr(value, "tolist"):
            try:
                return make_serializable(value.tolist())
            except Exception:
                pass
        if hasattr(value, "item"):
            try:
                return value.item()
            except Exception:
                pass
        return str(value)


def _convert_vector(value: Any, expected_dim: Optional[int]) -> Optional[List[float]]:
    """Convert potentially numpy-based sequence to a list[float] with expected dimension."""
    v = value
    if hasattr(v, "tolist"):
        try:
            v = v.tolist()
        except Exception:
            pass
    if isinstance(v, (list, tuple)):
        out: List[float] = []
        for elem in v:
            try:
                if hasattr(elem, "item"):
                    elem = elem.item()
                out.append(float(elem))
            except Exception:
                return None
        if expected_dim is not None and len(out) != int(expected_dim):
            return None
        return out
    return None


def _binary_bytes_to_bit_floats(value: Any, expected_dim: Optional[int]) -> Optional[List[float]]:
    """Convert Milvus binary vector representation to 0/1 float list of length expected_dim.

    Accepts bytes/bytearray or sequences convertible to bytes. Uses big-endian bit order.
    """
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


def doc_to_source(doc: Dict[str, Any], vector_fields: List[Tuple[str, Optional[int], DataType]]) -> Optional[Dict[str, Any]]:
    result: Dict[str, Any] = {}
    float_dims = {name: dim for name, dim, dt in vector_fields if dt == DataType.FLOAT_VECTOR and name}
    binary_dims = {name: dim for name, dim, dt in vector_fields if dt == DataType.BINARY_VECTOR and name}
    for k, v in doc.items():
        if k in float_dims:
            vec = _convert_vector(v, float_dims[k])
            if vec is None:
                return None
            result[k] = vec
        elif k in binary_dims:
            vecb = _binary_bytes_to_bit_floats(v, binary_dims[k])
            if vecb is None:
                return None
            result[k] = vecb
        else:
            result[k] = make_serializable(v)
    return result


# -----------------------------
# Additional Milvus helpers
# -----------------------------

def list_all_databases_with_fallback(milvus_client) -> List[str]:
    """List all databases via MilvusClient or fallback to pymilvus.db.

    Returns a non-empty list with at least ['default'] on failure.
    """
    # Try MilvusClient methods first
    for attr in ("list_databases", "list_database"):
        try:
            method = getattr(milvus_client, attr, None)
            if callable(method):
                res = method()
                if isinstance(res, dict):
                    for key in ("db_names", "databases", "names"):
                        if key in res and isinstance(res[key], list):
                            vals = res[key]
                            return [str(x) for x in vals] if vals else ["default"]
                if isinstance(res, list) and res:
                    return [str(x) for x in res]
        except Exception:
            continue

    # Fallback to pymilvus.db module
    try:
        from pymilvus import db as milvus_db
        for attr in ("list_databases", "list_database"):
            try:
                method = getattr(milvus_db, attr, None)
                if callable(method):
                    res = method()
                    if isinstance(res, dict):
                        for key in ("db_names", "databases", "names"):
                            if key in res and isinstance(res[key], list):
                                vals = res[key]
                                return [str(x) for x in vals] if vals else ["default"]
                    if isinstance(res, list) and res:
                        return [str(x) for x in res]
            except Exception:
                continue
    except Exception:
        pass

    return ["default"]


# -----------------------------
# Migration
# -----------------------------

def migrate_all_collections(milvus_cfg: Dict[str, Any], os_cfg: Dict[str, Any],
                            os_index_prefix: str,
                            collections_filter: Optional[List[str]] = None,
                            batch_size: int = 500,
                            recreate_indices: bool = False,
                            errors_dir: Optional[str] = None) -> None:
    logger.info("Connecting to Milvus...")
    try:
        # Create a temporary client to discover databases
        temp_client = MilvusClient(
            uri=milvus_cfg["uri"],
            user=milvus_cfg.get("user"),
            password=milvus_cfg.get("password"),
            db_name=milvus_cfg.get("db", "default"),
            secure=bool(milvus_cfg.get("secure", False)),
            server_pem_path=milvus_cfg.get("pem_path"),
            server_name=milvus_cfg.get("server"),
        )
        configured_db = (milvus_cfg.get("db") or "default")
        if isinstance(configured_db, str) and configured_db.lower() == "all":
            databases = list_all_databases_with_fallback(temp_client)
        else:
            databases = [configured_db]
        if not databases:
            databases = ["default"]
        logger.info("Milvus connected; databases to process: %s", databases)
    except Exception as e:
        logger.error("Failed to connect to Milvus: %s", e)
        sys.exit(1)

    os_client = create_opensearch_client(os_cfg)

    include_db_in_index_name = len(databases) > 1

    for db_name in databases:
        logger.info("Switching to database: %s", db_name)
        try:
            milvus_client = MilvusClient(
                uri=milvus_cfg["uri"],
                user=milvus_cfg.get("user"),
                password=milvus_cfg.get("password"),
                db_name=db_name,
                secure=bool(milvus_cfg.get("secure", False)),
                server_pem_path=milvus_cfg.get("pem_path"),
                server_name=milvus_cfg.get("server"),
            )
            collections = milvus_client.list_collections()
            logger.info("Database '%s' connected; found %d collections", db_name, len(collections))
        except Exception as e:
            logger.error("[%s] Failed to connect or list collections: %s", db_name, e)
            continue

        # Optionally filter collections
        if collections_filter:
            filtered = [c for c in collections if c in collections_filter]
            logger.info("[%s] Filtered collections: %s", db_name, filtered)
            collections = filtered

        for coll_name in collections:
            logger.info("Processing collection: %s.%s", db_name, coll_name)
            try:
                desc = milvus_client.describe_collection(collection_name=coll_name)
                fields = desc.get("fields", [])

                vector_fields = detect_vector_fields(fields)
                if not vector_fields:
                    logger.warning("[%s.%s] No vector fields found—skipping.", db_name, coll_name)
                    continue

                pk_field = detect_primary_key_field(fields)
                logger.info("[%s.%s] Detected %d vector field(s); PK field: %s", db_name, coll_name, len(vector_fields), pk_field)

                # Ensure the collection is loaded before querying
                try:
                    milvus_client.load_collection(collection_name=coll_name)
                except Exception as e:
                    logger.debug("[%s.%s] load_collection ignored: %s", db_name, coll_name, e)

                # Probe for data presence
                try:
                    sample = milvus_client.query(
                        collection_name=coll_name,
                        output_fields=["*"],
                        limit=1
                    )
                except Exception:
                    sample = []
                if not sample:
                    logger.warning("[%s.%s] No data to migrate—skipping.", db_name, coll_name)
                    continue
            except Exception as e:
                logger.error("[%s.%s] Schema detection failed: %s", db_name, coll_name, e)
                continue

            # Include database name in the index when processing multiple databases to avoid collisions
            if include_db_in_index_name:
                os_index = sanitize_index_name(f"{os_index_prefix}_{db_name}_{coll_name}")
            else:
                os_index = sanitize_index_name(f"{os_index_prefix}_{coll_name}")

            try:
                mapping = build_index_mapping(vector_fields, fields)
            except Exception as e:
                logger.error("[%s.%s] Mapping build failed: %s", db_name, coll_name, e)
                continue

            try:
                if recreate_indices:
                    try:
                        if os_client.indices.exists(index=os_index):
                            logger.info("[%s.%s] Deleting existing index '%s' before recreation", db_name, coll_name, os_index)
                            os_client.indices.delete(index=os_index)
                            # Wait for deletion to propagate
                            time.sleep(1)
                    except Exception as e:
                        logger.warning("[%s.%s] Failed checking/deleting existing index '%s': %s", db_name, coll_name, os_index, e)
                # Create index if not exists; ignore 400 (already exists)
                os_client.indices.create(index=os_index, body=mapping, ignore=400)
                logger.info("[%s.%s] OpenSearch index '%s' created or exists.", db_name, coll_name, os_index)
            except os_exceptions.OpenSearchException as e:
                logger.error("[%s.%s] Index creation failed: %s", db_name, coll_name, e)
                continue

            # Capture existing index settings to restore later
            try:
                settings_before = os_client.indices.get_settings(index=os_index)
                original_settings = settings_before.get(os_index, {}).get("settings", {}).get("index", {})
            except Exception:
                original_settings = {}

            # Speed up ingestion: disable refresh and replicas
            try:
                os_client.indices.put_settings(index=os_index, body={
                    "index": {
                        "refresh_interval": "-1",
                        "number_of_replicas": 0,
                    }
                })
            except Exception as e:
                logger.warning("[%s.%s] Failed to adjust index settings for ingestion speed: %s", db_name, coll_name, e)

            # Rows and iterator
            try:
                stats = milvus_client.get_collection_stats(coll_name)
                total = int(stats.get("row_count", 0))
            except Exception:
                total = 0

            # Adjust batch size for very high-dimensional vectors
            effective_batch_size = int(milvus_cfg.get("batch_size", batch_size))
            try:
                max_dim = max(int(d or 0) for _, d, _ in vector_fields)
            except Exception:
                max_dim = 0
            if max_dim > 512 and effective_batch_size > 200:
                effective_batch_size = 200

            # Ensure the collection is loaded before creating iterator
            try:
                milvus_client.load_collection(collection_name=coll_name)
            except Exception as e:
                logger.debug("[%s.%s] load_collection before iterator ignored: %s", db_name, coll_name, e)

            try:
                iterator = milvus_client.query_iterator(
                    collection_name=coll_name,
                    batch_size=effective_batch_size,
                    output_fields=["*"],
                )
            except Exception as e:
                logger.error("[%s.%s] Failed to create query iterator: %s", db_name, coll_name, e)
                continue

            pbar = tqdm(total=total or None, desc=f"Migrating {db_name}.{coll_name}", unit="docs")
            total_indexed = 0
            total_errors = 0
            start = time.time()

            try:
                # Prepare errors directory if requested
                if errors_dir:
                    try:
                        os.makedirs(errors_dir, exist_ok=True)
                    except Exception as e:
                        logger.warning("[%s.%s] Failed to create errors_dir '%s': %s", db_name, coll_name, errors_dir, e)

                while True:
                    batch = iterator.next()
                    if not batch:
                        break

                    actions = []
                    for doc in batch:
                        src = doc_to_source(doc, vector_fields)
                        if src is None:
                            logger.warning("[%s.%s] Skipping doc due to non-serializable or invalid vector field(s)", db_name, coll_name)
                            continue
                        action: Dict[str, Any] = {
                            "_index": os_index,
                            "_source": src,
                        }
                        # Use primary key as OpenSearch _id if available
                        if pk_field and pk_field in doc:
                            action["_id"] = str(doc[pk_field])
                        actions.append(action)

                    # Bulk index with error capture
                    try:
                        success_count, errors = helpers.bulk(
                            os_client,
                            actions,
                            chunk_size=effective_batch_size,
                            request_timeout=int(os_cfg.get("bulk_timeout", 300)),
                            raise_on_error=False,
                        )
                        total_indexed += int(success_count or 0)
                        if errors:
                            total_errors += len(errors)
                            # Write errors to JSONL for offline debugging
                            if errors_dir:
                                try:
                                    err_path = os.path.join(errors_dir, f"{os_index}_errors.jsonl")
                                    with open(err_path, "a", encoding="utf-8") as f:
                                        for err in errors:
                                            f.write(json.dumps(err) + "\n")
                                    logger.warning("[%s.%s] Wrote %d bulk errors to %s", db_name, coll_name, len(errors), err_path)
                                except Exception as e:  # noqa: BLE001
                                    logger.warning("[%s.%s] Failed to write bulk errors: %s", db_name, coll_name, e)
                            else:
                                # Log a compact summary at WARNING
                                logger.warning("[%s.%s] Bulk errors encountered: %d (enable errors_dir to persist)", db_name, coll_name, len(errors))
                    except Exception as e:
                        logger.error("[%s.%s] Bulk indexing error: %s", db_name, coll_name, e)
                        total_errors += len(actions)

                    pbar.update(len(batch))

                logger.info("[%s.%s] Migration complete. Indexed=%d Errors=%d Time=%.2fs", db_name, coll_name, total_indexed, total_errors, time.time() - start)
            except StopIteration:
                logger.info("[%s.%s] Finished all batches.", db_name, coll_name)
            except Exception as e:
                logger.error("[%s.%s] Error during iteration: %s", db_name, coll_name, e)
            finally:
                try:
                    iterator.close()
                except Exception:
                    pass
                pbar.close()

            # Restore index settings and refresh
            try:
                restore_settings: Dict[str, Any] = {"index": {}}
                # Restore refresh interval
                if "refresh_interval" in original_settings:
                    restore_settings["index"]["refresh_interval"] = original_settings["refresh_interval"]
                else:
                    restore_settings["index"]["refresh_interval"] = "1s"
                # Restore number_of_replicas
                if "number_of_replicas" in original_settings:
                    restore_settings["index"]["number_of_replicas"] = int(original_settings["number_of_replicas"])  # type: ignore
                else:
                    restore_settings["index"]["number_of_replicas"] = 1

                os_client.indices.put_settings(index=os_index, body=restore_settings)
                os_client.indices.refresh(index=os_index)
                logger.info("[%s.%s] Index settings restored and refreshed.", db_name, coll_name)
            except Exception as e:
                logger.warning("[%s.%s] Failed to restore index settings or refresh: %s", db_name, coll_name, e)


# -----------------------------
# CLI
# -----------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Migrate collections from Milvus to OpenSearch")

    # Milvus
    p.add_argument("--milvus-uri", default=os.getenv("MILVUS_URI", "http://localhost:19530"))
    p.add_argument("--milvus-user", default=os.getenv("MILVUS_USER"))
    p.add_argument("--milvus-password", default=os.getenv("MILVUS_PASSWORD"))
    p.add_argument("--milvus-db", default=os.getenv("MILVUS_DB", "default"), help="Database to migrate. Use 'all' to migrate all databases.")
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
    p.add_argument("--os-timeout", type=int, default=int(os.getenv("OS_TIMEOUT", "120")))
    p.add_argument("--os-max-retries", type=int, default=int(os.getenv("OS_MAX_RETRIES", "3")))
    p.add_argument("--os-bulk-timeout", type=int, default=int(os.getenv("OS_BULK_TIMEOUT", "300")))

    # General
    p.add_argument("--index-prefix", default=os.getenv("OS_INDEX_PREFIX", "milvus"))
    p.add_argument("--batch-size", type=int, default=int(os.getenv("BATCH_SIZE", "500")))
    p.add_argument("--collections", default=os.getenv("COLLECTIONS"), help="Comma-separated list of collections to migrate")
    p.add_argument("--recreate-indices", action="store_true", default=os.getenv("MIGRATE_RECREATE_INDICES", "false").lower() == "true")
    p.add_argument("--errors-dir", default=os.getenv("MIGRATE_ERRORS_DIR"))
    p.add_argument("--log-level", default=os.getenv("LOG_LEVEL", "INFO"))

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
        "batch_size": args.batch_size,
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
        "bulk_timeout": args.os_bulk_timeout,
    }

    collections_filter = [c.strip() for c in args.collections.split(',')] if args.collections else None

    migrate_all_collections(
        milvus_cfg=milvus_cfg,
        os_cfg=os_cfg,
        os_index_prefix=args.index_prefix,
        collections_filter=collections_filter,
        batch_size=args.batch_size,
        recreate_indices=args.recreate_indices,
        errors_dir=args.errors_dir,
    )


if __name__ == "__main__":
    main()
