# Milvus → OpenSearch Migration Tool

## Overview

This repository provides a **Python-based migration utility** to export data from **Milvus** into **OpenSearch**.

It supports both **scalar** and **vector** fields, ensuring type-safe conversion, JSON compatibility, and KNN-ready index creation.


---
## Features

- Supports Milvus **scalar**, **array**, **JSON**, and **vector** data types  
- Automatically creates OpenSearch indices with correct field mappings  
- Converts Milvus data to **JSON-serializable** and OpenSearch-compatible formats  
- Handles **FLOAT_VECTOR**, **BINARY_VECTOR**, and other KNN types  
- Efficient bulk ingestion with retries  
- Built-in logging and progress tracking (`tqdm`)

---

## Supported Milvus Data Types

| Category | Milvus DataType | OpenSearch Mapping | Status |
|-----------|------------------|--------------------|---------|
| Boolean | `BOOL` | `boolean` | ✅ |
| Integer | `INT8`, `INT16`, `INT32` | `integer` | ✅ |
| Long | `INT64` | `long` | ✅ |
| Floating point | `FLOAT`, `DOUBLE` | `float`, `double` | ✅ |
| Text | `VARCHAR`, `STRING` | `text + keyword` | ✅ |
| Arrays | `ARRAY` | element-type inferred (int/float/bool/keyword) | ✅ |
| JSON | `JSON` | `object`, `dynamic: true` | ✅ |
| Vectors | `FLOAT_VECTOR`, `FLOAT16_VECTOR`, `BFLOAT16_VECTOR`, `BINARY_VECTOR` | `knn_vector` | ✅ |
| Nulls | `None` | Skipped | ✅ |


⚠️ *No support:* `SPARSE_FLOAT_VECTOR`

---

## Requirements

**Python 3.8+**

Install dependencies:

```bash
pip install pymilvus opensearch-py tqdm numpy
```

---
## Configuration

You can configure the source (Milvus) and destination (OpenSearch) using **CLI arguments** or **environment variables**. CLI arguments take precedence over environment variables if both are set.


### Milvus Configuration

| Environment Variable | CLI Option | Description | Default |
|---------------------|------------|-------------|---------|
| `MILVUS_URI`         | `--milvus-uri` | Milvus server URI, including host and port | `http://localhost:19530` |
| `MILVUS_USER`        | `--milvus-user` | Username for Milvus authentication | None |
| `MILVUS_PASSWORD`    | `--milvus-password` | Password for Milvus authentication | None |
| `MILVUS_DB`          | `--milvus-db` | Database to migrate. Use `all` to migrate all databases | `default` |
| `MILVUS_SECURE`      | `--milvus-secure` | Enable secure connection (TLS) | `false` |
| `MILVUS_PEM_PATH`    | `--milvus-pem-path` | Path to PEM file for TLS | None |
| `MILVUS_SERVER_NAME` | `--milvus-server-name` | Server name for TLS verification | None |

---

### OpenSearch Configuration

| Environment Variable | CLI Option | Description | Default |
|---------------------|------------|-------------|---------|
| `OS_HOST`           | `--os-host` | OpenSearch host | `localhost` |
| `OS_PORT`           | `--os-port` | OpenSearch port | `9200` |
| `OS_USER`           | `--os-user` | Username for OpenSearch | None |
| `OS_PASSWORD`       | `--os-password` | Password for OpenSearch | None |
| `OS_USE_SSL`        | `--os-use-ssl` | Use SSL for OpenSearch connection | `false` |
| `OS_VERIFY_CERTS`   | `--os-verify-certs` | Verify SSL certificates | `false` |
| `OS_CA_CERTS`       | `--os-ca-certs` | Path to CA certificates file | None |
| `OS_TIMEOUT`        | `--os-timeout` | Timeout for OpenSearch requests (seconds) | `120` |
| `OS_MAX_RETRIES`    | `--os-max-retries` | Maximum retry attempts | `3` |
| `OS_BULK_TIMEOUT`   | `--os-bulk-timeout` | Bulk insert timeout (seconds) | `300` |

### General Options

| Environment Variable | CLI Option | Description | Default |
|---------------------|------------|-------------|---------|
| `OS_INDEX_PREFIX`    | `--index-prefix` | Prefix for OpenSearch indices | `milvus` |
| `BATCH_SIZE`         | `--batch-size` | Number of documents to migrate per batch | `500` |
| `COLLECTIONS`        | `--collections` | Comma-separated list of collections to migrate | All collections |
| `MIGRATE_RECREATE_INDICES` | `--recreate-indices` | Recreate indices in OpenSearch | `false` |
| `MIGRATE_ERRORS_DIR` | `--errors-dir` | Directory to store error logs | None |
| `LOG_LEVEL`          | `--log-level` | Logging level (`DEBUG`, `INFO`, `WARNING`, `ERROR`) | `INFO` |

### Examples

**Using environment variables:**

```bash
export MILVUS_URI="http://milvus.example.com:19530"
export MILVUS_USER="<user>"
export MILVUS_PASSWORD="<password>"
export MILVUS_DB="all"
export MILVUS_SECURE="true"
export MILVUS_PEM_PATH="/path/to/cert"
export MILVUS_SERVER_NAME="sample.com"

export OS_HOST="opensearch.example.com"
export OS_USER="admin"
export OS_PASSWORD="admin"
export OS_PORT="9200"
export OS_USE_SSL=True
export OS_VERIFY_CERTS=False
export OS_CA_CERTS=False
export MIGRATE_ERRORS_DIR="/root/openSearch"
export VALIDATE_SEARCH=true

python migrate.py
```

**Using CLI arguments:**
```bash
python migrate.py \
  --milvus-uri http://milvus.example.com:19530 \
  --milvus-db default \
  --os-host opensearch.example.com \
  --os-user admin \
  --os-password admin \
  --batch-size 1000 \
  --collections collection1,collection2
```

---
## Usage
Run the migration script:

```bash
python3 migration.py --config config.json
```

Optional arguments:
```bash
--db <database_name>           # Milvus database name (default: default)
--collection <collection_name> # Single collection to migrate
--batch-size <int>             # Batch size for fetching and indexing (default: 1000)
--verbose                      # Enable debug-level logs
```

---
## How It Works

- Connects to Milvus using pymilvus.MilvusClient

- Fetches all collections (or a specific one)

- Builds schema mappings for OpenSearch using:

- build_index_mapping() for field-to-field compatibility

- Transforms data via:

- make_serializable() to handle JSON, arrays, and numeric conversions

- Creates OpenSearch index with correct KNN settings

- Streams data in batches into OpenSearch using bulk helpers


---
## Core Functions

| Function                     | Description                                   |
| ---------------------------- | --------------------------------------------- |
| `create_opensearch_client()` | Initializes OpenSearch connection             |
| `make_serializable()`        | Converts Milvus data to JSON-compatible types |
| `build_index_mapping()`      | Creates OpenSearch mapping per Milvus schema  |
| `fetch_milvus_data()`        | Retrieves data from Milvus in batches         |
| `migrate_all_collections()`  | Main orchestration function                   |

---
## Logging

Logs are written to console and (optionally) a file:

```bash
[2025-10-08 12:30:15,102] INFO Migration: Created index my_collection
[2025-10-08 12:30:15,203] INFO Indexed 5000 records successfully
```

Enable debug mode for more detail:

```bash
python3 migration.py --verbose
```

---
## Best Practices
- Ensure OpenSearch has KNN plugin enabled for vector fields

- Use smaller batch sizes (500–2000) for large datasets

- Always validate mappings before production ingestion

- If JSON fields contain mixed types, set "dynamic": true in mappings (already enabled)

- You can later restore original OpenSearch index settings (replicas, refresh intervals) after ingestion

---
## Limitations
- SPARSE_FLOAT_VECTOR and VARBINARY are not natively supported by OpenSearch

- ARRAY of vectors (nested embeddings) is not supported

- Ensure consistent field schemas between Milvus and OpenSearch
