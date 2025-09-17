from pymilvus import (
    connections,
    FieldSchema, CollectionSchema, DataType,
    Collection,
    utility,
)
import numpy as np
import random

# Detect numpy bfloat16 support
try:
    BFLOAT16_DTYPE = np.dtype('bfloat16')  # type: ignore
    HAS_BFLOAT16 = True
except Exception:
    HAS_BFLOAT16 = False

# Detect SciPy sparse support
try:
    import scipy.sparse as sp  # type: ignore
    HAS_SCIPY = True
except Exception:
    HAS_SCIPY = False

# Connect to Milvus server (change host/port if needed)
connections.connect(
    uri='https://admin:V2BXxbOY3E2g@ibm-lh-lakehouse-milvus536.milvus.apps.enginecluster-aurora-3.cp.fyre.ibm.com:443',
    user='admin',
    password='V2BXxbOY3E2g',
    server_pem_path='/root/hello_milvus/aurora3.crt',
)


# -----------------------------
# Collections (original)
# -----------------------------

def create_collection_all_types():
    """
    Collection with many scalar types + one float vector + one binary vector
    """
    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=False),
        FieldSchema(name="int8_field", dtype=DataType.INT8),
        FieldSchema(name="int16_field", dtype=DataType.INT16),
        FieldSchema(name="int32_field", dtype=DataType.INT32),
        FieldSchema(name="int64_field", dtype=DataType.INT64),
        FieldSchema(name="float_field", dtype=DataType.FLOAT),
        FieldSchema(name="double_field", dtype=DataType.DOUBLE),
        FieldSchema(name="bool_field", dtype=DataType.BOOL),
        FieldSchema(name="string_field", dtype=DataType.VARCHAR, max_length=100),
        FieldSchema(name="float_vector", dtype=DataType.FLOAT_VECTOR, dim=128),
        FieldSchema(name="binary_vector", dtype=DataType.BINARY_VECTOR, dim=128),
    ]
    schema = CollectionSchema(fields, description="Collection with common field types")
    collection_name = "collection_all_types"
    if utility.has_collection(collection_name):
        Collection(collection_name).drop()
    collection = Collection(name=collection_name, schema=schema)
    return collection


def create_collection_multiple_vectors():
    """
    Collection with multiple vector fields only (float + binary)
    """
    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=False),
        FieldSchema(name="float_vector1", dtype=DataType.FLOAT_VECTOR, dim=64),
        FieldSchema(name="float_vector2", dtype=DataType.FLOAT_VECTOR, dim=128),
        FieldSchema(name="binary_vector1", dtype=DataType.BINARY_VECTOR, dim=256),
    ]
    schema = CollectionSchema(fields, description="Collection with multiple vector fields")
    collection_name = "collection_multiple_vectors"
    if utility.has_collection(collection_name):
        Collection(collection_name).drop()
    collection = Collection(name=collection_name, schema=schema)
    return collection


def create_collection_mixed_data():
    """
    Collection with mixed data: scalar + multiple vector fields + string
    """
    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=False),
        FieldSchema(name="int_field", dtype=DataType.INT32),
        FieldSchema(name="float_vector1", dtype=DataType.FLOAT_VECTOR, dim=128),
        FieldSchema(name="float_vector2", dtype=DataType.FLOAT_VECTOR, dim=64),
        FieldSchema(name="binary_vector", dtype=DataType.BINARY_VECTOR, dim=128),
        FieldSchema(name="string_field", dtype=DataType.VARCHAR, max_length=256),
        FieldSchema(name="bool_field", dtype=DataType.BOOL),
    ]
    schema = CollectionSchema(fields, description="Collection with mixed data types")
    collection_name = "collection_mixed_data"
    if utility.has_collection(collection_name):
        Collection(collection_name).drop()
    collection = Collection(name=collection_name, schema=schema)
    return collection


# -----------------------------
# New: Milvus 2.5.12 all-supported showcase
# -----------------------------

def create_collection_all_supported_25():
    """
    Comprehensive collection demonstrating many Milvus 2.5.12 types:
    - Scalars (ints, float, double, bool, string)
    - JSON
    - ARRAY<float>
    - Multiple FLOAT_VECTORs
    - Multiple BINARY_VECTORs
    - FLOAT16_VECTOR, BFLOAT16_VECTOR (conditionally added if numpy supports bfloat16)
    - SPARSE_FLOAT_VECTOR
    """
    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=False),
        # Scalars
        FieldSchema(name="int8_field", dtype=DataType.INT8),
        FieldSchema(name="int16_field", dtype=DataType.INT16),
        FieldSchema(name="int32_field", dtype=DataType.INT32),
        FieldSchema(name="int64_field", dtype=DataType.INT64),
        FieldSchema(name="float_field", dtype=DataType.FLOAT),
        FieldSchema(name="double_field", dtype=DataType.DOUBLE),
        FieldSchema(name="bool_field", dtype=DataType.BOOL),
        FieldSchema(name="string_field", dtype=DataType.VARCHAR, max_length=256),
        # JSON
        FieldSchema(name="json_field", dtype=DataType.JSON),
        # ARRAY<float>
        FieldSchema(name="float_array", dtype=DataType.ARRAY, element_type=DataType.FLOAT, max_capacity=32),
        # Dense vector types
        FieldSchema(name="float_vector1", dtype=DataType.FLOAT_VECTOR, dim=128),
        FieldSchema(name="float_vector2", dtype=DataType.FLOAT_VECTOR, dim=64),
        FieldSchema(name="float16_vector", dtype=DataType.FLOAT16_VECTOR, dim=128),
    ]
    if HAS_BFLOAT16:
        fields.append(FieldSchema(name="bfloat16_vector", dtype=DataType.BFLOAT16_VECTOR, dim=128))
    # Multiple binary and sparse vectors
    fields.extend([
        FieldSchema(name="binary_vector1", dtype=DataType.BINARY_VECTOR, dim=128),
        FieldSchema(name="binary_vector2", dtype=DataType.BINARY_VECTOR, dim=256),
    ])
    if HAS_SCIPY:
        fields.append(FieldSchema(name="sparse_vector", dtype=DataType.SPARSE_FLOAT_VECTOR))

    schema = CollectionSchema(fields, description="All supported types showcase for Milvus 2.5.12")
    collection_name = "collection_all_supported_25"
    if utility.has_collection(collection_name):
        Collection(collection_name).drop()
    collection = Collection(name=collection_name, schema=schema)
    return collection


# -----------------------------
# Data generators
# -----------------------------

def generate_data_all_types(num_rows):
    ids = list(range(num_rows))
    data = [
        ids,                                         # id
        np.random.randint(-128, 127, num_rows).tolist(),   # int8_field
        np.random.randint(-32768, 32767, num_rows).tolist(), # int16_field
        np.random.randint(-2147483648, 2147483647, num_rows).tolist(), # int32_field
        np.random.randint(-2**62, 2**62 - 1, num_rows).tolist(), # int64_field (safer range)
        np.random.rand(num_rows).astype(np.float32).tolist(),  # float_field
        np.random.rand(num_rows).astype(np.float64).tolist(),  # double_field
        [random.choice([True, False]) for _ in range(num_rows)], # bool_field
        [f"str_{i}" for i in range(num_rows)],  # string_field
        np.random.rand(num_rows, 128).astype(np.float32).tolist(),  # float_vector
    ]

    # binary_vector must be list of bytes (16 bytes = 128 bits per vector)
    binary_vectors = []
    for _ in range(num_rows):
        vector = np.random.randint(0, 256, 16).astype(np.uint8).tobytes()
        binary_vectors.append(vector)
    data.append(binary_vectors)

    return data


def generate_data_multiple_vectors(num_rows):
    ids = list(range(num_rows))
    float_vector1 = np.random.rand(num_rows, 64).astype(np.float32).tolist()
    float_vector2 = np.random.rand(num_rows, 128).astype(np.float32).tolist()
    binary_vectors = []
    for _ in range(num_rows):
        vector = np.random.randint(0, 256, 32).astype(np.uint8).tobytes()  # 256 bits = 32 bytes
        binary_vectors.append(vector)
    data = [
        ids,
        float_vector1,
        float_vector2,
        binary_vectors,
    ]
    return data


def generate_data_mixed(num_rows):
    ids = list(range(num_rows))
    int_field = np.random.randint(-2147483648, 2147483647, num_rows).tolist()
    float_vector1 = np.random.rand(num_rows, 128).astype(np.float32).tolist()
    float_vector2 = np.random.rand(num_rows, 64).astype(np.float32).tolist()
    binary_vectors = []
    for _ in range(num_rows):
        vector = np.random.randint(0, 256, 16).astype(np.uint8).tobytes()  # 128 bits = 16 bytes
        binary_vectors.append(vector)
    string_field = [f"text_{i}" for i in range(num_rows)]
    bool_field = [random.choice([True, False]) for _ in range(num_rows)]

    data = [
        ids,
        int_field,
        float_vector1,
        float_vector2,
        binary_vectors,
        string_field,
        bool_field,
    ]
    return data


def generate_sparse_matrix(num_rows: int, dim: int = 1024, nnz: int = 5):
    """Generate a SciPy CSR sparse matrix of shape (num_rows, dim) with nnz non-zeros per row."""
    if not HAS_SCIPY:
        raise RuntimeError("SciPy is required to generate SPARSE_FLOAT_VECTOR data")
    rows = []
    cols = []
    data_vals = []
    for i in range(num_rows):
        idxs = sorted(random.sample(range(dim), k=nnz))
        vals = np.random.rand(nnz).astype(np.float32)
        rows.extend([i] * nnz)
        cols.extend(idxs)
        data_vals.extend(vals.tolist())
    mat = sp.csr_matrix((data_vals, (rows, cols)), shape=(num_rows, dim), dtype=np.float32)
    return mat


def generate_data_all_supported_25(num_rows):
    ids = list(range(num_rows))
    int8_field = np.random.randint(-128, 127, num_rows).tolist()
    int16_field = np.random.randint(-32768, 32767, num_rows).tolist()
    int32_field = np.random.randint(-2147483648, 2147483647, num_rows).tolist()
    int64_field = np.random.randint(-2**62, 2**62 - 1, num_rows).tolist()
    float_field = np.random.rand(num_rows).astype(np.float32).tolist()
    double_field = np.random.rand(num_rows).astype(np.float64).tolist()
    bool_field = [random.choice([True, False]) for _ in range(num_rows)]
    string_field = [f"s_{i}" for i in range(num_rows)]

    # JSON
    json_field = [{"k": i, "tags": ["a", "b"], "flag": (i % 2 == 0)} for i in range(num_rows)]

    # ARRAY<float> (random length per row up to capacity 32)
    float_array = []
    for _ in range(num_rows):
        ln = random.randint(0, 10)
        float_array.append(np.random.rand(ln).astype(np.float32).tolist())

    # Dense FLOAT vectors (float32 lists OK)
    float_vector1 = np.random.rand(num_rows, 128).astype(np.float32).tolist()
    float_vector2 = np.random.rand(num_rows, 64).astype(np.float32).tolist()

    # FLOAT16 vectors: must be list of np.ndarray(dtype=float16)
    float16_vector = [np.random.rand(128).astype(np.float16) for _ in range(num_rows)]

    # BFLOAT16 vectors: include only if numpy supports bfloat16 dtype
    if HAS_BFLOAT16:
        bfloat16_vector = [np.random.rand(128).astype(BFLOAT16_DTYPE) for _ in range(num_rows)]

    # Binary vectors (bytes): 128b -> 16B, 256b -> 32B
    binary_vector1 = [np.random.randint(0, 256, 16).astype(np.uint8).tobytes() for _ in range(num_rows)]
    binary_vector2 = [np.random.randint(0, 256, 32).astype(np.uint8).tobytes() for _ in range(num_rows)]

    # Sparse vectors (CSR matrix) if SciPy available
    if HAS_SCIPY:
        sparse_vector = generate_sparse_matrix(num_rows, dim=1024, nnz=5)

    # Assemble in the same order as schema
    data = [
        ids,
        int8_field,
        int16_field,
        int32_field,
        int64_field,
        float_field,
        double_field,
        bool_field,
        string_field,
        json_field,
        float_array,
        float_vector1,
        float_vector2,
        float16_vector,
    ]
    if HAS_BFLOAT16:
        data.append(bfloat16_vector)
    data.extend([
        binary_vector1,
        binary_vector2,
    ])
    if HAS_SCIPY:
        data.append(sparse_vector)
    return data


# -----------------------------
# Index creation helper
# -----------------------------

def create_indexes(collection, vector_fields_with_params):
    """
    Create indexes on vector fields.
    vector_fields_with_params: dict {field_name: index_params}
    """
    for field_name, index_params in vector_fields_with_params.items():
        try:
            print(f"Creating index on '{field_name}' ...")
            collection.create_index(field_name=field_name, index_params=index_params)
        except Exception as e:
            print(f"Warning: failed to create index on '{field_name}': {e}")
    print("Indexes created.")


# -----------------------------
# Main
# -----------------------------

def main():
    num_rows = 10

    # Create collections
    col_all = create_collection_all_types()
    col_multi_vec = create_collection_multiple_vectors()
    col_mixed = create_collection_mixed_data()
    col_all_supported = create_collection_all_supported_25()

    # Insert data
    print("Inserting data into collection_all_types...")
    data_all = generate_data_all_types(num_rows)
    col_all.insert(data_all)

    print("Inserting data into collection_multiple_vectors...")
    data_multi_vec = generate_data_multiple_vectors(num_rows)
    col_multi_vec.insert(data_multi_vec)

    print("Inserting data into collection_mixed_data...")
    data_mixed = generate_data_mixed(num_rows)
    col_mixed.insert(data_mixed)

    print("Inserting data into collection_all_supported_25...")
    data_all_sup = generate_data_all_supported_25(num_rows)
    col_all_supported.insert(data_all_sup)

    # Create indexes before loading collections (only on vector fields)
    create_indexes(col_all, {
        "float_vector": {
            "index_type": "IVF_FLAT",
            "metric_type": "L2",
            "params": {"nlist": 128},
        },
        "binary_vector": {
            "index_type": "BIN_FLAT",
            "metric_type": "HAMMING",
        }
    })

    create_indexes(col_multi_vec, {
        "float_vector1": {
            "index_type": "IVF_FLAT",
            "metric_type": "L2",
            "params": {"nlist": 128},
        },
        "float_vector2": {
            "index_type": "IVF_FLAT",
            "metric_type": "L2",
            "params": {"nlist": 128},
        },
        "binary_vector1": {
            "index_type": "BIN_FLAT",
            "metric_type": "HAMMING",
        }
    })

    create_indexes(col_mixed, {
        "float_vector1": {
            "index_type": "IVF_FLAT",
            "metric_type": "L2",
            "params": {"nlist": 128},
        },
        "float_vector2": {
            "index_type": "IVF_FLAT",
            "metric_type": "L2",
            "params": {"nlist": 128},
        },
        "binary_vector": {
            "index_type": "BIN_FLAT",
            "metric_type": "HAMMING",
        }
    })

    # Indexes for the all-supported collection
    indexes_all_sup = {
        # dense float
        "float_vector1": {"index_type": "IVF_FLAT", "metric_type": "L2", "params": {"nlist": 128}},
        "float_vector2": {"index_type": "IVF_FLAT", "metric_type": "L2", "params": {"nlist": 128}},
        # float16/bfloat16 use same index interface; server will handle dtype
        "float16_vector": {"index_type": "IVF_FLAT", "metric_type": "L2", "params": {"nlist": 128}},
        # bfloat16 added only if present in schema
    }
    if HAS_BFLOAT16:
        indexes_all_sup["bfloat16_vector"] = {"index_type": "IVF_FLAT", "metric_type": "L2", "params": {"nlist": 128}}
    indexes_all_sup.update({
        # binary
        "binary_vector1": {"index_type": "BIN_FLAT", "metric_type": "HAMMING"},
        "binary_vector2": {"index_type": "BIN_FLAT", "metric_type": "HAMMING"},
    })
    if HAS_SCIPY:
        indexes_all_sup["sparse_vector"] = {"index_type": "SPARSE_WAND", "metric_type": "IP"}
    create_indexes(col_all_supported, indexes_all_sup)

    # Load collections after indexes created
    print("Loading collections into memory...")
    col_all.load()
    col_multi_vec.load()
    col_mixed.load()
    col_all_supported.load()

    print("All collections created, data inserted, indexes created, and loaded.")

    # utility.drop_collection("collection_all_supported_25")
    # utility.drop_collection("collection_all_types")
    # utility.drop_collection("collection_multiple_vectors")
    # utility.drop_collection("collection_mixed_data")
    # res = utility.list_collections()
    # print(res)


if __name__ == "__main__":
    main()
