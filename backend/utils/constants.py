from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]

# PAPERS PATH
DATA_PATH = BASE_DIR / "data" / "arxiv_cs_ai.json"

# MODEL NAMES
MODEL_NAME_MINI = "all-MiniLM-L6-v2"
MODEL_NAME_MPNET = "all-mpnet-base-v2"

# EMBEDDING PATHS
DEFAULT_EMBED_PATH = "data/embeddings.npy"
SECONDARY_EMBED_PATH = "data/embeddings2.npy"
TFIDF_EMBED_PATH = "data/tfidf"
HYBRID_EMBED_PATH = "data/hybrid_emb.npz"

# INDEX PATHS
MINI_INDEX_PATH = "data/mini.index"
MPNET_INDEX_PATH = "data/mpnet.index"