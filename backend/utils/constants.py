from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]

# PAPERS PATH
DATA_PATH = "data/arxiv_cs_ai.json"

# MODEL NAMES
MODEL_NAME_MINI = "all-MiniLM-L6-v2"
MODEL_NAME_MPNET = "all-mpnet-base-v2"

# EMBEDDING PATHS
MINI_EMBED_PATH = "data/embeddings/sbert_mini"
MPNET_EMBED_PATH = "data/embeddings/sbert_mpnet"
TFIDF_EMBED_PATH = "data/embeddings/tfidf"
HYBRID_EMBED_PATH = "data/hybrid_emb.npz"

# INDEX PATHS
MINI_INDEX_PATH = "data/indexes/sbert_mini"
MPNET_INDEX_PATH = "data/indexes/sbert_mpnet"