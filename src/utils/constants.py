from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]
DATA_PATH = BASE_DIR / "data" / "arxiv_cs_ai.json"

MODEL_NAME_MINI = "all-MiniLM-L6-v2"
DEFAULT_EMBED_PATH = "data/embeddings.npy"
MODEL_NAME_MPNET = "all-mpnet-base-v2"
SECONDARY_EMBED_PATH = "data/embeddings2.npy"
TFIDF_EMBED_PATH = "data/tfidf"
HYBRID_EMBED_PATH = "data/hybrid_emb.npz"