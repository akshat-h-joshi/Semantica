import faiss
import numpy as np
from sklearn.preprocessing import normalize

def build_faiss_index(embeddings: np.ndarray):
    """
    Build a FAISS index for cosine similarity
    (assumes embeddings are NOT yet normalized)
    """
    embeddings = normalize(embeddings).astype("float32")
    dim = embeddings.shape[1]

    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    return index


def faiss_search(index, query_emb, k=5):
    query_emb = normalize(query_emb).astype("float32")
    scores, indices = index.search(query_emb, k)
    return indices[0], scores[0]


def save_index(index, path):
    faiss.write_index(index, str(path))


def load_index(path):
    return faiss.read_index(str(path))