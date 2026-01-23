import os
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import normalize
from .recommender_base import RecommenderBase
from ..indexing.faiss_index import (
    build_faiss_index,
    faiss_search,
    load_index,
    save_index
)

class SBERTFaissRecommender(RecommenderBase):
    def __init__(
        self,
        model_name: str,
        embeddings: np.ndarray,
        index_path: str
    ):
        super().__init__(name=model_name)
        self.embeddings = embeddings
        self.model = SentenceTransformer(model_name)

        if os.path.exists(index_path):
            self.index = load_index(index_path)
        else:
            self.index = build_faiss_index(self.embeddings)
            save_index(self.index, index_path)

    def embed(self, texts):
        return normalize(
            self.model.encode(texts, convert_to_numpy=True)
        )

    def recommend_indices(self, query, top_k=5):
        query_emb = self.embed([query])
        indices, scores = faiss_search(self.index, query_emb, k=top_k)
        return list(zip(indices, scores))