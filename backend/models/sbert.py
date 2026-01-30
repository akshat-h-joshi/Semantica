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
from ..explainability.query_expansion import expand_query

class SBERTFaissRecommender(RecommenderBase):
    def __init__(
        self,
        model_name: str,
        title_embeddings: np.ndarray,
        abstract_embeddings: np.ndarray,
        index_dir: str
    ):
        super().__init__(name=model_name)
        self.model = SentenceTransformer(model_name)

        os.makedirs(index_dir, exist_ok=True)

        self.title_index = self._load_or_build(
            title_embeddings,
            os.path.join(index_dir, "title.index")
        )

        self.abstract_index = self._load_or_build(
            abstract_embeddings,
            os.path.join(index_dir, "abstract.index")
        )

    def _load_or_build(self, embeddings, path):
        if os.path.exists(path):
            return load_index(path)
        index = build_faiss_index(embeddings)
        save_index(index, path)
        return index

    def embed(self, query):
        return normalize(
            self.model.encode(query, convert_to_numpy=True)
        )

    def recommend_indices(self, query, top_k=5):
        expanded_query = expand_query(query)
        query_text = " ".join(expanded_query)
        query_emb = self.embed([query_text])

        t_idx, t_scores = faiss_search(self.title_index, query_emb, k=top_k * 3)
        a_idx, a_scores = faiss_search(self.abstract_index, query_emb, k=top_k * 3)

        scores = {}

        for i, s in zip(t_idx, t_scores):
            scores[i] = scores.get(i, 0) + 0.7 * s

        for i, s in zip(a_idx, a_scores):
            scores[i] = scores.get(i, 0) + 0.3 * s

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return ranked[:top_k]