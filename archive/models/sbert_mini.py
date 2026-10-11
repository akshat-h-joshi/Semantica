from sentence_transformers import SentenceTransformer, util
import numpy as np
from .recommender_base import RecommenderBase
from ..utils.constants import MODEL_NAME_MINI
from ..indexing.faiss_index import build_faiss_index, faiss_search
from sklearn.preprocessing import normalize

class SBERTMini(RecommenderBase):
    def __init__(self, embeddings):
        super().__init__(name=MODEL_NAME_MINI)
        self.embeddings = embeddings
        self.model = SentenceTransformer(MODEL_NAME_MINI)
        self.index = build_faiss_index(embeddings)

    def embed(self, texts):
        return normalize(self.model.encode(texts, convert_to_numpy=True))

    def score(self, query, top_k=5):
        query_emb = self.embed([query])
        indices, scores = faiss_search(self.index, query_emb, k=top_k)
        return indices, scores 

    def recommend(self, query, papers, top_k=5):
        indices, scores = self.score(query, top_k=top_k)

        return [
            (papers[i]["title"], float(scores[idx]))
            for idx, i in enumerate(indices)
        ]