from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from .recommender_base import RecommenderBase

class HybridRecommender(RecommenderBase):
    def __init__(self, primary_model, secondary_model, weights, k=50):
        super().__init__(name="hybrid")
        self.primary_model = primary_model
        self.secondary_model = secondary_model
        self.weights = weights
        self.k = k
        
    def recommend_indices(self, query, top_k=5):
        w1, w2 = self.weights

        p = dict(self.primary_model.recommend_indices(query, self.k))
        s = dict(self.secondary_model.recommend_indices(query, self.k))

        candidates = set(p) | set(s)

        final = {}
        for i in candidates:
            final[i] = (
                w1 * p.get(i, 0.0) +
                w2 * s.get(i, 0.0)
            )

        ranked = sorted(
            final.items(),
            key=lambda x: x[1],
            reverse=True
        )[:top_k]

        return [
            (i, float(score))
            for i, score in ranked
        ]