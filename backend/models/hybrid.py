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

        p_raw = self.primary_model.recommend_indices(query, self.k)
        s_raw = self.secondary_model.recommend_indices(query, self.k)

        p_scores = {r["index"]: r["score"] for r in p_raw}
        s_scores = {r["index"]: r["score"] for r in s_raw}

        candidates = set(p_scores) | set(s_scores)

        final = {}
        for i in candidates:
            final[i] = (
                w1 * p_scores.get(i, 0.0) +
                w2 * s_scores.get(i, 0.0)
            )

        ranked = sorted(
            final.items(),
            key=lambda x: x[1],
            reverse=True
        )[:top_k]

        p_expl = {r["index"]: r["explanation"] for r in p_raw}
        s_expl = {r["index"]: r["explanation"] for r in s_raw}

        results = []

        for i, score in ranked:

            components = {}

            p_component = p_expl.get(i)
            if p_component is not None:
                components[self.primary_model.name] = p_component

            s_component = s_expl.get(i)
            if s_component is not None:
                components[self.secondary_model.name] = s_component
                
            results.append({
                "index": i,
                "score": float(score),
                "explanation": {
                    "model": "hybrid",
                    "components": components,
                    "reason": (
                        f"Ranked using a weighted combination of "
                        f"{self.primary_model.name}: {w1} and {self.secondary_model.name}: {w2}"
                    )
                }
            })

        return results