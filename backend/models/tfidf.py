from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from .recommender_base import RecommenderBase
import numpy as np
from ..explainability.query_expansion import expand_query

class TFIDFRecommender(RecommenderBase):
    def __init__(self, title_vectorizer, title_embeddings, abstract_vectorizer, abstract_embeddings):
        super().__init__(name="tfidf")
        self.tv = title_vectorizer
        self.te = title_embeddings
        self.av = abstract_vectorizer
        self.ae = abstract_embeddings


    def _top_terms(self, query_vec, doc_vec, vectorizer, k=3):
        contrib = query_vec.multiply(doc_vec)
        if contrib.nnz == 0:
            return []

        indices = contrib.indices
        scores = contrib.data

        top = sorted(zip(indices, scores), key=lambda x: -x[1])[:k]
        feature_names = vectorizer.get_feature_names_out()

        return [feature_names[i] for i, _ in top]


    def embed(self, query):
        return self.vectorizer.transform(query)


    def score(self, query):
        expanded_query = expand_query(query)
        query_text = " ".join(expanded_query)

        q_title = self.tv.transform([query_text])
        q_abs = self.av.transform([query_text])

        title_sim = cosine_similarity(q_title, self.te)[0]
        abstract_sim = cosine_similarity(q_abs, self.ae)[0]
        
        return title_sim, abstract_sim, q_title, q_abs


    def recommend_indices(self, query, top_k=5):
        title_sim, abstract_sim, q_title, q_abs = self.score(query)
        scores =  title_sim * 0.7 + abstract_sim * 0.3
        top = np.argsort(-scores)[:top_k]
        top = [i for i in top if scores[i] > 0][:top_k]

        results = []

        for i in top:
            title_signal = max(title_sim[i], 0)
            abstract_signal = max(abstract_sim[i], 0)

            signal_sum = title_signal + abstract_signal
            
            fields = {}
            if signal_sum > 0:
                fields["title"] = round(title_signal / signal_sum, 4)
                fields["abstract"] = round(abstract_signal / signal_sum, 4)

            title_terms = self._top_terms(q_title, self.te[i], self.tv)
            abstract_terms = self._top_terms(q_abs, self.ae[i], self.av)

            dominant = max(fields, key=fields.get)

            explanation = {
                "model": "tfidf",
                "fields": fields,
                "matched_terms": {
                    "title": title_terms,
                    "abstract": abstract_terms,
                },
                "reason": (
                    "Ranked due to keyword overlap with the query, "
                    f"primarily driven by the {dominant}."
                )
            }

            results.append({
                "index": i,
                "score": float(scores[i]),
                "explanation": explanation
            })

        return results