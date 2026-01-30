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

    def embed(self, query):
        return self.vectorizer.transform(query)

    def score(self, query):
        expanded_query = expand_query(query)
        query_text = " ".join(expanded_query)

        q_title = self.tv.transform([query_text])
        q_abs = self.av.transform([query_text])

        title_sim = cosine_similarity(q_title, self.te)[0]
        abstract_sim = cosine_similarity(q_abs, self.ae)[0]

        scores = 0.7 * title_sim + 0.3 * abstract_sim

        return scores

    def recommend_indices(self, query, top_k=5):
        scores = self.score(query)
        top = np.argsort(-scores)[:top_k]
        return [(i, float(scores[i])) for i in top]