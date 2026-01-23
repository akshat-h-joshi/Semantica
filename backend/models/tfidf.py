from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from .recommender_base import RecommenderBase
import numpy as np

class TFIDFRecommender(RecommenderBase):
    def __init__(self, vectorizer, embeddings):
        super().__init__(name="tfidf")
        self.vectorizer = vectorizer
        self.embeddings = embeddings

    def embed(self, texts):
        return self.vectorizer.transform(texts)

    def score(self, query):
        query_emb = self.embed([query])
        sim = cosine_similarity(query_emb, self.embeddings)
        return sim[0]

    def recommend_indices(self, query, top_k=5):
        sims = self.score(query)
        top = np.argsort(-sims)[:top_k]
        return [(i, float(sims[i])) for i in top]