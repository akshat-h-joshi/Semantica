from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from .recommender_base import RecommenderBase

class TFIDFRecommender(RecommenderBase):
    def __init__(self, vectorizer, embeddings):
        super().__init__(name="tfidf")
        self.vectorizer = vectorizer
        self.embeddings = embeddings

    def embed(self, texts):
        return self.vectorizer.transform(texts)

    def score(self, query):
        query_embedding = self.embed([query])
        sim = cosine_similarity(query_embedding, self.embeddings)
        return sim[0]