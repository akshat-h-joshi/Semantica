from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from scipy.sparse import csr_matrix, hstack
from .recommender_base import RecommenderBase

class HybridRecommender(RecommenderBase):
    def __init__(self, mpnet_model, tfidf_vectorizer, hybrid_embeddings, weights):
        super().__init__(name="hybrid")
        self.mpnet_model = mpnet_model
        self.tfidf_vectorizer = tfidf_vectorizer
        self.embeddings = hybrid_embeddings
        self.weights = weights

    def embed(self, texts):
        mpnet_emb = self.mpnet_model.embed(texts)
        tfidf_emb = self.tfidf_vectorizer.transform(texts)

        mpnet_emb = normalize(mpnet_emb)
        tfidf_emb = normalize(tfidf_emb)

        w_mpnet, w_tfidf = self.weights

        mpnet_sparse = csr_matrix(mpnet_emb)

        return hstack([
            w_mpnet * mpnet_sparse,
            w_tfidf * tfidf_emb
        ], format="csr")

    def score(self, query):
        query_emb = self.embed([query])
        return cosine_similarity(query_emb, self.embeddings)[0].tolist()