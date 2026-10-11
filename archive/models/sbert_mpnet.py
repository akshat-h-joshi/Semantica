from sentence_transformers import SentenceTransformer, util
import numpy as np
from .recommender_base import RecommenderBase
from ..utils.constants import MODEL_NAME_MPNET

class SBERTMpnet(RecommenderBase):
    def __init__(self, embeddings):
        super().__init__(name=MODEL_NAME_MPNET)
        self.model = SentenceTransformer(MODEL_NAME_MPNET)
        self.embeddings = embeddings

    def embed(self, texts):
        return self.model.encode(texts, convert_to_numpy=True)

    def score(self, query):
        query_embedding = self.embed([query])
        return util.cos_sim(query_embedding, self.embeddings)[0].tolist()