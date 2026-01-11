from abc import ABC, abstractmethod
from typing import List, Any, Dict
import numpy as np

class RecommenderBase(ABC):
    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def embed(self, texts: List[str]) -> Any:
        pass

    @abstractmethod
    def score(self, query_embedding: Any, item_embeddings: Any) -> List[float]:
        pass

    def preprocess(self, text: str) -> str:
        return text

    def recommend(
        self,
        query: str,
        papers: List[Dict],
        top_k: int = 5
    ) -> List[Dict]:
    
        query = self.preprocess(query)

        scores = self.score(query)

        ranked = sorted(
            zip(papers, scores),
            key=lambda x: x[1],
            reverse=True
        )

        return [
            (paper["title"], float(score))
            for paper, score in ranked[:top_k]
        ]