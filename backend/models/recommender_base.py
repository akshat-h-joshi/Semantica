from abc import ABC, abstractmethod
from typing import List, Any, Dict, Tuple
import numpy as np

class RecommenderBase(ABC):
    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def recommend_indices(
        self,
        query: str,
        top_k: int = 5
    ) -> List[Tuple[int, float]]:
    
        pass

    def recommend(
        self,
        query: str,
        papers: List[Dict],
        top_k: int = 5
    ) -> List[Tuple[str, float]]:
        ranked = self.recommend_indices(query, top_k)

        return [
            (papers[i]["title"], float(score))
            for i, score in ranked
        ]