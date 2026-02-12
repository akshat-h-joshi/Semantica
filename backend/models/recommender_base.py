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
    ) -> List[Dict[str, Any]]:
    
        pass

    def recommend(
        self,
        query: str,
        papers: List[Dict],
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        ranked = self.recommend_indices(query, top_k)
        
        results = []
        for r in ranked:
            i = r["index"]

            results.append({
                "paper_id": str(i),
                "title": papers[i]["title"],
                "abstract": papers[i]["abstract"],
                "score": float(r["score"]),
                "link": papers[i]["link"],
                "explanation": r.get("explanation"),
            })

        return results