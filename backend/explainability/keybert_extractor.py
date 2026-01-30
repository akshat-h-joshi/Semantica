from keybert import KeyBERT
from sentence_transformers import SentenceTransformer
from typing import List, Tuple
from functools import lru_cache

class KeyBERTExtractor:
    def __init__(self, model_name: str):
        self.embedding_model = SentenceTransformer(model_name)
        self.kb = KeyBERT(model=self.embedding_model)

    def extract_keywords(
        self,
        abstract: str,
        query: str,
        top_k: int = 8,
        ngram_range: Tuple[int, int] = (1, 2)
    ) -> List[str]:
        """Returns keywords ranked by semantic relevance to the query"""

        keywords = self.kb.extract_keywords(
            docs=abstract,
            keyphrase_ngram_range=ngram_range,
            stop_words="english",
            use_mmr=True,
            diversity=0.5,
            top_n=top_k,
            seed_keywords=query.split()
        )

        return [kw for kw, _ in keywords]

    # def extract_batch(
    #     self,
    #     abstracts: List[str],
    #     query: str,
    #     top_k: int = 8
    # ):
    #     return [
    #         self.extract_keywords(abstract, query, top_k)
    #         for abstract in abstracts
    #     ]

    @lru_cache(maxsize=2048)
    def extract_keywords_cached(
        self,
        abstract: str,
        query: str
    ):
        return tuple(self.extract_keywords(abstract, query))