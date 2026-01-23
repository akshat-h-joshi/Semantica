import numpy as np 
from sklearn.metrics.pairwise import cosine_similarity
import json

def run_evaluation(model_name, papers, recommenders, weights=None):   
    recommender = recommenders[model_name] 

    return {
        "category_purity": category_purity(papers, recommender),
        "mrr": mean_reciprocal_rank(papers, recommender)
    }

def category_purity(papers, recommender, k=5, num_queries=200):
    scores = []

    for i, paper in enumerate(papers[:num_queries]):
        query = paper["title"]

        results = recommender.recommend_indices(
            query=query,
            top_k=k + 1
        )

        # Remove self if present
        ranked_indices = [
            idx for idx, _ in results
            if idx != i
        ][:k]

        same_cat = sum(
            papers[j]["category"] == paper["category"]
            for j in ranked_indices
        )

        scores.append(same_cat / k)

    return float(np.mean(scores))


def reciprocal_rank(ranked_indices, relevant_indices):
    for rank, idx in enumerate(ranked_indices, start=1):
        if idx in relevant_indices:
            return 1.0 / rank
    return 0.0


def mean_reciprocal_rank(
    papers,
    recommender,
    k=10,
    num_queries=100
):
    mrr_scores = []

    for i in range(num_queries):
        query = papers[i]["title"]
        query_cat = papers[i]["category"]

        results = recommender.recommend_indices(
            query=query,
            top_k=k + 1
        )

        ranked_indices = [
            idx for idx, _ in results
            if idx != i
        ][:k]

        relevant_indices = {
            j for j, p in enumerate(papers)
            if p["category"] == query_cat and j != i
        }

        rr = reciprocal_rank(ranked_indices, relevant_indices)
        mrr_scores.append(rr)

    return float(np.mean(mrr_scores))