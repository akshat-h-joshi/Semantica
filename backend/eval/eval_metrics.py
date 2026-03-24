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
            r["index"] for r in results
            if r["index"] != i
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
            r["index"] for r in results
            if r["index"] != i
        ][:k]

        relevant_indices = {
            j for j, p in enumerate(papers)
            if p["category"] == query_cat and j != i
        }

        rr = reciprocal_rank(ranked_indices, relevant_indices)
        mrr_scores.append(rr)

    return float(np.mean(mrr_scores))


def build_rank_map(results):
    """
    results: output of recommend_indices
    returns: dict[index -> rank]
    """
    return {
        item["index"]: rank + 1
        for rank, item in enumerate(results)
    }


def compare_ranks(base_results, other_results):
    base_ranks = build_rank_map(base_results)
    other_ranks = build_rank_map(other_results)

    comparison = {}

    for idx, base_rank in base_ranks.items():
        other_rank = other_ranks.get(idx)

        if other_rank is None:
            comparison[idx] = {
                "base_rank": base_rank,
                "other_rank": None,
                "delta": None,
                "status": "missing"
            }
        else:
            delta = other_rank - base_rank
            comparison[idx] = {
                "base_rank": base_rank,
                "other_rank": other_rank,
                "delta": delta,
                "status": "present"
            }

    return comparison