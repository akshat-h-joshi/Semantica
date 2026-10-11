import numpy as np 
from sentence_transformers import SentenceTransformer 
from sklearn.metrics.pairwise import cosine_similarity
import json
from ..input.data_loader import load_or_create_embeddings, load_or_create_tfidf_embeddings, load_or_fetch_papers
from ..utils.constants import DEFAULT_EMBED_PATH, MODEL_NAME_MINI, SECONDARY_EMBED_PATH, MODEL_NAME_MPNET, TFIDF_EMBED_PATH
from ..models.hybrid import HybridRecommender
from ..models.sbert_mpnet import SBERTMpnet
from ..models.tfidf import TFIDFRecommender

def run_evaluation(model_name, papers, recommenders, weights=None):    
    return {
        "category_purity": category_purity(papers, recommenders[model_name].embeddings),
        "mrr": mean_reciprocal_rank(papers, recommenders[model_name].embeddings)
    }


def category_purity(papers, embeddings, k=5):
    scores = []
    for i, paper in enumerate(papers[:200]):  # subset for speed
        sims = cosine_similarity(
            embeddings[i:i+1], embeddings
        )[0]

        top_k = np.argsort(-sims)[1:k+1]  # skip itself
        same_cat = sum(
            papers[j]['category'] == paper['category']
            for j in top_k
        )
        scores.append(same_cat / k)

    return np.mean(scores)


def reciprocal_rank(ranked_indices, relevant_indices):
    for rank, idx in enumerate(ranked_indices, start=1):
        if idx in relevant_indices:
            return 1.0 / rank
    return 0.0


def mean_reciprocal_rank(papers, embeddings, k=10, num_queries=100):
    mrr_scores = []

    for i in range(num_queries):
        query_emb = embeddings[i].reshape(1,-1)
        query_cat = papers[i]["category"]

        sims = cosine_similarity(query_emb, embeddings)[0]
        ranked_indices = np.argsort(-sims)

        # Remove the query itself
        ranked_indices = ranked_indices[ranked_indices != i][:k]

        # Relevant papers = same category
        relevant_indices = {
            j for j, p in enumerate(papers)
            if p["category"] == query_cat and j != i
        }

        rr = reciprocal_rank(ranked_indices, relevant_indices)
        mrr_scores.append(rr)

    return np.mean(mrr_scores)
