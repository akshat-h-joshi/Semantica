from ..input.data_loader import load_or_create_embeddings, load_or_fetch_papers, load_or_create_tfidf_embeddings
from .sbert_mini import SBERTMini
from .sbert_mpnet import SBERTMpnet
from ..constants import DEFAULT_EMBED_PATH, MODEL_NAME_MINI, MODEL_NAME_MPNET, SECONDARY_EMBED_PATH, TFIDF_EMBED_PATH
from .factory import get_model
from .hybrid import HybridRecommender
from .tfidf import TFIDFRecommender

def run_recommendation(query, model, top_k=5):
    papers = load_or_fetch_papers()

    if model == "hybrid":
        sbert_emb = load_or_create_embeddings(
            papers, MODEL_NAME_MPNET, SECONDARY_EMBED_PATH
        )

        vectorizer, tfidf_emb = load_or_create_tfidf_embeddings(
            papers, TFIDF_EMBED_PATH
        )

        hybrid = HybridRecommender(
            recommenders=[
                SBERTMpnet(),
                TFIDFRecommender(vectorizer, tfidf_emb)
            ],
            weights=[0.7, 0.3]
        )

        results = hybrid.recommend(
            query=query,
            item_embeddings={
                "all-mpnet-base-v2": sbert_emb,
                "tfidf": tfidf_emb
            },
            papers=papers,
            top_k=top_k
        )
    
    else:
        if model == "mini":    
            embeddings = load_or_create_embeddings(papers, MODEL_NAME_MINI, DEFAULT_EMBED_PATH)
        elif model == "mpnet":
            embeddings = load_or_create_embeddings(papers, MODEL_NAME_MPNET, SECONDARY_EMBED_PATH)

        recommender = get_model(model)

        results = recommender.recommend(
            query=query,
            item_embeddings=embeddings,
            papers=papers,
            top_k=top_k
        )

    return results
