# http://127.0.0.1:8000/docs
# uvicorn src.api.app:app --reload

from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager

from .schemas import (
    RecommendRequest,
    RecommendResponse,
    RecommendationItem,
    ModelsResponse,
    ModelInfo,
    EvaluateRequest,
    EvaluateResponse
)

from ..input.data_loader import (
    load_or_fetch_papers,
    load_or_create_embeddings,
    load_or_create_tfidf_embeddings,
    load_or_create_hybrid_embeddings
)

from ..models.sbert_mpnet import SBERTMpnet
from ..models.sbert_mini import SBERTMini
from ..models.tfidf import TFIDFRecommender
from ..models.hybrid import HybridRecommender

from ..eval.evaluation import run_evaluation

from ..utils.constants import (
    MODEL_NAME_MPNET,
    MODEL_NAME_MINI,
    TFIDF_EMBED_PATH,
    SECONDARY_EMBED_PATH,
    DEFAULT_EMBED_PATH,
    HYBRID_EMBED_PATH
)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    app.state.papers = load_or_fetch_papers()

    app.state.embeddings = {
        MODEL_NAME_MPNET: load_or_create_embeddings(
            app.state.papers, MODEL_NAME_MPNET, SECONDARY_EMBED_PATH
        ),
        MODEL_NAME_MINI: load_or_create_embeddings(
            app.state.papers, MODEL_NAME_MINI, DEFAULT_EMBED_PATH
        )
    }

    app.state.tfidf_vectorizer, app.state.tfidf_embeddings = (
        load_or_create_tfidf_embeddings(app.state.papers, TFIDF_EMBED_PATH)
    )

    app.state.hybrid_embeddings = load_or_create_hybrid_embeddings(app.state.papers, HYBRID_EMBED_PATH)

    # Initialise recommenders
    app.state.recommenders = {
        "mpnet": SBERTMpnet(app.state.embeddings[MODEL_NAME_MPNET]),
        "mini": SBERTMini(app.state.embeddings[MODEL_NAME_MINI]),
        "tfidf": TFIDFRecommender(app.state.tfidf_vectorizer, app.state.tfidf_embeddings)
    }

    app.state.recommenders["hybrid"] = HybridRecommender(app.state.recommenders["mpnet"], app.state.tfidf_vectorizer, app.state.hybrid_embeddings, (0.7, 0.3))

    yield  # ---- app runs here ----


app = FastAPI(
    title="Semantica API",
    version="1.0.0",
    description="Research paper recommendation API",
    lifespan=lifespan
)


@app.get("/api/v1/health")
def health():
    return {"status": "ok"}


@app.get("/api/v1/models", response_model=ModelsResponse)
def get_models():
    return ModelsResponse(
        models=[
            ModelInfo(
                name="mpnet",
                type="semantic",
                description="SBERT all-mpnet-base-v2"
            ),
            ModelInfo(
                name="mini",
                type="semantic",
                description="SBERT all-MiniLM-L6-v2"
            ),
            ModelInfo(
                name="tfidf",
                type="lexical",
                description="TF-IDF baseline"
            ),
            ModelInfo(
                name="hybrid",
                type="ensemble",
                description="Weighted SBERT + TF-IDF"
            )
        ]
    )


@app.post("/api/v1/recommend", response_model=RecommendResponse)
def recommend(req: RecommendRequest):
    model_name = req.model.lower()

    if model_name not in {"mpnet", "mini", "tfidf", "hybrid"}:
        raise HTTPException(status_code=400, detail="Unknown model")

    papers = app.state.papers

    recommender = app.state.recommenders[model_name]
    results = recommender.recommend(
        query=req.query,
        papers=papers,
        top_k=req.top_k
    )


    return RecommendResponse(
        query=req.query,
        model=model_name,
        results=[
            RecommendationItem(title=title, score=score)
            for title, score in results
        ]
    )


@app.post("/api/v1/evaluate", response_model=EvaluateResponse)
def evaluate(req: EvaluateRequest):
    results = {}
    papers = app.state.papers
    hybrid_weights = req.hybrid_weights

    for model_name in req.models:
        model_name = model_name.lower()

        if model_name not in {"mpnet", "mini", "tfidf", "hybrid"}:
            raise HTTPException(status_code=400, detail="Unknown model")

        results[model_name] = run_evaluation(model_name, papers, app.state.recommenders)

    return EvaluateResponse(
        results=results
    )
