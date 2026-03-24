# http://127.0.0.1:8000/docs
# uvicorn backend.api.app:app --reload

from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager

from .schemas import (
    RecommendRequest,
    RecommendResponse,
    RecommendationItem,
    ModelsResponse,
    ModelInfo,
    CompareRequest,
    ModelMetrics,
    RankChange,
    CompareResponse
)

from ..input.data_loader import (
    load_or_fetch_papers,
    load_or_create_embeddings,
    load_or_create_tfidf_embeddings,
)

from ..models.sbert import SBERTFaissRecommender
from ..models.tfidf import TFIDFRecommender
from ..models.hybrid import HybridRecommender

from ..eval.eval_metrics import run_evaluation, compare_ranks

from ..utils.constants import (
    MODEL_NAME_MPNET,
    MODEL_NAME_MINI,
    TFIDF_EMBED_PATH,
    MPNET_EMBED_PATH,
    MINI_EMBED_PATH,
    HYBRID_EMBED_PATH,
    MINI_INDEX_PATH,
    MPNET_INDEX_PATH
)

from ..explainability.keybert_extractor import KeyBERTExtractor

from fastapi.middleware.cors import CORSMiddleware

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    app.state.papers = load_or_fetch_papers()

    app.state.abstract_embeddings = {
        MODEL_NAME_MINI: load_or_create_embeddings(
            app.state.papers, MODEL_NAME_MINI, MINI_EMBED_PATH, "abstract"
        ),
        MODEL_NAME_MPNET: load_or_create_embeddings(
            app.state.papers, MODEL_NAME_MPNET, MPNET_EMBED_PATH, "abstract"
        )
    }

    app.state.title_embeddings = {
        MODEL_NAME_MINI: load_or_create_embeddings(
            app.state.papers, MODEL_NAME_MINI, MINI_EMBED_PATH, "title"
        ),
        MODEL_NAME_MPNET: load_or_create_embeddings(app.state.papers, MODEL_NAME_MPNET, MPNET_EMBED_PATH, "title")
    }

    app.state.tfidf_title_vectorizer, app.state.tfidf_title_embeddings = (
        load_or_create_tfidf_embeddings(app.state.papers, TFIDF_EMBED_PATH, "title")
    )

    app.state.tfidf_abstract_vectorizer, app.state.tfidf_abstract_embeddings = (
        load_or_create_tfidf_embeddings(app.state.papers, TFIDF_EMBED_PATH, "abstract")
    )

    # Initialise recommenders
    app.state.recommenders = {
        "mini": SBERTFaissRecommender(MODEL_NAME_MINI, app.state.title_embeddings[MODEL_NAME_MINI], app.state.abstract_embeddings[MODEL_NAME_MINI], MINI_INDEX_PATH),
        "mpnet": SBERTFaissRecommender(MODEL_NAME_MPNET, app.state.title_embeddings[MODEL_NAME_MPNET], app.state.abstract_embeddings[MODEL_NAME_MPNET], MPNET_INDEX_PATH),
        "tfidf": TFIDFRecommender(app.state.tfidf_title_vectorizer, app.state.tfidf_title_embeddings, app.state.tfidf_abstract_vectorizer, app.state.tfidf_abstract_embeddings)
    }

    app.state.recommenders["hybrid"] = HybridRecommender(app.state.recommenders["mpnet"], app.state.recommenders["tfidf"], (0.7, 0.3))

    app.state.keyword_extractor = KeyBERTExtractor(MODEL_NAME_MPNET)

    yield  # ---- app runs here ----


app = FastAPI(
    title="Semantica API",
    version="1.0.0",
    description="Research paper recommendation API",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
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

    for item in results:
        item["keywords"] = list(
            app.state.keyword_extractor.extract_keywords_cached(
                item["abstract"],
                req.query
            )
        )

    return RecommendResponse(
        query=req.query,
        model=model_name,
        results=[
            RecommendationItem(
                paper_id=item["paper_id"], 
                title=item["title"], 
                score=item["score"],
                keywords=item["keywords"],
                abstract=item["abstract"],
                link=item["link"],
                explanation=item["explanation"]
            )
            for item in results
        ]
    )

@app.post("/api/v2/evaluate", response_model=CompareResponse)
def compare_models(req: CompareRequest):
    papers = app.state.papers
    recommenders = app.state.recommenders

    baseline = req.baseline_model.lower()
    compare = req.compare_model.lower()

    allowed = {"mpnet", "mini", "tfidf", "hybrid"}
    if baseline not in allowed or compare not in allowed:
        raise HTTPException(status_code=400, detail="Unknown model")

    if baseline == compare:
        raise HTTPException(status_code=400, detail="Models must be different")

    baseline_metrics = run_evaluation(baseline, papers, recommenders)
    compare_metrics = run_evaluation(compare, papers, recommenders)

    query = papers[0]["title"] 

    base_results = recommenders[baseline].recommend_indices(query, top_k=50)
    compare_results = recommenders[compare].recommend_indices(query, top_k=50)

    rank_changes = compare_ranks(base_results, compare_results)

    return CompareResponse(
        baseline_model=baseline,
        compare_model=compare,
        metrics={
            baseline: baseline_metrics,
            compare: compare_metrics,
        },
        rank_changes=rank_changes
    )
