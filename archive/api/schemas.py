from pydantic import BaseModel, Field
from typing import List, Dict, Optional 

class RecommendRequest(BaseModel):
    query: str = Field(
        ...,
        description="Natural language query describing the topic of interest"
    )

    model: str = Field(
        default="mpnet",
        description="Recommender model to use: mpnet | mini | tfidf | hybrid"
    )

    top_k: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Number of papers to return"
    )

class RecommendationItem(BaseModel):
    paper_id: Optional[str] = Field(
        default=None,
        description="Unique identifier for the paper"
    )

    title: str = Field(
        ...,
        description="Title of the paper"
    )

    score: float = Field(
        ...,
        description="Relevance score (model-specific)"
    )

class RecommendResponse(BaseModel):
    query: str
    model: str
    results: List[RecommendationItem]

class ModelInfo(BaseModel):
    name: str
    type: str
    description: str

class ModelsResponse(BaseModel):
    models: List[ModelInfo]

class EvaluateRequest(BaseModel):
    models: List[str]
    # hybrid_weights: Optional[List[float]] = None

class EvaluateResponse(BaseModel):
    results: Dict[str, Dict[str, float]]

