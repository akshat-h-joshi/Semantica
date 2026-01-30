from pydantic import BaseModel, Field
from typing import List, Dict, Optional 

class RecommendRequest(BaseModel):
    query: str = Field(
        ...,
        description="Natural language query describing the topic of interest"
    )

    model: str = Field(
        default="mini",
        description="Recommender model to use: mini | mpnet | tfidf | hybrid"
    )

    top_k: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Number of papers to return"
    )

class RecommendationItem(BaseModel):
    paper_id: str = Field(
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

    keywords: List[str] = Field(
        ...,
        description="Keywords most semantically relevant to query from paper"
    )

    abstract: str = Field(
        ...,
        description="Abstract of paper"
    )

    link: str = Field(
        ...,
        description="URL that links to paper"
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

class EvaluateResponse(BaseModel):
    results: Dict[str, Dict[str, float]]

