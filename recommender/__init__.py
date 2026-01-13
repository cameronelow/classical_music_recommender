"""
Classical Music Recommendation System

A production-grade content-based recommendation system for classical music works.

Main components:
- config: Configuration management
- data_loader: Data loading and validation
- features: Feature extraction and engineering
- recommender: Core recommendation engine
- evaluation: Quality metrics and testing
- service: Production API layer
- semantic_search: Mood-based natural language search
- semantic_evaluation: Semantic search quality metrics

Quick start:
    >>> from recommender.service import initialize_service, get_service
    >>> initialize_service()
    >>> service = get_service()
    >>> # Traditional similarity-based recommendations
    >>> recommendations = service.recommend_by_query("Brandenburg Concerto", n=5)
    >>> # Semantic search by mood
    >>> results = service.search_by_mood("dark and dramatic", n=5)
"""

from recommender.config import Config, get_config
from recommender.data_loader import DataLoader, MusicDataset
from recommender.features import FeatureExtractor, FeatureMatrix
from recommender.recommender import MusicRecommender, Recommendation
from recommender.evaluation import RecommenderEvaluator
from recommender.service import RecommenderService, initialize_service, get_service
from recommender.semantic_search import (
    SemanticSearchEngine,
    WorkDescriptionGenerator,
    SemanticSearchResult,
)
from recommender.semantic_evaluation import SemanticSearchEvaluator

__version__ = "1.1.0"  # Bumped for semantic search feature

__all__ = [
    # Config
    "Config",
    "get_config",
    # Data
    "DataLoader",
    "MusicDataset",
    # Features
    "FeatureExtractor",
    "FeatureMatrix",
    # Recommender
    "MusicRecommender",
    "Recommendation",
    # Evaluation
    "RecommenderEvaluator",
    # Service
    "RecommenderService",
    "initialize_service",
    "get_service",
    # Semantic Search
    "SemanticSearchEngine",
    "WorkDescriptionGenerator",
    "SemanticSearchResult",
    "SemanticSearchEvaluator",
]
