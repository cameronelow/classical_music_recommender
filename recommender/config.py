"""
Configuration module for the classical music recommendation system.

This module centralizes all configuration parameters, making them easily
adjustable without modifying core logic. Parameters can be overridden via
environment variables for production deployments.
"""

import os
from pathlib import Path
from typing import Optional
from pydantic import BaseModel, Field, field_validator


class FeatureWeights(BaseModel):
    """Feature importance weights for similarity computation.

    Higher weights mean the feature has more influence on recommendations.
    Weights are normalized internally, so absolute values don't matter -
    only relative proportions.

    Design decision: Composer gets highest weight because "more Bach" is
    usually what users want. Period is second because era similarity matters
    more than specific instrumentation for classical music.
    """
    composer: float = Field(default=5.0, description="Composer similarity weight")
    period: float = Field(default=3.0, description="Period/era similarity weight")
    work_type: float = Field(default=2.0, description="Work type (symphony, concerto, etc.) weight")
    key: float = Field(default=1.0, description="Musical key similarity weight")
    tags: float = Field(default=2.5, description="Tags (genre, instrumentation, mood) weight")
    catalog_pattern: float = Field(default=0.5, description="Catalog number pattern weight")

    @field_validator('*')
    @classmethod
    def validate_positive(cls, v: float) -> float:
        if v < 0:
            raise ValueError("Feature weights must be non-negative")
        return v


class PathConfig(BaseModel):
    """File paths for data and artifacts.

    All paths are relative to the project root unless absolute paths provided.
    """
    # Data paths
    works_parquet: Path = Field(
        default=Path("data/processed/works.parquet"),
        description="Works metadata parquet file"
    )
    composers_parquet: Path = Field(
        default=Path("data/processed/composers.parquet"),
        description="Composers metadata parquet file"
    )
    work_tags_parquet: Path = Field(
        default=Path("data/processed/work_tags_enhanced.parquet"),
        description="Work tags parquet file (includes MusicBrainz + auto-tagger enhanced tags)"
    )

    # Cache paths
    cache_dir: Path = Field(
        default=Path("data/cache/recommender"),
        description="Directory for caching feature vectors and similarity matrices"
    )
    similarity_matrix_cache: Path = Field(
        default=Path("data/cache/recommender/similarity_matrix.npz"),
        description="Pre-computed similarity matrix cache"
    )
    feature_vectors_cache: Path = Field(
        default=Path("data/cache/recommender/feature_vectors.npz"),
        description="Pre-computed feature vectors cache"
    )

    @field_validator('*')
    @classmethod
    def resolve_path(cls, v: Path) -> Path:
        """Convert relative paths to absolute from project root."""
        if not v.is_absolute():
            # Find project root (contains requirements.txt)
            current = Path.cwd()
            while current != current.parent:
                if (current / 'requirements.txt').exists():
                    return current / v
                current = current.parent
            # Fallback to current directory
            return Path.cwd() / v
        return v


class RecommenderConfig(BaseModel):
    """Core recommendation engine parameters."""

    # Similarity computation
    similarity_metric: str = Field(
        default="cosine",
        description="Similarity metric (cosine, euclidean, manhattan)"
    )
    min_similarity_threshold: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Minimum similarity score to include in results"
    )

    # Diversity parameters
    diversity_weight: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description="Weight for diversity vs pure similarity (0=pure similarity, 1=pure diversity)"
    )
    max_same_composer_ratio: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Maximum ratio of recommendations from same composer as seed work"
    )

    # Performance parameters
    use_approximate_neighbors: bool = Field(
        default=False,
        description="Use FAISS for approximate nearest neighbors (faster for large datasets)"
    )
    faiss_index_type: str = Field(
        default="IndexFlatIP",
        description="FAISS index type (IndexFlatIP, IndexIVFFlat, etc.)"
    )
    faiss_nprobe: int = Field(
        default=10,
        ge=1,
        description="Number of clusters to search in FAISS (higher=more accurate, slower)"
    )

    # Caching
    enable_cache: bool = Field(
        default=True,
        description="Enable caching of feature vectors and similarity matrices"
    )
    cache_ttl_seconds: int = Field(
        default=86400,  # 24 hours
        ge=0,
        description="Cache time-to-live in seconds (0=never expire)"
    )

    # Explanation generation
    generate_explanations: bool = Field(
        default=True,
        description="Generate human-readable explanations for recommendations"
    )

    @field_validator('similarity_metric')
    @classmethod
    def validate_metric(cls, v: str) -> str:
        valid_metrics = {'cosine', 'euclidean', 'manhattan', 'dot'}
        if v.lower() not in valid_metrics:
            raise ValueError(f"similarity_metric must be one of {valid_metrics}")
        return v.lower()


class DataLoaderConfig(BaseModel):
    """Data loading and validation parameters."""

    validate_on_load: bool = Field(
        default=True,
        description="Perform data quality checks when loading"
    )
    handle_missing_composers: str = Field(
        default="create_unknown",
        description="How to handle works with missing composers (drop, create_unknown, keep_null)"
    )
    min_tag_count: int = Field(
        default=1,
        ge=0,
        description="Minimum number of tags a work should have (0=no minimum)"
    )
    deduplicate_works: bool = Field(
        default=True,
        description="Remove duplicate works based on title+composer"
    )

    @field_validator('handle_missing_composers')
    @classmethod
    def validate_missing_strategy(cls, v: str) -> str:
        valid_strategies = {'drop', 'create_unknown', 'keep_null'}
        if v not in valid_strategies:
            raise ValueError(f"handle_missing_composers must be one of {valid_strategies}")
        return v


class FeatureEngineeringConfig(BaseModel):
    """Feature extraction and engineering parameters."""

    # Text feature parameters
    tfidf_max_features: int = Field(
        default=100,
        ge=1,
        description="Maximum number of TF-IDF features for text fields"
    )
    tfidf_ngram_range: tuple[int, int] = Field(
        default=(1, 2),
        description="N-gram range for TF-IDF (e.g., (1,2) for unigrams and bigrams)"
    )
    tfidf_min_df: int = Field(
        default=1,
        ge=1,
        description="Minimum document frequency for TF-IDF terms"
    )

    # Catalog number extraction
    extract_catalog_patterns: bool = Field(
        default=True,
        description="Extract patterns from catalog numbers (Op., BWV, K., etc.)"
    )

    # Key relationship encoding
    use_circle_of_fifths: bool = Field(
        default=True,
        description="Encode musical keys using circle of fifths relationships"
    )

    # Composite features
    create_composite_features: bool = Field(
        default=True,
        description="Create composite features (e.g., 'baroque_concerto')"
    )
    composite_combinations: list[tuple[str, str]] = Field(
        default=[
            ('period', 'work_type'),
            ('composer', 'work_type'),
            ('period', 'key'),
        ],
        description="Field pairs to combine into composite features"
    )

    # Missing value handling
    fill_missing_text: str = Field(
        default="unknown",
        description="Placeholder for missing text values"
    )
    fill_missing_numeric: float = Field(
        default=0.0,
        description="Placeholder for missing numeric values"
    )


class SemanticSearchConfig(BaseModel):
    """Semantic search configuration for mood-based recommendations."""

    # Model settings
    embedding_model: str = Field(
        default="all-MiniLM-L6-v2",
        description="Sentence transformer model to use for embeddings"
    )
    embedding_batch_size: int = Field(
        default=32,
        ge=1,
        description="Batch size for encoding embeddings"
    )

    # Cache settings
    embedding_cache_dir: Path = Field(
        default=Path("data/embeddings"),
        description="Directory for caching embeddings"
    )
    enable_embedding_cache: bool = Field(
        default=True,
        description="Enable caching of work embeddings"
    )

    # Search settings
    default_search_results: int = Field(
        default=10,
        ge=1,
        description="Default number of search results to return"
    )
    min_similarity_threshold: float = Field(
        default=0.15,
        ge=0.0,
        le=1.0,
        description="Minimum similarity score to include in results (balanced to allow good matches while filtering weak ones)"
    )
    enable_query_enhancement: bool = Field(
        default=True,
        description="Enable query enhancement with synonyms and mappings"
    )

    # Diversity settings
    enable_diversity_sampling: bool = Field(
        default=True,
        description="Enable weighted random sampling for result diversity"
    )
    diversity_candidate_multiplier: int = Field(
        default=3,
        ge=1,
        le=5,
        description="How many candidates to consider for diversity (2 = moderate, 3 = high diversity)"
    )

    # Description generation weights (how many times to repeat each feature)
    composer_weight: int = Field(
        default=3,
        ge=1,
        description="Repeat composer name N times in description for emphasis"
    )
    period_weight: int = Field(
        default=2,
        ge=1,
        description="Repeat period N times in description"
    )
    work_type_weight: int = Field(
        default=2,
        ge=1,
        description="Repeat work type N times in description"
    )
    key_mood_weight: int = Field(
        default=2,
        ge=1,
        description="Repeat key mood associations N times in description"
    )
    tag_weight: int = Field(
        default=3,
        ge=1,
        description="Repeat tags N times in description (increased from 1 to 3 to emphasize mood/character in semantic search)"
    )

    @field_validator('embedding_cache_dir')
    @classmethod
    def resolve_path(cls, v: Path) -> Path:
        """Convert relative paths to absolute from project root."""
        if not v.is_absolute():
            # Find project root (contains requirements.txt)
            current = Path.cwd()
            while current != current.parent:
                if (current / 'requirements.txt').exists():
                    return current / v
                current = current.parent
            # Fallback to current directory
            return Path.cwd() / v
        return v


class Config(BaseModel):
    """Master configuration class combining all config sections."""

    feature_weights: FeatureWeights = Field(default_factory=FeatureWeights)
    paths: PathConfig = Field(default_factory=PathConfig)
    recommender: RecommenderConfig = Field(default_factory=RecommenderConfig)
    data_loader: DataLoaderConfig = Field(default_factory=DataLoaderConfig)
    feature_engineering: FeatureEngineeringConfig = Field(default_factory=FeatureEngineeringConfig)
    semantic_search: SemanticSearchConfig = Field(default_factory=SemanticSearchConfig)

    # Global settings
    random_seed: int = Field(
        default=42,
        description="Random seed for reproducibility"
    )
    log_level: str = Field(
        default="INFO",
        description="Logging level (DEBUG, INFO, WARNING, ERROR)"
    )

    @classmethod
    def from_env(cls) -> "Config":
        """Create configuration from environment variables.

        Environment variables should be prefixed with RECOMMENDER_
        and use double underscores for nested fields.

        Example:
            RECOMMENDER_FEATURE_WEIGHTS__COMPOSER=6.0
            RECOMMENDER_RECOMMENDER__DIVERSITY_WEIGHT=0.4
        """
        # For now, just use defaults. Can extend to parse env vars.
        return cls()

    def ensure_cache_dirs(self) -> None:
        """Create cache directories if they don't exist."""
        self.paths.cache_dir.mkdir(parents=True, exist_ok=True)
        self.semantic_search.embedding_cache_dir.mkdir(parents=True, exist_ok=True)


# Global config instance
_config: Optional[Config] = None


def get_config() -> Config:
    """Get global configuration instance (singleton pattern).

    Returns:
        Config: Global configuration object

    Example:
        >>> config = get_config()
        >>> print(config.feature_weights.composer)
        5.0
    """
    global _config
    if _config is None:
        _config = Config.from_env()
        _config.ensure_cache_dirs()
    return _config


def set_config(config: Config) -> None:
    """Override global configuration (useful for testing).

    Args:
        config: New configuration object to use globally
    """
    global _config
    _config = config


# Convenience function for tests
def reset_config() -> None:
    """Reset configuration to default values."""
    global _config
    _config = None
