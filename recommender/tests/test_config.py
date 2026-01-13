"""Tests for configuration module."""

import pytest
from pathlib import Path

from recommender.config import (
    Config,
    FeatureWeights,
    PathConfig,
    RecommenderConfig,
    get_config,
    set_config,
    reset_config
)


class TestFeatureWeights:
    """Tests for FeatureWeights configuration."""

    def test_default_weights(self):
        """Test default feature weights are set correctly."""
        weights = FeatureWeights()
        assert weights.composer == 5.0
        assert weights.period == 3.0
        assert weights.work_type == 2.0
        assert weights.key == 1.0
        assert weights.tags == 2.5
        assert weights.catalog_pattern == 0.5

    def test_custom_weights(self):
        """Test custom feature weights."""
        weights = FeatureWeights(
            composer=10.0,
            period=5.0,
            work_type=3.0
        )
        assert weights.composer == 10.0
        assert weights.period == 5.0
        assert weights.work_type == 3.0

    def test_negative_weight_validation(self):
        """Test that negative weights are rejected."""
        with pytest.raises(ValueError):
            FeatureWeights(composer=-1.0)


class TestPathConfig:
    """Tests for PathConfig configuration."""

    def test_default_paths(self):
        """Test default paths are set."""
        paths = PathConfig()
        assert paths.works_parquet is not None
        assert paths.composers_parquet is not None
        assert paths.work_tags_parquet is not None
        assert paths.cache_dir is not None


class TestRecommenderConfig:
    """Tests for RecommenderConfig."""

    def test_default_config(self):
        """Test default recommender configuration."""
        config = RecommenderConfig()
        assert config.similarity_metric == "cosine"
        assert config.diversity_weight == 0.3
        assert config.enable_cache is True

    def test_similarity_metric_validation(self):
        """Test similarity metric validation."""
        # Valid metrics should work
        config = RecommenderConfig(similarity_metric="euclidean")
        assert config.similarity_metric == "euclidean"

        # Invalid metric should raise error
        with pytest.raises(ValueError):
            RecommenderConfig(similarity_metric="invalid_metric")

    def test_diversity_weight_bounds(self):
        """Test diversity weight is bounded [0, 1]."""
        with pytest.raises(ValueError):
            RecommenderConfig(diversity_weight=-0.1)

        with pytest.raises(ValueError):
            RecommenderConfig(diversity_weight=1.5)


class TestConfig:
    """Tests for master Config class."""

    def test_default_config(self):
        """Test default configuration."""
        config = Config()
        assert config.feature_weights is not None
        assert config.paths is not None
        assert config.recommender is not None
        assert config.random_seed == 42

    def test_custom_config(self):
        """Test custom configuration."""
        weights = FeatureWeights(composer=10.0)
        config = Config(feature_weights=weights)
        assert config.feature_weights.composer == 10.0

    def test_get_config_singleton(self):
        """Test get_config returns singleton."""
        reset_config()
        config1 = get_config()
        config2 = get_config()
        assert config1 is config2

    def test_set_config(self):
        """Test set_config overrides global config."""
        custom_config = Config(random_seed=123)
        set_config(custom_config)

        retrieved = get_config()
        assert retrieved.random_seed == 123

        # Clean up
        reset_config()

    def test_reset_config(self):
        """Test reset_config clears global config."""
        get_config()  # Initialize
        reset_config()

        # Next call should create new instance
        config = get_config()
        assert config is not None
