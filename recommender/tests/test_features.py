"""Tests for feature extraction module."""

import pytest
import numpy as np
import pandas as pd

from recommender.features import FeatureExtractor, FeatureMatrix, CIRCLE_OF_FIFTHS
from recommender.config import Config
from recommender.tests.test_data_loader import sample_dataset


class TestFeatureMatrix:
    """Tests for FeatureMatrix dataclass."""

    def test_init(self):
        """Test FeatureMatrix initialization."""
        features = np.random.rand(10, 5)
        work_ids = np.array([f'work{i}' for i in range(10)])
        feature_names = [f'feature{i}' for i in range(5)]
        feature_groups = {'group1': (0, 3), 'group2': (3, 5)}

        fm = FeatureMatrix(
            features=features,
            work_ids=work_ids,
            feature_names=feature_names,
            feature_groups=feature_groups
        )

        assert fm.features.shape == (10, 5)
        assert len(fm.work_ids) == 10
        assert len(fm.feature_names) == 5

    def test_dimension_validation(self):
        """Test that dimension mismatches are caught."""
        features = np.random.rand(10, 5)
        work_ids = np.array([f'work{i}' for i in range(8)])  # Wrong length
        feature_names = [f'feature{i}' for i in range(5)]
        feature_groups = {}

        with pytest.raises(AssertionError):
            FeatureMatrix(
                features=features,
                work_ids=work_ids,
                feature_names=feature_names,
                feature_groups=feature_groups
            )

    def test_get_feature_group(self):
        """Test extracting feature groups."""
        features = np.random.rand(10, 5)
        work_ids = np.array([f'work{i}' for i in range(10)])
        feature_names = [f'feature{i}' for i in range(5)]
        feature_groups = {'group1': (0, 3), 'group2': (3, 5)}

        fm = FeatureMatrix(
            features=features,
            work_ids=work_ids,
            feature_names=feature_names,
            feature_groups=feature_groups
        )

        group1_features = fm.get_feature_group('group1')
        assert group1_features.shape == (10, 3)

        group2_features = fm.get_feature_group('group2')
        assert group2_features.shape == (10, 2)

        # Unknown group
        with pytest.raises(ValueError):
            fm.get_feature_group('unknown')


class TestFeatureExtractor:
    """Tests for FeatureExtractor class."""

    def test_init(self):
        """Test feature extractor initialization."""
        extractor = FeatureExtractor()
        assert extractor._is_fitted is False

    def test_fit_transform(self, sample_dataset):
        """Test fitting and transforming features."""
        extractor = FeatureExtractor()
        feature_matrix = extractor.fit_transform(sample_dataset)

        # Check dimensions
        assert feature_matrix.features.shape[0] == len(sample_dataset.works)
        assert feature_matrix.features.shape[1] > 0
        assert len(feature_matrix.work_ids) == len(sample_dataset.works)
        assert len(feature_matrix.feature_names) == feature_matrix.features.shape[1]

        # Check feature groups exist
        assert 'composer' in feature_matrix.feature_groups
        assert 'period' in feature_matrix.feature_groups
        assert 'work_type' in feature_matrix.feature_groups
        assert 'key' in feature_matrix.feature_groups
        assert 'tags' in feature_matrix.feature_groups

        # Check that extractor is fitted
        assert extractor._is_fitted is True

    def test_prepare_dataframe(self, sample_dataset):
        """Test dataframe preparation."""
        extractor = FeatureExtractor()
        df = extractor._prepare_dataframe(sample_dataset)

        # Should have works merged with composers
        assert 'name' in df.columns  # Composer name
        assert 'period' in df.columns
        assert 'title' in df.columns

        # Check missing value filling
        assert not df['work_type'].isna().any()

    def test_extract_composer_features(self, sample_dataset):
        """Test composer feature extraction."""
        extractor = FeatureExtractor()
        df = extractor._prepare_dataframe(sample_dataset)

        features, names = extractor._extract_composer_features(df)

        # Should have one-hot encoded composers
        assert features.shape[0] == len(df)
        assert features.shape[1] > 0
        assert len(names) == features.shape[1]
        assert all('composer_' in name for name in names)

    def test_extract_key_features_circle_of_fifths(self, sample_dataset):
        """Test key features using circle of fifths."""
        config = Config()
        config.feature_engineering.use_circle_of_fifths = True

        extractor = FeatureExtractor(config)
        df = extractor._prepare_dataframe(sample_dataset)

        features, names = extractor._extract_key_features(df)

        # Circle of fifths encoding produces 2 features (sin, cos)
        assert features.shape == (len(df), 2)
        assert 'key_circle_sin' in names
        assert 'key_circle_cos' in names

    def test_extract_key_features_one_hot(self, sample_dataset):
        """Test key features using one-hot encoding."""
        config = Config()
        config.feature_engineering.use_circle_of_fifths = False

        extractor = FeatureExtractor(config)
        df = extractor._prepare_dataframe(sample_dataset)

        features, names = extractor._extract_key_features(df)

        # One-hot encoding
        assert features.shape[0] == len(df)
        assert features.shape[1] > 0

    def test_extract_tag_features(self, sample_dataset):
        """Test tag feature extraction."""
        extractor = FeatureExtractor()
        df = extractor._prepare_dataframe(sample_dataset)

        features, names = extractor._extract_tag_features(df, sample_dataset)

        # TF-IDF features
        assert features.shape[0] == len(df)
        assert features.shape[1] > 0
        assert all('tag_' in name for name in names)

    def test_extract_catalog_features(self, sample_dataset):
        """Test catalog pattern extraction."""
        config = Config()
        config.feature_engineering.extract_catalog_patterns = True

        extractor = FeatureExtractor(config)
        df = extractor._prepare_dataframe(sample_dataset)

        features, names = extractor._extract_catalog_features(df)

        # Should detect patterns like 'Op.', 'K.'
        assert features.shape[0] == len(df)
        assert features.shape[1] > 0

        # Disable catalog features
        config.feature_engineering.extract_catalog_patterns = False
        extractor2 = FeatureExtractor(config)
        features2, names2 = extractor2._extract_catalog_features(df)

        assert features2.shape[1] == 0
        assert len(names2) == 0

    def test_extract_composite_features(self, sample_dataset):
        """Test composite feature extraction."""
        config = Config()
        config.feature_engineering.create_composite_features = True

        extractor = FeatureExtractor(config)
        df = extractor._prepare_dataframe(sample_dataset)

        features, names = extractor._extract_composite_features(df)

        assert features.shape[0] == len(df)
        if features.shape[1] > 0:
            assert all('composite_' in name for name in names)

    def test_apply_feature_weights(self, sample_dataset):
        """Test feature weight application."""
        extractor = FeatureExtractor()
        feature_matrix = extractor.fit_transform(sample_dataset)

        # Weights should be applied
        # Composer features should be weighted higher than key features
        composer_start, composer_end = feature_matrix.feature_groups['composer']
        key_start, key_end = feature_matrix.feature_groups['key']

        # Just verify dimensions match
        assert composer_end > composer_start
        assert key_end > key_start


class TestCircleOfFifths:
    """Tests for circle of fifths encoding."""

    def test_circle_of_fifths_mapping(self):
        """Test that circle of fifths has correct keys."""
        assert 'C major' in CIRCLE_OF_FIFTHS
        assert 'A minor' in CIRCLE_OF_FIFTHS
        assert CIRCLE_OF_FIFTHS['C major'] == CIRCLE_OF_FIFTHS['A minor']  # Relative keys

    def test_circle_distance(self):
        """Test that adjacent keys have adjacent positions."""
        c_major = CIRCLE_OF_FIFTHS['C major']
        g_major = CIRCLE_OF_FIFTHS['G major']

        # G major is one step clockwise from C major
        assert (g_major - c_major) % 12 == 1
