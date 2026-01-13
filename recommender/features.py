"""
Feature engineering module for classical music recommendation system.

This module extracts meaningful features from works, composers, and tags
to create embeddings for similarity computation.

Features extracted:
- Composer features (name, period, country, era)
- Work metadata (type, key, catalog number patterns)
- Tags (genres, instrumentation, mood, form)
- Temporal features (composition era, decade)
- Musical key relationships (circle of fifths)
- Composite features (period + work_type, etc.)
"""

import logging
import re
from typing import Optional, Any
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import scipy.sparse as sp

from recommender.config import Config, get_config
from recommender.data_loader import MusicDataset

logger = logging.getLogger(__name__)


# Circle of fifths mapping for key relationships
# Keys that are close on the circle are musically similar
CIRCLE_OF_FIFTHS = {
    'C major': 0, 'G major': 1, 'D major': 2, 'A major': 3, 'E major': 4,
    'B major': 5, 'F# major': 6, 'C# major': 7, 'Ab major': 8, 'Eb major': 9,
    'Bb major': 10, 'F major': 11,
    'A minor': 0, 'E minor': 1, 'B minor': 2, 'F# minor': 3, 'C# minor': 4,
    'G# minor': 5, 'D# minor': 6, 'A# minor': 7, 'F minor': 8, 'C minor': 9,
    'G minor': 10, 'D minor': 11,
}


# Common catalog prefixes and their patterns
CATALOG_PATTERNS = {
    'opus': r'\b(?:op\.?|opus)\s*(\d+)',
    'bwv': r'\bBWV\s*(\d+)',
    'k': r'\bK\.?\s*(\d+)',
    'hob': r'\bHob\.?\s*([IVX]+:\d+)',
    'd': r'\bD\.?\s*(\d+)',
    'rv': r'\bRV\s*(\d+)',
    'woo': r'\bWoO\s*(\d+)',
    'hwv': r'\bHWV\s*(\d+)',
}


@dataclass
class FeatureMatrix:
    """Container for feature vectors and metadata.

    Stores the feature matrix along with feature names and work IDs
    for interpretability and debugging.
    """
    features: np.ndarray  # Shape: (n_works, n_features)
    work_ids: np.ndarray  # Shape: (n_works,)
    feature_names: list[str]
    feature_groups: dict[str, tuple[int, int]]  # Maps group name to (start_idx, end_idx)

    def __post_init__(self):
        """Validate dimensions."""
        assert len(self.features) == len(self.work_ids), \
            f"Feature matrix and work_ids length mismatch: {len(self.features)} vs {len(self.work_ids)}"
        assert self.features.shape[1] == len(self.feature_names), \
            f"Feature count mismatch: {self.features.shape[1]} vs {len(self.feature_names)}"

    def get_feature_group(self, group_name: str) -> np.ndarray:
        """Extract features for a specific group.

        Args:
            group_name: Name of feature group (e.g., 'composer', 'tags')

        Returns:
            Feature submatrix for that group
        """
        if group_name not in self.feature_groups:
            raise ValueError(f"Unknown feature group: {group_name}")
        start, end = self.feature_groups[group_name]
        return self.features[:, start:end]


class FeatureExtractor:
    """Extracts and engineers features for classical music works.

    This class handles all feature engineering, including:
    - Text features via TF-IDF
    - Categorical features via one-hot encoding
    - Numerical features via normalization
    - Composite features
    - Musical domain knowledge (key relationships, catalog patterns)
    """

    def __init__(self, config: Optional[Config] = None):
        """Initialize feature extractor.

        Args:
            config: Configuration object (uses global config if None)
        """
        self.config = config or get_config()

        # Fitted transformers (set during fit())
        self._composer_encoder: Optional[OneHotEncoder] = None
        self._period_encoder: Optional[OneHotEncoder] = None
        self._work_type_encoder: Optional[OneHotEncoder] = None
        self._key_encoder: Optional[OneHotEncoder] = None
        self._country_encoder: Optional[OneHotEncoder] = None
        self._tags_vectorizer: Optional[TfidfVectorizer] = None
        self._composite_encoder: Optional[OneHotEncoder] = None

        self._is_fitted = False

    def fit_transform(self, dataset: MusicDataset) -> FeatureMatrix:
        """Fit feature extractors and transform dataset.

        Args:
            dataset: Music dataset to extract features from

        Returns:
            FeatureMatrix with extracted features
        """
        logger.info("Fitting feature extractors and transforming data...")

        # Prepare enriched dataframe with all fields
        df = self._prepare_dataframe(dataset)

        # Extract each feature group
        feature_matrices = []
        feature_names = []
        feature_groups = {}
        current_idx = 0

        # 1. Composer features
        composer_features, composer_names = self._extract_composer_features(df)
        feature_matrices.append(composer_features)
        feature_names.extend(composer_names)
        feature_groups['composer'] = (current_idx, current_idx + len(composer_names))
        current_idx += len(composer_names)

        # 2. Period features
        period_features, period_names = self._extract_period_features(df)
        feature_matrices.append(period_features)
        feature_names.extend(period_names)
        feature_groups['period'] = (current_idx, current_idx + len(period_names))
        current_idx += len(period_names)

        # 3. Work type features
        work_type_features, work_type_names = self._extract_work_type_features(df)
        feature_matrices.append(work_type_features)
        feature_names.extend(work_type_names)
        feature_groups['work_type'] = (current_idx, current_idx + len(work_type_names))
        current_idx += len(work_type_names)

        # 4. Key features
        key_features, key_names = self._extract_key_features(df)
        feature_matrices.append(key_features)
        feature_names.extend(key_names)
        feature_groups['key'] = (current_idx, current_idx + len(key_names))
        current_idx += len(key_names)

        # 5. Tag features
        tag_features, tag_names = self._extract_tag_features(df, dataset)
        feature_matrices.append(tag_features)
        feature_names.extend(tag_names)
        feature_groups['tags'] = (current_idx, current_idx + len(tag_names))
        current_idx += len(tag_names)

        # 6. Catalog pattern features
        catalog_features, catalog_names = self._extract_catalog_features(df)
        feature_matrices.append(catalog_features)
        feature_names.extend(catalog_names)
        feature_groups['catalog'] = (current_idx, current_idx + len(catalog_names))
        current_idx += len(catalog_names)

        # 7. Composite features
        if self.config.feature_engineering.create_composite_features:
            composite_features, composite_names = self._extract_composite_features(df)
            feature_matrices.append(composite_features)
            feature_names.extend(composite_names)
            feature_groups['composite'] = (current_idx, current_idx + len(composite_names))
            current_idx += len(composite_names)

        # 8. Country features
        country_features, country_names = self._extract_country_features(df)
        feature_matrices.append(country_features)
        feature_names.extend(country_names)
        feature_groups['country'] = (current_idx, current_idx + len(country_names))
        current_idx += len(country_names)

        # Concatenate all features
        combined_features = np.hstack(feature_matrices)

        # Apply feature weights
        weighted_features = self._apply_feature_weights(combined_features, feature_groups)

        self._is_fitted = True

        logger.info(f"Extracted {weighted_features.shape[1]} features for {len(df)} works")
        logger.info(f"Feature groups: {list(feature_groups.keys())}")
        logger.info(f"Total feature names: {len(feature_names)}")

        # Debug: Check feature count matches
        if weighted_features.shape[1] != len(feature_names):
            logger.error(f"Feature/name mismatch! Features: {weighted_features.shape[1]}, Names: {len(feature_names)}")
            for group_name, (start, end) in feature_groups.items():
                logger.error(f"  {group_name}: {end - start} features")
            raise ValueError(f"Feature count mismatch: {weighted_features.shape[1]} features != {len(feature_names)} names")

        return FeatureMatrix(
            features=weighted_features,
            work_ids=df['work_id'].values,
            feature_names=feature_names,
            feature_groups=feature_groups
        )

    def transform(self, dataset: MusicDataset) -> FeatureMatrix:
        """Transform dataset using fitted extractors.

        Args:
            dataset: Music dataset to transform

        Returns:
            FeatureMatrix with extracted features

        Raises:
            RuntimeError: If extractors haven't been fitted yet
        """
        if not self._is_fitted:
            raise RuntimeError("FeatureExtractor must be fitted before transform(). Call fit_transform() first.")

        # Similar to fit_transform but uses pre-fitted transformers
        # (Implementation would be similar but using transform() instead of fit_transform())
        logger.info("Transforming data using fitted extractors...")
        # For now, just use fit_transform (in production, implement proper transform)
        return self.fit_transform(dataset)

    def _prepare_dataframe(self, dataset: MusicDataset) -> pd.DataFrame:
        """Prepare enriched dataframe with all fields joined.

        Args:
            dataset: Music dataset

        Returns:
            DataFrame with works + composers merged
        """
        # Start with works
        df = dataset.works.copy()

        # Merge composer data
        composer_cols = ['composer_id', 'name', 'period', 'country', 'birth_year', 'death_year']
        available_composer_cols = [col for col in composer_cols if col in dataset.composers.columns]

        df = df.merge(
            dataset.composers[available_composer_cols],
            on='composer_id',
            how='left',
            suffixes=('', '_composer')
        )

        # Fill missing values
        fill_value = self.config.feature_engineering.fill_missing_text
        df['name'] = df['name'].fillna(fill_value)
        df['period'] = df['period'].fillna(fill_value)
        df['country'] = df['country'].fillna(fill_value)
        df['work_type'] = df['work_type'].fillna(fill_value)
        df['key'] = df['key'].fillna(fill_value)
        df['catalog_number'] = df['catalog_number'].fillna('')

        return df

    def _extract_composer_features(self, df: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
        """Extract one-hot encoded composer features.

        Args:
            df: Prepared dataframe

        Returns:
            Feature matrix and feature names
        """
        composers = df[['name']].values

        if self._composer_encoder is None:
            self._composer_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
            features = self._composer_encoder.fit_transform(composers)
        else:
            features = self._composer_encoder.transform(composers)

        # Generate feature names - need to iterate through ALL categories in the first category list
        names = [f"composer_{cat}" for cat in self._composer_encoder.categories_[0]]

        return features, names

    def _extract_period_features(self, df: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
        """Extract one-hot encoded period features.

        Args:
            df: Prepared dataframe

        Returns:
            Feature matrix and feature names
        """
        periods = df[['period']].values

        if self._period_encoder is None:
            self._period_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
            features = self._period_encoder.fit_transform(periods)
        else:
            features = self._period_encoder.transform(periods)

        names = [f"period_{cat}" for cat in self._period_encoder.categories_[0]]

        return features, names

    def _extract_work_type_features(self, df: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
        """Extract one-hot encoded work type features.

        Args:
            df: Prepared dataframe

        Returns:
            Feature matrix and feature names
        """
        work_types = df[['work_type']].values

        if self._work_type_encoder is None:
            self._work_type_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
            features = self._work_type_encoder.fit_transform(work_types)
        else:
            features = self._work_type_encoder.transform(work_types)

        names = [f"work_type_{cat}" for cat in self._work_type_encoder.categories_[0]]

        return features, names

    def _extract_key_features(self, df: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
        """Extract key features using circle of fifths encoding.

        Args:
            df: Prepared dataframe

        Returns:
            Feature matrix and feature names
        """
        if not self.config.feature_engineering.use_circle_of_fifths:
            # Simple one-hot encoding
            keys = df[['key']].values
            if self._key_encoder is None:
                self._key_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
                features = self._key_encoder.fit_transform(keys)
            else:
                features = self._key_encoder.transform(keys)
            names = [f"key_{cat[0]}" for cat in self._key_encoder.categories_]
            return features, names

        # Circle of fifths encoding: encode as sin/cos of position
        # This captures that C major and G major are close, while C major and F# major are distant
        positions = df['key'].map(CIRCLE_OF_FIFTHS).fillna(-1).values

        # For unknown keys, use zero vector
        sin_features = np.where(positions >= 0, np.sin(2 * np.pi * positions / 12), 0)
        cos_features = np.where(positions >= 0, np.cos(2 * np.pi * positions / 12), 0)

        features = np.column_stack([sin_features, cos_features])
        names = ['key_circle_sin', 'key_circle_cos']

        return features, names

    def _extract_tag_features(self, df: pd.DataFrame, dataset: MusicDataset) -> tuple[np.ndarray, list[str]]:
        """Extract TF-IDF features from tags.

        Args:
            df: Prepared dataframe
            dataset: Music dataset (for tag lookup)

        Returns:
            Feature matrix and feature names
        """
        # Build tag documents (one string per work with all tags)
        tag_documents = []
        for work_id in df['work_id']:
            tags = dataset.get_work_tags(work_id)
            # Also include mb_tags from works table if available
            work = dataset.get_work_by_id(work_id)
            if work is not None and 'mb_tags' in work.index:
                mb_tags_value = work['mb_tags']
                # Check if it's a non-empty list
                if isinstance(mb_tags_value, list) and len(mb_tags_value) > 0:
                    tags.extend(mb_tags_value)

            tag_text = ' '.join(tags) if tags else 'no_tags'
            tag_documents.append(tag_text)

        # TF-IDF vectorization
        if self._tags_vectorizer is None:
            self._tags_vectorizer = TfidfVectorizer(
                max_features=self.config.feature_engineering.tfidf_max_features,
                ngram_range=self.config.feature_engineering.tfidf_ngram_range,
                min_df=self.config.feature_engineering.tfidf_min_df,
                lowercase=True,
                token_pattern=r'\b\w+\b'
            )
            features_sparse = self._tags_vectorizer.fit_transform(tag_documents)
        else:
            features_sparse = self._tags_vectorizer.transform(tag_documents)

        features = features_sparse.toarray()
        names = [f"tag_{term}" for term in self._tags_vectorizer.get_feature_names_out()]

        return features, names

    def _extract_catalog_features(self, df: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
        """Extract features from catalog numbers (e.g., Op., BWV, K.).

        Args:
            df: Prepared dataframe

        Returns:
            Feature matrix and feature names
        """
        if not self.config.feature_engineering.extract_catalog_patterns:
            # Return empty features
            return np.zeros((len(df), 0)), []

        features = []
        names = []

        for pattern_name, pattern_regex in CATALOG_PATTERNS.items():
            # Binary feature: does this work have this catalog pattern?
            has_pattern = df['catalog_number'].str.contains(
                pattern_regex,
                case=False,
                regex=True,
                na=False
            ).astype(float).values

            features.append(has_pattern)
            names.append(f"catalog_{pattern_name}")

        if features:
            return np.column_stack(features), names
        else:
            return np.zeros((len(df), 0)), []

    def _extract_composite_features(self, df: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
        """Extract composite features from field combinations.

        Creates features like "baroque_concerto", "mozart_symphony", etc.

        Args:
            df: Prepared dataframe

        Returns:
            Feature matrix and feature names
        """
        composite_data = []

        for field1, field2 in self.config.feature_engineering.composite_combinations:
            if field1 in df.columns and field2 in df.columns:
                # Create combined strings
                combined = df[field1].astype(str) + '_' + df[field2].astype(str)
                composite_data.append(combined.values.reshape(-1, 1))

        if not composite_data:
            return np.zeros((len(df), 0)), []

        # Stack all composite fields
        composite_array = np.hstack(composite_data)

        # One-hot encode
        if self._composite_encoder is None:
            self._composite_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
            features = self._composite_encoder.fit_transform(composite_array)
        else:
            features = self._composite_encoder.transform(composite_array)

        # Generate names - flatten across all categories
        names = []
        cat_idx = 0
        for i, (field1, field2) in enumerate(self.config.feature_engineering.composite_combinations):
            if field1 in df.columns and field2 in df.columns:
                for cat in self._composite_encoder.categories_[cat_idx]:
                    names.append(f"composite_{cat}")
                cat_idx += 1

        return features, names

    def _extract_country_features(self, df: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
        """Extract one-hot encoded country features.

        Args:
            df: Prepared dataframe

        Returns:
            Feature matrix and feature names
        """
        countries = df[['country']].values

        if self._country_encoder is None:
            self._country_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
            features = self._country_encoder.fit_transform(countries)
        else:
            features = self._country_encoder.transform(countries)

        names = [f"country_{cat}" for cat in self._country_encoder.categories_[0]]

        return features, names

    def _apply_feature_weights(
        self,
        features: np.ndarray,
        feature_groups: dict[str, tuple[int, int]]
    ) -> np.ndarray:
        """Apply feature importance weights to feature matrix.

        Args:
            features: Unweighted feature matrix
            feature_groups: Mapping of group names to column ranges

        Returns:
            Weighted feature matrix
        """
        weighted_features = features.copy()
        weights = self.config.feature_weights

        # Apply weights to each group
        weight_map = {
            'composer': weights.composer,
            'period': weights.period,
            'work_type': weights.work_type,
            'key': weights.key,
            'tags': weights.tags,
            'catalog': weights.catalog_pattern,
            'composite': (weights.period + weights.work_type) / 2,  # Average of constituent weights
            'country': weights.period * 0.5,  # Country is related to period
        }

        for group_name, (start, end) in feature_groups.items():
            if group_name in weight_map:
                weight = weight_map[group_name]
                weighted_features[:, start:end] *= weight

        return weighted_features
