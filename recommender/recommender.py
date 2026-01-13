"""
Core recommendation engine for classical music works.

This module provides the main recommendation functionality:
- Finding similar works by ID
- Searching and recommending by query
- Filtering recommendations by composer, period, etc.
- Diverse recommendations to avoid clustering
- Explanation generation
"""

import logging
from typing import Optional, List, Dict, Any
from dataclasses import dataclass
from pathlib import Path
import pickle

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances, manhattan_distances

from recommender.config import Config, get_config
from recommender.data_loader import DataLoader, MusicDataset
from recommender.features import FeatureExtractor, FeatureMatrix
from recommender.explanation_formatter import ExplanationFormatter

logger = logging.getLogger(__name__)


@dataclass
class Recommendation:
    """A single recommendation with metadata.

    Represents one recommended work with its similarity score,
    rank, and explanation of why it was recommended.
    """
    work_id: str
    title: str
    composer: str
    work_type: Optional[str]
    key: Optional[str]
    period: Optional[str]
    similarity_score: float
    rank: int
    explanation: Optional[str] = None
    tags: Optional[List[str]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'work_id': self.work_id,
            'title': self.title,
            'composer': self.composer,
            'work_type': self.work_type,
            'key': self.key,
            'period': self.period,
            'similarity_score': float(self.similarity_score),
            'rank': self.rank,
            'explanation': self.explanation,
            'tags': self.tags
        }


class MusicRecommender:
    """Classical music recommendation engine.

    This class handles:
    - Loading and caching data
    - Computing similarity between works
    - Generating recommendations with diversity
    - Explaining why works were recommended
    """

    def __init__(self, config: Optional[Config] = None):
        """Initialize recommender.

        Args:
            config: Configuration object (uses global config if None)
        """
        self.config = config or get_config()
        self.explanation_formatter = ExplanationFormatter()

        # Data and features (loaded lazily)
        self._dataset: Optional[MusicDataset] = None
        self._features: Optional[FeatureMatrix] = None
        self._similarity_matrix: Optional[np.ndarray] = None

        # Index for fast lookups
        self._work_id_to_idx: Optional[Dict[str, int]] = None

        self._is_ready = False

    def load(self, force_rebuild: bool = False) -> None:
        """Load data and build feature vectors.

        Args:
            force_rebuild: If True, rebuild features even if cache exists
        """
        logger.info("Loading recommender system...")

        # Load dataset
        loader = DataLoader(self.config)
        self._dataset = loader.load()

        # Try to load from cache
        if self.config.recommender.enable_cache and not force_rebuild:
            if self._load_from_cache():
                logger.info("Loaded features from cache")
                self._build_indices()
                self._is_ready = True
                return

        # Build features from scratch
        logger.info("Building features from scratch...")
        extractor = FeatureExtractor(self.config)
        self._features = extractor.fit_transform(self._dataset)

        # Pre-compute similarity matrix if enabled
        if self.config.recommender.enable_cache:
            logger.info("Pre-computing similarity matrix...")
            self._similarity_matrix = self._compute_all_similarities(self._features.features)
            self._save_to_cache()

        self._build_indices()
        self._is_ready = True
        logger.info("Recommender system ready")

    def recommend_similar(
        self,
        work_id: str,
        n: int = 10,
        exclude_same_work: bool = True
    ) -> List[Recommendation]:
        """Find similar works by work ID.

        Args:
            work_id: ID of the seed work
            n: Number of recommendations to return
            exclude_same_work: Whether to exclude the seed work from results

        Returns:
            List of Recommendation objects, sorted by similarity

        Raises:
            ValueError: If work_id not found
            RuntimeError: If recommender not loaded
        """
        self._ensure_ready()

        if work_id not in self._work_id_to_idx:
            raise ValueError(f"Work ID not found: {work_id}")

        work_idx = self._work_id_to_idx[work_id]

        # Get similarity scores
        if self._similarity_matrix is not None:
            # Use pre-computed matrix
            similarities = self._similarity_matrix[work_idx]
        else:
            # Compute on demand
            work_vector = self._features.features[work_idx:work_idx+1]
            similarities = self._compute_similarity(work_vector, self._features.features)[0]

        # Get top-N similar works
        if exclude_same_work:
            # Set self-similarity to -inf to exclude it
            similarities = similarities.copy()
            similarities[work_idx] = -np.inf

        top_indices = np.argsort(similarities)[::-1][:n]
        top_scores = similarities[top_indices]

        # Build recommendations
        recommendations = []
        seed_work = self._dataset.get_work_by_id(work_id)

        for rank, (idx, score) in enumerate(zip(top_indices, top_scores), start=1):
            rec_work_id = self._features.work_ids[idx]
            rec = self._build_recommendation(
                rec_work_id,
                score,
                rank,
                seed_work=seed_work
            )
            recommendations.append(rec)

        return recommendations

    def recommend_by_query(
        self,
        query: str,
        n: int = 10
    ) -> List[Recommendation]:
        """Search for works and recommend similar ones.

        Args:
            query: Search query (title substring match)
            n: Number of recommendations to return

        Returns:
            List of recommendations for the best-matching work

        Raises:
            ValueError: If no works match the query
            RuntimeError: If recommender not loaded
        """
        self._ensure_ready()

        # Search for matching works
        matches = self._dataset.search_works(query, limit=1)

        if len(matches) == 0:
            raise ValueError(f"No works found matching query: '{query}'")

        # Use the best match as seed
        seed_work_id = matches.iloc[0]['work_id']
        logger.info(f"Query '{query}' matched work: {matches.iloc[0]['title']}")

        return self.recommend_similar(seed_work_id, n=n, exclude_same_work=True)

    def recommend_by_filters(
        self,
        composer: Optional[str] = None,
        period: Optional[str] = None,
        work_type: Optional[str] = None,
        key: Optional[str] = None,
        n: int = 10
    ) -> List[Recommendation]:
        """Get recommendations filtered by criteria.

        This is useful for browsing (e.g., "show me baroque concertos").

        Args:
            composer: Filter by composer name
            period: Filter by period/era
            work_type: Filter by work type
            key: Filter by musical key
            n: Number of recommendations to return

        Returns:
            List of recommendations matching filters

        Raises:
            ValueError: If no works match the filters
            RuntimeError: If recommender not loaded
        """
        self._ensure_ready()

        # Find composer_id if composer name provided
        composer_id = None
        if composer:
            composer_match = self._dataset.composers[
                self._dataset.composers['name'].str.contains(composer, case=False, na=False)
            ]
            if len(composer_match) > 0:
                composer_id = composer_match.iloc[0]['composer_id']
            else:
                raise ValueError(f"Composer not found: '{composer}'")

        # Filter works
        filtered = self._dataset.filter_works(
            composer_id=composer_id,
            work_type=work_type,
            key=key,
            period=period
        )

        if len(filtered) == 0:
            raise ValueError("No works match the specified filters")

        # Return top N by some criteria (for now, just return first N)
        # In production, could rank by popularity, tags, etc.
        recommendations = []
        for rank, (idx, row) in enumerate(filtered.head(n).iterrows(), start=1):
            rec = self._build_recommendation(
                row['work_id'],
                score=1.0,  # No similarity score for filtered browse
                rank=rank,
                seed_work=None
            )
            recommendations.append(rec)

        return recommendations

    def recommend_diverse(
        self,
        work_id: str,
        n: int = 10,
        diversity_weight: Optional[float] = None
    ) -> List[Recommendation]:
        """Get diverse recommendations that avoid clustering.

        Uses Maximal Marginal Relevance (MMR) to balance similarity
        with diversity, avoiding recommending 10 nearly identical works.

        Args:
            work_id: ID of the seed work
            n: Number of recommendations to return
            diversity_weight: Weight for diversity (0=pure similarity, 1=pure diversity)
                            If None, uses config default

        Returns:
            List of diverse recommendations

        Raises:
            ValueError: If work_id not found
            RuntimeError: If recommender not loaded
        """
        self._ensure_ready()

        if diversity_weight is None:
            diversity_weight = self.config.recommender.diversity_weight

        if work_id not in self._work_id_to_idx:
            raise ValueError(f"Work ID not found: {work_id}")

        work_idx = self._work_id_to_idx[work_id]

        # Get similarity to seed work
        if self._similarity_matrix is not None:
            similarities_to_seed = self._similarity_matrix[work_idx].copy()
        else:
            work_vector = self._features.features[work_idx:work_idx+1]
            similarities_to_seed = self._compute_similarity(
                work_vector,
                self._features.features
            )[0]

        # Exclude seed work
        similarities_to_seed[work_idx] = -np.inf

        # MMR algorithm
        selected_indices = []
        selected_scores = []
        candidate_indices = list(range(len(self._features.features)))
        candidate_indices.remove(work_idx)

        for _ in range(n):
            if not candidate_indices:
                break

            # Compute MMR scores for all candidates
            mmr_scores = []
            for candidate_idx in candidate_indices:
                # Similarity to seed (relevance)
                relevance = similarities_to_seed[candidate_idx]

                # Max similarity to already selected items (redundancy)
                if selected_indices:
                    similarities_to_selected = []
                    for selected_idx in selected_indices:
                        if self._similarity_matrix is not None:
                            sim = self._similarity_matrix[candidate_idx, selected_idx]
                        else:
                            vec1 = self._features.features[candidate_idx:candidate_idx+1]
                            vec2 = self._features.features[selected_idx:selected_idx+1]
                            sim = self._compute_similarity(vec1, vec2)[0, 0]
                        similarities_to_selected.append(sim)
                    redundancy = max(similarities_to_selected)
                else:
                    redundancy = 0

                # MMR = λ * relevance - (1-λ) * redundancy
                mmr = diversity_weight * relevance - (1 - diversity_weight) * redundancy
                mmr_scores.append(mmr)

            # Select best MMR candidate
            best_candidate_position = np.argmax(mmr_scores)
            best_idx = candidate_indices[best_candidate_position]

            selected_indices.append(best_idx)
            selected_scores.append(similarities_to_seed[best_idx])
            candidate_indices.pop(best_candidate_position)

        # Build recommendations
        recommendations = []
        seed_work = self._dataset.get_work_by_id(work_id)

        for rank, (idx, score) in enumerate(zip(selected_indices, selected_scores), start=1):
            rec_work_id = self._features.work_ids[idx]
            rec = self._build_recommendation(
                rec_work_id,
                score,
                rank,
                seed_work=seed_work
            )
            recommendations.append(rec)

        return recommendations

    def get_work_info(self, work_id: str) -> Optional[Dict[str, Any]]:
        """Get detailed information about a work.

        Args:
            work_id: Work identifier

        Returns:
            Dictionary with work metadata or None if not found
        """
        self._ensure_ready()

        work = self._dataset.get_work_by_id(work_id)
        if work is None:
            return None

        tags = self._dataset.get_work_tags(work_id)
        composer_name = self._dataset.get_composer_name(work['composer_id'])

        return {
            'work_id': work_id,
            'title': work['title'],
            'composer': composer_name,
            'work_type': work.get('work_type'),
            'key': work.get('key'),
            'catalog_number': work.get('catalog_number'),
            'tags': tags
        }

    def _build_recommendation(
        self,
        work_id: str,
        score: float,
        rank: int,
        seed_work: Optional[pd.Series] = None
    ) -> Recommendation:
        """Build a Recommendation object with metadata and explanation.

        Args:
            work_id: Recommended work ID
            score: Similarity score
            rank: Rank in recommendation list
            seed_work: Original seed work (for explanation generation)

        Returns:
            Recommendation object
        """
        work = self._dataset.get_work_by_id(work_id)
        if work is None:
            raise ValueError(f"Work not found: {work_id}")

        composer_name = self._dataset.get_composer_name(work['composer_id'])
        tags = self._dataset.get_work_tags(work_id)

        # Get period if available
        period = None
        if work['composer_id']:
            composer = self._dataset.composers[
                self._dataset.composers['composer_id'] == work['composer_id']
            ]
            if len(composer) > 0 and 'period' in composer.columns:
                period = composer.iloc[0]['period']

        # Generate explanation
        explanation = None
        if self.config.recommender.generate_explanations and seed_work is not None:
            explanation = self._generate_explanation(seed_work, work, tags)

        return Recommendation(
            work_id=work_id,
            title=work['title'],
            composer=composer_name,
            work_type=work.get('work_type'),
            key=work.get('key'),
            period=period,
            similarity_score=score,
            rank=rank,
            explanation=explanation,
            tags=tags[:5] if tags else None  # Limit to top 5 tags
        )

    def _generate_explanation(
        self,
        seed_work: pd.Series,
        rec_work: pd.Series,
        rec_tags: List[str]
    ) -> str:
        """Generate human-readable explanation for recommendation.

        Args:
            seed_work: Original seed work
            rec_work: Recommended work
            rec_tags: Tags for recommended work

        Returns:
            Explanation string
        """
        reasons = []

        # Same composer
        if seed_work['composer_id'] == rec_work['composer_id']:
            reasons.append("same composer")

        # Same work type
        if seed_work.get('work_type') and rec_work.get('work_type'):
            if seed_work['work_type'] == rec_work['work_type']:
                reasons.append(f"both {rec_work['work_type']}s")

        # Same key
        if seed_work.get('key') and rec_work.get('key'):
            if seed_work['key'] == rec_work['key']:
                reasons.append(f"both in {rec_work['key']}")

        # Same period
        seed_composer = self._dataset.composers[
            self._dataset.composers['composer_id'] == seed_work['composer_id']
        ]
        rec_composer = self._dataset.composers[
            self._dataset.composers['composer_id'] == rec_work['composer_id']
        ]

        if len(seed_composer) > 0 and len(rec_composer) > 0:
            if 'period' in seed_composer.columns and 'period' in rec_composer.columns:
                seed_period = seed_composer.iloc[0]['period']
                rec_period = rec_composer.iloc[0]['period']
                if seed_period == rec_period and seed_period:
                    reasons.append(f"both {rec_period}")

        # Common tags
        seed_tags = self._dataset.get_work_tags(seed_work['work_id'])
        common_tags = set(seed_tags).intersection(set(rec_tags))
        if common_tags:
            # Pick most interesting common tag
            tag = list(common_tags)[0]
            reasons.append(f"similar style ({tag})")

        if not reasons:
            return self.explanation_formatter.format_similarity_explanation(
                reasons=[],
                seed_work=seed_work,
                rec_work=rec_work
            )

        return self.explanation_formatter.format_similarity_explanation(
            reasons=reasons,
            seed_work=seed_work,
            rec_work=rec_work
        )

    def _compute_similarity(
        self,
        features1: np.ndarray,
        features2: np.ndarray
    ) -> np.ndarray:
        """Compute similarity between feature vectors.

        Args:
            features1: First feature matrix (n_samples1, n_features)
            features2: Second feature matrix (n_samples2, n_features)

        Returns:
            Similarity matrix (n_samples1, n_samples2)
        """
        metric = self.config.recommender.similarity_metric

        if metric == 'cosine':
            return cosine_similarity(features1, features2)
        elif metric == 'euclidean':
            # Convert distance to similarity (inverse)
            distances = euclidean_distances(features1, features2)
            return 1 / (1 + distances)
        elif metric == 'manhattan':
            distances = manhattan_distances(features1, features2)
            return 1 / (1 + distances)
        elif metric == 'dot':
            return np.dot(features1, features2.T)
        else:
            raise ValueError(f"Unknown similarity metric: {metric}")

    def _compute_all_similarities(self, features: np.ndarray) -> np.ndarray:
        """Compute full similarity matrix.

        Args:
            features: Feature matrix (n_works, n_features)

        Returns:
            Similarity matrix (n_works, n_works)
        """
        return self._compute_similarity(features, features)

    def _build_indices(self) -> None:
        """Build lookup indices for fast access."""
        self._work_id_to_idx = {
            work_id: idx
            for idx, work_id in enumerate(self._features.work_ids)
        }

    def _ensure_ready(self) -> None:
        """Ensure recommender is loaded and ready.

        Raises:
            RuntimeError: If recommender not loaded
        """
        if not self._is_ready:
            raise RuntimeError(
                "Recommender not loaded. Call load() first."
            )

    def _load_from_cache(self) -> bool:
        """Load features and similarity matrix from cache.

        Returns:
            True if successfully loaded from cache, False otherwise
        """
        feature_cache = self.config.paths.feature_vectors_cache
        similarity_cache = self.config.paths.similarity_matrix_cache

        if not feature_cache.exists():
            return False

        try:
            # Load features
            with open(feature_cache, 'rb') as f:
                self._features = pickle.load(f)

            # Load similarity matrix if exists
            if similarity_cache.exists():
                similarity_data = np.load(similarity_cache)
                self._similarity_matrix = similarity_data['similarity_matrix']

            return True
        except Exception as e:
            logger.warning(f"Failed to load from cache: {e}")
            return False

    def _save_to_cache(self) -> None:
        """Save features and similarity matrix to cache."""
        try:
            feature_cache = self.config.paths.feature_vectors_cache
            similarity_cache = self.config.paths.similarity_matrix_cache

            # Ensure cache directory exists
            self.config.paths.cache_dir.mkdir(parents=True, exist_ok=True)

            # Save features
            with open(feature_cache, 'wb') as f:
                pickle.dump(self._features, f)

            # Save similarity matrix
            if self._similarity_matrix is not None:
                np.savez_compressed(
                    similarity_cache,
                    similarity_matrix=self._similarity_matrix
                )

            logger.info("Saved features and similarity matrix to cache")
        except Exception as e:
            logger.warning(f"Failed to save to cache: {e}")
