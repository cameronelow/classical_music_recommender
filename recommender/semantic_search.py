"""
Semantic search module for mood-based classical music recommendations.

This module provides natural language search capabilities using sentence
transformers to enable queries like:
- "I'm feeling moody"
- "bright and cheery but not too energetic"
- "studying music"
- "rainy Sunday morning vibes"
"""

import logging
import json
import pickle
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from recommender.config import Config, get_config
from recommender.data_loader import MusicDataset
from recommender.explanation_formatter import ExplanationFormatter

logger = logging.getLogger(__name__)


# Key-to-mood mapping for all 24 major/minor keys
KEY_MOOD_MAP = {
    # Major keys
    'C major': 'bright cheerful simple pure innocent straightforward',
    'C# major': 'brilliant passionate intense',
    'Db major': 'brilliant passionate intense',
    'D major': 'triumphant festive brilliant joyful celebratory',
    'Eb major': 'heroic noble majestic dignified grand',
    'E major': 'radiant warm luminous hopeful',
    'F major': 'pastoral calm peaceful gentle serene natural',
    'F# major': 'dreamy ethereal mystical',
    'Gb major': 'dreamy ethereal mystical',
    'G major': 'joyful energetic lively playful spirited',
    'Ab major': 'tender loving gentle warm',
    'A major': 'confident bright optimistic',
    'Bb major': 'noble heroic majestic powerful',
    'B major': 'brilliant harsh intense',

    # Minor keys
    'C minor': 'dark tragic serious dramatic ominous',
    'C# minor': 'passionate intense emotional',
    'D minor': 'melancholic serious solemn reflective somber',
    'Eb minor': 'sad anxious troubled',
    'E minor': 'sad lonely contemplative moody introspective wistful',
    'F minor': 'dark brooding melancholic depressing',
    'F# minor': 'sad mysterious somber',
    'G minor': 'serious melancholic pensive sad',
    'G# minor': 'dark depressing tragic',
    'Ab minor': 'dark depressing tragic',
    'A minor': 'melancholic tender sad gentle moody reflective',
    'Bb minor': 'dark tragic horrific',
    'B minor': 'lonely sad isolated contemplative',
}


# Query enhancement mappings
QUERY_ENHANCEMENTS = {
    # Activities
    'studying': 'calm focused minimal instrumental moderate-tempo no-vocals ambient concentration',
    'working out': 'energetic high-energy fast-tempo motivating powerful rhythmic vigorous',
    'relaxing': 'calm peaceful serene gentle slow-tempo soothing tranquil',
    'dinner party': 'elegant sophisticated moderate-tempo not-too-dramatic pleasant background',
    'morning': 'gentle awakening bright moderate-tempo uplifting fresh',
    'evening': 'calm reflective gentle peaceful settling',
    'sleeping': 'calm peaceful slow very-quiet gentle soothing lullaby',
    'reading': 'calm quiet minimal instrumental gentle background ambient',
    'meditation': 'calm peaceful slow minimal serene contemplative',
    'cooking': 'pleasant moderate-tempo uplifting cheerful light',

    # Moods (expand with synonyms)
    'happy': 'joyful cheerful uplifting bright optimistic',
    'sad': 'melancholic sorrowful tragic dark contemplative',
    'moody': 'melancholic introspective contemplative moody brooding',
    'energetic': 'vibrant lively dynamic spirited animated',
    'calm': 'peaceful serene tranquil gentle soothing',
    'dramatic': 'intense powerful emotional passionate theatrical',
    'romantic': 'tender loving passionate intimate emotional',
    'dark': 'ominous brooding somber tragic',
    'bright': 'cheerful radiant luminous joyful',
    'contemplative': 'reflective meditative introspective thoughtful',
    'triumphant': 'victorious celebratory majestic powerful',

    # Times/Weather
    'rainy': 'melancholic calm reflective gentle moody',
    'sunny': 'bright cheerful radiant joyful uplifting',
    'cloudy': 'contemplative moody subdued gentle',
    'winter': 'cold contemplative intimate quiet',
    'spring': 'fresh awakening uplifting bright gentle',
    'summer': 'bright warm energetic joyful',
    'autumn': 'reflective melancholic warm contemplative',

    # Modern/Casual phrases
    'main character': 'triumphant confident bold dramatic powerful heroic grand majestic',
    'villain': 'dark dramatic ominous powerful intense menacing',
    'goofy': 'playful lighthearted cheerful whimsical bright humorous',
    'silly': 'playful lighthearted cheerful bright whimsical',
    'vibing': 'relaxed chill peaceful calm flowing smooth',
    'chill': 'calm relaxed peaceful easy-going mellow',
    'hype': 'energetic exciting powerful dynamic vigorous',
    'cozy': 'warm intimate peaceful gentle comforting',
    'aesthetic': 'beautiful elegant atmospheric dreamy ethereal',
    'sad girl': 'melancholic emotional tender introspective gentle sorrowful',
    'badass': 'powerful bold intense confident dramatic strong',
    'epic': 'grand majestic powerful dramatic triumphant heroic',
    'spicy': 'passionate intense dramatic fiery energetic',
    'elegant': 'sophisticated refined graceful beautiful gentle',
}


@dataclass
class SemanticSearchResult:
    """A single semantic search result with metadata."""
    work_id: str
    title: str
    composer: str
    work_type: Optional[str]
    key: Optional[str]
    period: Optional[str]
    similarity_score: float
    rank: int
    explanation: str
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


@dataclass
class EmbeddingCacheMetadata:
    """Metadata for embedding cache versioning."""
    model_name: str
    creation_date: str
    num_works: int
    embedding_dim: int
    cache_version: str = "1.0"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'model_name': self.model_name,
            'creation_date': self.creation_date,
            'num_works': self.num_works,
            'embedding_dim': self.embedding_dim,
            'cache_version': self.cache_version,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'EmbeddingCacheMetadata':
        """Create from dictionary."""
        return cls(
            model_name=data['model_name'],
            creation_date=data['creation_date'],
            num_works=data['num_works'],
            embedding_dim=data['embedding_dim'],
            cache_version=data.get('cache_version', '1.0')
        )


class WorkDescriptionGenerator:
    """Generates rich text descriptions optimized for semantic search."""

    def __init__(self, dataset: MusicDataset, config: Optional[Config] = None):
        """Initialize description generator.

        Args:
            dataset: Music dataset with works, composers, and tags
            config: Configuration object
        """
        self.dataset = dataset
        self.config = config or get_config()

    def create_work_description(
        self,
        work_id: str,
        composer_name: str,
        period: Optional[str] = None,
        tags: Optional[List[str]] = None
    ) -> str:
        """Generate a rich text description optimized for semantic search.

        Args:
            work_id: Work identifier
            composer_name: Composer name
            period: Musical period (Baroque, Classical, etc.)
            tags: List of tags for the work

        Returns:
            Rich text description combining all metadata

        Example output:
            "Symphony No. 9 in E minor by Antonín Dvořák. Romantic period symphony.
             E minor key: melancholic, contemplative, moody. From the New World.
             Tags: romantic-era, orchestral, dramatic, nostalgic, triumphant.
             Large orchestra, four movements, moderately fast tempo."
        """
        work = self.dataset.get_work_by_id(work_id)
        if work is None:
            return ""

        parts = []

        # Basic info: title and composer (weighted)
        composer_repeated = " ".join([composer_name] * self.config.semantic_search.composer_weight)
        parts.append(f"{work['title']} by {composer_repeated}")

        # Period and work type (weighted)
        if period:
            period_repeated = " ".join([period] * self.config.semantic_search.period_weight)
            parts.append(f"{period_repeated} period")

        if work.get('work_type'):
            work_type_repeated = " ".join([work['work_type']] * self.config.semantic_search.work_type_weight)
            parts.append(work_type_repeated)

        # Musical key with mood associations (weighted)
        if work.get('key'):
            key_moods = self._get_key_moods(work['key'])
            if key_moods:
                key_mood_text = f"{work['key']}: {key_moods}"
                key_mood_repeated = " ".join([key_mood_text] * self.config.semantic_search.key_mood_weight)
                parts.append(key_mood_repeated)

        # Catalog number for context
        if work.get('catalog_number'):
            parts.append(f"Catalog: {work['catalog_number']}")

        # Tags (weighted)
        if tags:
            tags_text = " ".join(tags)
            tags_repeated = " ".join([tags_text] * self.config.semantic_search.tag_weight)
            parts.append(f"Tags: {tags_repeated}")

        # Derive additional features from catalog patterns
        catalog_info = self._extract_catalog_features(work.get('catalog_number', ''))
        if catalog_info:
            parts.append(catalog_info)

        return ". ".join(parts) + "."

    def _get_key_moods(self, key: str) -> str:
        """Get mood associations for a musical key.

        Args:
            key: Musical key (e.g., "E minor", "D major")

        Returns:
            Space-separated mood keywords
        """
        # Normalize key format
        key_normalized = key.strip()

        # Try exact match
        if key_normalized in KEY_MOOD_MAP:
            return KEY_MOOD_MAP[key_normalized]

        # Try case-insensitive match
        for k, v in KEY_MOOD_MAP.items():
            if k.lower() == key_normalized.lower():
                return v

        # Try to infer major/minor
        if 'minor' in key_normalized.lower():
            return 'melancholic contemplative moody introspective'
        elif 'major' in key_normalized.lower():
            return 'bright cheerful uplifting optimistic'

        return ''

    def _extract_catalog_features(self, catalog_number: str) -> str:
        """Extract tempo/character hints from catalog numbers.

        Args:
            catalog_number: Catalog number (e.g., "Op. 95", "BWV 1006")

        Returns:
            Additional descriptive text
        """
        if not catalog_number:
            return ""

        features = []

        # BWV = Bach's works
        if 'BWV' in catalog_number.upper():
            features.append("Baroque era counterpoint")

        # K. or KV = Mozart's works
        if 'K.' in catalog_number or 'KV' in catalog_number.upper():
            features.append("Classical era elegance")

        # Op. = Opus number (common in Romantic era)
        if 'Op.' in catalog_number:
            features.append("structured composition")

        # D. = Schubert
        if catalog_number.startswith('D.'):
            features.append("Romantic lyricism")

        return " ".join(features)


class SemanticSearchEngine:
    """Semantic search engine for mood-based music recommendations.

    Uses sentence transformers to enable natural language queries
    for finding music by mood, activity, or vibe.
    """

    def __init__(
        self,
        dataset: MusicDataset,
        config: Optional[Config] = None
    ):
        """Initialize semantic search engine.

        Args:
            dataset: Music dataset with works, composers, and tags
            config: Configuration object
        """
        self.dataset = dataset
        self.config = config or get_config()
        self.description_generator = WorkDescriptionGenerator(dataset, config)
        self.explanation_formatter = ExplanationFormatter()

        # Model and embeddings (loaded lazily)
        self.model: Optional[SentenceTransformer] = None
        self.work_embeddings: Optional[np.ndarray] = None
        self.work_descriptions: Optional[List[str]] = None
        self.work_id_mapping: Optional[List[str]] = None

        self._is_ready = False

    def load(self, force_rebuild: bool = False) -> None:
        """Load model and embeddings.

        Args:
            force_rebuild: If True, rebuild embeddings even if cache exists
        """
        logger.info("Loading semantic search engine...")

        # Load sentence transformer model
        logger.info(f"Loading model: {self.config.semantic_search.embedding_model}")
        self.model = SentenceTransformer(self.config.semantic_search.embedding_model)

        # Try to load from cache
        if self.config.semantic_search.enable_embedding_cache and not force_rebuild:
            if self._load_from_cache():
                logger.info("Loaded embeddings from cache")
                self._is_ready = True
                return

        # Generate embeddings from scratch
        logger.info("Generating embeddings from scratch...")
        self._generate_embeddings()

        # Save to cache
        if self.config.semantic_search.enable_embedding_cache:
            self._save_to_cache()

        self._is_ready = True
        logger.info("Semantic search engine ready")

    def search_by_mood(
        self,
        query: str,
        n: int = 10,
        min_score: Optional[float] = None
    ) -> List[SemanticSearchResult]:
        """Search for works matching a mood/vibe query.

        Args:
            query: Natural language mood query (e.g., "I'm feeling moody")
            n: Number of results to return
            min_score: Minimum similarity score (uses config default if None)

        Returns:
            List of semantic search results sorted by relevance

        Example:
            >>> engine.search_by_mood("bright and cheerful", n=5)
        """
        self._ensure_ready()

        if min_score is None:
            min_score = self.config.semantic_search.min_similarity_threshold

        # Enhance query if enabled
        if self.config.semantic_search.enable_query_enhancement:
            enhanced_query = self._enhance_query(query)
        else:
            enhanced_query = query

        # Encode query
        query_embedding = self.model.encode([enhanced_query])[0]

        # Calculate similarities
        similarities = cosine_similarity(
            [query_embedding],
            self.work_embeddings
        )[0]

        # Filter by minimum score
        valid_indices = np.where(similarities >= min_score)[0]
        valid_similarities = similarities[valid_indices]

        # Sort by similarity
        sorted_positions = np.argsort(valid_similarities)[::-1]

        # Apply diversity sampling if enabled
        if self.config.semantic_search.enable_diversity_sampling and len(sorted_positions) > n:
            # Take top candidates for diversity pool
            candidate_multiplier = self.config.semantic_search.diversity_candidate_multiplier
            candidate_count = min(candidate_multiplier * n, len(sorted_positions))
            candidate_positions = sorted_positions[:candidate_count]
            candidate_scores = valid_similarities[candidate_positions]

            # Weighted random sample (higher scores = higher probability)
            # Normalize scores to probabilities
            probabilities = candidate_scores / candidate_scores.sum()

            # Randomly select n items based on probabilities
            selected_positions = np.random.choice(
                candidate_positions,
                size=n,
                replace=False,  # Don't pick same item twice
                p=probabilities
            )

            # Re-sort selected items by score (best first)
            selected_similarities = valid_similarities[selected_positions]
            re_sorted = np.argsort(selected_similarities)[::-1]

            top_indices = valid_indices[selected_positions[re_sorted]]
            top_scores = selected_similarities[re_sorted]
        else:
            # Not enough candidates or diversity disabled, use standard sorting
            top_indices = valid_indices[sorted_positions[:n]]
            top_scores = valid_similarities[sorted_positions[:n]]

        # Build results
        results = []
        for rank, (idx, score) in enumerate(zip(top_indices, top_scores), start=1):
            result = self._build_search_result(
                work_id=self.work_id_mapping[idx],
                similarity_score=score,
                rank=rank,
                query=query
            )
            results.append(result)

        return results

    def search_by_activity(
        self,
        activity: str,
        context: str = "",
        n: int = 10
    ) -> List[SemanticSearchResult]:
        """Search for works suitable for an activity.

        Args:
            activity: Activity name (e.g., "studying", "working out")
            context: Optional additional context
            n: Number of results to return

        Returns:
            List of activity-appropriate recommendations

        Example:
            >>> engine.search_by_activity("studying", context="need to focus")
        """
        # Combine activity and context into query
        if context:
            query = f"{activity} {context}"
        else:
            query = activity

        return self.search_by_mood(query, n=n)

    def search_by_description(
        self,
        query: str,
        n: int = 10
    ) -> List[SemanticSearchResult]:
        """General natural language search.

        Args:
            query: Natural language description
            n: Number of results to return

        Returns:
            List of matching works
        """
        return self.search_by_mood(query, n=n)

    def combine_semantic_and_similarity(
        self,
        query: str,
        similar_to_work_id: str,
        n: int = 10,
        semantic_weight: float = 0.5
    ) -> List[SemanticSearchResult]:
        """Hybrid search combining semantic search with work similarity.

        Args:
            query: Natural language query
            similar_to_work_id: Work ID to find similar works to
            n: Number of results to return
            semantic_weight: Weight for semantic score (0-1)

        Returns:
            Combined recommendations

        Example:
            >>> engine.combine_semantic_and_similarity(
            ...     "moody pieces",
            ...     similar_to_work_id="chopin-nocturne-op9-no2"
            ... )
        """
        self._ensure_ready()

        # Get work index
        try:
            work_idx = self.work_id_mapping.index(similar_to_work_id)
        except ValueError:
            raise ValueError(f"Work ID not found: {similar_to_work_id}")

        # Enhance and encode query
        if self.config.semantic_search.enable_query_enhancement:
            enhanced_query = self._enhance_query(query)
        else:
            enhanced_query = query

        query_embedding = self.model.encode([enhanced_query])[0]

        # Semantic similarities
        semantic_similarities = cosine_similarity(
            [query_embedding],
            self.work_embeddings
        )[0]

        # Work-to-work similarities
        work_similarities = cosine_similarity(
            self.work_embeddings[work_idx:work_idx+1],
            self.work_embeddings
        )[0]

        # Combine scores
        combined_scores = (
            semantic_weight * semantic_similarities +
            (1 - semantic_weight) * work_similarities
        )

        # Exclude the seed work
        combined_scores[work_idx] = -np.inf

        # Get top N
        top_indices = np.argsort(combined_scores)[::-1][:n]
        top_scores = combined_scores[top_indices]

        # Build results
        results = []
        for rank, (idx, score) in enumerate(zip(top_indices, top_scores), start=1):
            result = self._build_search_result(
                work_id=self.work_id_mapping[idx],
                similarity_score=score,
                rank=rank,
                query=query
            )
            results.append(result)

        return results

    def get_embedding_quality_metrics(self) -> Dict[str, Any]:
        """Assess embedding quality.

        Returns:
            Dictionary with quality metrics
        """
        self._ensure_ready()

        # Calculate pairwise similarities
        pairwise_sims = cosine_similarity(self.work_embeddings)

        # Remove diagonal (self-similarity)
        n = len(pairwise_sims)
        mask = ~np.eye(n, dtype=bool)
        non_diag_sims = pairwise_sims[mask]

        metrics = {
            'num_works': len(self.work_embeddings),
            'embedding_dim': self.work_embeddings.shape[1],
            'model_name': self.config.semantic_search.embedding_model,
            'similarity_distribution': {
                'mean': float(np.mean(non_diag_sims)),
                'std': float(np.std(non_diag_sims)),
                'min': float(np.min(non_diag_sims)),
                'max': float(np.max(non_diag_sims)),
                'median': float(np.median(non_diag_sims)),
                'q25': float(np.percentile(non_diag_sims, 25)),
                'q75': float(np.percentile(non_diag_sims, 75)),
            },
            'description_stats': {
                'avg_length': float(np.mean([len(d) for d in self.work_descriptions])),
                'min_length': min(len(d) for d in self.work_descriptions),
                'max_length': max(len(d) for d in self.work_descriptions),
            }
        }

        return metrics

    def _generate_embeddings(self) -> None:
        """Generate embeddings for all works in the dataset."""
        logger.info("Generating work descriptions...")

        descriptions = []
        work_ids = []

        for _, work in self.dataset.works.iterrows():
            work_id = work['work_id']
            composer_name = self.dataset.get_composer_name(work['composer_id'])

            # Get period from composer
            period = None
            if work['composer_id']:
                composer = self.dataset.composers[
                    self.dataset.composers['composer_id'] == work['composer_id']
                ]
                if len(composer) > 0 and 'period' in composer.columns:
                    period = composer.iloc[0]['period']

            # Get tags
            tags = self.dataset.get_work_tags(work_id)

            # Generate description
            description = self.description_generator.create_work_description(
                work_id=work_id,
                composer_name=composer_name,
                period=period,
                tags=tags
            )

            descriptions.append(description)
            work_ids.append(work_id)

        logger.info(f"Encoding {len(descriptions)} descriptions...")

        # Batch encode for efficiency
        embeddings = self.model.encode(
            descriptions,
            batch_size=self.config.semantic_search.embedding_batch_size,
            show_progress_bar=True,
            convert_to_numpy=True
        )

        self.work_embeddings = embeddings
        self.work_descriptions = descriptions
        self.work_id_mapping = work_ids

        logger.info(f"Generated {len(embeddings)} embeddings with dimension {embeddings.shape[1]}")

    def _enhance_query(self, query: str) -> str:
        """Enhance query with synonyms and activity mappings.

        Args:
            query: Original query

        Returns:
            Enhanced query with additional terms
        """
        query_lower = query.lower()
        enhancements = []

        # Check for known keywords
        for keyword, expansion in QUERY_ENHANCEMENTS.items():
            if keyword in query_lower:
                enhancements.append(expansion)

        if enhancements:
            return f"{query} {' '.join(enhancements)}"

        return query

    def _build_search_result(
        self,
        work_id: str,
        similarity_score: float,
        rank: int,
        query: str
    ) -> SemanticSearchResult:
        """Build a semantic search result with metadata.

        Args:
            work_id: Work identifier
            similarity_score: Similarity score
            rank: Result rank
            query: Original query

        Returns:
            SemanticSearchResult object
        """
        work = self.dataset.get_work_by_id(work_id)
        if work is None:
            raise ValueError(f"Work not found: {work_id}")

        composer_name = self.dataset.get_composer_name(work['composer_id'])
        tags = self.dataset.get_work_tags(work_id)

        # Get period
        period = None
        if work['composer_id']:
            composer = self.dataset.composers[
                self.dataset.composers['composer_id'] == work['composer_id']
            ]
            if len(composer) > 0 and 'period' in composer.columns:
                period = composer.iloc[0]['period']

        # Generate explanation
        explanation = self._generate_explanation(work, query, tags)

        return SemanticSearchResult(
            work_id=work_id,
            title=work['title'],
            composer=composer_name,
            work_type=work.get('work_type'),
            key=work.get('key'),
            period=period,
            similarity_score=similarity_score,
            rank=rank,
            explanation=explanation,
            tags=tags[:5] if tags else None
        )

    def _generate_explanation(
        self,
        work: Any,
        query: str,
        tags: List[str]
    ) -> str:
        """Generate explanation for why this work matches the query.

        Args:
            work: Work data
            query: Search query
            tags: Work tags

        Returns:
            Explanation string
        """
        reasons = []
        query_lower = query.lower()

        # Check if key mood matches query (exact match)
        # Also check title for additional keys (handles multi-work collections)
        primary_key = work.get('key')
        all_keys = [primary_key] if primary_key else []

        # Extract additional keys from title (e.g., "Partita no. 5 in G major / Partita no. 6 in E minor")
        if work.get('title'):
            title = work['title']
            # Look for key patterns like "in E minor", "in G major", etc.
            import re
            key_pattern = r'in ([A-G][#b]? (?:major|minor))'
            found_keys = re.findall(key_pattern, title, re.IGNORECASE)
            for key in found_keys:
                if key not in all_keys:
                    all_keys.append(key)

        # Check each key for matches
        # First check if query explicitly mentions the key (e.g., "e minor", "g major")
        for key in all_keys:
            if key and key.lower() in query_lower:
                # Direct key match in query
                key_moods = self.description_generator._get_key_moods(key)
                if key_moods:
                    mood_words = key_moods.split()[:2]
                    reasons.append(f"{key} ({', '.join(mood_words)})")
                else:
                    reasons.append(f"{key}")
                break  # Only show first matching key

        # If no direct key match, check for mood word matches
        if not reasons:
            for key in all_keys:
                if key:
                    key_moods = self.description_generator._get_key_moods(key)
                    if key_moods:
                        mood_words = key_moods.split()
                        matching_moods = [m for m in mood_words if m in query_lower]
                        if matching_moods:
                            reasons.append(f"{key} ({', '.join(matching_moods[:2])})")
                            break  # Only show first matching key

        # Check for matching tags (exact match)
        if tags:
            matching_tags = [t for t in tags if t.lower() in query_lower]
            if matching_tags:
                reasons.append(f"tags: {', '.join(matching_tags[:2])}")

        # Check period (exact match)
        period_match = False
        period = None
        if work.get('composer_id'):
            composer = self.dataset.composers[
                self.dataset.composers['composer_id'] == work['composer_id']
            ]
            if len(composer) > 0 and 'period' in composer.columns:
                period = composer.iloc[0]['period']
                if period and period.lower() in query_lower:
                    reasons.append(f"{period} period")
                    period_match = True

        # If we found exact matches, use formatter with exact match mode
        if reasons:
            return self.explanation_formatter.format_semantic_explanation(
                reasons=reasons,
                work=work,
                query=query,
                is_exact_match=True
            )

        # No exact matches - generate contextual explanation based on work characteristics
        # This provides meaningful explanations for semantic similarity matches

        # Build description based on key + mood
        # Use all_keys to handle multi-work collections
        best_key = all_keys[0] if all_keys else None
        if best_key:
            key_moods = self.description_generator._get_key_moods(best_key)
            if key_moods:
                # Extract first 2 mood descriptors
                mood_words = key_moods.split()[:2]
                mood_desc = " and ".join(mood_words)
                reasons.append(f"{mood_desc} ({best_key})")

        # Add most relevant tags (up to 2)
        if tags:
            # Prioritize mood/character tags
            priority_tags = [t for t in tags[:5] if any(
                keyword in t.lower() for keyword in [
                    'dramatic', 'lyrical', 'peaceful', 'intense', 'gentle',
                    'energetic', 'calm', 'passionate', 'melancholic', 'bright',
                    'dark', 'triumphant', 'romantic', 'playful', 'somber'
                ]
            )]
            if priority_tags:
                reasons.extend(priority_tags[:2])
            elif len(tags) > 0:
                reasons.extend(tags[:2])

        # Add work type (include symphony/concerto if we don't have much else)
        work_type = None
        if work.get('work_type'):
            # If we have very few reasons, include all work types
            # Otherwise only include distinctive ones
            work_type_lower = work['work_type'].lower()
            if len(reasons) < 2 or work_type_lower not in ['symphony', 'concerto']:
                work_type = work_type_lower

        # Add period context if available
        if period and not period_match:
            reasons.append(f"{period} era")

        # Use formatter to create natural explanation
        return self.explanation_formatter.format_semantic_explanation(
            reasons=reasons,
            work={'work_type': work_type or work.get('work_type')},
            query=query,
            is_exact_match=False
        )

    def _load_from_cache(self) -> bool:
        """Load embeddings from cache.

        Returns:
            True if successfully loaded, False otherwise
        """
        cache_dir = self.config.semantic_search.embedding_cache_dir

        embeddings_file = cache_dir / "work_embeddings.npz"
        descriptions_file = cache_dir / "work_descriptions.pkl"
        mapping_file = cache_dir / "work_id_mapping.pkl"
        metadata_file = cache_dir / "embedding_metadata.json"

        # Check if all files exist
        if not all(f.exists() for f in [embeddings_file, descriptions_file, mapping_file, metadata_file]):
            logger.info("Cache files not found")
            return False

        try:
            # Load metadata
            with open(metadata_file, 'r') as f:
                metadata_dict = json.load(f)
            metadata = EmbeddingCacheMetadata.from_dict(metadata_dict)

            # Check if model has changed
            if metadata.model_name != self.config.semantic_search.embedding_model:
                logger.info(f"Model changed from {metadata.model_name} to {self.config.semantic_search.embedding_model}")
                return False

            # Check if number of works has changed
            if metadata.num_works != len(self.dataset.works):
                logger.info(f"Dataset size changed from {metadata.num_works} to {len(self.dataset.works)}")
                return False

            # Load embeddings
            embeddings_data = np.load(embeddings_file)
            self.work_embeddings = embeddings_data['embeddings']

            # Load descriptions
            with open(descriptions_file, 'rb') as f:
                self.work_descriptions = pickle.load(f)

            # Load work ID mapping
            with open(mapping_file, 'rb') as f:
                self.work_id_mapping = pickle.load(f)

            logger.info(f"Loaded cache created on {metadata.creation_date}")
            return True

        except Exception as e:
            logger.warning(f"Failed to load from cache: {e}")
            return False

    def _save_to_cache(self) -> None:
        """Save embeddings to cache."""
        cache_dir = self.config.semantic_search.embedding_cache_dir
        cache_dir.mkdir(parents=True, exist_ok=True)

        try:
            # Save embeddings
            embeddings_file = cache_dir / "work_embeddings.npz"
            np.savez_compressed(
                embeddings_file,
                embeddings=self.work_embeddings
            )

            # Save descriptions
            descriptions_file = cache_dir / "work_descriptions.pkl"
            with open(descriptions_file, 'wb') as f:
                pickle.dump(self.work_descriptions, f)

            # Save work ID mapping
            mapping_file = cache_dir / "work_id_mapping.pkl"
            with open(mapping_file, 'wb') as f:
                pickle.dump(self.work_id_mapping, f)

            # Save metadata
            metadata = EmbeddingCacheMetadata(
                model_name=self.config.semantic_search.embedding_model,
                creation_date=datetime.now().isoformat(),
                num_works=len(self.work_embeddings),
                embedding_dim=self.work_embeddings.shape[1]
            )
            metadata_file = cache_dir / "embedding_metadata.json"
            with open(metadata_file, 'w') as f:
                json.dump(metadata.to_dict(), f, indent=2)

            logger.info(f"Saved embeddings to cache: {cache_dir}")

        except Exception as e:
            logger.warning(f"Failed to save to cache: {e}")

    def _ensure_ready(self) -> None:
        """Ensure engine is loaded and ready.

        Raises:
            RuntimeError: If engine not loaded
        """
        if not self._is_ready:
            raise RuntimeError(
                "Semantic search engine not loaded. Call load() first."
            )
