"""
Service layer for production deployment.

This module provides a clean API interface for the recommendation system:
- Singleton pattern for recommender (load once, reuse)
- Async support for I/O operations
- Proper error handling and logging
- Health checks
- Metrics and monitoring hooks
"""

import logging
import time
from typing import List, Dict, Any, Optional
from functools import wraps
from datetime import datetime
import asyncio

from recommender.config import Config, get_config
from recommender.recommender import MusicRecommender, Recommendation
from recommender.data_loader import DataLoader
from recommender.semantic_search import SemanticSearchEngine

logger = logging.getLogger(__name__)


# Global recommender instance (singleton)
_recommender_instance: Optional[MusicRecommender] = None
_semantic_search_instance: Optional[SemanticSearchEngine] = None
_load_time: Optional[datetime] = None


class ServiceMetrics:
    """Tracks service metrics for monitoring.

    In production, these would be exported to a monitoring system
    like Prometheus, DataDog, etc.
    """

    def __init__(self):
        self.recommendation_count = 0
        self.error_count = 0
        self.cache_hits = 0
        self.total_latency = 0.0
        self.request_count = 0

    def record_request(self, latency: float, cached: bool = False, error: bool = False):
        """Record a request with its latency.

        Args:
            latency: Request latency in seconds
            cached: Whether result was cached
            error: Whether request resulted in error
        """
        self.request_count += 1
        self.total_latency += latency

        if cached:
            self.cache_hits += 1

        if error:
            self.error_count += 1
        else:
            self.recommendation_count += 1

    def get_stats(self) -> Dict[str, Any]:
        """Get current metrics.

        Returns:
            Dictionary with metric statistics
        """
        avg_latency = (
            self.total_latency / self.request_count
            if self.request_count > 0 else 0.0
        )

        cache_hit_rate = (
            self.cache_hits / self.request_count
            if self.request_count > 0 else 0.0
        )

        error_rate = (
            self.error_count / self.request_count
            if self.request_count > 0 else 0.0
        )

        return {
            'total_requests': self.request_count,
            'successful_recommendations': self.recommendation_count,
            'errors': self.error_count,
            'cache_hits': self.cache_hits,
            'avg_latency_ms': avg_latency * 1000,
            'cache_hit_rate': cache_hit_rate,
            'error_rate': error_rate,
        }

    def reset(self):
        """Reset all metrics."""
        self.recommendation_count = 0
        self.error_count = 0
        self.cache_hits = 0
        self.total_latency = 0.0
        self.request_count = 0


# Global metrics instance
_metrics = ServiceMetrics()


def track_metrics(func):
    """Decorator to track request metrics.

    Args:
        func: Function to wrap

    Returns:
        Wrapped function that tracks metrics
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        error = False

        try:
            result = func(*args, **kwargs)
            return result
        except Exception as e:
            error = True
            raise
        finally:
            latency = time.time() - start_time
            _metrics.record_request(latency, error=error)

    return wrapper


def track_metrics_async(func):
    """Async version of track_metrics decorator.

    Args:
        func: Async function to wrap

    Returns:
        Wrapped async function that tracks metrics
    """
    @wraps(func)
    async def wrapper(*args, **kwargs):
        start_time = time.time()
        error = False

        try:
            result = await func(*args, **kwargs)
            return result
        except Exception as e:
            error = True
            raise
        finally:
            latency = time.time() - start_time
            _metrics.record_request(latency, error=error)

    return wrapper


class RecommenderService:
    """Production service interface for music recommendations.

    This class provides a high-level API with proper error handling,
    logging, and monitoring for production use.
    """

    def __init__(self, config: Optional[Config] = None):
        """Initialize service.

        Args:
            config: Configuration object (uses global config if None)
        """
        self.config = config or get_config()

    def initialize(
        self,
        force_rebuild: bool = False,
        enable_semantic_search: bool = True
    ) -> None:
        """Initialize the recommender system.

        This loads data, builds features, and prepares the system.
        Should be called once at application startup.

        Args:
            force_rebuild: If True, rebuild features even if cache exists
            enable_semantic_search: If True, initialize semantic search engine

        Raises:
            RuntimeError: If initialization fails
        """
        global _recommender_instance, _semantic_search_instance, _load_time

        if _recommender_instance is not None and not force_rebuild:
            logger.info("Recommender already initialized")
            if enable_semantic_search and _semantic_search_instance is None:
                # Initialize semantic search if not already done
                self._initialize_semantic_search(force_rebuild)
            return

        try:
            logger.info("Initializing recommender service...")
            start_time = time.time()

            recommender = MusicRecommender(self.config)
            recommender.load(force_rebuild=force_rebuild)

            _recommender_instance = recommender
            _load_time = datetime.now()

            elapsed = time.time() - start_time
            logger.info(f"Recommender initialized in {elapsed:.2f}s")

            # Initialize semantic search if enabled
            if enable_semantic_search:
                self._initialize_semantic_search(force_rebuild)

        except Exception as e:
            logger.error(f"Failed to initialize recommender: {e}", exc_info=True)
            raise RuntimeError(f"Recommender initialization failed: {e}")

    def _initialize_semantic_search(self, force_rebuild: bool = False) -> None:
        """Initialize semantic search engine.

        Args:
            force_rebuild: If True, rebuild embeddings even if cache exists
        """
        global _semantic_search_instance

        try:
            logger.info("Initializing semantic search engine...")
            start_time = time.time()

            # Get dataset from recommender
            recommender = self.get_recommender()
            dataset = recommender._dataset

            # Create and load semantic search engine
            semantic_search = SemanticSearchEngine(dataset, self.config)
            semantic_search.load(force_rebuild=force_rebuild)

            _semantic_search_instance = semantic_search

            elapsed = time.time() - start_time
            logger.info(f"Semantic search engine initialized in {elapsed:.2f}s")

        except Exception as e:
            logger.error(f"Failed to initialize semantic search: {e}", exc_info=True)
            logger.warning("Continuing without semantic search capabilities")
            _semantic_search_instance = None

    def get_recommender(self) -> MusicRecommender:
        """Get the singleton recommender instance.

        Returns:
            MusicRecommender instance

        Raises:
            RuntimeError: If recommender not initialized
        """
        if _recommender_instance is None:
            raise RuntimeError(
                "Recommender not initialized. Call initialize() first."
            )
        return _recommender_instance

    def get_semantic_search(self) -> Optional[SemanticSearchEngine]:
        """Get the singleton semantic search instance.

        Returns:
            SemanticSearchEngine instance or None if not initialized
        """
        return _semantic_search_instance

    @track_metrics
    def recommend_similar(
        self,
        work_id: str,
        n: int = 10,
        diverse: bool = False,
        diversity_weight: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        """Get recommendations for a work.

        Args:
            work_id: ID of the seed work
            n: Number of recommendations
            diverse: Whether to use diversity-aware recommendations
            diversity_weight: Weight for diversity (only if diverse=True)

        Returns:
            List of recommendation dictionaries

        Raises:
            ValueError: If work_id not found
            RuntimeError: If service not initialized
        """
        recommender = self.get_recommender()

        try:
            if diverse:
                recommendations = recommender.recommend_diverse(
                    work_id,
                    n=n,
                    diversity_weight=diversity_weight
                )
            else:
                recommendations = recommender.recommend_similar(
                    work_id,
                    n=n
                )

            return [rec.to_dict() for rec in recommendations]

        except ValueError as e:
            logger.warning(f"Invalid work_id: {work_id} - {e}")
            raise
        except Exception as e:
            logger.error(f"Recommendation failed for {work_id}: {e}", exc_info=True)
            raise RuntimeError(f"Recommendation failed: {e}")

    @track_metrics
    def recommend_by_query(
        self,
        query: str,
        n: int = 10
    ) -> List[Dict[str, Any]]:
        """Get recommendations based on search query.

        Args:
            query: Search query for work title
            n: Number of recommendations

        Returns:
            List of recommendation dictionaries

        Raises:
            ValueError: If query doesn't match any works
            RuntimeError: If service not initialized
        """
        recommender = self.get_recommender()

        try:
            recommendations = recommender.recommend_by_query(query, n=n)
            return [rec.to_dict() for rec in recommendations]

        except ValueError as e:
            logger.warning(f"Query '{query}' matched no works: {e}")
            raise
        except Exception as e:
            logger.error(f"Query recommendation failed: {e}", exc_info=True)
            raise RuntimeError(f"Query recommendation failed: {e}")

    @track_metrics
    def recommend_by_filters(
        self,
        composer: Optional[str] = None,
        period: Optional[str] = None,
        work_type: Optional[str] = None,
        key: Optional[str] = None,
        n: int = 10
    ) -> List[Dict[str, Any]]:
        """Get recommendations filtered by criteria.

        Args:
            composer: Filter by composer name
            period: Filter by period/era
            work_type: Filter by work type
            key: Filter by musical key
            n: Number of recommendations

        Returns:
            List of recommendation dictionaries

        Raises:
            ValueError: If filters don't match any works
            RuntimeError: If service not initialized
        """
        recommender = self.get_recommender()

        try:
            recommendations = recommender.recommend_by_filters(
                composer=composer,
                period=period,
                work_type=work_type,
                key=key,
                n=n
            )
            return [rec.to_dict() for rec in recommendations]

        except ValueError as e:
            logger.warning(f"Filters matched no works: {e}")
            raise
        except Exception as e:
            logger.error(f"Filter recommendation failed: {e}", exc_info=True)
            raise RuntimeError(f"Filter recommendation failed: {e}")

    @track_metrics
    def get_work_info(self, work_id: str) -> Dict[str, Any]:
        """Get detailed information about a work.

        Args:
            work_id: Work identifier

        Returns:
            Dictionary with work metadata

        Raises:
            ValueError: If work_id not found
            RuntimeError: If service not initialized
        """
        recommender = self.get_recommender()

        try:
            info = recommender.get_work_info(work_id)
            if info is None:
                raise ValueError(f"Work not found: {work_id}")
            return info

        except ValueError:
            raise
        except Exception as e:
            logger.error(f"Failed to get work info for {work_id}: {e}", exc_info=True)
            raise RuntimeError(f"Failed to get work info: {e}")

    def health_check(self) -> Dict[str, Any]:
        """Check service health.

        Returns:
            Dictionary with health status

        Example:
            >>> service = RecommenderService()
            >>> service.initialize()
            >>> health = service.health_check()
            >>> print(health['status'])
            'healthy'
        """
        status = {
            'status': 'unhealthy',
            'recommender_loaded': False,
            'load_time': None,
            'dataset_size': 0,
            'error': None
        }

        try:
            if _recommender_instance is None:
                status['error'] = 'Recommender not initialized'
                return status

            status['recommender_loaded'] = True
            status['load_time'] = _load_time.isoformat() if _load_time else None

            # Check dataset
            dataset = _recommender_instance._dataset
            if dataset:
                status['dataset_size'] = len(dataset.works)

            # Check features
            if _recommender_instance._features is not None:
                status['feature_count'] = _recommender_instance._features.features.shape[1]

            status['status'] = 'healthy'

        except Exception as e:
            status['error'] = str(e)
            logger.error(f"Health check failed: {e}", exc_info=True)

        return status

    def get_metrics(self) -> Dict[str, Any]:
        """Get service metrics.

        Returns:
            Dictionary with service metrics
        """
        metrics = _metrics.get_stats()

        # Add health status
        health = self.health_check()
        metrics['health'] = health['status']
        metrics['dataset_size'] = health.get('dataset_size', 0)

        return metrics

    def reset_metrics(self) -> None:
        """Reset service metrics."""
        _metrics.reset()
        logger.info("Service metrics reset")

    # Semantic search methods

    @track_metrics
    def search_by_mood(
        self,
        query: str,
        n: int = 10,
        min_score: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        """Search for works by mood/vibe using natural language.

        Args:
            query: Natural language mood query (e.g., "I'm feeling moody")
            n: Number of results to return
            min_score: Minimum similarity score threshold

        Returns:
            List of recommendation dictionaries

        Raises:
            RuntimeError: If semantic search not initialized

        Example:
            >>> service.search_by_mood("bright and cheerful", n=5)
        """
        semantic_search = self.get_semantic_search()
        if semantic_search is None:
            raise RuntimeError(
                "Semantic search not initialized. "
                "Call initialize(enable_semantic_search=True) first."
            )

        try:
            results = semantic_search.search_by_mood(query, n=n, min_score=min_score)
            return [r.to_dict() for r in results]

        except Exception as e:
            logger.error(f"Mood search failed for '{query}': {e}", exc_info=True)
            raise RuntimeError(f"Mood search failed: {e}")

    @track_metrics
    def search_by_activity(
        self,
        activity: str,
        context: str = "",
        n: int = 10
    ) -> List[Dict[str, Any]]:
        """Search for works suitable for an activity.

        Args:
            activity: Activity name (e.g., "studying", "working out")
            context: Optional additional context
            n: Number of results to return

        Returns:
            List of recommendation dictionaries

        Raises:
            RuntimeError: If semantic search not initialized

        Example:
            >>> service.search_by_activity("studying", context="need to focus")
        """
        semantic_search = self.get_semantic_search()
        if semantic_search is None:
            raise RuntimeError(
                "Semantic search not initialized. "
                "Call initialize(enable_semantic_search=True) first."
            )

        try:
            results = semantic_search.search_by_activity(activity, context=context, n=n)
            return [r.to_dict() for r in results]

        except Exception as e:
            logger.error(f"Activity search failed for '{activity}': {e}", exc_info=True)
            raise RuntimeError(f"Activity search failed: {e}")

    @track_metrics
    def search_by_description(
        self,
        query: str,
        n: int = 10
    ) -> List[Dict[str, Any]]:
        """General natural language search for works.

        Args:
            query: Natural language description
            n: Number of results to return

        Returns:
            List of recommendation dictionaries

        Raises:
            RuntimeError: If semantic search not initialized

        Example:
            >>> service.search_by_description("rainy Sunday morning vibes")
        """
        semantic_search = self.get_semantic_search()
        if semantic_search is None:
            raise RuntimeError(
                "Semantic search not initialized. "
                "Call initialize(enable_semantic_search=True) first."
            )

        try:
            results = semantic_search.search_by_description(query, n=n)
            return [r.to_dict() for r in results]

        except Exception as e:
            logger.error(f"Description search failed for '{query}': {e}", exc_info=True)
            raise RuntimeError(f"Description search failed: {e}")

    @track_metrics
    def hybrid_search(
        self,
        query: str,
        similar_to_work_id: Optional[str] = None,
        n: int = 10,
        semantic_weight: float = 0.5
    ) -> List[Dict[str, Any]]:
        """Hybrid search combining semantic search with work similarity.

        Args:
            query: Natural language query
            similar_to_work_id: Optional work ID to find similar works to
            n: Number of results to return
            semantic_weight: Weight for semantic score vs similarity (0-1)

        Returns:
            List of recommendation dictionaries

        Raises:
            RuntimeError: If semantic search not initialized

        Example:
            >>> service.hybrid_search(
            ...     "moody pieces",
            ...     similar_to_work_id="some-work-id",
            ...     semantic_weight=0.7
            ... )
        """
        semantic_search = self.get_semantic_search()
        if semantic_search is None:
            raise RuntimeError(
                "Semantic search not initialized. "
                "Call initialize(enable_semantic_search=True) first."
            )

        try:
            if similar_to_work_id:
                results = semantic_search.combine_semantic_and_similarity(
                    query=query,
                    similar_to_work_id=similar_to_work_id,
                    n=n,
                    semantic_weight=semantic_weight
                )
            else:
                # Fall back to pure semantic search
                results = semantic_search.search_by_mood(query, n=n)

            return [r.to_dict() for r in results]

        except ValueError as e:
            logger.warning(f"Hybrid search failed: {e}")
            raise
        except Exception as e:
            logger.error(f"Hybrid search failed: {e}", exc_info=True)
            raise RuntimeError(f"Hybrid search failed: {e}")

    def get_embedding_quality_metrics(self) -> Dict[str, Any]:
        """Get quality metrics for semantic search embeddings.

        Returns:
            Dictionary with quality metrics

        Raises:
            RuntimeError: If semantic search not initialized
        """
        semantic_search = self.get_semantic_search()
        if semantic_search is None:
            raise RuntimeError(
                "Semantic search not initialized. "
                "Call initialize(enable_semantic_search=True) first."
            )

        try:
            return semantic_search.get_embedding_quality_metrics()
        except Exception as e:
            logger.error(f"Failed to get embedding metrics: {e}", exc_info=True)
            raise RuntimeError(f"Failed to get embedding metrics: {e}")

    # Async versions of main methods

    async def recommend_similar_async(
        self,
        work_id: str,
        n: int = 10,
        diverse: bool = False,
        diversity_weight: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        """Async version of recommend_similar.

        Runs the recommendation in a thread pool to avoid blocking.
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            self.recommend_similar,
            work_id,
            n,
            diverse,
            diversity_weight
        )

    async def recommend_by_query_async(
        self,
        query: str,
        n: int = 10
    ) -> List[Dict[str, Any]]:
        """Async version of recommend_by_query."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            self.recommend_by_query,
            query,
            n
        )

    async def recommend_by_filters_async(
        self,
        composer: Optional[str] = None,
        period: Optional[str] = None,
        work_type: Optional[str] = None,
        key: Optional[str] = None,
        n: int = 10
    ) -> List[Dict[str, Any]]:
        """Async version of recommend_by_filters."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            self.recommend_by_filters,
            composer,
            period,
            work_type,
            key,
            n
        )


# Convenience functions for quick usage

def initialize_service(config: Optional[Config] = None, force_rebuild: bool = False) -> None:
    """Initialize the recommendation service.

    Args:
        config: Configuration object (uses global config if None)
        force_rebuild: If True, rebuild features even if cache exists
    """
    service = RecommenderService(config)
    service.initialize(force_rebuild=force_rebuild)


def get_service(config: Optional[Config] = None) -> RecommenderService:
    """Get a RecommenderService instance.

    Args:
        config: Configuration object (uses global config if None)

    Returns:
        RecommenderService instance
    """
    return RecommenderService(config)
