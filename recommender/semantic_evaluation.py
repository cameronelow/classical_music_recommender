"""
Evaluation and quality metrics for semantic search.

This module provides tools to assess the quality of semantic search
results and embeddings.
"""

import logging
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass

from recommender.semantic_search import SemanticSearchEngine

logger = logging.getLogger(__name__)


# Test queries with expected features
TEST_QUERIES = [
    # Mood queries
    ("moody and contemplative", ["minor", "slow", "melancholic", "introspective"]),
    ("bright and cheerful", ["major", "joyful", "uplifting"]),
    ("dark and dramatic", ["minor", "tragic", "dramatic", "ominous"]),
    ("peaceful and calm", ["calm", "peaceful", "serene", "gentle"]),
    ("energetic and lively", ["energetic", "lively", "fast", "spirited"]),

    # Activity queries
    ("studying music", ["calm", "instrumental", "minimal", "focused"]),
    ("workout music", ["energetic", "fast", "rhythmic", "powerful"]),
    ("relaxing music", ["calm", "peaceful", "slow", "soothing"]),
    ("dinner party music", ["elegant", "sophisticated", "moderate"]),
    ("morning music", ["gentle", "bright", "uplifting", "awakening"]),

    # Complex/contextual queries
    ("rainy Sunday morning", ["calm", "reflective", "gentle", "melancholic"]),
    ("romantic dinner", ["tender", "intimate", "elegant"]),
    ("triumphant celebration", ["triumphant", "celebratory", "majestic", "joyful"]),
    ("late night contemplation", ["calm", "introspective", "quiet", "reflective"]),
    ("spring awakening", ["fresh", "bright", "awakening", "uplifting"]),

    # Period-specific
    ("baroque counterpoint", ["Baroque", "counterpoint", "structured"]),
    ("romantic era drama", ["Romantic", "dramatic", "emotional", "passionate"]),
    ("classical elegance", ["Classical", "elegant", "refined", "structured"]),

    # Instrument/texture
    ("solo piano", ["piano", "solo", "intimate"]),
    ("full orchestra", ["orchestral", "grand", "large", "powerful"]),
    ("chamber music", ["chamber", "intimate", "small"]),
]


@dataclass
class QueryEvaluation:
    """Evaluation result for a single query."""
    query: str
    expected_features: List[str]
    num_results: int
    avg_similarity: float
    features_found: Dict[str, int]
    feature_coverage: float
    top_results: List[Dict[str, Any]]


class SemanticSearchEvaluator:
    """Evaluates semantic search quality."""

    def __init__(self, semantic_search: SemanticSearchEngine):
        """Initialize evaluator.

        Args:
            semantic_search: Semantic search engine to evaluate
        """
        self.semantic_search = semantic_search

    def evaluate_query(
        self,
        query: str,
        expected_features: List[str],
        n: int = 10
    ) -> QueryEvaluation:
        """Evaluate a single query.

        Args:
            query: Test query
            expected_features: Features expected in results
            n: Number of results to retrieve

        Returns:
            QueryEvaluation with metrics
        """
        # Get search results
        results = self.semantic_search.search_by_mood(query, n=n)

        if not results:
            return QueryEvaluation(
                query=query,
                expected_features=expected_features,
                num_results=0,
                avg_similarity=0.0,
                features_found={},
                feature_coverage=0.0,
                top_results=[]
            )

        # Calculate average similarity
        avg_similarity = sum(r.similarity_score for r in results) / len(results)

        # Check which expected features appear in results
        features_found = {feature: 0 for feature in expected_features}

        for result in results:
            # Check in work description (key, period, work type, tags)
            result_text = " ".join([
                str(result.key or ""),
                str(result.period or ""),
                str(result.work_type or ""),
                " ".join(result.tags or [])
            ]).lower()

            for feature in expected_features:
                if feature.lower() in result_text:
                    features_found[feature] += 1

        # Calculate feature coverage (% of expected features found)
        features_with_matches = sum(1 for count in features_found.values() if count > 0)
        feature_coverage = features_with_matches / len(expected_features) if expected_features else 0.0

        # Format top results
        top_results = [r.to_dict() for r in results[:5]]

        return QueryEvaluation(
            query=query,
            expected_features=expected_features,
            num_results=len(results),
            avg_similarity=avg_similarity,
            features_found=features_found,
            feature_coverage=feature_coverage,
            top_results=top_results
        )

    def evaluate_all_test_queries(
        self,
        n: int = 10
    ) -> Dict[str, Any]:
        """Evaluate all predefined test queries.

        Args:
            n: Number of results per query

        Returns:
            Dictionary with aggregate evaluation metrics
        """
        logger.info(f"Evaluating {len(TEST_QUERIES)} test queries...")

        evaluations = []
        for query, expected_features in TEST_QUERIES:
            try:
                eval_result = self.evaluate_query(query, expected_features, n=n)
                evaluations.append(eval_result)
            except Exception as e:
                logger.warning(f"Failed to evaluate query '{query}': {e}")

        if not evaluations:
            return {
                'total_queries': len(TEST_QUERIES),
                'successful_queries': 0,
                'error': 'All queries failed'
            }

        # Aggregate metrics
        avg_similarity = sum(e.avg_similarity for e in evaluations) / len(evaluations)
        avg_coverage = sum(e.feature_coverage for e in evaluations) / len(evaluations)
        avg_results = sum(e.num_results for e in evaluations) / len(evaluations)

        # Queries with good coverage (>= 50%)
        good_coverage_queries = sum(1 for e in evaluations if e.feature_coverage >= 0.5)

        # Queries with high similarity (>= 0.5)
        high_similarity_queries = sum(1 for e in evaluations if e.avg_similarity >= 0.5)

        return {
            'total_queries': len(TEST_QUERIES),
            'successful_queries': len(evaluations),
            'avg_similarity_score': avg_similarity,
            'avg_feature_coverage': avg_coverage,
            'avg_results_per_query': avg_results,
            'queries_with_good_coverage': good_coverage_queries,
            'queries_with_high_similarity': high_similarity_queries,
            'coverage_rate': good_coverage_queries / len(evaluations),
            'high_similarity_rate': high_similarity_queries / len(evaluations),
            'individual_results': [
                {
                    'query': e.query,
                    'similarity': e.avg_similarity,
                    'coverage': e.feature_coverage,
                    'num_results': e.num_results,
                }
                for e in evaluations
            ]
        }

    def evaluate_query_consistency(
        self,
        query: str,
        runs: int = 5,
        n: int = 10
    ) -> Dict[str, Any]:
        """Evaluate consistency of results for the same query.

        Args:
            query: Test query
            runs: Number of times to run the query
            n: Number of results per run

        Returns:
            Consistency metrics
        """
        logger.info(f"Evaluating consistency for '{query}' over {runs} runs...")

        all_results = []
        for _ in range(runs):
            results = self.semantic_search.search_by_mood(query, n=n)
            all_results.append([r.work_id for r in results])

        if not all_results:
            return {
                'query': query,
                'error': 'No results'
            }

        # Calculate overlap between runs
        # For each pair of runs, count how many work IDs overlap
        overlaps = []
        for i in range(len(all_results)):
            for j in range(i + 1, len(all_results)):
                set_i = set(all_results[i])
                set_j = set(all_results[j])
                overlap = len(set_i.intersection(set_j))
                overlap_ratio = overlap / n if n > 0 else 0
                overlaps.append(overlap_ratio)

        avg_overlap = sum(overlaps) / len(overlaps) if overlaps else 0.0

        # Perfect consistency would be 1.0 (100% overlap)
        return {
            'query': query,
            'runs': runs,
            'avg_overlap_ratio': avg_overlap,
            'consistency_score': avg_overlap,
            'note': '1.0 = perfect consistency, 0.0 = no consistency'
        }

    def evaluate_catalog_coverage(
        self,
        queries: List[str],
        n: int = 10
    ) -> Dict[str, Any]:
        """Evaluate how much of the catalog is accessible via queries.

        Args:
            queries: List of test queries
            n: Number of results per query

        Returns:
            Coverage metrics
        """
        logger.info(f"Evaluating catalog coverage across {len(queries)} queries...")

        # Collect all unique work IDs returned
        all_work_ids = set()

        for query in queries:
            try:
                results = self.semantic_search.search_by_mood(query, n=n)
                for result in results:
                    all_work_ids.add(result.work_id)
            except Exception as e:
                logger.warning(f"Failed query '{query}': {e}")

        # Total works in dataset
        total_works = len(self.semantic_search.dataset.works)

        coverage_ratio = len(all_work_ids) / total_works if total_works > 0 else 0.0

        return {
            'total_works_in_catalog': total_works,
            'works_returned_by_queries': len(all_work_ids),
            'coverage_ratio': coverage_ratio,
            'coverage_percentage': coverage_ratio * 100,
            'queries_tested': len(queries),
            'note': 'Higher coverage = more of catalog is discoverable via queries'
        }

    def get_diversity_metrics(
        self,
        query: str,
        n: int = 10
    ) -> Dict[str, Any]:
        """Evaluate diversity of results for a query.

        Args:
            query: Test query
            n: Number of results

        Returns:
            Diversity metrics
        """
        results = self.semantic_search.search_by_mood(query, n=n)

        if not results:
            return {
                'query': query,
                'error': 'No results'
            }

        # Count unique composers, periods, work types, keys
        composers = set(r.composer for r in results if r.composer)
        periods = set(r.period for r in results if r.period)
        work_types = set(r.work_type for r in results if r.work_type)
        keys = set(r.key for r in results if r.key)

        return {
            'query': query,
            'num_results': len(results),
            'unique_composers': len(composers),
            'unique_periods': len(periods),
            'unique_work_types': len(work_types),
            'unique_keys': len(keys),
            'composer_diversity_ratio': len(composers) / len(results) if results else 0,
            'period_diversity_ratio': len(periods) / len(results) if results else 0,
            'work_type_diversity_ratio': len(work_types) / len(results) if results else 0,
        }

    def run_comprehensive_evaluation(self) -> Dict[str, Any]:
        """Run comprehensive evaluation suite.

        Returns:
            Dictionary with all evaluation metrics
        """
        logger.info("Running comprehensive semantic search evaluation...")

        # 1. Test query quality
        query_eval = self.evaluate_all_test_queries(n=10)

        # 2. Consistency check (on a few queries)
        sample_queries = [q for q, _ in TEST_QUERIES[:3]]
        consistency_results = []
        for query in sample_queries:
            consistency = self.evaluate_query_consistency(query, runs=3, n=10)
            consistency_results.append(consistency)

        avg_consistency = sum(c['consistency_score'] for c in consistency_results) / len(consistency_results)

        # 3. Catalog coverage
        all_queries = [q for q, _ in TEST_QUERIES]
        coverage = self.evaluate_catalog_coverage(all_queries, n=10)

        # 4. Diversity metrics (on a few queries)
        diversity_results = []
        for query in sample_queries:
            diversity = self.get_diversity_metrics(query, n=10)
            diversity_results.append(diversity)

        avg_composer_diversity = sum(
            d['composer_diversity_ratio'] for d in diversity_results
        ) / len(diversity_results)

        # 5. Embedding quality
        embedding_metrics = self.semantic_search.get_embedding_quality_metrics()

        return {
            'query_quality': query_eval,
            'consistency': {
                'avg_consistency_score': avg_consistency,
                'individual_results': consistency_results,
            },
            'catalog_coverage': coverage,
            'diversity': {
                'avg_composer_diversity': avg_composer_diversity,
                'individual_results': diversity_results,
            },
            'embedding_quality': embedding_metrics,
            'overall_health': self._calculate_overall_health(
                query_eval,
                avg_consistency,
                coverage,
                avg_composer_diversity
            )
        }

    def _calculate_overall_health(
        self,
        query_eval: Dict[str, Any],
        consistency: float,
        coverage: Dict[str, Any],
        diversity: float
    ) -> str:
        """Calculate overall health rating.

        Args:
            query_eval: Query evaluation results
            consistency: Consistency score
            coverage: Coverage metrics
            diversity: Diversity score

        Returns:
            Health rating: excellent, good, fair, poor
        """
        # Calculate composite score
        avg_similarity = query_eval.get('avg_similarity_score', 0)
        coverage_ratio = coverage.get('coverage_ratio', 0)

        # Weighted average
        score = (
            0.4 * avg_similarity +
            0.3 * consistency +
            0.2 * coverage_ratio +
            0.1 * diversity
        )

        if score >= 0.75:
            return "excellent"
        elif score >= 0.6:
            return "good"
        elif score >= 0.4:
            return "fair"
        else:
            return "poor"
