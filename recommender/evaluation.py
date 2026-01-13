"""
Evaluation module for assessing recommendation quality.

This module provides tools to evaluate the recommendation system:
- Diversity metrics (how varied are recommendations?)
- Coverage metrics (what % of catalog gets recommended?)
- Similarity distribution analysis
- Manual test cases with expected results
- A/B testing support
"""

import logging
from typing import List, Dict, Any, Optional, Set, Tuple
from dataclasses import dataclass
from collections import defaultdict

import numpy as np
import pandas as pd

from recommender.config import Config, get_config
from recommender.recommender import MusicRecommender, Recommendation

logger = logging.getLogger(__name__)


@dataclass
class DiversityMetrics:
    """Metrics measuring recommendation diversity.

    High diversity means recommendations are varied rather than all similar.
    """
    # Intra-list diversity (diversity within a single recommendation list)
    avg_pairwise_dissimilarity: float  # Higher = more diverse
    unique_composers_ratio: float  # % of unique composers in list
    unique_work_types_ratio: float  # % of unique work types
    unique_periods_ratio: float  # % of unique periods

    # Composer concentration
    max_same_composer_ratio: float  # Max % from single composer
    gini_coefficient: float  # Distribution inequality (0=equal, 1=concentrated)

    def __str__(self) -> str:
        return (
            f"Diversity Metrics:\n"
            f"  Avg pairwise dissimilarity: {self.avg_pairwise_dissimilarity:.3f}\n"
            f"  Unique composers: {self.unique_composers_ratio:.1%}\n"
            f"  Unique work types: {self.unique_work_types_ratio:.1%}\n"
            f"  Unique periods: {self.unique_periods_ratio:.1%}\n"
            f"  Max same composer: {self.max_same_composer_ratio:.1%}\n"
            f"  Gini coefficient: {self.gini_coefficient:.3f}"
        )


@dataclass
class CoverageMetrics:
    """Metrics measuring catalog coverage.

    High coverage means many different works get recommended.
    """
    total_catalog_size: int
    works_ever_recommended: int
    coverage_ratio: float  # % of catalog that gets recommended

    # Per-category coverage
    composers_covered: int
    composers_total: int
    work_types_covered: int
    work_types_total: int
    periods_covered: int
    periods_total: int

    # Popularity bias
    avg_recommendations_per_work: float
    median_recommendations_per_work: float
    most_recommended_works: List[Tuple[str, int]]  # (work_id, count)

    def __str__(self) -> str:
        return (
            f"Coverage Metrics:\n"
            f"  Catalog coverage: {self.works_ever_recommended}/{self.total_catalog_size} "
            f"({self.coverage_ratio:.1%})\n"
            f"  Composers: {self.composers_covered}/{self.composers_total}\n"
            f"  Work types: {self.work_types_covered}/{self.work_types_total}\n"
            f"  Periods: {self.periods_covered}/{self.periods_total}\n"
            f"  Avg recommendations per work: {self.avg_recommendations_per_work:.2f}\n"
            f"  Median: {self.median_recommendations_per_work:.1f}\n"
            f"  Top 5 most recommended: {', '.join([f'{wid[:8]}({cnt})' for wid, cnt in self.most_recommended_works[:5]])}"
        )


@dataclass
class TestCase:
    """A manual test case with expected characteristics.

    Used to validate that recommendations make sense for known inputs.
    """
    name: str
    seed_work_query: str  # Query to find seed work
    expected_characteristics: Dict[str, Any]  # Expected properties of recommendations
    min_similarity_threshold: float = 0.1  # Minimum acceptable similarity

    def __str__(self) -> str:
        return f"Test: {self.name} (seed: '{self.seed_work_query}')"


class RecommenderEvaluator:
    """Evaluates recommendation system quality.

    This class provides comprehensive evaluation of the recommender,
    including automated metrics and manual test cases.
    """

    def __init__(
        self,
        recommender: MusicRecommender,
        config: Optional[Config] = None
    ):
        """Initialize evaluator.

        Args:
            recommender: Loaded recommender instance to evaluate
            config: Configuration object (uses global config if None)
        """
        self.recommender = recommender
        self.config = config or get_config()

        if not recommender._is_ready:
            raise ValueError("Recommender must be loaded before evaluation")

    def evaluate_diversity(
        self,
        recommendations: List[Recommendation]
    ) -> DiversityMetrics:
        """Evaluate diversity of a recommendation list.

        Args:
            recommendations: List of recommendations to evaluate

        Returns:
            DiversityMetrics with diversity scores
        """
        if not recommendations:
            raise ValueError("Cannot evaluate empty recommendation list")

        n = len(recommendations)

        # Extract metadata
        composers = [r.composer for r in recommendations]
        work_types = [r.work_type for r in recommendations if r.work_type]
        periods = [r.period for r in recommendations if r.period]
        scores = [r.similarity_score for r in recommendations]

        # Unique ratios
        unique_composers_ratio = len(set(composers)) / n
        unique_work_types_ratio = len(set(work_types)) / len(work_types) if work_types else 0.0
        unique_periods_ratio = len(set(periods)) / len(periods) if periods else 0.0

        # Composer concentration
        composer_counts = pd.Series(composers).value_counts()
        max_same_composer_ratio = composer_counts.max() / n

        # Gini coefficient (measure of inequality)
        gini = self._compute_gini(composer_counts.values)

        # Pairwise dissimilarity
        # Dissimilarity = 1 - similarity
        dissimilarities = []
        for i in range(n):
            for j in range(i + 1, n):
                # Approximate dissimilarity from scores
                # Works with similar scores are likely similar to each other
                score_diff = abs(scores[i] - scores[j])
                dissimilarities.append(score_diff)

        avg_pairwise_dissimilarity = np.mean(dissimilarities) if dissimilarities else 0.0

        return DiversityMetrics(
            avg_pairwise_dissimilarity=avg_pairwise_dissimilarity,
            unique_composers_ratio=unique_composers_ratio,
            unique_work_types_ratio=unique_work_types_ratio,
            unique_periods_ratio=unique_periods_ratio,
            max_same_composer_ratio=max_same_composer_ratio,
            gini_coefficient=gini
        )

    def evaluate_coverage(
        self,
        n_samples: int = 100,
        n_recommendations: int = 10
    ) -> CoverageMetrics:
        """Evaluate catalog coverage by sampling recommendations.

        Args:
            n_samples: Number of random works to use as seeds
            n_recommendations: Number of recommendations per seed

        Returns:
            CoverageMetrics with coverage statistics
        """
        dataset = self.recommender._dataset
        all_work_ids = dataset.works['work_id'].values

        # Sample seed works
        np.random.seed(self.config.random_seed)
        sample_size = min(n_samples, len(all_work_ids))
        seed_work_ids = np.random.choice(all_work_ids, size=sample_size, replace=False)

        # Track which works get recommended
        recommended_works = defaultdict(int)
        recommended_composers = set()
        recommended_work_types = set()
        recommended_periods = set()

        logger.info(f"Sampling {sample_size} works to evaluate coverage...")

        for seed_id in seed_work_ids:
            try:
                recommendations = self.recommender.recommend_similar(
                    seed_id,
                    n=n_recommendations,
                    exclude_same_work=True
                )

                for rec in recommendations:
                    recommended_works[rec.work_id] += 1
                    recommended_composers.add(rec.composer)
                    if rec.work_type:
                        recommended_work_types.add(rec.work_type)
                    if rec.period:
                        recommended_periods.add(rec.period)

            except Exception as e:
                logger.warning(f"Failed to get recommendations for {seed_id}: {e}")

        # Calculate metrics
        works_ever_recommended = len(recommended_works)
        coverage_ratio = works_ever_recommended / len(all_work_ids)

        # Per-category coverage
        all_composers = set(dataset.composers['name'])
        all_work_types = set(dataset.works['work_type'].dropna())
        all_periods = set(dataset.composers['period'].dropna()) if 'period' in dataset.composers.columns else set()

        # Recommendation distribution
        rec_counts = list(recommended_works.values())
        avg_recs = np.mean(rec_counts) if rec_counts else 0.0
        median_recs = np.median(rec_counts) if rec_counts else 0.0

        most_recommended = sorted(
            recommended_works.items(),
            key=lambda x: x[1],
            reverse=True
        )[:10]

        return CoverageMetrics(
            total_catalog_size=len(all_work_ids),
            works_ever_recommended=works_ever_recommended,
            coverage_ratio=coverage_ratio,
            composers_covered=len(recommended_composers),
            composers_total=len(all_composers),
            work_types_covered=len(recommended_work_types),
            work_types_total=len(all_work_types),
            periods_covered=len(recommended_periods),
            periods_total=len(all_periods),
            avg_recommendations_per_work=avg_recs,
            median_recommendations_per_work=median_recs,
            most_recommended_works=most_recommended
        )

    def evaluate_similarity_distribution(
        self,
        n_samples: int = 100,
        n_recommendations: int = 10
    ) -> Dict[str, Any]:
        """Analyze distribution of similarity scores.

        Args:
            n_samples: Number of random works to sample
            n_recommendations: Number of recommendations per work

        Returns:
            Dictionary with distribution statistics
        """
        dataset = self.recommender._dataset
        all_work_ids = dataset.works['work_id'].values

        np.random.seed(self.config.random_seed)
        sample_size = min(n_samples, len(all_work_ids))
        seed_work_ids = np.random.choice(all_work_ids, size=sample_size, replace=False)

        all_scores = []

        logger.info(f"Sampling {sample_size} works to analyze similarity distribution...")

        for seed_id in seed_work_ids:
            try:
                recommendations = self.recommender.recommend_similar(
                    seed_id,
                    n=n_recommendations,
                    exclude_same_work=True
                )
                scores = [r.similarity_score for r in recommendations]
                all_scores.extend(scores)
            except Exception as e:
                logger.warning(f"Failed for {seed_id}: {e}")

        if not all_scores:
            return {"error": "No scores collected"}

        scores_array = np.array(all_scores)

        return {
            'mean': float(np.mean(scores_array)),
            'median': float(np.median(scores_array)),
            'std': float(np.std(scores_array)),
            'min': float(np.min(scores_array)),
            'max': float(np.max(scores_array)),
            'q25': float(np.percentile(scores_array, 25)),
            'q75': float(np.percentile(scores_array, 75)),
            'n_samples': len(all_scores)
        }

    def run_test_cases(self) -> Dict[str, Any]:
        """Run predefined test cases.

        Returns:
            Dictionary with test results
        """
        test_cases = self._get_test_cases()
        results = {}

        logger.info(f"Running {len(test_cases)} test cases...")

        for test_case in test_cases:
            try:
                result = self._run_test_case(test_case)
                results[test_case.name] = result
                status = "PASS" if result['passed'] else "FAIL"
                logger.info(f"  {test_case.name}: {status}")
            except Exception as e:
                logger.error(f"  {test_case.name}: ERROR - {e}")
                results[test_case.name] = {
                    'passed': False,
                    'error': str(e)
                }

        # Summary
        passed = sum(1 for r in results.values() if r.get('passed', False))
        total = len(results)

        return {
            'summary': {
                'total': total,
                'passed': passed,
                'failed': total - passed,
                'pass_rate': passed / total if total > 0 else 0.0
            },
            'details': results
        }

    def _run_test_case(self, test_case: TestCase) -> Dict[str, Any]:
        """Run a single test case.

        Args:
            test_case: Test case to run

        Returns:
            Dictionary with test results
        """
        # Find seed work
        recommendations = self.recommender.recommend_by_query(
            test_case.seed_work_query,
            n=10
        )

        if not recommendations:
            return {
                'passed': False,
                'reason': 'No recommendations generated'
            }

        # Check expected characteristics
        failures = []

        expected = test_case.expected_characteristics

        # Check minimum similarity
        min_score = min(r.similarity_score for r in recommendations)
        if min_score < test_case.min_similarity_threshold:
            failures.append(f"Minimum similarity {min_score:.3f} below threshold {test_case.min_similarity_threshold}")

        # Check expected composer presence
        if 'should_include_composer' in expected:
            expected_composer = expected['should_include_composer']
            composers = [r.composer for r in recommendations]
            if expected_composer not in composers:
                failures.append(f"Expected composer '{expected_composer}' not in recommendations")

        # Check expected work type presence
        if 'should_include_work_type' in expected:
            expected_type = expected['should_include_work_type']
            work_types = [r.work_type for r in recommendations if r.work_type]
            if expected_type not in work_types:
                failures.append(f"Expected work type '{expected_type}' not in recommendations")

        # Check diversity constraints
        if 'max_same_composer' in expected:
            max_allowed = expected['max_same_composer']
            composers = [r.composer for r in recommendations]
            composer_counts = pd.Series(composers).value_counts()
            max_count = composer_counts.max()
            if max_count > max_allowed:
                failures.append(f"Too many works from same composer: {max_count} > {max_allowed}")

        return {
            'passed': len(failures) == 0,
            'failures': failures,
            'recommendations': [r.to_dict() for r in recommendations[:3]]  # Include top 3 for inspection
        }

    def _get_test_cases(self) -> List[TestCase]:
        """Define test cases.

        Returns:
            List of test cases to run
        """
        # These test cases are generic - in production, customize based on actual data
        return [
            TestCase(
                name="baroque_concerto_similarity",
                seed_work_query="concerto",
                expected_characteristics={
                    'should_include_work_type': 'concerto',
                    'max_same_composer': 7  # Allow some same-composer but not all
                },
                min_similarity_threshold=0.05
            ),
            TestCase(
                name="symphony_diversity",
                seed_work_query="symphony",
                expected_characteristics={
                    'max_same_composer': 6  # Ensure some diversity
                },
                min_similarity_threshold=0.05
            ),
            TestCase(
                name="opera_recommendations",
                seed_work_query="opera",
                expected_characteristics={
                    'should_include_work_type': 'opera',
                },
                min_similarity_threshold=0.05
            ),
        ]

    def generate_full_report(
        self,
        n_samples: int = 50,
        n_recommendations: int = 10
    ) -> str:
        """Generate comprehensive evaluation report.

        Args:
            n_samples: Number of samples for coverage/diversity evaluation
            n_recommendations: Number of recommendations per sample

        Returns:
            Formatted report string
        """
        lines = [
            "=" * 80,
            "RECOMMENDER SYSTEM EVALUATION REPORT",
            "=" * 80,
            ""
        ]

        # Similarity distribution
        lines.append("SIMILARITY DISTRIBUTION")
        lines.append("-" * 80)
        sim_dist = self.evaluate_similarity_distribution(n_samples, n_recommendations)
        for key, value in sim_dist.items():
            if key != 'n_samples':
                lines.append(f"  {key}: {value:.4f}")
        lines.append(f"  Samples: {sim_dist.get('n_samples', 0)}")
        lines.append("")

        # Coverage
        lines.append("COVERAGE METRICS")
        lines.append("-" * 80)
        coverage = self.evaluate_coverage(n_samples, n_recommendations)
        lines.append(str(coverage))
        lines.append("")

        # Diversity (sample a few recommendation lists)
        lines.append("DIVERSITY METRICS (Sample of 5 recommendation lists)")
        lines.append("-" * 80)
        dataset = self.recommender._dataset
        sample_work_ids = np.random.choice(
            dataset.works['work_id'].values,
            size=min(5, len(dataset.works)),
            replace=False
        )

        diversity_scores = []
        for work_id in sample_work_ids:
            try:
                recs = self.recommender.recommend_similar(work_id, n=n_recommendations)
                div = self.evaluate_diversity(recs)
                diversity_scores.append(div)
            except:
                pass

        if diversity_scores:
            avg_dissim = np.mean([d.avg_pairwise_dissimilarity for d in diversity_scores])
            avg_composer_ratio = np.mean([d.unique_composers_ratio for d in diversity_scores])
            avg_gini = np.mean([d.gini_coefficient for d in diversity_scores])

            lines.append(f"  Avg pairwise dissimilarity: {avg_dissim:.3f}")
            lines.append(f"  Avg unique composer ratio: {avg_composer_ratio:.1%}")
            lines.append(f"  Avg Gini coefficient: {avg_gini:.3f}")
        lines.append("")

        # Test cases
        lines.append("TEST CASES")
        lines.append("-" * 80)
        test_results = self.run_test_cases()
        summary = test_results['summary']
        lines.append(f"  Total: {summary['total']}")
        lines.append(f"  Passed: {summary['passed']}")
        lines.append(f"  Failed: {summary['failed']}")
        lines.append(f"  Pass rate: {summary['pass_rate']:.1%}")
        lines.append("")

        # Details of failures
        if summary['failed'] > 0:
            lines.append("Failed Tests:")
            for name, result in test_results['details'].items():
                if not result.get('passed', False):
                    lines.append(f"\n  {name}:")
                    if 'error' in result:
                        lines.append(f"    Error: {result['error']}")
                    elif 'failures' in result:
                        for failure in result['failures']:
                            lines.append(f"    - {failure}")

        lines.append("=" * 80)

        return "\n".join(lines)

    @staticmethod
    def _compute_gini(values: np.ndarray) -> float:
        """Compute Gini coefficient (measure of inequality).

        Args:
            values: Array of values (e.g., counts per category)

        Returns:
            Gini coefficient (0 = perfect equality, 1 = perfect inequality)
        """
        if len(values) == 0:
            return 0.0

        sorted_values = np.sort(values)
        n = len(values)
        index = np.arange(1, n + 1)
        return (2 * np.sum(index * sorted_values)) / (n * np.sum(sorted_values)) - (n + 1) / n
