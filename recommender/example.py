"""
Example script demonstrating the classical music recommendation system.

This script shows how to:
1. Initialize the recommender system
2. Get recommendations for specific works
3. Search and recommend by query
4. Filter recommendations by criteria
5. Use diversity-aware recommendations
6. Evaluate system quality

Run this script:
    python -m recommender.example
"""

import logging
from recommender.service import initialize_service, get_service
from recommender import MusicRecommender, RecommenderEvaluator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


def example_1_basic_usage():
    """Example 1: Basic recommendation usage."""
    print("\n" + "=" * 80)
    print("EXAMPLE 1: Basic Recommendation Usage")
    print("=" * 80)

    # Initialize the service (happens once)
    logger.info("Initializing recommendation service...")
    initialize_service()

    # Get the service
    service = get_service()

    # Check health
    health = service.health_check()
    print(f"\nSystem Status: {health['status']}")
    print(f"Dataset Size: {health['dataset_size']} works")

    # Get all works to find a sample
    recommender = service.get_recommender()
    sample_works = recommender._dataset.works.head(1)

    if len(sample_works) > 0:
        sample_work_id = sample_works.iloc[0]['work_id']
        sample_title = sample_works.iloc[0]['title']

        print(f"\nGetting recommendations for: {sample_title}")
        print("-" * 80)

        # Get recommendations
        recommendations = service.recommend_similar(
            work_id=sample_work_id,
            n=5
        )

        for rec in recommendations:
            print(f"\n{rec['rank']}. {rec['title']}")
            print(f"   Composer: {rec['composer']}")
            if rec['work_type']:
                print(f"   Type: {rec['work_type']}")
            if rec['key']:
                print(f"   Key: {rec['key']}")
            print(f"   Similarity: {rec['similarity_score']:.3f}")
            if rec['explanation']:
                print(f"   {rec['explanation']}")


def example_2_search_and_recommend():
    """Example 2: Search by query and get recommendations."""
    print("\n" + "=" * 80)
    print("EXAMPLE 2: Search and Recommend")
    print("=" * 80)

    service = get_service()

    # Try searching for common classical works
    search_queries = ["concerto", "symphony", "sonata"]

    for query in search_queries:
        try:
            print(f"\nSearching for: '{query}'")
            print("-" * 80)

            recommendations = service.recommend_by_query(query, n=3)

            for rec in recommendations:
                print(f"{rec['rank']}. {rec['title']} by {rec['composer']}")
                print(f"   Score: {rec['similarity_score']:.3f}")

            break  # Just show first successful search

        except ValueError as e:
            print(f"No results for '{query}'")
            continue


def example_3_filtered_browsing():
    """Example 3: Browse by filtering criteria."""
    print("\n" + "=" * 80)
    print("EXAMPLE 3: Filter-Based Browsing")
    print("=" * 80)

    service = get_service()
    recommender = service.get_recommender()

    # Get available composers
    composers = recommender._dataset.composers['name'].head(3).tolist()

    for composer_name in composers:
        try:
            print(f"\nWorks by {composer_name}:")
            print("-" * 80)

            recommendations = service.recommend_by_filters(
                composer=composer_name,
                n=5
            )

            for rec in recommendations:
                print(f"{rec['rank']}. {rec['title']}")
                if rec['work_type']:
                    print(f"   Type: {rec['work_type']}")

            break  # Show first successful filter

        except ValueError:
            continue


def example_4_diverse_recommendations():
    """Example 4: Compare standard vs diverse recommendations."""
    print("\n" + "=" * 80)
    print("EXAMPLE 4: Diverse Recommendations")
    print("=" * 80)

    service = get_service()
    recommender = service.get_recommender()

    # Get a sample work
    sample_works = recommender._dataset.works.head(1)

    if len(sample_works) > 0:
        sample_work_id = sample_works.iloc[0]['work_id']
        sample_title = sample_works.iloc[0]['title']

        print(f"\nSeed work: {sample_title}")

        # Standard recommendations
        print("\nStandard Recommendations (similarity-focused):")
        print("-" * 80)
        standard_recs = service.recommend_similar(
            work_id=sample_work_id,
            n=5,
            diverse=False
        )

        composer_counts_standard = {}
        for rec in standard_recs:
            composer = rec['composer']
            composer_counts_standard[composer] = composer_counts_standard.get(composer, 0) + 1
            print(f"{rec['rank']}. {rec['title']} by {composer}")

        # Diverse recommendations
        print("\nDiverse Recommendations (diversity-aware):")
        print("-" * 80)
        diverse_recs = service.recommend_similar(
            work_id=sample_work_id,
            n=5,
            diverse=True,
            diversity_weight=0.4
        )

        composer_counts_diverse = {}
        for rec in diverse_recs:
            composer = rec['composer']
            composer_counts_diverse[composer] = composer_counts_diverse.get(composer, 0) + 1
            print(f"{rec['rank']}. {rec['title']} by {composer}")

        print("\nComposer distribution:")
        print(f"  Standard: {len(composer_counts_standard)} unique composers")
        print(f"  Diverse: {len(composer_counts_diverse)} unique composers")


def example_5_evaluation():
    """Example 5: Evaluate recommendation quality."""
    print("\n" + "=" * 80)
    print("EXAMPLE 5: System Evaluation")
    print("=" * 80)

    # Get the recommender
    service = get_service()
    recommender = service.get_recommender()

    # Create evaluator
    evaluator = RecommenderEvaluator(recommender)

    print("\nRunning evaluation (this may take a moment)...")

    # Run test cases
    test_results = evaluator.run_test_cases()
    summary = test_results['summary']

    print("\nTest Results:")
    print(f"  Total tests: {summary['total']}")
    print(f"  Passed: {summary['passed']}")
    print(f"  Failed: {summary['failed']}")
    print(f"  Pass rate: {summary['pass_rate']:.1%}")

    # Get similarity distribution
    sim_dist = evaluator.evaluate_similarity_distribution(n_samples=20, n_recommendations=5)

    print("\nSimilarity Score Distribution:")
    print(f"  Mean: {sim_dist.get('mean', 0):.3f}")
    print(f"  Median: {sim_dist.get('median', 0):.3f}")
    print(f"  Min: {sim_dist.get('min', 0):.3f}")
    print(f"  Max: {sim_dist.get('max', 0):.3f}")
    print(f"  Std: {sim_dist.get('std', 0):.3f}")

    # Get service metrics
    metrics = service.get_metrics()
    print("\nService Metrics:")
    print(f"  Total requests: {metrics['total_requests']}")
    print(f"  Successful: {metrics['successful_recommendations']}")
    print(f"  Errors: {metrics['errors']}")
    print(f"  Avg latency: {metrics['avg_latency_ms']:.2f}ms")


def main():
    """Run all examples."""
    print("\n" + "=" * 80)
    print("CLASSICAL MUSIC RECOMMENDATION SYSTEM - EXAMPLES")
    print("=" * 80)

    try:
        example_1_basic_usage()
        example_2_search_and_recommend()
        example_3_filtered_browsing()
        example_4_diverse_recommendations()
        example_5_evaluation()

        print("\n" + "=" * 80)
        print("All examples completed successfully!")
        print("=" * 80 + "\n")

    except Exception as e:
        logger.error(f"Example failed: {e}", exc_info=True)
        print(f"\nError: {e}")
        print("Note: Make sure you have data files in data/processed/")


if __name__ == "__main__":
    main()
