#!/usr/bin/env python3
"""
Standalone example script for the classical music recommendation system.

This script can be run directly from the project root:
    python3 run_recommender_example.py

Or as a module:
    python3 -m run_recommender_example
"""

import sys
from pathlib import Path

# Add project root to path so we can import recommender package
project_root = Path(__file__).parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

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

    # Get all works to find a sample with a known composer
    recommender = service.get_recommender()

    # Find a work with a known composer for better demo
    works_with_composer = recommender._dataset.works[
        recommender._dataset.works['composer_id'].notna() &
        (recommender._dataset.works['composer_id'] != 'unknown-composer')
    ]

    if len(works_with_composer) > 0:
        sample_work_id = works_with_composer.iloc[0]['work_id']
        sample_title = works_with_composer.iloc[0]['title']
    else:
        # Fallback to first work if no composer found
        sample_works = recommender._dataset.works.head(1)
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


def example_3_diverse_recommendations():
    """Example 3: Compare standard vs diverse recommendations."""
    print("\n" + "=" * 80)
    print("EXAMPLE 3: Diverse Recommendations")
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


def example_4_evaluation():
    """Example 4: Quick evaluation metrics."""
    print("\n" + "=" * 80)
    print("EXAMPLE 4: System Evaluation")
    print("=" * 80)

    service = get_service()
    recommender = service.get_recommender()

    # Create evaluator
    evaluator = RecommenderEvaluator(recommender)

    print("\nRunning evaluation (this may take a moment)...")

    # Get similarity distribution
    sim_dist = evaluator.evaluate_similarity_distribution(n_samples=20, n_recommendations=5)

    print("\nSimilarity Score Distribution:")
    print(f"  Mean: {sim_dist.get('mean', 0):.3f}")
    print(f"  Median: {sim_dist.get('median', 0):.3f}")
    print(f"  Min: {sim_dist.get('min', 0):.3f}")
    print(f"  Max: {sim_dist.get('max', 0):.3f}")

    # Get service metrics
    metrics = service.get_metrics()
    print("\nService Metrics:")
    print(f"  Total requests: {metrics['total_requests']}")
    print(f"  Successful: {metrics['successful_recommendations']}")
    print(f"  Avg latency: {metrics['avg_latency_ms']:.2f}ms")


def example_5_semantic_search():
    """Example 5: Semantic search by mood and activity."""
    print("\n" + "=" * 80)
    print("EXAMPLE 5: Semantic Search - Mood and Activity Based")
    print("=" * 80)

    service = get_service()

    # Check if semantic search is available
    if service.get_semantic_search() is None:
        print("\nSemantic search not available (requires sentence-transformers)")
        print("Install with: pip install sentence-transformers torch")
        return

    # Test mood-based searches
    mood_queries = [
        "I'm feeling moody and contemplative",
        "bright and cheerful music",
        "rainy Sunday morning vibes",
    ]

    for query in mood_queries:
        try:
            print(f"\nSearching: '{query}'")
            print("-" * 80)

            results = service.search_by_mood(query, n=3)

            for rec in results:
                print(f"{rec['rank']}. {rec['title']} by {rec['composer']}")
                if rec['key']:
                    print(f"   Key: {rec['key']}")
                print(f"   Similarity: {rec['similarity_score']:.3f}")
                print(f"   {rec['explanation']}")

        except Exception as e:
            logger.warning(f"Search failed for '{query}': {e}")

    # Test activity-based searches
    print("\n" + "=" * 80)
    print("Activity-Based Search")
    print("=" * 80)

    activities = [
        ("studying", "need to focus"),
        ("relaxing", ""),
        ("morning", "uplifting start to the day"),
    ]

    for activity, context in activities:
        try:
            query_desc = f"{activity}" + (f" ({context})" if context else "")
            print(f"\nSearching for: {query_desc}")
            print("-" * 80)

            results = service.search_by_activity(activity, context=context, n=3)

            for rec in results:
                print(f"{rec['rank']}. {rec['title']} by {rec['composer']}")
                print(f"   Score: {rec['similarity_score']:.3f}")

        except Exception as e:
            logger.warning(f"Activity search failed: {e}")


def example_6_hybrid_search():
    """Example 6: Hybrid semantic + similarity search."""
    print("\n" + "=" * 80)
    print("EXAMPLE 6: Hybrid Search (Semantic + Similarity)")
    print("=" * 80)

    service = get_service()

    if service.get_semantic_search() is None:
        print("\nSemantic search not available")
        return

    recommender = service.get_recommender()

    # Get a sample work
    sample_works = recommender._dataset.works.head(1)
    if len(sample_works) == 0:
        print("No works available")
        return

    sample_work_id = sample_works.iloc[0]['work_id']
    sample_title = sample_works.iloc[0]['title']

    print(f"\nSeed work: {sample_title}")
    print("\nSearching for: 'dark and dramatic pieces similar to this work'")
    print("-" * 80)

    try:
        results = service.hybrid_search(
            query="dark and dramatic",
            similar_to_work_id=sample_work_id,
            n=5,
            semantic_weight=0.6
        )

        for rec in results:
            print(f"{rec['rank']}. {rec['title']} by {rec['composer']}")
            if rec['key']:
                print(f"   Key: {rec['key']}")
            print(f"   Score: {rec['similarity_score']:.3f}")

    except Exception as e:
        logger.warning(f"Hybrid search failed: {e}")


def example_7_embedding_quality():
    """Example 7: Semantic search quality metrics."""
    print("\n" + "=" * 80)
    print("EXAMPLE 7: Semantic Search Quality Metrics")
    print("=" * 80)

    service = get_service()

    if service.get_semantic_search() is None:
        print("\nSemantic search not available")
        return

    try:
        metrics = service.get_embedding_quality_metrics()

        print(f"\nEmbedding Model: {metrics['model_name']}")
        print(f"Number of Works: {metrics['num_works']}")
        print(f"Embedding Dimension: {metrics['embedding_dim']}")

        print("\nSimilarity Distribution:")
        sim_dist = metrics['similarity_distribution']
        print(f"  Mean: {sim_dist['mean']:.3f}")
        print(f"  Median: {sim_dist['median']:.3f}")
        print(f"  Range: [{sim_dist['min']:.3f}, {sim_dist['max']:.3f}]")

        print("\nDescription Statistics:")
        desc_stats = metrics['description_stats']
        print(f"  Avg Length: {desc_stats['avg_length']:.0f} characters")
        print(f"  Range: [{desc_stats['min_length']}, {desc_stats['max_length']}]")

    except Exception as e:
        logger.warning(f"Failed to get metrics: {e}")


def main():
    """Run all examples."""
    print("\n" + "=" * 80)
    print("CLASSICAL MUSIC RECOMMENDATION SYSTEM - EXAMPLES")
    print("=" * 80)

    try:
        example_1_basic_usage()
        example_2_search_and_recommend()
        example_3_diverse_recommendations()
        example_4_evaluation()

        # Semantic search examples (may not be available)
        print("\n" + "=" * 80)
        print("SEMANTIC SEARCH EXAMPLES")
        print("=" * 80)

        example_5_semantic_search()
        example_6_hybrid_search()
        example_7_embedding_quality()

        print("\n" + "=" * 80)
        print("All examples completed successfully!")
        print("=" * 80 + "\n")

    except Exception as e:
        logger.error(f"Example failed: {e}", exc_info=True)
        print(f"\nError: {e}")
        print("Note: Make sure you have data files in data/processed/")


if __name__ == "__main__":
    main()
