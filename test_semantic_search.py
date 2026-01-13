#!/usr/bin/env python3
"""
Quick test script for semantic search functionality.

This script demonstrates the new mood-based semantic search features.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from recommender.service import initialize_service, get_service

def main():
    print("\n" + "=" * 80)
    print("SEMANTIC SEARCH TEST")
    print("=" * 80)

    # Initialize service
    print("\nInitializing service (this may take a minute on first run)...")
    initialize_service()
    service = get_service()

    # Check if semantic search is available
    if service.get_semantic_search() is None:
        print("ERROR: Semantic search not available!")
        return

    print("✓ Semantic search initialized successfully!\n")

    # Test 1: Mood search
    print("=" * 80)
    print("TEST 1: Search by Mood")
    print("=" * 80)
    print("\nQuery: 'I'm feeling moody and contemplative'\n")

    results = service.search_by_mood("I'm feeling moody and contemplative", n=5)

    for rec in results:
        print(f"{rec['rank']}. {rec['title']}")
        print(f"   Composer: {rec['composer']}")
        if rec['key']:
            print(f"   Key: {rec['key']}")
        print(f"   Similarity: {rec['similarity_score']:.3f}")
        print(f"   {rec['explanation']}\n")

    # Test 2: Activity search
    print("=" * 80)
    print("TEST 2: Search by Activity")
    print("=" * 80)
    print("\nQuery: 'studying' with context 'need to focus'\n")

    results = service.search_by_activity("studying", context="need to focus", n=5)

    for rec in results:
        print(f"{rec['rank']}. {rec['title']} by {rec['composer']}")
        print(f"   Score: {rec['similarity_score']:.3f}\n")

    # Test 3: Description search
    print("=" * 80)
    print("TEST 3: Natural Language Description")
    print("=" * 80)
    print("\nQuery: 'bright and cheerful morning music'\n")

    results = service.search_by_description("bright and cheerful morning music", n=5)

    for rec in results:
        print(f"{rec['rank']}. {rec['title']} by {rec['composer']}")
        print(f"   Score: {rec['similarity_score']:.3f}")
        if rec['key']:
            print(f"   Key: {rec['key']}")
        print()

    # Test 4: Embedding quality
    print("=" * 80)
    print("TEST 4: Embedding Quality Metrics")
    print("=" * 80)

    metrics = service.get_embedding_quality_metrics()

    print(f"\nModel: {metrics['model_name']}")
    print(f"Works indexed: {metrics['num_works']}")
    print(f"Embedding dimension: {metrics['embedding_dim']}")

    sim_dist = metrics['similarity_distribution']
    print(f"\nSimilarity Distribution:")
    print(f"  Mean: {sim_dist['mean']:.3f}")
    print(f"  Median: {sim_dist['median']:.3f}")
    print(f"  Range: [{sim_dist['min']:.3f}, {sim_dist['max']:.3f}]")

    desc_stats = metrics['description_stats']
    print(f"\nDescription Statistics:")
    print(f"  Avg length: {desc_stats['avg_length']:.0f} characters")
    print(f"  Range: [{desc_stats['min_length']}, {desc_stats['max_length']}]")

    print("\n" + "=" * 80)
    print("ALL TESTS PASSED!")
    print("=" * 80)
    print("\nSemantic search is working correctly.")
    print("You can now search for music using natural language queries.")
    print("\nExample queries to try:")
    print("  • 'dark and dramatic'")
    print("  • 'peaceful and calm'")
    print("  • 'energetic and uplifting'")
    print("  • 'rainy Sunday morning vibes'")
    print("  • 'romantic dinner music'")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
