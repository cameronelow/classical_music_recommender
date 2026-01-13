#!/usr/bin/env python3
"""
Test script specifically for checking explanation quality.
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
    print("EXPLANATION QUALITY TEST")
    print("=" * 80)

    # Initialize service
    print("\nInitializing service...")
    initialize_service()
    service = get_service()

    if service.get_semantic_search() is None:
        print("ERROR: Semantic search not available!")
        return

    print("✓ Service ready!\n")

    # Test various queries to see explanation quality
    test_queries = [
        "I'm feeling moody",
        "bright and cheerful",
        "studying music",
        "rainy Sunday",
        "epic and dramatic",
        "calm peaceful evening",
        "energetic workout",
        "romantic dinner",
        "dark and brooding",
        "morning coffee vibes"
    ]

    for query in test_queries:
        print("=" * 80)
        print(f"Query: '{query}'")
        print("=" * 80)

        results = service.search_by_mood(query, n=3)

        for rec in results:
            print(f"\n{rec['rank']}. {rec['title']}")
            print(f"   Composer: {rec['composer']}")
            print(f"   Score: {rec['similarity_score']:.3f}")
            print(f"   Explanation: {rec['explanation']}")
            if rec.get('tags'):
                print(f"   Tags: {', '.join(rec['tags'][:3])}")

        print()

    print("\n" + "=" * 80)
    print("TEST COMPLETE")
    print("=" * 80)
    print("\nCheck above for explanation quality.")
    print("Look for:")
    print("  ✓ Specific mood descriptors (e.g., 'melancholic and contemplative')")
    print("  ✓ Key information (e.g., 'E minor')")
    print("  ✓ Tags and period information")
    print("  ✗ Generic fallbacks like 'This piece matches your vibe'")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
