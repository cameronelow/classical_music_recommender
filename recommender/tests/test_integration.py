"""
Integration tests for the recommendation system.

These tests verify that the full system works end-to-end with real data.
They require the parquet files to be present in data/processed/.
"""

import pytest
from pathlib import Path

from recommender.service import RecommenderService, initialize_service, get_service
from recommender import MusicRecommender, Config


# Mark all tests in this file as integration tests
pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def initialized_service():
    """Initialize service once for all integration tests."""
    config = Config()

    # Check if data files exist
    if not config.paths.works_parquet.exists():
        pytest.skip(f"Data file not found: {config.paths.works_parquet}")

    service = RecommenderService(config)
    try:
        service.initialize()
        yield service
    except FileNotFoundError as e:
        pytest.skip(f"Required data files not found: {e}")


class TestEndToEndRecommendations:
    """End-to-end tests with real data."""

    def test_service_initialization(self, initialized_service):
        """Test that service initializes successfully."""
        health = initialized_service.health_check()
        assert health['status'] == 'healthy'
        assert health['recommender_loaded'] is True
        assert health['dataset_size'] > 0

    def test_recommend_similar_real_data(self, initialized_service):
        """Test recommendations with real data."""
        # Get a sample work ID
        recommender = initialized_service.get_recommender()
        sample_work_id = recommender._dataset.works.iloc[0]['work_id']

        # Get recommendations
        recommendations = initialized_service.recommend_similar(
            work_id=sample_work_id,
            n=5
        )

        # Verify recommendations
        assert len(recommendations) > 0
        assert len(recommendations) <= 5

        # Check structure
        for rec in recommendations:
            assert 'work_id' in rec
            assert 'title' in rec
            assert 'composer' in rec
            assert 'similarity_score' in rec
            assert 'rank' in rec
            assert rec['similarity_score'] >= 0

        # Check ranking order
        scores = [r['similarity_score'] for r in recommendations]
        assert scores == sorted(scores, reverse=True), "Recommendations should be sorted by score"

    def test_search_recommendations_real_data(self, initialized_service):
        """Test search-based recommendations."""
        recommender = initialized_service.get_recommender()

        # Get a sample work title
        sample_title = recommender._dataset.works.iloc[0]['title']

        # Extract a word from the title for searching
        search_term = sample_title.split()[0]

        try:
            recommendations = initialized_service.recommend_by_query(
                query=search_term,
                n=3
            )

            assert len(recommendations) > 0
            assert len(recommendations) <= 3

        except ValueError:
            # If search term doesn't match anything, that's acceptable
            pytest.skip(f"Search term '{search_term}' didn't match any works")

    def test_diverse_recommendations_real_data(self, initialized_service):
        """Test diversity-aware recommendations."""
        recommender = initialized_service.get_recommender()

        # Find a work from a composer with multiple works
        composer_work_counts = recommender._dataset.works.groupby('composer_id').size()
        composer_with_multiple = composer_work_counts[composer_work_counts >= 2].index[0] if len(composer_work_counts[composer_work_counts >= 2]) > 0 else None

        if composer_with_multiple is None:
            pytest.skip("Need a composer with multiple works for this test")

        works_by_composer = recommender._dataset.works[
            recommender._dataset.works['composer_id'] == composer_with_multiple
        ]
        sample_work_id = works_by_composer.iloc[0]['work_id']

        # Get standard recommendations
        standard_recs = initialized_service.recommend_similar(
            work_id=sample_work_id,
            n=10,
            diverse=False
        )

        # Get diverse recommendations
        diverse_recs = initialized_service.recommend_similar(
            work_id=sample_work_id,
            n=10,
            diverse=True,
            diversity_weight=0.5
        )

        # Both should return recommendations
        assert len(standard_recs) > 0
        assert len(diverse_recs) > 0

        # Diverse recommendations should have more variety (if dataset is large enough)
        if len(recommender._dataset.works) > 20:
            standard_composers = set(r['composer'] for r in standard_recs)
            diverse_composers = set(r['composer'] for r in diverse_recs)

            # Diverse mode should generally have more unique composers
            # (This may not always be true with small datasets)
            print(f"Standard: {len(standard_composers)} composers, Diverse: {len(diverse_composers)} composers")

    def test_get_work_info_real_data(self, initialized_service):
        """Test retrieving work information."""
        recommender = initialized_service.get_recommender()
        sample_work_id = recommender._dataset.works.iloc[0]['work_id']

        info = initialized_service.get_work_info(sample_work_id)

        assert info is not None
        assert info['work_id'] == sample_work_id
        assert 'title' in info
        assert 'composer' in info

    def test_metrics_tracking(self, initialized_service):
        """Test that metrics are tracked correctly."""
        # Reset metrics
        initialized_service.reset_metrics()

        # Get initial metrics
        initial_metrics = initialized_service.get_metrics()
        assert initial_metrics['total_requests'] == 0

        # Make a request
        recommender = initialized_service.get_recommender()
        sample_work_id = recommender._dataset.works.iloc[0]['work_id']

        initialized_service.recommend_similar(sample_work_id, n=5)

        # Check metrics updated
        updated_metrics = initialized_service.get_metrics()
        assert updated_metrics['total_requests'] > initial_metrics['total_requests']
        assert updated_metrics['successful_recommendations'] > 0

    def test_error_handling_invalid_work_id(self, initialized_service):
        """Test error handling for invalid work ID."""
        with pytest.raises(ValueError, match="Work ID not found"):
            initialized_service.recommend_similar(
                work_id="nonexistent-work-id-12345",
                n=5
            )

    def test_error_handling_invalid_query(self, initialized_service):
        """Test error handling for queries with no matches."""
        with pytest.raises(ValueError, match="No works found"):
            initialized_service.recommend_by_query(
                query="zzznonexistentqueryzzzz",
                n=5
            )


class TestRecommenderDirectUsage:
    """Tests using the recommender directly (not via service)."""

    def test_direct_recommender_usage(self):
        """Test using MusicRecommender directly."""
        config = Config()

        if not config.paths.works_parquet.exists():
            pytest.skip("Data files not available")

        recommender = MusicRecommender(config)

        try:
            recommender.load()
        except FileNotFoundError:
            pytest.skip("Required data files not found")

        # Verify loaded
        assert recommender._is_ready is True
        assert recommender._dataset is not None
        assert recommender._features is not None

        # Get recommendations
        sample_work_id = recommender._dataset.works.iloc[0]['work_id']
        recommendations = recommender.recommend_similar(sample_work_id, n=3)

        assert len(recommendations) > 0
        assert all(hasattr(r, 'title') for r in recommendations)
        assert all(hasattr(r, 'composer') for r in recommendations)
        assert all(hasattr(r, 'similarity_score') for r in recommendations)
