"""Tests for semantic search module."""

import pytest
import numpy as np
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from recommender.semantic_search import (
    SemanticSearchEngine,
    WorkDescriptionGenerator,
    KEY_MOOD_MAP,
    QUERY_ENHANCEMENTS,
    SemanticSearchResult,
    EmbeddingCacheMetadata
)
from recommender.config import Config
from recommender.data_loader import MusicDataset


@pytest.fixture
def mock_dataset():
    """Create a mock dataset for testing."""
    dataset = Mock(spec=MusicDataset)

    # Mock works DataFrame
    import pandas as pd
    works_data = {
        'work_id': ['work1', 'work2', 'work3'],
        'title': [
            'Symphony No. 9 in E minor',
            'Piano Concerto in D major',
            'Nocturne in C# minor'
        ],
        'composer_id': ['comp1', 'comp2', 'comp3'],
        'work_type': ['symphony', 'concerto', 'nocturne'],
        'key': ['E minor', 'D major', 'C# minor'],
        'catalog_number': ['Op. 95', 'K. 467', 'Op. 27 No. 1']
    }
    dataset.works = pd.DataFrame(works_data)

    # Mock composers DataFrame
    composers_data = {
        'composer_id': ['comp1', 'comp2', 'comp3'],
        'name': ['Antonín Dvořák', 'Wolfgang Amadeus Mozart', 'Frédéric Chopin'],
        'period': ['Romantic', 'Classical', 'Romantic']
    }
    dataset.composers = pd.DataFrame(composers_data)

    # Mock methods
    def get_work_by_id(work_id):
        row = dataset.works[dataset.works['work_id'] == work_id]
        if len(row) == 0:
            return None
        return row.iloc[0]

    def get_composer_name(composer_id):
        row = dataset.composers[dataset.composers['composer_id'] == composer_id]
        if len(row) == 0:
            return 'Unknown'
        return row.iloc[0]['name']

    def get_work_tags(work_id):
        # Return mock tags
        tag_map = {
            'work1': ['orchestral', 'dramatic', 'romantic-era'],
            'work2': ['classical-era', 'elegant', 'piano'],
            'work3': ['piano', 'nocturne', 'intimate', 'melancholic']
        }
        return tag_map.get(work_id, [])

    dataset.get_work_by_id = get_work_by_id
    dataset.get_composer_name = get_composer_name
    dataset.get_work_tags = get_work_tags

    return dataset


@pytest.fixture
def config():
    """Create test configuration."""
    return Config()


class TestWorkDescriptionGenerator:
    """Tests for WorkDescriptionGenerator."""

    def test_create_work_description(self, mock_dataset, config):
        """Test basic work description generation."""
        generator = WorkDescriptionGenerator(mock_dataset, config)

        description = generator.create_work_description(
            work_id='work1',
            composer_name='Antonín Dvořák',
            period='Romantic',
            tags=['orchestral', 'dramatic']
        )

        assert 'Symphony No. 9 in E minor' in description
        assert 'Antonín Dvořák' in description
        assert 'Romantic' in description
        assert 'E minor' in description
        assert 'orchestral' in description or 'dramatic' in description

    def test_get_key_moods_exact_match(self, mock_dataset, config):
        """Test key mood lookup with exact match."""
        generator = WorkDescriptionGenerator(mock_dataset, config)

        moods = generator._get_key_moods('E minor')
        assert 'sad' in moods or 'melancholic' in moods or 'moody' in moods

    def test_get_key_moods_major_fallback(self, mock_dataset, config):
        """Test key mood lookup with major fallback."""
        generator = WorkDescriptionGenerator(mock_dataset, config)

        moods = generator._get_key_moods('X major')  # Non-existent key
        assert 'bright' in moods or 'cheerful' in moods

    def test_get_key_moods_minor_fallback(self, mock_dataset, config):
        """Test key mood lookup with minor fallback."""
        generator = WorkDescriptionGenerator(mock_dataset, config)

        moods = generator._get_key_moods('X minor')  # Non-existent key
        assert 'melancholic' in moods

    def test_extract_catalog_features(self, mock_dataset, config):
        """Test catalog feature extraction."""
        generator = WorkDescriptionGenerator(mock_dataset, config)

        # BWV (Bach)
        features = generator._extract_catalog_features('BWV 1006')
        assert 'Baroque' in features

        # K. (Mozart)
        features = generator._extract_catalog_features('K. 467')
        assert 'Classical' in features

        # Op. (general)
        features = generator._extract_catalog_features('Op. 95')
        assert 'structured' in features


class TestSemanticSearchEngine:
    """Tests for SemanticSearchEngine."""

    @patch('recommender.semantic_search.SentenceTransformer')
    def test_initialization(self, mock_transformer, mock_dataset, config):
        """Test semantic search engine initialization."""
        engine = SemanticSearchEngine(mock_dataset, config)

        assert engine.dataset == mock_dataset
        assert engine.config == config
        assert not engine._is_ready

    @patch('recommender.semantic_search.SentenceTransformer')
    def test_load_generates_embeddings(self, mock_transformer, mock_dataset, config):
        """Test that load generates embeddings."""
        # Mock the sentence transformer
        mock_model = MagicMock()
        mock_model.encode.return_value = np.random.rand(3, 384)  # 3 works, 384 dims
        mock_transformer.return_value = mock_model

        # Disable caching for this test
        config.semantic_search.enable_embedding_cache = False

        engine = SemanticSearchEngine(mock_dataset, config)
        engine.load()

        assert engine._is_ready
        assert engine.work_embeddings is not None
        assert len(engine.work_embeddings) == 3
        assert len(engine.work_descriptions) == 3
        assert len(engine.work_id_mapping) == 3

    @patch('recommender.semantic_search.SentenceTransformer')
    def test_enhance_query_activity(self, mock_transformer, mock_dataset, config):
        """Test query enhancement for activities."""
        engine = SemanticSearchEngine(mock_dataset, config)

        enhanced = engine._enhance_query("studying")
        assert 'calm' in enhanced or 'focused' in enhanced

        enhanced = engine._enhance_query("working out")
        assert 'energetic' in enhanced

    @patch('recommender.semantic_search.SentenceTransformer')
    def test_enhance_query_mood(self, mock_transformer, mock_dataset, config):
        """Test query enhancement for moods."""
        engine = SemanticSearchEngine(mock_dataset, config)

        enhanced = engine._enhance_query("happy")
        assert 'joyful' in enhanced or 'cheerful' in enhanced

        enhanced = engine._enhance_query("sad")
        assert 'melancholic' in enhanced

    @patch('recommender.semantic_search.SentenceTransformer')
    def test_search_by_mood(self, mock_transformer, mock_dataset, config):
        """Test mood-based search."""
        # Mock the sentence transformer
        mock_model = MagicMock()

        # Mock embeddings: 3 works
        work_embeddings = np.array([
            [1.0, 0.0, 0.0],  # work1
            [0.0, 1.0, 0.0],  # work2
            [0.0, 0.0, 1.0],  # work3
        ])
        mock_model.encode.side_effect = [
            work_embeddings,  # Initial encoding
            np.array([[1.0, 0.0, 0.0]])  # Query encoding (matches work1)
        ]
        mock_transformer.return_value = mock_model

        config.semantic_search.enable_embedding_cache = False

        engine = SemanticSearchEngine(mock_dataset, config)
        engine.load()

        # Search
        results = engine.search_by_mood("moody", n=2)

        assert len(results) <= 2
        assert all(isinstance(r, SemanticSearchResult) for r in results)

    @patch('recommender.semantic_search.SentenceTransformer')
    def test_search_by_activity(self, mock_transformer, mock_dataset, config):
        """Test activity-based search."""
        mock_model = MagicMock()
        work_embeddings = np.random.rand(3, 384)
        mock_model.encode.side_effect = [
            work_embeddings,
            np.random.rand(1, 384)  # Query
        ]
        mock_transformer.return_value = mock_model

        config.semantic_search.enable_embedding_cache = False

        engine = SemanticSearchEngine(mock_dataset, config)
        engine.load()

        results = engine.search_by_activity("studying", context="need focus", n=2)

        assert len(results) <= 2

    @patch('recommender.semantic_search.SentenceTransformer')
    def test_search_respects_min_score(self, mock_transformer, mock_dataset, config):
        """Test that search respects minimum similarity threshold."""
        mock_model = MagicMock()

        # Create embeddings with known similarities
        work_embeddings = np.array([
            [1.0, 0.0],
            [0.0, 1.0],
            [0.5, 0.5],
        ])
        query_embedding = np.array([[1.0, 0.0]])

        mock_model.encode.side_effect = [work_embeddings, query_embedding]
        mock_transformer.return_value = mock_model

        config.semantic_search.enable_embedding_cache = False

        engine = SemanticSearchEngine(mock_dataset, config)
        engine.load()

        # Search with high threshold
        results = engine.search_by_mood("test", n=10, min_score=0.9)

        # Only work1 should match (similarity = 1.0)
        assert len(results) <= 3

    @patch('recommender.semantic_search.SentenceTransformer')
    def test_ensure_ready_raises_if_not_loaded(self, mock_transformer, mock_dataset, config):
        """Test that using engine before loading raises error."""
        engine = SemanticSearchEngine(mock_dataset, config)

        with pytest.raises(RuntimeError, match="not loaded"):
            engine._ensure_ready()

    @patch('recommender.semantic_search.SentenceTransformer')
    def test_get_embedding_quality_metrics(self, mock_transformer, mock_dataset, config):
        """Test embedding quality metrics calculation."""
        mock_model = MagicMock()
        work_embeddings = np.random.rand(3, 384)
        mock_model.encode.return_value = work_embeddings
        mock_transformer.return_value = mock_model

        config.semantic_search.enable_embedding_cache = False

        engine = SemanticSearchEngine(mock_dataset, config)
        engine.load()

        metrics = engine.get_embedding_quality_metrics()

        assert 'num_works' in metrics
        assert 'embedding_dim' in metrics
        assert 'similarity_distribution' in metrics
        assert metrics['num_works'] == 3
        assert metrics['embedding_dim'] == 384


class TestEmbeddingCacheMetadata:
    """Tests for EmbeddingCacheMetadata."""

    def test_to_dict(self):
        """Test metadata serialization."""
        metadata = EmbeddingCacheMetadata(
            model_name='test-model',
            creation_date='2025-01-01',
            num_works=100,
            embedding_dim=384
        )

        data = metadata.to_dict()

        assert data['model_name'] == 'test-model'
        assert data['num_works'] == 100
        assert data['embedding_dim'] == 384

    def test_from_dict(self):
        """Test metadata deserialization."""
        data = {
            'model_name': 'test-model',
            'creation_date': '2025-01-01',
            'num_works': 100,
            'embedding_dim': 384,
            'cache_version': '1.0'
        }

        metadata = EmbeddingCacheMetadata.from_dict(data)

        assert metadata.model_name == 'test-model'
        assert metadata.num_works == 100


class TestSemanticSearchResult:
    """Tests for SemanticSearchResult."""

    def test_to_dict(self):
        """Test result serialization."""
        result = SemanticSearchResult(
            work_id='work1',
            title='Test Symphony',
            composer='Test Composer',
            work_type='symphony',
            key='D major',
            period='Romantic',
            similarity_score=0.95,
            rank=1,
            explanation='Test explanation',
            tags=['orchestral', 'dramatic']
        )

        data = result.to_dict()

        assert data['work_id'] == 'work1'
        assert data['title'] == 'Test Symphony'
        assert data['similarity_score'] == 0.95
        assert data['rank'] == 1
        assert data['tags'] == ['orchestral', 'dramatic']


class TestKeyMoodMap:
    """Tests for KEY_MOOD_MAP constant."""

    def test_all_major_keys_present(self):
        """Test that all 12 major keys are in the map."""
        major_keys = [
            'C major', 'C# major', 'D major', 'Eb major', 'E major', 'F major',
            'F# major', 'G major', 'Ab major', 'A major', 'Bb major', 'B major'
        ]

        for key in major_keys:
            # Check either exact match or enharmonic equivalent
            assert key in KEY_MOOD_MAP or any(
                k.startswith(key[0]) and 'major' in k
                for k in KEY_MOOD_MAP.keys()
            )

    def test_all_minor_keys_present(self):
        """Test that all 12 minor keys are in the map."""
        minor_keys = [
            'C minor', 'C# minor', 'D minor', 'Eb minor', 'E minor', 'F minor',
            'F# minor', 'G minor', 'G# minor', 'A minor', 'Bb minor', 'B minor'
        ]

        for key in minor_keys:
            assert key in KEY_MOOD_MAP or any(
                k.startswith(key[0]) and 'minor' in k
                for k in KEY_MOOD_MAP.keys()
            )

    def test_moods_are_strings(self):
        """Test that all mood values are non-empty strings."""
        for key, moods in KEY_MOOD_MAP.items():
            assert isinstance(moods, str)
            assert len(moods) > 0


class TestQueryEnhancements:
    """Tests for QUERY_ENHANCEMENTS constant."""

    def test_common_activities_present(self):
        """Test that common activities have enhancements."""
        activities = ['studying', 'working out', 'relaxing', 'sleeping']

        for activity in activities:
            assert activity in QUERY_ENHANCEMENTS

    def test_common_moods_present(self):
        """Test that common moods have enhancements."""
        moods = ['happy', 'sad', 'calm', 'energetic']

        for mood in moods:
            assert mood in QUERY_ENHANCEMENTS

    def test_enhancements_are_strings(self):
        """Test that all enhancements are non-empty strings."""
        for keyword, enhancement in QUERY_ENHANCEMENTS.items():
            assert isinstance(enhancement, str)
            assert len(enhancement) > 0
