"""
Tests for the auto_tagger module.
"""

import pytest
import pandas as pd
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

import sys
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from tagging.auto_tagger import ClassicalMusicAutoTagger, TAG_TAXONOMY
from tagging.tagging_config import TaggingConfig


@pytest.fixture
def mock_config(tmp_path):
    """Create a mock configuration for testing."""
    config = TaggingConfig()
    config.anthropic_api_key = "test-api-key"
    config.works_file = tmp_path / "works.csv"
    config.composers_file = tmp_path / "composers.csv"
    config.output_file = tmp_path / "output.csv"
    config.checkpoint_file = tmp_path / "checkpoint.json"
    config.corrections_file = tmp_path / "corrections.jsonl"
    config.delay_between_requests = 0  # No delay in tests
    return config


@pytest.fixture
def sample_work():
    """Sample work for testing."""
    return {
        'work_id': 'test-work-123',
        'title': 'Piano Sonata No. 14 in C-sharp minor, Op. 27 No. 2 "Moonlight"',
        'composer_id': 'beethoven-123',
        'work_type': 'Sonata',
        'key': 'C-sharp minor',
        'catalog_number': 'Op. 27 No. 2'
    }


def test_tag_taxonomy_structure():
    """Test that TAG_TAXONOMY has the expected structure."""
    expected_categories = [
        'mood', 'character', 'tempo', 'instrumentation',
        'form', 'complexity', 'popularity'
    ]

    for category in expected_categories:
        assert category in TAG_TAXONOMY
        assert isinstance(TAG_TAXONOMY[category], list)
        assert len(TAG_TAXONOMY[category]) > 0


def test_auto_tagger_initialization(mock_config):
    """Test that auto-tagger initializes correctly."""
    with patch('tagging.auto_tagger.Anthropic'):
        tagger = ClassicalMusicAutoTagger(mock_config)

        assert tagger.config == mock_config
        assert tagger.total_requests == 0
        assert tagger.total_input_tokens == 0
        assert tagger.total_output_tokens == 0


def test_validate_tags(mock_config):
    """Test tag validation."""
    with patch('tagging.auto_tagger.Anthropic'):
        tagger = ClassicalMusicAutoTagger(mock_config)

        # Valid tags
        valid_tags = ['dramatic', 'virtuosic', 'famous']
        result = tagger._validate_tags(valid_tags)
        assert result == valid_tags

        # Mix of valid and invalid tags
        mixed_tags = ['dramatic', 'invalid-tag', 'famous']
        result = tagger._validate_tags(mixed_tags)
        assert result == ['dramatic', 'famous']

        # All invalid tags
        invalid_tags = ['not-a-tag', 'also-invalid']
        result = tagger._validate_tags(invalid_tags)
        assert result == []


def test_build_tagging_prompt(mock_config, sample_work):
    """Test prompt building."""
    with patch('tagging.auto_tagger.Anthropic'):
        tagger = ClassicalMusicAutoTagger(mock_config)

        prompt = tagger._build_tagging_prompt(sample_work)

        # Check that prompt includes key information
        assert 'Piano Sonata No. 14' in prompt
        assert 'Sonata' in prompt
        assert 'C-sharp minor' in prompt
        assert 'Op. 27 No. 2' in prompt

        # Check that taxonomy is included
        assert 'mood' in prompt
        assert 'dramatic' in prompt
        assert 'famous' in prompt


@patch('tagging.auto_tagger.Anthropic')
def test_tag_work_success(mock_anthropic, mock_config, sample_work):
    """Test successful tagging of a work."""
    # Mock API response
    mock_response = Mock()
    mock_response.content = [Mock(text='{"tags": ["dramatic", "famous", "solo-piano"]}')]
    mock_response.usage = Mock(input_tokens=500, output_tokens=200)

    mock_client = Mock()
    mock_client.messages.create.return_value = mock_response
    mock_anthropic.return_value = mock_client

    tagger = ClassicalMusicAutoTagger(mock_config)
    tags, error = tagger.tag_work(sample_work)

    assert error is None
    assert len(tags) == 3
    assert 'dramatic' in tags
    assert 'famous' in tags
    assert 'solo-piano' in tags


@patch('tagging.auto_tagger.Anthropic')
def test_tag_work_with_markdown_response(mock_anthropic, mock_config, sample_work):
    """Test handling of markdown-wrapped JSON response."""
    # Mock API response with markdown
    mock_response = Mock()
    mock_response.content = [Mock(text='```json\n{"tags": ["dramatic", "famous"]}\n```')]
    mock_response.usage = Mock(input_tokens=500, output_tokens=200)

    mock_client = Mock()
    mock_client.messages.create.return_value = mock_response
    mock_anthropic.return_value = mock_client

    tagger = ClassicalMusicAutoTagger(mock_config)
    tags, error = tagger.tag_work(sample_work)

    assert error is None
    assert len(tags) == 2
    assert 'dramatic' in tags


@patch('tagging.auto_tagger.Anthropic')
def test_tag_work_invalid_json(mock_anthropic, mock_config, sample_work):
    """Test handling of invalid JSON response."""
    # Mock API response with invalid JSON
    mock_response = Mock()
    mock_response.content = [Mock(text='This is not JSON')]
    mock_response.usage = Mock(input_tokens=500, output_tokens=200)

    mock_client = Mock()
    mock_client.messages.create.return_value = mock_response
    mock_anthropic.return_value = mock_client

    mock_config.retry_attempts = 1  # Only retry once for faster test

    tagger = ClassicalMusicAutoTagger(mock_config)
    tags, error = tagger.tag_work(sample_work)

    assert error is not None
    assert 'JSON parse error' in error
    assert len(tags) == 0


def test_estimate_cost(mock_config):
    """Test cost estimation."""
    with patch('tagging.auto_tagger.Anthropic'):
        tagger = ClassicalMusicAutoTagger(mock_config)

        estimate = tagger.estimate_cost(100)

        assert estimate['total_works'] == 100
        assert estimate['estimated_tokens'] == 70000  # 100 * 700
        assert estimate['estimated_cost_usd'] > 0
        assert 'estimated_time_minutes' in estimate
        assert 'within_budget' in estimate


def test_get_usage_stats(mock_config):
    """Test usage statistics tracking."""
    with patch('tagging.auto_tagger.Anthropic'):
        tagger = ClassicalMusicAutoTagger(mock_config)

        # Simulate some usage
        tagger.total_requests = 10
        tagger.failed_requests = 2
        tagger.total_input_tokens = 5000
        tagger.total_output_tokens = 2000

        stats = tagger.get_usage_stats()

        assert stats['total_requests'] == 10
        assert stats['failed_requests'] == 2
        assert stats['success_rate'] == 0.8
        assert stats['total_input_tokens'] == 5000
        assert stats['total_output_tokens'] == 2000
        assert stats['total_tokens'] == 7000
        assert stats['total_cost_usd'] > 0


@patch('tagging.auto_tagger.Anthropic')
def test_tag_batch(mock_anthropic, mock_config, tmp_path):
    """Test batch tagging."""
    # Mock API response
    mock_response = Mock()
    mock_response.content = [Mock(text='{"tags": ["dramatic", "famous"]}')]
    mock_response.usage = Mock(input_tokens=500, output_tokens=200)

    mock_client = Mock()
    mock_client.messages.create.return_value = mock_response
    mock_anthropic.return_value = mock_client

    # Create sample works
    works = [
        {'work_id': f'work-{i}', 'title': f'Work {i}'}
        for i in range(5)
    ]

    tagger = ClassicalMusicAutoTagger(mock_config)
    result_df = tagger.tag_batch(works, save_progress=False)

    assert len(result_df) == 10  # 5 works * 2 tags each
    assert 'work_id' in result_df.columns
    assert 'tag' in result_df.columns
    assert 'source' in result_df.columns


def test_checkpoint_save_and_load(mock_config, tmp_path):
    """Test checkpoint saving and loading."""
    with patch('tagging.auto_tagger.Anthropic'):
        tagger = ClassicalMusicAutoTagger(mock_config)

        # Save checkpoint
        checkpoint_data = {
            'processed_work_ids': ['work-1', 'work-2'],
            'total_works': 10
        }
        tagger._save_checkpoint(checkpoint_data)

        # Load checkpoint
        loaded = tagger._load_checkpoint()

        assert loaded['processed_work_ids'] == ['work-1', 'work-2']
        assert loaded['total_works'] == 10

        # Clear checkpoint
        tagger.clear_checkpoint()
        loaded = tagger._load_checkpoint()
        assert loaded == {}
