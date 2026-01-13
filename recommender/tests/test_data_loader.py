"""Tests for data loader module."""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path

from recommender.data_loader import DataLoader, MusicDataset, DataQualityReport
from recommender.config import Config, DataLoaderConfig


@pytest.fixture
def sample_works_df():
    """Create sample works dataframe."""
    return pd.DataFrame({
        'work_id': ['work1', 'work2', 'work3'],
        'composer_id': ['comp1', 'comp1', 'comp2'],
        'title': ['Symphony No. 1', 'Symphony No. 2', 'Concerto in D'],
        'work_type': ['symphony', 'symphony', 'concerto'],
        'key': ['C major', 'D major', 'D major'],
        'catalog_number': ['Op. 1', 'Op. 2', 'K. 123'],
        'mb_tags': [[], ['classical'], ['baroque', 'concerto']]
    })


@pytest.fixture
def sample_composers_df():
    """Create sample composers dataframe."""
    return pd.DataFrame({
        'composer_id': ['comp1', 'comp2'],
        'name': ['Johann Sebastian Bach', 'Wolfgang Amadeus Mozart'],
        'sort_name': ['Bach, Johann Sebastian', 'Mozart, Wolfgang Amadeus'],
        'period': ['Baroque', 'Classical'],
        'country': ['Germany', 'Austria'],
        'birth_year': [1685, 1756],
        'death_year': [1750, 1791],
        'annotation': [None, None],
        'mb_tags': [[], []],
        'total_works_count': [2, 1]
    })


@pytest.fixture
def sample_tags_df():
    """Create sample tags dataframe."""
    return pd.DataFrame({
        'work_id': ['work1', 'work1', 'work2', 'work3'],
        'tag': ['baroque', 'symphony', 'classical', 'concerto'],
        'source': ['musicbrainz', 'musicbrainz', 'musicbrainz', 'musicbrainz']
    })


@pytest.fixture
def sample_dataset(sample_works_df, sample_composers_df, sample_tags_df):
    """Create sample MusicDataset."""
    quality_report = DataQualityReport(
        total_works=3,
        total_composers=2,
        total_tags=4,
        works_missing_composer=0,
        works_missing_title=0,
        works_missing_work_type=0,
        works_missing_key=0,
        works_with_no_tags=0,
        duplicate_works=0,
        orphaned_tags=0,
        composers_with_no_works=0,
        avg_tags_per_work=1.33,
        median_tags_per_work=1.0,
        unique_tags=4,
        most_common_tags=[('baroque', 1), ('symphony', 1)],
        work_type_counts={'symphony': 2, 'concerto': 1},
        key_counts={'C major': 1, 'D major': 2},
        period_counts={'Baroque': 2, 'Classical': 1}
    )

    return MusicDataset(
        sample_works_df,
        sample_composers_df,
        sample_tags_df,
        quality_report
    )


class TestMusicDataset:
    """Tests for MusicDataset class."""

    def test_init(self, sample_dataset):
        """Test dataset initialization."""
        assert len(sample_dataset.works) == 3
        assert len(sample_dataset.composers) == 2
        assert len(sample_dataset.tags) == 4

    def test_get_work_by_id(self, sample_dataset):
        """Test retrieving work by ID."""
        work = sample_dataset.get_work_by_id('work1')
        assert work is not None
        assert work['title'] == 'Symphony No. 1'

        # Non-existent work
        work = sample_dataset.get_work_by_id('nonexistent')
        assert work is None

    def test_get_work_tags(self, sample_dataset):
        """Test retrieving tags for a work."""
        tags = sample_dataset.get_work_tags('work1')
        assert 'baroque' in tags
        assert 'symphony' in tags

        # Work with no tags
        tags = sample_dataset.get_work_tags('work_no_tags')
        assert tags == []

    def test_get_composer_name(self, sample_dataset):
        """Test retrieving composer name."""
        name = sample_dataset.get_composer_name('comp1')
        assert name == 'Johann Sebastian Bach'

        # Non-existent composer
        name = sample_dataset.get_composer_name('nonexistent')
        assert name == 'Unknown'

        # None composer_id
        name = sample_dataset.get_composer_name(None)
        assert name == 'Unknown'

    def test_search_works(self, sample_dataset):
        """Test searching works by title."""
        results = sample_dataset.search_works('Symphony')
        assert len(results) == 2

        results = sample_dataset.search_works('Concerto')
        assert len(results) == 1

        # Case insensitive
        results = sample_dataset.search_works('symphony')
        assert len(results) == 2

        # No matches
        results = sample_dataset.search_works('Nonexistent')
        assert len(results) == 0

    def test_filter_works(self, sample_dataset):
        """Test filtering works by criteria."""
        # Filter by composer
        results = sample_dataset.filter_works(composer_id='comp1')
        assert len(results) == 2

        # Filter by work type
        results = sample_dataset.filter_works(work_type='symphony')
        assert len(results) == 2

        # Filter by key
        results = sample_dataset.filter_works(key='D major')
        assert len(results) == 2

        # Multiple filters
        results = sample_dataset.filter_works(
            composer_id='comp1',
            work_type='symphony'
        )
        assert len(results) == 2


class TestDataLoader:
    """Tests for DataLoader class."""

    def test_validate_and_clean_duplicates(self, sample_works_df, sample_composers_df, sample_tags_df):
        """Test deduplication of works."""
        # Add a duplicate work
        duplicate = sample_works_df.iloc[0].copy()
        works_with_dup = pd.concat([sample_works_df, duplicate.to_frame().T], ignore_index=True)

        config = Config()
        config.data_loader.deduplicate_works = True

        loader = DataLoader(config)
        works, composers, tags = loader._validate_and_clean(
            works_with_dup,
            sample_composers_df,
            sample_tags_df
        )

        # Should remove the duplicate
        assert len(works) == len(sample_works_df)

    def test_validate_and_clean_orphaned_tags(self, sample_works_df, sample_composers_df, sample_tags_df):
        """Test removal of orphaned tags."""
        # Add orphaned tag
        orphaned = pd.DataFrame({
            'work_id': ['nonexistent_work'],
            'tag': ['orphan_tag'],
            'source': ['test']
        })
        tags_with_orphan = pd.concat([sample_tags_df, orphaned], ignore_index=True)

        loader = DataLoader()
        works, composers, tags = loader._validate_and_clean(
            sample_works_df,
            sample_composers_df,
            tags_with_orphan
        )

        # Should remove orphaned tag
        assert len(tags) == len(sample_tags_df)

    def test_generate_quality_report(self, sample_works_df, sample_composers_df, sample_tags_df):
        """Test quality report generation."""
        loader = DataLoader()
        report = loader._generate_quality_report(
            sample_works_df,
            sample_composers_df,
            sample_tags_df
        )

        assert report.total_works == 3
        assert report.total_composers == 2
        assert report.total_tags == 4
        assert report.unique_tags == 4
        assert isinstance(report.most_common_tags, list)
        assert isinstance(report.work_type_counts, dict)

    def test_quality_report_str(self, sample_dataset):
        """Test quality report string representation."""
        report_str = str(sample_dataset.quality_report)
        assert 'DATA QUALITY REPORT' in report_str
        assert 'Dataset Size' in report_str
        assert 'Missing Data' in report_str
