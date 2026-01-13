"""
Data loading and validation module for classical music recommendation system.

This module handles:
- Loading works, composers, and tags from parquet files
- Data validation and quality checks
- Merging data correctly
- Handling missing values and data quality issues
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd
import numpy as np
from pydantic import BaseModel

from recommender.config import Config, get_config

logger = logging.getLogger(__name__)


@dataclass
class DataQualityReport:
    """Report of data quality metrics and issues.

    This helps identify data problems early and track
    dataset health over time.
    """
    total_works: int
    total_composers: int
    total_tags: int

    # Missing data statistics
    works_missing_composer: int
    works_missing_title: int
    works_missing_work_type: int
    works_missing_key: int
    works_with_no_tags: int

    # Data quality metrics
    duplicate_works: int
    orphaned_tags: int  # Tags pointing to non-existent works
    composers_with_no_works: int
    avg_tags_per_work: float
    median_tags_per_work: float

    # Tag coverage
    unique_tags: int
    most_common_tags: list[tuple[str, int]]

    # Work type distribution
    work_type_counts: dict[str, int]

    # Key distribution
    key_counts: dict[str, int]

    # Period distribution
    period_counts: dict[str, int]

    def __str__(self) -> str:
        """Generate human-readable report."""
        lines = [
            "=" * 60,
            "DATA QUALITY REPORT",
            "=" * 60,
            f"\nDataset Size:",
            f"  Total works: {self.total_works:,}",
            f"  Total composers: {self.total_composers:,}",
            f"  Total tag associations: {self.total_tags:,}",
            f"\nMissing Data:",
            f"  Works missing composer: {self.works_missing_composer} ({self.works_missing_composer/self.total_works*100:.1f}%)",
            f"  Works missing title: {self.works_missing_title}",
            f"  Works missing work_type: {self.works_missing_work_type} ({self.works_missing_work_type/self.total_works*100:.1f}%)",
            f"  Works missing key: {self.works_missing_key} ({self.works_missing_key/self.total_works*100:.1f}%)",
            f"  Works with no tags: {self.works_with_no_tags} ({self.works_with_no_tags/self.total_works*100:.1f}%)",
            f"\nData Quality Issues:",
            f"  Duplicate works: {self.duplicate_works}",
            f"  Orphaned tags: {self.orphaned_tags}",
            f"  Composers with no works: {self.composers_with_no_works}",
            f"\nTag Coverage:",
            f"  Unique tags: {self.unique_tags}",
            f"  Avg tags per work: {self.avg_tags_per_work:.2f}",
            f"  Median tags per work: {self.median_tags_per_work:.1f}",
            f"  Top 10 tags: {', '.join([f'{tag}({count})' for tag, count in self.most_common_tags[:10]])}",
            f"\nWork Type Distribution:",
        ]

        for work_type, count in sorted(self.work_type_counts.items(), key=lambda x: x[1], reverse=True)[:10]:
            lines.append(f"  {work_type or 'Unknown'}: {count}")

        lines.append(f"\nPeriod Distribution:")
        for period, count in sorted(self.period_counts.items(), key=lambda x: x[1], reverse=True):
            lines.append(f"  {period or 'Unknown'}: {count}")

        lines.append("=" * 60)
        return "\n".join(lines)


class MusicDataset:
    """Container for classical music works dataset.

    This class holds the merged dataset and provides
    convenient access to works, composers, and tags.
    """

    def __init__(
        self,
        works: pd.DataFrame,
        composers: pd.DataFrame,
        tags: pd.DataFrame,
        quality_report: DataQualityReport
    ):
        """Initialize dataset.

        Args:
            works: Works dataframe with columns [work_id, composer_id, title, work_type, catalog_number, key, mb_tags]
            composers: Composers dataframe
            tags: Work tags dataframe
            quality_report: Data quality metrics
        """
        self.works = works
        self.composers = composers
        self.tags = tags
        self.quality_report = quality_report

        # Create lookup dictionaries for fast access
        self._work_id_to_idx = {work_id: idx for idx, work_id in enumerate(works['work_id'])}
        self._composer_id_to_name = dict(zip(composers['composer_id'], composers['name']))

        logger.info(f"Dataset initialized with {len(works)} works, {len(composers)} composers")

    def get_work_by_id(self, work_id: str) -> Optional[pd.Series]:
        """Get work by ID.

        Args:
            work_id: Work identifier

        Returns:
            Work data as Series or None if not found
        """
        idx = self._work_id_to_idx.get(work_id)
        if idx is not None:
            return self.works.iloc[idx]
        return None

    def get_work_tags(self, work_id: str) -> list[str]:
        """Get tags for a specific work.

        Args:
            work_id: Work identifier

        Returns:
            List of tag strings
        """
        work_tags = self.tags[self.tags['work_id'] == work_id]['tag'].tolist()
        return work_tags

    def get_composer_name(self, composer_id: Optional[str]) -> str:
        """Get composer name by ID.

        Args:
            composer_id: Composer identifier

        Returns:
            Composer name or 'Unknown' if not found
        """
        if composer_id is None or pd.isna(composer_id):
            return "Unknown"
        return self._composer_id_to_name.get(composer_id, "Unknown")

    def search_works(self, query: str, limit: int = 10) -> pd.DataFrame:
        """Search works by title.

        Args:
            query: Search query (case-insensitive substring match)
            limit: Maximum number of results

        Returns:
            DataFrame of matching works
        """
        query_lower = query.lower()
        mask = self.works['title'].str.lower().str.contains(query_lower, na=False)
        return self.works[mask].head(limit)

    def filter_works(
        self,
        composer_id: Optional[str] = None,
        work_type: Optional[str] = None,
        key: Optional[str] = None,
        period: Optional[str] = None
    ) -> pd.DataFrame:
        """Filter works by various criteria.

        Args:
            composer_id: Filter by composer
            work_type: Filter by work type
            key: Filter by musical key
            period: Filter by period/era

        Returns:
            Filtered DataFrame
        """
        result = self.works.copy()

        if composer_id is not None:
            result = result[result['composer_id'] == composer_id]

        if work_type is not None:
            result = result[result['work_type'] == work_type]

        if key is not None:
            result = result[result['key'] == key]

        if period is not None:
            # Need to join with composers for period
            result = result.merge(self.composers[['composer_id', 'period']], on='composer_id', how='left')
            result = result[result['period'] == period]

        return result


class DataLoader:
    """Loads and validates classical music data from parquet files."""

    def __init__(self, config: Optional[Config] = None):
        """Initialize data loader.

        Args:
            config: Configuration object (uses global config if None)
        """
        self.config = config or get_config()

    def load(self) -> MusicDataset:
        """Load and validate all data files.

        Returns:
            MusicDataset: Validated and merged dataset

        Raises:
            FileNotFoundError: If data files don't exist
            ValueError: If data validation fails
        """
        logger.info("Loading data files...")

        # Load parquet files
        works = self._load_works()
        composers = self._load_composers()
        tags = self._load_tags()

        logger.info(f"Loaded {len(works)} works, {len(composers)} composers, {len(tags)} tag associations")

        # Try to infer missing composer links from work titles
        works = self._infer_composer_links(works, composers)

        # Validate data if configured
        if self.config.data_loader.validate_on_load:
            works, composers, tags = self._validate_and_clean(works, composers, tags)

        # Generate quality report
        quality_report = self._generate_quality_report(works, composers, tags)
        logger.info(f"\n{quality_report}")

        return MusicDataset(works, composers, tags, quality_report)

    def _load_works(self) -> pd.DataFrame:
        """Load works parquet file."""
        path = self.config.paths.works_parquet
        if not path.exists():
            raise FileNotFoundError(f"Works file not found: {path}")

        df = pd.read_parquet(path)

        # Validate required columns
        required_cols = ['work_id', 'title']
        missing_cols = set(required_cols) - set(df.columns)
        if missing_cols:
            raise ValueError(f"Works file missing required columns: {missing_cols}")

        return df

    def _load_composers(self) -> pd.DataFrame:
        """Load composers parquet file."""
        path = self.config.paths.composers_parquet
        if not path.exists():
            raise FileNotFoundError(f"Composers file not found: {path}")

        df = pd.read_parquet(path)

        # Validate required columns
        required_cols = ['composer_id', 'name']
        missing_cols = set(required_cols) - set(df.columns)
        if missing_cols:
            raise ValueError(f"Composers file missing required columns: {missing_cols}")

        return df

    def _load_tags(self) -> pd.DataFrame:
        """Load work tags parquet file."""
        path = self.config.paths.work_tags_parquet
        if not path.exists():
            raise FileNotFoundError(f"Work tags file not found: {path}")

        df = pd.read_parquet(path)

        # Validate required columns
        required_cols = ['work_id', 'tag']
        missing_cols = set(required_cols) - set(df.columns)
        if missing_cols:
            raise ValueError(f"Tags file missing required columns: {missing_cols}")

        return df

    def _infer_composer_links(self, works: pd.DataFrame, composers: pd.DataFrame) -> pd.DataFrame:
        """Infer missing composer links by matching work titles to composer names.

        This is a workaround for ETL data quality issues where composer_id wasn't populated.

        Args:
            works: Works dataframe
            composers: Composers dataframe

        Returns:
            Works dataframe with inferred composer_ids
        """
        works = works.copy()
        missing_composer = works['composer_id'].isna()

        if missing_composer.sum() == 0:
            return works  # All works already have composers

        logger.info(f"Inferring composer links for {missing_composer.sum()} works...")

        # Create mapping of composer last names to composer_ids
        # Handle common name formats
        composer_mapping = {}
        for _, composer in composers.iterrows():
            name = composer['name']
            composer_id = composer['composer_id']

            # Add full name
            composer_mapping[name.lower()] = composer_id

            # Add last name (split on space, take last part)
            parts = name.split()
            if len(parts) > 1:
                last_name = parts[-1].lower()
                if last_name not in composer_mapping:  # Avoid overwriting
                    composer_mapping[last_name] = composer_id

                # Also try "Last, First" format
                if ',' not in name:
                    sort_name = f"{parts[-1]}, {' '.join(parts[:-1])}"
                    composer_mapping[sort_name.lower()] = composer_id

        # Add catalog number patterns for specific composers
        catalog_patterns = {
            'bwv': '24f1766e-9635-4d58-a4d4-9413f9f98a4c',  # Bach
            'rv': 'ad79836d-9849-44df-8789-180bbc823f3c',   # Vivaldi
            'k.': 'b972f589-fb0e-474e-b64a-803b0364fa75',   # Mozart
            'op.': None,  # Generic, skip
            'hob': None,  # Haydn - not in our list
            'd.': 'f91e3a88-24ee-4563-8963-fab73d2765ed',   # Schubert
        }

        # Match works to composers based on title
        matched_count = 0
        for idx in works[missing_composer].index:
            title = works.at[idx, 'title']
            if pd.isna(title):
                continue

            title_lower = title.lower()

            # Try to find composer name in title
            matched = False
            for composer_name, composer_id in composer_mapping.items():
                if composer_name in title_lower:
                    works.at[idx, 'composer_id'] = composer_id
                    matched_count += 1
                    matched = True
                    break

            # If no match, try catalog patterns
            if not matched:
                for pattern, composer_id in catalog_patterns.items():
                    if composer_id and pattern in title_lower:
                        works.at[idx, 'composer_id'] = composer_id
                        matched_count += 1
                        break

        logger.info(f"Successfully inferred {matched_count} composer links from work titles")

        return works

    def _validate_and_clean(
        self,
        works: pd.DataFrame,
        composers: pd.DataFrame,
        tags: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Validate and clean data.

        Args:
            works: Works dataframe
            composers: Composers dataframe
            tags: Tags dataframe

        Returns:
            Cleaned versions of all dataframes
        """
        logger.info("Validating and cleaning data...")

        # Handle duplicate works
        if self.config.data_loader.deduplicate_works:
            original_count = len(works)
            works = works.drop_duplicates(subset=['title', 'composer_id'], keep='first')
            removed = original_count - len(works)
            if removed > 0:
                logger.warning(f"Removed {removed} duplicate works")

        # Handle missing composers
        strategy = self.config.data_loader.handle_missing_composers
        works_missing_composer = works['composer_id'].isna().sum()

        if works_missing_composer > 0:
            if strategy == 'drop':
                logger.warning(f"Dropping {works_missing_composer} works with missing composers")
                works = works[works['composer_id'].notna()]
            elif strategy == 'create_unknown':
                logger.info(f"Creating 'Unknown' composer for {works_missing_composer} works")
                # Create unknown composer
                unknown_composer = pd.DataFrame({
                    'composer_id': ['unknown-composer'],
                    'name': ['Unknown'],
                    'sort_name': ['Unknown'],
                    'period': ['Unknown'],
                    'country': [pd.NA],
                    'birth_year': [pd.NA],
                    'death_year': [pd.NA],
                    'annotation': [pd.NA],
                    'mb_tags': [[]],
                    'total_works_count': [works_missing_composer]
                })
                composers = pd.concat([composers, unknown_composer], ignore_index=True)
                works = works.copy()  # Avoid SettingWithCopyWarning
                works.loc[works['composer_id'].isna(), 'composer_id'] = 'unknown-composer'
            # else: keep_null - do nothing

        # Remove orphaned tags (tags pointing to non-existent works)
        valid_work_ids = set(works['work_id'])
        orphaned_mask = ~tags['work_id'].isin(valid_work_ids)
        orphaned_count = orphaned_mask.sum()
        if orphaned_count > 0:
            logger.warning(f"Removing {orphaned_count} orphaned tags")
            tags = tags[~orphaned_mask]

        # Normalize text fields (make copies to avoid SettingWithCopyWarning)
        works = works.copy()
        works['title'] = works['title'].str.strip()
        if 'work_type' in works.columns:
            works['work_type'] = works['work_type'].str.strip().str.lower()
        if 'key' in works.columns:
            works['key'] = works['key'].str.strip()

        composers = composers.copy()
        composers['name'] = composers['name'].str.strip()
        if 'period' in composers.columns:
            composers['period'] = composers['period'].str.strip()

        tags = tags.copy()
        tags['tag'] = tags['tag'].str.strip().str.lower()

        return works, composers, tags

    def _generate_quality_report(
        self,
        works: pd.DataFrame,
        composers: pd.DataFrame,
        tags: pd.DataFrame
    ) -> DataQualityReport:
        """Generate data quality report.

        Args:
            works: Works dataframe
            composers: Composers dataframe
            tags: Tags dataframe

        Returns:
            DataQualityReport with metrics
        """
        # Count tags per work
        tags_per_work = tags.groupby('work_id').size()

        # Missing data stats
        works_missing_composer = works['composer_id'].isna().sum()
        works_missing_title = works['title'].isna().sum()
        works_missing_work_type = works['work_type'].isna().sum() if 'work_type' in works.columns else len(works)
        works_missing_key = works['key'].isna().sum() if 'key' in works.columns else len(works)

        # Works with no tags
        works_with_tags = set(tags['work_id'])
        works_with_no_tags = len(works) - len(works_with_tags.intersection(set(works['work_id'])))

        # Duplicate works
        duplicate_works = works.duplicated(subset=['title', 'composer_id']).sum()

        # Orphaned tags
        valid_work_ids = set(works['work_id'])
        orphaned_tags = (~tags['work_id'].isin(valid_work_ids)).sum()

        # Composers with no works
        work_composer_ids = set(works['composer_id'].dropna())
        composers_with_no_works = len(composers) - len(work_composer_ids.intersection(set(composers['composer_id'])))

        # Tag statistics
        unique_tags = tags['tag'].nunique()
        tag_counts = tags['tag'].value_counts()
        most_common_tags = list(zip(tag_counts.index[:20], tag_counts.values[:20]))

        avg_tags_per_work = tags_per_work.mean() if len(tags_per_work) > 0 else 0.0
        median_tags_per_work = tags_per_work.median() if len(tags_per_work) > 0 else 0.0

        # Work type distribution
        if 'work_type' in works.columns:
            work_type_counts = works['work_type'].fillna('Unknown').value_counts().to_dict()
        else:
            work_type_counts = {'Unknown': len(works)}

        # Key distribution
        if 'key' in works.columns:
            key_counts = works['key'].fillna('Unknown').value_counts().to_dict()
        else:
            key_counts = {'Unknown': len(works)}

        # Period distribution (from composers)
        works_with_composers = works.merge(
            composers[['composer_id', 'period']],
            on='composer_id',
            how='left'
        )
        if 'period' in works_with_composers.columns:
            period_counts = works_with_composers['period'].fillna('Unknown').value_counts().to_dict()
        else:
            period_counts = {'Unknown': len(works)}

        return DataQualityReport(
            total_works=len(works),
            total_composers=len(composers),
            total_tags=len(tags),
            works_missing_composer=int(works_missing_composer),
            works_missing_title=int(works_missing_title),
            works_missing_work_type=int(works_missing_work_type),
            works_missing_key=int(works_missing_key),
            works_with_no_tags=int(works_with_no_tags),
            duplicate_works=int(duplicate_works),
            orphaned_tags=int(orphaned_tags),
            composers_with_no_works=int(composers_with_no_works),
            avg_tags_per_work=float(avg_tags_per_work),
            median_tags_per_work=float(median_tags_per_work),
            unique_tags=int(unique_tags),
            most_common_tags=most_common_tags,
            work_type_counts=work_type_counts,
            key_counts=key_counts,
            period_counts=period_counts
        )
