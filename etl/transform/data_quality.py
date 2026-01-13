"""
Data quality validation and reporting.
"""

import pandas as pd
from typing import Dict, Any, List
from dataclasses import dataclass, field
import logging

logger = logging.getLogger(__name__)


@dataclass
class QualityMetrics:
    """Data quality metrics."""

    # Counts
    total_composers: int = 0
    total_works: int = 0
    total_recordings: int = 0

    # Completeness metrics
    composers_with_birth_year: int = 0
    composers_with_death_year: int = 0
    composers_with_period: int = 0
    composers_with_tags: int = 0

    works_with_catalog_number: int = 0
    works_with_work_type: int = 0
    works_with_key: int = 0
    works_with_tags: int = 0
    works_with_recordings: int = 0

    recordings_with_audio_features: int = 0
    recordings_with_popularity: int = 0

    # Data quality issues
    duplicate_composers: int = 0
    duplicate_works: int = 0
    duplicate_recordings: int = 0
    missing_composer_ids: int = 0
    missing_work_ids: int = 0

    # Errors and warnings
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary."""
        return {
            'counts': {
                'composers': int(self.total_composers),
                'works': int(self.total_works),
                'recordings': int(self.total_recordings),
            },
            'completeness': {
                'composers': {
                    'with_birth_year': int(self.composers_with_birth_year),
                    'with_death_year': int(self.composers_with_death_year),
                    'with_period': int(self.composers_with_period),
                    'with_tags': int(self.composers_with_tags),
                    'birth_year_pct': float(self._percentage(self.composers_with_birth_year, self.total_composers)),
                    'death_year_pct': float(self._percentage(self.composers_with_death_year, self.total_composers)),
                    'period_pct': float(self._percentage(self.composers_with_period, self.total_composers)),
                    'tags_pct': float(self._percentage(self.composers_with_tags, self.total_composers)),
                },
                'works': {
                    'with_catalog_number': int(self.works_with_catalog_number),
                    'with_work_type': int(self.works_with_work_type),
                    'with_key': int(self.works_with_key),
                    'with_tags': int(self.works_with_tags),
                    'with_recordings': int(self.works_with_recordings),
                    'catalog_number_pct': float(self._percentage(self.works_with_catalog_number, self.total_works)),
                    'work_type_pct': float(self._percentage(self.works_with_work_type, self.total_works)),
                    'key_pct': float(self._percentage(self.works_with_key, self.total_works)),
                    'tags_pct': float(self._percentage(self.works_with_tags, self.total_works)),
                    'recordings_pct': float(self._percentage(self.works_with_recordings, self.total_works)),
                },
                'recordings': {
                    'with_audio_features': int(self.recordings_with_audio_features),
                    'with_popularity': int(self.recordings_with_popularity),
                    'audio_features_pct': float(self._percentage(self.recordings_with_audio_features, self.total_recordings)),
                    'popularity_pct': float(self._percentage(self.recordings_with_popularity, self.total_recordings)),
                },
            },
            'data_quality': {
                'duplicate_composers': int(self.duplicate_composers),
                'duplicate_works': int(self.duplicate_works),
                'duplicate_recordings': int(self.duplicate_recordings),
                'missing_composer_ids': int(self.missing_composer_ids),
                'missing_work_ids': int(self.missing_work_ids),
            },
            'issues': {
                'errors': self.errors,
                'warnings': self.warnings,
            }
        }

    @staticmethod
    def _percentage(count: int, total: int) -> float:
        """Calculate percentage."""
        if total == 0:
            return 0.0
        return round(100 * count / total, 2)

    def print_report(self):
        """Print formatted quality report."""
        print("\n" + "=" * 80)
        print("DATA QUALITY REPORT")
        print("=" * 80)

        print("\nCOUNTS:")
        print(f"  Composers: {self.total_composers}")
        print(f"  Works: {self.total_works}")
        print(f"  Recordings: {self.total_recordings}")

        print("\nCOMPOSER COMPLETENESS:")
        print(f"  Birth year: {self.composers_with_birth_year}/{self.total_composers} "
              f"({self._percentage(self.composers_with_birth_year, self.total_composers)}%)")
        print(f"  Death year: {self.composers_with_death_year}/{self.total_composers} "
              f"({self._percentage(self.composers_with_death_year, self.total_composers)}%)")
        print(f"  Period: {self.composers_with_period}/{self.total_composers} "
              f"({self._percentage(self.composers_with_period, self.total_composers)}%)")
        print(f"  Tags: {self.composers_with_tags}/{self.total_composers} "
              f"({self._percentage(self.composers_with_tags, self.total_composers)}%)")

        print("\nWORK COMPLETENESS:")
        print(f"  Catalog number: {self.works_with_catalog_number}/{self.total_works} "
              f"({self._percentage(self.works_with_catalog_number, self.total_works)}%)")
        print(f"  Work type: {self.works_with_work_type}/{self.total_works} "
              f"({self._percentage(self.works_with_work_type, self.total_works)}%)")
        print(f"  Key: {self.works_with_key}/{self.total_works} "
              f"({self._percentage(self.works_with_key, self.total_works)}%)")
        print(f"  Tags: {self.works_with_tags}/{self.total_works} "
              f"({self._percentage(self.works_with_tags, self.total_works)}%)")
        print(f"  Recordings: {self.works_with_recordings}/{self.total_works} "
              f"({self._percentage(self.works_with_recordings, self.total_works)}%)")

        if self.total_recordings > 0:
            print("\nRECORDING COMPLETENESS:")
            print(f"  Audio features: {self.recordings_with_audio_features}/{self.total_recordings} "
                  f"({self._percentage(self.recordings_with_audio_features, self.total_recordings)}%)")
            print(f"  Popularity: {self.recordings_with_popularity}/{self.total_recordings} "
                  f"({self._percentage(self.recordings_with_popularity, self.total_recordings)}%)")

        print("\nDATA QUALITY ISSUES:")
        print(f"  Duplicate composers: {self.duplicate_composers}")
        print(f"  Duplicate works: {self.duplicate_works}")
        print(f"  Duplicate recordings: {self.duplicate_recordings}")
        print(f"  Missing composer IDs: {self.missing_composer_ids}")
        print(f"  Missing work IDs: {self.missing_work_ids}")

        if self.warnings:
            print(f"\nWARNINGS ({len(self.warnings)}):")
            for warning in self.warnings[:10]:
                print(f"  - {warning}")
            if len(self.warnings) > 10:
                print(f"  ... and {len(self.warnings) - 10} more warnings")

        if self.errors:
            print(f"\nERRORS ({len(self.errors)}):")
            for error in self.errors[:10]:
                print(f"  - {error}")
            if len(self.errors) > 10:
                print(f"  ... and {len(self.errors) - 10} more errors")

        print("\n" + "=" * 80)


class DataQualityValidator:
    """Validate data quality and generate reports."""

    @staticmethod
    def validate_composers(df: pd.DataFrame) -> QualityMetrics:
        """
        Validate composers table.

        Args:
            df: Composers DataFrame

        Returns:
            Quality metrics
        """
        metrics = QualityMetrics()
        metrics.total_composers = len(df)

        # Check for required fields
        if 'composer_id' not in df.columns:
            metrics.errors.append("Missing required column: composer_id")
            return metrics

        # Check duplicates
        duplicates = df['composer_id'].duplicated()
        metrics.duplicate_composers = duplicates.sum()
        if metrics.duplicate_composers > 0:
            metrics.warnings.append(f"Found {metrics.duplicate_composers} duplicate composer IDs")

        # Check null IDs
        null_ids = df['composer_id'].isnull()
        metrics.missing_composer_ids = null_ids.sum()
        if metrics.missing_composer_ids > 0:
            metrics.errors.append(f"Found {metrics.missing_composer_ids} composers with null IDs")

        # Completeness checks
        if 'birth_year' in df.columns:
            metrics.composers_with_birth_year = df['birth_year'].notna().sum()

        if 'death_year' in df.columns:
            metrics.composers_with_death_year = df['death_year'].notna().sum()

        if 'period' in df.columns:
            metrics.composers_with_period = df['period'].notna().sum()

        if 'mb_tags' in df.columns:
            metrics.composers_with_tags = df['mb_tags'].apply(
                lambda x: len(x) > 0 if isinstance(x, list) else False
            ).sum()

        return metrics

    @staticmethod
    def validate_works(df: pd.DataFrame) -> QualityMetrics:
        """
        Validate works table.

        Args:
            df: Works DataFrame

        Returns:
            Quality metrics
        """
        metrics = QualityMetrics()
        metrics.total_works = len(df)

        # Check for required fields
        if 'work_id' not in df.columns:
            metrics.errors.append("Missing required column: work_id")
            return metrics

        # Check duplicates
        duplicates = df['work_id'].duplicated()
        metrics.duplicate_works = duplicates.sum()
        if metrics.duplicate_works > 0:
            metrics.warnings.append(f"Found {metrics.duplicate_works} duplicate work IDs")

        # Check null IDs
        null_ids = df['work_id'].isnull()
        metrics.missing_work_ids = null_ids.sum()
        if metrics.missing_work_ids > 0:
            metrics.errors.append(f"Found {metrics.missing_work_ids} works with null IDs")

        # Completeness checks
        if 'catalog_number' in df.columns:
            metrics.works_with_catalog_number = df['catalog_number'].notna().sum()

        if 'work_type' in df.columns:
            metrics.works_with_work_type = df['work_type'].notna().sum()

        if 'key' in df.columns:
            metrics.works_with_key = df['key'].notna().sum()

        if 'mb_tags' in df.columns:
            metrics.works_with_tags = df['mb_tags'].apply(
                lambda x: len(x) > 0 if isinstance(x, list) else False
            ).sum()

        if 'recording_count' in df.columns:
            metrics.works_with_recordings = (df['recording_count'] > 0).sum()

        return metrics

    @staticmethod
    def validate_recordings(df: pd.DataFrame) -> QualityMetrics:
        """
        Validate recordings table.

        Args:
            df: Recordings DataFrame

        Returns:
            Quality metrics
        """
        metrics = QualityMetrics()
        metrics.total_recordings = len(df)

        if len(df) == 0:
            return metrics

        # Check duplicates
        if 'recording_id' in df.columns:
            duplicates = df['recording_id'].duplicated()
            metrics.duplicate_recordings = duplicates.sum()
            if metrics.duplicate_recordings > 0:
                metrics.warnings.append(f"Found {metrics.duplicate_recordings} duplicate recording IDs")

        # Completeness checks
        if 'popularity' in df.columns:
            metrics.recordings_with_popularity = df['popularity'].notna().sum()

        return metrics

    @staticmethod
    def validate_all(
        composers_df: pd.DataFrame,
        works_df: pd.DataFrame,
        recordings_df: pd.DataFrame,
        audio_features_df: pd.DataFrame,
        work_tags_df: pd.DataFrame
    ) -> QualityMetrics:
        """
        Validate all tables and combine metrics.

        Args:
            composers_df: Composers DataFrame
            works_df: Works DataFrame
            recordings_df: Recordings DataFrame
            audio_features_df: Audio features DataFrame
            work_tags_df: Work tags DataFrame

        Returns:
            Combined quality metrics
        """
        metrics = QualityMetrics()

        # Validate composers
        composer_metrics = DataQualityValidator.validate_composers(composers_df)
        metrics.total_composers = composer_metrics.total_composers
        metrics.composers_with_birth_year = composer_metrics.composers_with_birth_year
        metrics.composers_with_death_year = composer_metrics.composers_with_death_year
        metrics.composers_with_period = composer_metrics.composers_with_period
        metrics.composers_with_tags = composer_metrics.composers_with_tags
        metrics.duplicate_composers = composer_metrics.duplicate_composers
        metrics.missing_composer_ids = composer_metrics.missing_composer_ids
        metrics.errors.extend(composer_metrics.errors)
        metrics.warnings.extend(composer_metrics.warnings)

        # Validate works
        work_metrics = DataQualityValidator.validate_works(works_df)
        metrics.total_works = work_metrics.total_works
        metrics.works_with_catalog_number = work_metrics.works_with_catalog_number
        metrics.works_with_work_type = work_metrics.works_with_work_type
        metrics.works_with_key = work_metrics.works_with_key
        metrics.works_with_tags = work_metrics.works_with_tags
        metrics.works_with_recordings = work_metrics.works_with_recordings
        metrics.duplicate_works = work_metrics.duplicate_works
        metrics.missing_work_ids = work_metrics.missing_work_ids
        metrics.errors.extend(work_metrics.errors)
        metrics.warnings.extend(work_metrics.warnings)

        # Validate recordings
        recording_metrics = DataQualityValidator.validate_recordings(recordings_df)
        metrics.total_recordings = recording_metrics.total_recordings
        metrics.recordings_with_popularity = recording_metrics.recordings_with_popularity
        metrics.duplicate_recordings = recording_metrics.duplicate_recordings
        metrics.errors.extend(recording_metrics.errors)
        metrics.warnings.extend(recording_metrics.warnings)

        # Check audio features
        if len(audio_features_df) > 0:
            metrics.recordings_with_audio_features = len(audio_features_df)

        return metrics
