"""
Main data transformation pipeline.

Transforms raw extracted data into normalized Parquet tables.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import pandas as pd
import numpy as np

from .parsers import (
    CatalogNumberParser,
    WorkTypeParser,
    KeyParser,
    PeriodClassifier
)
from .data_quality import DataQualityValidator, QualityMetrics

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class MusicDataTransformer:
    """
    Transform raw music data from MusicBrainz into normalized tables.

    Transforms MusicBrainz release groups (albums) and their metadata into a structured
    format with composers, works, and associated tags.
    """

    def __init__(self, input_dir: Path, output_dir: Path):
        """
        Initialize transformer.

        Args:
            input_dir: Directory containing raw data files
            output_dir: Directory for output Parquet files
        """
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Data containers
        self.composers_data: List[Dict[str, Any]] = []
        self.works_data: List[Dict[str, Any]] = []
        self.work_tags_data: List[Dict[str, Any]] = []

        logger.info(f"Initialized transformer: {input_dir} -> {output_dir}")

    def transform_all(self, artist_names: Optional[List[str]] = None) -> QualityMetrics:
        """
        Transform all artist data.

        Args:
            artist_names: List of artist names to process. If None, auto-detect from files.

        Returns:
            Quality metrics
        """
        # Auto-detect artists if not specified
        if artist_names is None:
            artist_names = self._detect_artists()

        if not artist_names:
            logger.warning("No artist files found to process")
            return QualityMetrics()

        logger.info(f"Processing {len(artist_names)} artists: {', '.join(artist_names)}")

        # Process each artist
        for artist_name in artist_names:
            self._process_artist(artist_name)

        # Create DataFrames
        logger.info("Creating normalized tables...")
        composers_df = self._create_composers_table()
        works_df = self._create_works_table()
        work_tags_df = self._create_work_tags_table()

        # Create empty DataFrames for validation compatibility
        recordings_df = pd.DataFrame()
        audio_features_df = pd.DataFrame()

        # Validate data quality
        logger.info("Validating data quality...")
        metrics = DataQualityValidator.validate_all(
            composers_df,
            works_df,
            recordings_df,
            audio_features_df,
            work_tags_df
        )

        # Save to Parquet (only MusicBrainz tables)
        logger.info("Saving Parquet files...")
        self._save_parquet(composers_df, 'composers.parquet')
        self._save_parquet(works_df, 'works.parquet')
        self._save_parquet(work_tags_df, 'work_tags.parquet')

        logger.info("Transformation complete!")
        return metrics

    def _detect_artists(self) -> List[str]:
        """Auto-detect artist names from combined JSON files."""
        artists = []
        for file in self.input_dir.glob("*_combined.json"):
            # Extract artist name from filename (e.g., "Bach_combined.json" -> "Bach")
            artist_name = file.stem.replace('_combined', '')
            artists.append(artist_name)

        return sorted(artists)

    def _process_artist(self, artist_name: str):
        """
        Process a single artist's data.

        Args:
            artist_name: Artist name
        """
        logger.info(f"Processing artist: {artist_name}")

        # Load combined data
        combined_file = self.input_dir / f"{artist_name}_combined.json"
        if not combined_file.exists():
            logger.warning(f"No combined file found for {artist_name}: {combined_file}")
            return

        with open(combined_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Process artist/composer
        self._process_composer(data['artist'])

        # Process works (albums in current data structure)
        # Support both field names: musicbrainz_id (old) and artist_id (new)
        composer_id = data['artist'].get('musicbrainz_id') or data['artist'].get('artist_id')
        composer_name = data['artist'].get('name', artist_name)
        for work in data.get('works', []):
            self._process_work(work, composer_id, composer_name)

    def _process_composer(self, artist_data: Dict[str, Any]):
        """
        Process composer data.

        Args:
            artist_data: Artist data dictionary
        """
        # Support both field names: musicbrainz_id (old) and artist_id (new)
        composer_id = artist_data.get('musicbrainz_id') or artist_data.get('artist_id')
        if not composer_id:
            logger.warning(f"Artist missing MusicBrainz ID: {artist_data.get('name')}")
            return

        # Extract life span
        life_span = artist_data.get('life_span', {})
        birth_date = life_span.get('begin')
        death_date = life_span.get('end')

        birth_year = PeriodClassifier._extract_year(birth_date)
        death_year = PeriodClassifier._extract_year(death_date)

        # Classify period
        period = PeriodClassifier.classify(birth_year, death_year)

        # Extract annotation
        annotation_data = artist_data.get('annotation', '')
        if isinstance(annotation_data, dict):
            annotation = annotation_data.get('text', '')
        else:
            annotation = annotation_data

        # Get tags
        mb_tags = artist_data.get('musicbrainz_tags', [])

        # Convert to list if needed
        if not isinstance(mb_tags, list):
            mb_tags = []

        composer = {
            'composer_id': composer_id,
            'name': artist_data.get('name'),
            'sort_name': artist_data.get('sort_name'),
            'birth_year': birth_year,
            'death_year': death_year,
            'country': artist_data.get('country'),
            'period': period,
            'annotation': annotation,
            'mb_tags': mb_tags,
            'total_works_count': 0,  # Will be updated later
        }

        self.composers_data.append(composer)

    def _process_work(self, work_data: Dict[str, Any], composer_id: str, composer_name: str):
        """
        Process work/album data.

        NOTE: Currently processes albums/release groups as works.
        When track data becomes available, this should be updated to process actual musical works.

        Args:
            work_data: Work data dictionary
            composer_id: Composer MusicBrainz ID
            composer_name: Composer name
        """
        # Support both field names: musicbrainz_id (old) and release_group_id (new)
        work_id = work_data.get('musicbrainz_id') or work_data.get('release_group_id')
        if not work_id:
            logger.debug(f"Work missing MusicBrainz ID: {work_data.get('title')}")
            return

        title = work_data.get('title', '')

        # Parse structured information from title
        catalog_number = CatalogNumberParser.parse(title)
        work_type = WorkTypeParser.parse(title)
        key = KeyParser.parse(title)

        # Extract tags
        tags = work_data.get('tags', [])
        if not isinstance(tags, list):
            tags = []

        # Process tags for work_tags table
        source = work_data.get('source', 'musicbrainz')
        for tag in tags:
            self.work_tags_data.append({
                'work_id': work_id,
                'tag': tag,
                'source': source
            })

        # Create work record (MusicBrainz data only)
        work = {
            'work_id': work_id,
            'composer_id': composer_id,
            'title': title,
            'work_type': work_type,
            'catalog_number': catalog_number,
            'key': key,
            'mb_tags': tags,
        }

        self.works_data.append(work)

    def _create_composers_table(self) -> pd.DataFrame:
        """
        Create composers table.

        Returns:
            Composers DataFrame
        """
        if not self.composers_data:
            return pd.DataFrame()

        df = pd.DataFrame(self.composers_data)

        # Calculate total works count
        works_df = pd.DataFrame(self.works_data)
        if not works_df.empty and 'composer_id' in works_df.columns:
            works_count = works_df.groupby('composer_id').size().to_dict()
            df['total_works_count'] = df['composer_id'].map(works_count).fillna(0).astype(int)

        # Ensure correct types
        df['birth_year'] = pd.to_numeric(df['birth_year'], errors='coerce').astype('Int64')
        df['death_year'] = pd.to_numeric(df['death_year'], errors='coerce').astype('Int64')

        # Order columns
        column_order = [
            'composer_id', 'name', 'sort_name', 'birth_year', 'death_year',
            'country', 'period', 'annotation', 'mb_tags', 'total_works_count'
        ]
        df = df[[col for col in column_order if col in df.columns]]

        logger.info(f"Created composers table: {len(df)} rows")
        return df

    def _create_works_table(self) -> pd.DataFrame:
        """
        Create works table.

        Returns:
            Works DataFrame
        """
        if not self.works_data:
            return pd.DataFrame()

        df = pd.DataFrame(self.works_data)

        # Order columns
        column_order = [
            'work_id', 'composer_id', 'title', 'work_type', 'catalog_number',
            'key', 'mb_tags'
        ]
        df = df[[col for col in column_order if col in df.columns]]

        logger.info(f"Created works table: {len(df)} rows")
        return df

    def _create_work_tags_table(self) -> pd.DataFrame:
        """
        Create work tags table.

        Returns:
            Work tags DataFrame
        """
        if not self.work_tags_data:
            return pd.DataFrame(columns=['work_id', 'tag', 'source'])

        df = pd.DataFrame(self.work_tags_data)

        # Remove duplicates
        df = df.drop_duplicates()

        logger.info(f"Created work_tags table: {len(df)} rows")
        return df

    def _save_parquet(self, df: pd.DataFrame, filename: str):
        """
        Save DataFrame to Parquet file.

        Args:
            df: DataFrame to save
            filename: Output filename
        """
        output_path = self.output_dir / filename

        if df.empty:
            logger.warning(f"Skipping empty table: {filename}")
            # Still save the empty file with schema
            df.to_parquet(output_path, index=False, engine='pyarrow')
            return

        df.to_parquet(output_path, index=False, engine='pyarrow')
        logger.info(f"Saved {filename}: {len(df)} rows, {df.memory_usage(deep=True).sum() / 1024:.2f} KB")
