"""
ETL Extract Package for Classical Music Recommender

This package provides extraction capabilities for classical music data from
the MusicBrainz API.

Main exports:
- BatchProcessor: Process multiple artists through the ETL pipeline
- MusicBrainzExtractor: Extract from MusicBrainz API
- Config: Configuration management

Usage:
    from etl.extract import BatchProcessor, Config

    config = Config()
    config.validate()

    processor = BatchProcessor(config)
    processor.process_single_artist("Bach")
"""

from .config import Config
from .batch_processor import BatchProcessor
from .musicbrainz_extractor import MusicBrainzExtractor
from .schemas import (
    CombinedArtistProfile,
    ArtistData,
    WorkData,
    Metadata,
    validate_combined_data,
    export_schema_to_file
)

__all__ = [
    # Main classes
    'Config',
    'BatchProcessor',
    'MusicBrainzExtractor',

    # Schemas
    'CombinedArtistProfile',
    'ArtistData',
    'WorkData',
    'Metadata',

    # Utilities
    'validate_combined_data',
    'export_schema_to_file',
]

__version__ = '0.1.0'
