"""
ETL Transform Module

This module transforms raw extracted data into normalized Parquet tables
optimized for the classical music recommendation system.
"""

from .transformer import MusicDataTransformer
from .parsers import (
    CatalogNumberParser,
    WorkTypeParser,
    KeyParser,
    PeriodClassifier
)
from .data_quality import DataQualityValidator

__all__ = [
    'MusicDataTransformer',
    'CatalogNumberParser',
    'WorkTypeParser',
    'KeyParser',
    'PeriodClassifier',
    'DataQualityValidator'
]
