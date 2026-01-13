"""Abstract base class for API extractors."""

import json
import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any
from pathlib import Path
from datetime import datetime

from .config import Config
from .rate_limiter import RateLimiter


class BaseExtractor(ABC):
    """Abstract base class for all extractors."""

    def __init__(self, config: Config, rate_limiter: RateLimiter):
        """
        Initialize extractor.

        Args:
            config: Configuration object
            rate_limiter: Rate limiter for API calls
        """
        self.config = config
        self.rate_limiter = rate_limiter
        self.logger = self._setup_logger()

    def _setup_logger(self) -> logging.Logger:
        """Configure logging with consistent format."""
        logger = logging.getLogger(self.__class__.__name__)

        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)

        return logger

    @abstractmethod
    def extract_artist(self, artist_identifier: str) -> Optional[Dict[str, Any]]:
        """
        Extract artist information.

        Args:
            artist_identifier: Artist name or ID

        Returns:
            Dict with artist data, or None if not found
        """
        pass

    @abstractmethod
    def extract_works(self, artist_id: str) -> List[Dict[str, Any]]:
        """
        Extract works/albums for an artist.

        Args:
            artist_id: Artist ID in the source system

        Returns:
            List of dicts with work data
        """
        pass

    def _handle_api_error(self, error: Exception, context: str) -> None:
        """
        Standardized error handling and logging.

        Args:
            error: The exception that occurred
            context: Context string describing what was being attempted
        """
        error_type = type(error).__name__
        self.logger.error(f"API error during {context}: {error_type} - {str(error)}")

    def save_raw_data(self, data: Dict[str, Any], filename: str) -> None:
        """
        Save raw API response to disk.

        Args:
            data: Data to save
            filename: Name of file (will be saved in raw data directory)
        """
        filepath = self.config.RAW_DATA_DIR / filename

        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            self.logger.debug(f"Saved raw data to {filepath}")

        except Exception as e:
            self.logger.error(f"Failed to save raw data to {filepath}: {e}")

    def extract_complete_artist_profile(self, artist_name: str) -> Optional[Dict[str, Any]]:
        """
        One-stop method to extract artist + all works.

        Args:
            artist_name: Name of artist to extract

        Returns:
            Dict with artist and works data, or None if artist not found
        """
        artist_data = self.extract_artist(artist_name)
        if not artist_data:
            return None

        # Get artist ID from the returned data
        # Subclasses should ensure 'artist_id' is in the returned dict
        artist_id = artist_data.get('artist_id')
        if not artist_id:
            self.logger.error(f"No artist_id in extracted data for {artist_name}")
            return None

        works_data = self.extract_works(artist_id)

        return {
            'artist': artist_data,
            'works': works_data,
            'extraction_timestamp': datetime.now().isoformat(),
        }
