"""Batch processing orchestration for multiple artists."""

import json
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path
from datetime import datetime

try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False

from .config import Config
from .rate_limiter import RateLimiter
from .musicbrainz_extractor import MusicBrainzExtractor


class BatchProcessor:
    """Process multiple artists through the ETL pipeline."""

    def __init__(self, config: Config):
        """
        Initialize batch processor.

        Args:
            config: Configuration object
        """
        self.config = config
        self.logger = self._setup_logger()

        # Initialize rate limiter
        mb_rate_limiter = RateLimiter(config.MUSICBRAINZ_RATE_LIMIT)

        # Initialize extractor
        self.mb_extractor = MusicBrainzExtractor(config, mb_rate_limiter)

        # Progress tracking
        self.progress_file = config.RAW_DATA_DIR / "batch_progress.json"

    def _setup_logger(self) -> logging.Logger:
        """Configure logging."""
        logger = logging.getLogger('BatchProcessor')

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

    def process_artists(
        self,
        artist_names: List[str],
        resume: bool = True
    ) -> Dict[str, Any]:
        """
        Process a list of artists.

        Args:
            artist_names: List of artist names to process
            resume: If True, skip already processed artists

        Returns:
            Dict with summary statistics
        """
        # Load progress if resuming
        completed = set()
        if resume and self.progress_file.exists():
            try:
                with open(self.progress_file, 'r') as f:
                    progress = json.load(f)
                    completed = set(progress.get('completed', []))
                self.logger.info(f"Loaded progress: {len(completed)} artists already completed")
            except Exception as e:
                self.logger.warning(f"Failed to load progress file: {e}")

        # Filter out completed artists
        to_process = [name for name in artist_names if name not in completed]

        self.logger.info(
            f"Processing {len(to_process)} artists "
            f"(skipping {len(completed)} already completed)"
        )

        stats = {
            'total': len(artist_names),
            'already_completed': len(completed),
            'successful': 0,
            'failed': 0,
            'errors': []
        }

        # Process with progress bar
        iterator = tqdm(to_process, desc="Extracting artists") if TQDM_AVAILABLE else to_process

        for artist_name in iterator:
            try:
                self.process_single_artist(artist_name)
                stats['successful'] += 1
                completed.add(artist_name)

                # Save progress periodically
                self._save_progress(completed)

            except Exception as e:
                self.logger.error(f"Failed to process {artist_name}: {e}")
                stats['failed'] += 1
                stats['errors'].append({
                    'artist': artist_name,
                    'error': str(e)
                })

        stats['completed'] = len(completed)

        # Save final progress
        self._save_progress(completed)

        # Print summary
        self.logger.info("=" * 60)
        self.logger.info("BATCH PROCESSING COMPLETE")
        self.logger.info(f"Total artists: {stats['total']}")
        self.logger.info(f"Successful: {stats['successful']}")
        self.logger.info(f"Failed: {stats['failed']}")
        self.logger.info(f"Total completed (including previous): {stats['completed']}")
        self.logger.info("=" * 60)

        return stats

    def process_single_artist(self, artist_name: str) -> None:
        """
        Process a single artist through the extraction pipeline.

        Pipeline:
        1. Extract from MusicBrainz
        2. Save outputs

        Args:
            artist_name: Name of artist to process
        """
        self.logger.info(f"Processing: {artist_name}")

        # Extract from MusicBrainz
        self.logger.info(f"  → Extracting from MusicBrainz...")
        mb_data = self.mb_extractor.extract_complete_artist_profile(artist_name)

        if mb_data:
            # MusicBrainz extractor already saves the data
            self.logger.info(f"✓ Successfully processed: {artist_name}")
        else:
            self.logger.warning(f"✗ No data found for: {artist_name}")
            raise ValueError(f"No data found for artist: {artist_name}")

    def _save_progress(self, completed: set) -> None:
        """
        Save progress to disk.

        Args:
            completed: Set of completed artist names
        """
        progress = {
            'completed': sorted(list(completed)),
            'last_update': datetime.now().isoformat(),
            'total_completed': len(completed)
        }

        try:
            with open(self.progress_file, 'w') as f:
                json.dump(progress, f, indent=2)
        except Exception as e:
            self.logger.error(f"Failed to save progress: {e}")

    def process_from_file(self, filepath: Path) -> Dict[str, Any]:
        """
        Process artists from a file (one artist name per line).

        Args:
            filepath: Path to file containing artist names

        Returns:
            Dict with summary statistics
        """
        self.logger.info(f"Reading artists from file: {filepath}")

        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                artist_names = [line.strip() for line in f if line.strip()]

            self.logger.info(f"Found {len(artist_names)} artists in file")
            return self.process_artists(artist_names)

        except Exception as e:
            self.logger.error(f"Failed to read file {filepath}: {e}")
            raise

    def clear_progress(self) -> None:
        """Clear saved progress (start fresh)."""
        if self.progress_file.exists():
            self.progress_file.unlink()
            self.logger.info("Progress file cleared")
        else:
            self.logger.info("No progress file to clear")

    def get_progress(self) -> Dict[str, Any]:
        """
        Get current progress.

        Returns:
            Dict with progress information
        """
        if not self.progress_file.exists():
            return {
                'completed': [],
                'total_completed': 0,
                'last_update': None
            }

        try:
            with open(self.progress_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            self.logger.error(f"Failed to read progress: {e}")
            return {
                'completed': [],
                'total_completed': 0,
                'last_update': None,
                'error': str(e)
            }
