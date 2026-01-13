"""Configuration management for ETL extract module."""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


class Config:
    """Configuration for ETL extraction system."""

    # MusicBrainz settings
    MUSICBRAINZ_USER_AGENT = "ClassicalMusicRecommender"
    MUSICBRAINZ_VERSION = "0.1"
    MUSICBRAINZ_EMAIL = os.getenv("MUSICBRAINZ_EMAIL", "email@gmail.com")
    MUSICBRAINZ_RATE_LIMIT = float(os.getenv("MUSICBRAINZ_RATE_LIMIT", "1.0"))  # seconds between requests


    # Data paths
    BASE_DIR = Path(__file__).parent.parent.parent
    RAW_DATA_DIR = BASE_DIR / "data" / "raw"
    PROCESSED_DATA_DIR = BASE_DIR / "data" / "processed"
    SCHEMAS_DIR = BASE_DIR / "data" / "schemas"

    # Output formats
    SAVE_JSON = os.getenv("SAVE_JSON", "true").lower() == "true"
    SAVE_PARQUET = os.getenv("SAVE_PARQUET", "true").lower() == "true"

    @classmethod
    def validate(cls):
        """Validate required configuration is present."""
        errors = []

        # Check MusicBrainz config
        if not cls.MUSICBRAINZ_EMAIL or cls.MUSICBRAINZ_EMAIL == "email@gmail.com":
            errors.append("MUSICBRAINZ_EMAIL is not set in environment variables")

        # Ensure data directories exist
        cls.RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
        cls.PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
        cls.SCHEMAS_DIR.mkdir(parents=True, exist_ok=True)

        if errors:
            error_msg = "Configuration validation failed:\n" + "\n".join(f"  - {err}" for err in errors)
            error_msg += "\n\nPlease set the required environment variables in a .env file or your environment."
            raise ValueError(error_msg)

        return True

    @classmethod
    def print_config(cls):
        """Print current configuration (hiding secrets)."""
        print("Configuration:")
        print(f"  MusicBrainz Email: {cls.MUSICBRAINZ_EMAIL}")
        print(f"  MusicBrainz Rate Limit: {cls.MUSICBRAINZ_RATE_LIMIT}s")
        print(f"  Raw Data Dir: {cls.RAW_DATA_DIR}")
        print(f"  Save JSON: {cls.SAVE_JSON}")
        print(f"  Save Parquet: {cls.SAVE_PARQUET}")
