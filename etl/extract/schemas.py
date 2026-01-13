"""Data schemas and validation using Pydantic."""

from pydantic import BaseModel, Field, field_validator
from typing import List, Optional, Dict, Any
from datetime import datetime
from pathlib import Path
import json


class ArtistData(BaseModel):
    """Schema for artist data."""

    name: str
    musicbrainz_id: Optional[str] = None
    sort_name: Optional[str] = None
    type: Optional[str] = None
    country: Optional[str] = None
    life_span: Optional[Dict[str, str]] = None
    musicbrainz_tags: List[str] = Field(default_factory=list)
    annotation: Optional[str] = None
    disambiguation: Optional[str] = None


class WorkData(BaseModel):
    """Schema for work/album data."""

    title: str
    musicbrainz_id: Optional[str] = None
    type: Optional[str] = None
    release_date: Optional[str] = None
    tags: List[str] = Field(default_factory=list)
    annotation: Optional[str] = None
    source: Optional[str] = None
    primary_type: Optional[str] = None
    secondary_types: List[str] = Field(default_factory=list)


class Metadata(BaseModel):
    """Schema for extraction metadata."""

    extraction_timestamp: str
    sources: List[str] = Field(default_factory=list)


class CombinedArtistProfile(BaseModel):
    """Complete artist profile schema."""

    artist: ArtistData
    works: List[WorkData] = Field(default_factory=list)
    metadata: Metadata

    @field_validator('works')
    @classmethod
    def validate_works(cls, v):
        """Ensure works is a list."""
        if v is None:
            return []
        return v


def validate_combined_data(data: Dict[str, Any]) -> CombinedArtistProfile:
    """
    Validate combined data against schema.

    Args:
        data: Dict to validate

    Returns:
        Validated CombinedArtistProfile instance

    Raises:
        ValidationError if data doesn't match schema
    """
    return CombinedArtistProfile(**data)


def export_schema_to_file(output_path: Path) -> None:
    """
    Export JSON schema to file.

    Args:
        output_path: Path where schema should be saved
    """
    schema = CombinedArtistProfile.model_json_schema()

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(schema, f, indent=2)
