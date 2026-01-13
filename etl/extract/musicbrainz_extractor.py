"""MusicBrainz API extractor - builds on testing.py patterns."""

import musicbrainzngs as mbz
from typing import Dict, List, Optional, Any
from datetime import datetime

from .base_extractor import BaseExtractor
from .config import Config
from .rate_limiter import RateLimiter


class MusicBrainzExtractor(BaseExtractor):
    """Extract data from MusicBrainz API."""

    def __init__(self, config: Config, rate_limiter: RateLimiter):
        """
        Initialize MusicBrainz extractor.

        Args:
            config: Configuration object
            rate_limiter: Rate limiter for API calls
        """
        super().__init__(config, rate_limiter)
        self._initialize_client()

    def _initialize_client(self) -> None:
        """Set up MusicBrainz client with user agent (from testing.py pattern)."""
        mbz.set_useragent(
            self.config.MUSICBRAINZ_USER_AGENT,
            self.config.MUSICBRAINZ_VERSION,
            self.config.MUSICBRAINZ_EMAIL
        )
        self.logger.info(f"Initialized MusicBrainz client with user agent: {self.config.MUSICBRAINZ_USER_AGENT}")

    def extract_artist(self, artist_name: str) -> Optional[Dict[str, Any]]:
        """
        Extract comprehensive artist information.

        Based on testing.py lines 12-33.

        Args:
            artist_name: Name of artist to search for

        Returns:
            Dict with artist data including:
            - artist_id, name, sort_name
            - tags (list of genre/style tags)
            - annotation (bio/description)
            - type (person/group), country, life_span
        """
        try:
            # Rate limit before API call
            self.rate_limiter.wait_if_needed()

            # Search for artist (similar to testing.py line 12)
            artist_search_results = mbz.search_artists(artist=artist_name)

            if not artist_search_results.get('artist-list'):
                self.logger.warning(f"No artist found with the name: {artist_name}")
                return None

            # Take top result
            artist_basic = artist_search_results['artist-list'][0]
            artist_id = artist_basic['id']
            artist_name_from_api = artist_basic['name']

            self.logger.info(f"Found artist: {artist_name_from_api} (ID: {artist_id})")

            # Get detailed info (similar to testing.py lines 23-26)
            self.rate_limiter.wait_if_needed()
            artist_details = mbz.get_artist_by_id(
                artist_id,
                includes=['tags', 'annotation', 'ratings']
            )

            # Parse and return structured data
            artist_data = self._parse_artist_data(artist_details['artist'])
            return artist_data

        except mbz.WebServiceError as e:
            self._handle_api_error(e, f"MusicBrainz artist search: {artist_name}")
            return None

        except Exception as e:
            self._handle_api_error(e, f"MusicBrainz artist extraction: {artist_name}")
            return None

    def extract_works(self, artist_id: str, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Extract release groups (works) for an artist.

        Based on testing.py lines 35-63.

        Args:
            artist_id: MusicBrainz artist ID
            limit: Maximum number of release groups to fetch

        Returns:
            List of dicts with:
            - release_group_id, title, type (album/single/etc)
            - first_release_date
            - tags, annotation
            - primary_type, secondary_types
        """
        try:
            # Rate limit before API call
            self.rate_limiter.wait_if_needed()

            # Get artist with release groups (similar to testing.py lines 23-26)
            artist_details = mbz.get_artist_by_id(
                artist_id,
                includes=['release-groups']
            )

            release_groups = artist_details['artist'].get('release-group-list', [])
            self.logger.info(f"Found {len(release_groups)} release groups for artist {artist_id}")

            works_data = []

            # Process each release group (similar to testing.py lines 38-63)
            for rg in release_groups[:limit]:
                rg_id = rg['id']

                try:
                    # Get detailed info for each release group
                    self.rate_limiter.wait_if_needed()
                    rg_details = mbz.get_release_group_by_id(
                        rg_id,
                        includes=['tags', 'annotation']
                    )

                    work_data = self._parse_release_group_data(rg_details['release-group'])
                    works_data.append(work_data)

                except mbz.WebServiceError as e:
                    self.logger.warning(f"Failed to fetch release group {rg_id}: {e}")
                    # Add basic info even if detailed fetch fails
                    work_data = {
                        'source': 'musicbrainz',
                        'release_group_id': rg['id'],
                        'title': rg.get('title'),
                        'primary_type': rg.get('type'),
                        'first_release_date': rg.get('first-release-date'),
                        'tags': [],
                        'annotation': '',
                    }
                    works_data.append(work_data)
                    continue

            return works_data

        except mbz.WebServiceError as e:
            self._handle_api_error(e, f"MusicBrainz works for artist: {artist_id}")
            return []

        except Exception as e:
            self._handle_api_error(e, f"MusicBrainz works extraction: {artist_id}")
            return []

    def _parse_artist_data(self, artist_raw: Dict) -> Dict[str, Any]:
        """
        Transform raw API response to structured format.

        Args:
            artist_raw: Raw artist dict from MusicBrainz API

        Returns:
            Structured artist data dict
        """
        # Extract tags (similar to testing.py line 28)
        tags = [tag['name'] for tag in artist_raw.get('tag-list', [])]

        return {
            'source': 'musicbrainz',
            'artist_id': artist_raw['id'],
            'name': artist_raw['name'],
            'sort_name': artist_raw.get('sort-name'),
            'type': artist_raw.get('type'),
            'country': artist_raw.get('country'),
            'life_span': artist_raw.get('life-span', {}),
            'tags': tags,
            'annotation': artist_raw.get('annotation', ''),
            'disambiguation': artist_raw.get('disambiguation', ''),
        }

    def _parse_release_group_data(self, rg_raw: Dict) -> Dict[str, Any]:
        """
        Transform raw release group to structured format.

        Args:
            rg_raw: Raw release group dict from MusicBrainz API

        Returns:
            Structured work data dict
        """
        # Extract tags (similar to testing.py lines 56-58)
        tags = [tag['name'] for tag in rg_raw.get('tag-list', [])]

        return {
            'source': 'musicbrainz',
            'release_group_id': rg_raw['id'],
            'title': rg_raw['title'],
            'primary_type': rg_raw.get('type'),
            'secondary_types': rg_raw.get('secondary-type-list', []),
            'first_release_date': rg_raw.get('first-release-date'),
            'tags': tags,
            'annotation': rg_raw.get('annotation', ''),
        }

    def extract_complete_artist_profile(self, artist_name: str) -> Optional[Dict[str, Any]]:
        """
        One-stop method to extract artist + all works.

        Args:
            artist_name: Name of artist to extract

        Returns:
            Dict with artist and works data
        """
        self.logger.info(f"Extracting complete MusicBrainz profile for: {artist_name}")

        artist_data = self.extract_artist(artist_name)
        if not artist_data:
            return None

        works_data = self.extract_works(artist_data['artist_id'])

        profile = {
            'artist': artist_data,
            'works': works_data,
            'extraction_timestamp': datetime.now().isoformat(),
        }

        # Optionally save raw data
        safe_name = "".join(c for c in artist_name if c.isalnum() or c in (' ', '-', '_'))
        safe_name = safe_name.replace(' ', '_')
        self.save_raw_data(profile, f"{safe_name}_musicbrainz.json")

        return profile
