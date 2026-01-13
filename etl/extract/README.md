# ETL Extract Module

Extract artist, works, and genre data from MusicBrainz and Spotify APIs for the Classical Music Recommender system.

## Features

- **MusicBrainz Integration**: Extract classical music metadata, works, and tags
- **Spotify Integration**: Extract genres, popularity, audio features
- **Genre Enrichment**: Combine tags from both sources for better categorization
- **Dual Output**: JSON (for inspection) + Parquet (for analytics)
- **Batch Processing**: Process multiple artists with progress tracking
- **Rate Limiting**: Automatic API rate limiting to prevent throttling
- **Resumable**: Continue batch processing from where you left off

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure API Credentials

Copy the example environment file and add your credentials:

```bash
cp .env.example .env
```

Edit `.env` and add your credentials:

```bash
# Your email for MusicBrainz
MUSICBRAINZ_EMAIL=your.email@example.com

# Get from https://developer.spotify.com/dashboard
SPOTIFY_CLIENT_ID=your_client_id_here
SPOTIFY_CLIENT_SECRET=your_client_secret_here
```

## Usage

### Quick Start

Process a single artist:

```python
from etl.extract import BatchProcessor, Config

config = Config()
config.validate()

processor = BatchProcessor(config)
processor.process_single_artist("Bach")
```

### Batch Processing

Process multiple artists:

```python
from etl.extract import BatchProcessor, Config

artists = ["Bach", "Mozart", "Beethoven", "Chopin"]

config = Config()
processor = BatchProcessor(config)
stats = processor.process_artists(artists, resume=True)

print(f"Processed: {stats['successful']}/{stats['total']}")
```

### From File

Process artists from a text file (one per line):

```python
from pathlib import Path
from etl.extract import BatchProcessor, Config

config = Config()
processor = BatchProcessor(config)
stats = processor.process_from_file(Path("artists.txt"))
```

### Individual Extractors

Use extractors directly for more control:

```python
from etl.extract import MusicBrainzExtractor, SpotifyExtractor, Config, RateLimiter

config = Config()

# Create extractors
mb_limiter = RateLimiter(config.MUSICBRAINZ_RATE_LIMIT)
mb_extractor = MusicBrainzExtractor(config, mb_limiter)

# Extract from MusicBrainz only
data = mb_extractor.extract_complete_artist_profile("Vivaldi")
```

## Output Format

### JSON Structure

```json
{
  "artist": {
    "name": "Johann Sebastian Bach",
    "musicbrainz_id": "...",
    "spotify_id": "...",
    "genres": ["baroque", "classical", "german baroque"],
    "popularity": 85,
    "life_span": {"begin": "1685", "end": "1750"}
  },
  "works": [
    {
      "title": "Brandenburg Concertos",
      "musicbrainz_id": "...",
      "spotify_id": "...",
      "audio_features": {
        "avg_tempo": 120.5,
        "avg_energy": 0.65
      }
    }
  ]
}
```

### Parquet Tables

- `{artist}_artist.parquet`: Flattened artist metadata
- `{artist}_works.parquet`: Flattened works with artist foreign keys

## File Structure

```
etl/extract/
├── __init__.py              # Package interface
├── config.py                # Configuration management
├── rate_limiter.py          # Rate limiting utilities
├── base_extractor.py        # Abstract base class
├── musicbrainz_extractor.py # MusicBrainz API extraction
├── spotify_extractor.py     # Spotify API extraction
├── combiner.py              # Data combination layer
├── batch_processor.py       # Batch processing orchestration
├── schemas.py               # Data schemas & validation
├── example_usage.py         # Usage examples
└── README.md                # This file
```

## Progress Tracking

The batch processor saves progress to `data/raw/batch_progress.json`:

```python
# Check progress
progress = processor.get_progress()
print(f"Completed: {progress['total_completed']} artists")

# Clear progress (start fresh)
processor.clear_progress()
```

## Error Handling

- Individual artist failures don't stop batch processing
- Errors are logged and included in statistics
- API rate limiting is automatic
- Progress is saved after each artist for resumption

## Advanced Configuration

Override defaults in `.env`:

```bash
# Rate limiting (seconds between requests)
MUSICBRAINZ_RATE_LIMIT=1.5
SPOTIFY_RATE_LIMIT=0.2

# Output formats
SAVE_JSON=true
SAVE_PARQUET=false
```

## Examples

See [example_usage.py](example_usage.py) for complete examples:

- Process single artist
- Batch processing
- Individual extractors
- Progress checking

Run examples:

```bash
python -m etl.extract.example_usage
```

## Data Sources

### MusicBrainz
- Artist metadata (name, country, life span)
- Release groups (works/compositions)
- Tags (genre/style)
- Annotations (descriptions)

### Spotify
- Genres
- Popularity metrics
- Audio features (tempo, energy, key, etc.)
- Album metadata
- Images

## Output Location

All data is saved to `data/raw/`:

- `{artist}_combined.json` - Combined data from both sources
- `{artist}_artist.parquet` - Artist metadata table
- `{artist}_works.parquet` - Works/albums table
- `{artist}_musicbrainz.json` - Raw MusicBrainz response
- `{artist}_spotify.json` - Raw Spotify response
- `batch_progress.json` - Progress tracking

## Reference

Built on patterns from [testing.py](testing.py), which demonstrates the MusicBrainz API integration approach.
