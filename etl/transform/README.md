# ETL Transform Module

This module transforms raw extracted classical music data into normalized Parquet tables optimized for building a recommendation system.

## Overview

The transformation pipeline processes data extracted from MusicBrainz and Spotify APIs, cleaning, normalizing, and structuring it into five main tables:

1. **composers.parquet** - Composer metadata
2. **works.parquet** - Musical works/compositions
3. **recordings.parquet** - Individual recordings/tracks
4. **audio_features.parquet** - Spotify audio features
5. **work_tags.parquet** - Tags associated with works

## Quick Start

### Basic Usage

```bash
cd etl/transform
python3 run_transform.py
```

This will:
- Auto-detect artists from files in `data/raw/`
- Transform all data
- Output Parquet files to `data/processed/`
- Display a quality report

### Advanced Usage

```bash
# Process specific artists
python3 run_transform.py --artists Bach Dvorak

# Custom input/output directories
python3 run_transform.py \
  --input-dir /path/to/raw/data \
  --output-dir /path/to/processed/data

# Save quality report to JSON
python3 run_transform.py --report data/quality_report.json
```

## Output Schema

### 1. composers.parquet

| Column | Type | Description |
|--------|------|-------------|
| composer_id | string | MusicBrainz composer ID (primary key) |
| name | string | Composer's full name |
| sort_name | string | Name in sortable format (e.g., "Bach, Johann Sebastian") |
| birth_year | int | Year of birth |
| death_year | int | Year of death |
| country | string | Country code (ISO 3166-1 alpha-2) |
| period | string | Musical period (Baroque, Classical, Romantic, etc.) |
| annotation | string | Biographical text from MusicBrainz |
| mb_tags | list[string] | Tags from MusicBrainz |
| total_works_count | int | Number of works in database |
| total_recordings_count | int | Number of recordings (currently 0) |
| avg_popularity | float | Average Spotify popularity (currently null) |

**Period Classification:**
- Medieval: before 1400
- Renaissance: 1400-1600
- Baroque: 1600-1750
- Classical: 1750-1820
- Romantic: 1820-1910
- Modern: 1910-2000
- Contemporary: 2000+

### 2. works.parquet

| Column | Type | Description |
|--------|------|-------------|
| work_id | string | MusicBrainz work ID (primary key) |
| composer_id | string | Foreign key to composers table |
| title | string | Work title |
| work_type | string | Type of work (Symphony, Concerto, Sonata, etc.) |
| catalog_number | string | Catalog number (BWV, Op., K., etc.) |
| key | string | Musical key (e.g., "D minor", "E♭ major") |
| mb_tags | list[string] | Tags from MusicBrainz |
| recording_count | int | Number of recordings (currently 0) |
| avg_duration_ms | float | Average duration across recordings |
| avg_tempo | float | Average tempo |
| avg_energy | float | Average energy (0-1) |
| avg_valence | float | Average valence/mood (0-1) |
| avg_acousticness | float | Average acousticness (0-1) |

**Note:** Currently, the works table contains albums/release groups from MusicBrainz since individual track data is not yet available. The schema is designed to support individual musical works once track-level extraction is implemented.

### 3. recordings.parquet

| Column | Type | Description |
|--------|------|-------------|
| recording_id | string | Spotify track ID (primary key) |
| work_id | string | Foreign key to works table (may be null) |
| composer_id | string | Foreign key to composers table |
| track_name | string | Track/recording name |
| artists | list[string] | Performer names |
| album_name | string | Album name |
| duration_ms | int | Duration in milliseconds |
| popularity | int | Spotify popularity score (0-100) |
| release_date | string | Release date (ISO format) |
| spotify_uri | string | Spotify URI |

**Note:** Currently empty. Will be populated when track-level data is extracted.

### 4. audio_features.parquet

| Column | Type | Description |
|--------|------|-------------|
| recording_id | string | Spotify track ID (foreign key) |
| tempo | float | Tempo in BPM |
| key | int | Musical key (0-11, Pitch Class notation) |
| mode | int | Modality (0 = minor, 1 = major) |
| time_signature | int | Time signature |
| acousticness | float | Acousticness (0-1) |
| danceability | float | Danceability (0-1) |
| energy | float | Energy (0-1) |
| instrumentalness | float | Instrumentalness (0-1) |
| liveness | float | Liveness (0-1) |
| loudness | float | Loudness in dB |
| speechiness | float | Speechiness (0-1) |
| valence | float | Valence/positivity (0-1) |

**Note:** Currently empty. Will be populated when track-level data is extracted.

### 5. work_tags.parquet

| Column | Type | Description |
|--------|------|-------------|
| work_id | string | Foreign key to works table |
| tag | string | Tag name |
| source | string | Tag source ('musicbrainz' or 'spotify') |

## Data Quality Report

The transformation process generates a comprehensive quality report showing:

- **Counts**: Number of composers, works, and recordings processed
- **Completeness Metrics**: Percentage of records with each field populated
- **Data Quality Issues**: Duplicates, missing IDs, etc.
- **Errors and Warnings**: Any issues encountered during transformation

Example output:
```
COUNTS:
  Composers: 2
  Works: 150
  Recordings: 0

COMPOSER COMPLETENESS:
  Birth year: 2/2 (100.0%)
  Death year: 2/2 (100.0%)
  Period: 2/2 (100.0%)
  Tags: 2/2 (100.0%)

WORK COMPLETENESS:
  Catalog number: 18/150 (12.0%)
  Work type: 29/150 (19.33%)
  Key: 11/150 (7.33%)
  Tags: 21/150 (14.0%)
```

## Parsing Features

### Catalog Number Extraction

Automatically extracts catalog numbers from work titles:

- **BWV** (Bach-Werke-Verzeichnis): `BWV 1048`
- **Op.** (Opus): `Op. 95`, `Opus 59 No. 3`
- **K.** (Köchel, Mozart): `K. 525`, `KV 331`
- **Hob.** (Hoboken, Haydn): `Hob. III:77`
- **D.** (Deutsch, Schubert): `D. 956`
- **RV** (Ryom, Vivaldi): `RV 315`
- **B.** (Dvořák catalog): `B. 178`

### Work Type Detection

Identifies work types from titles:

- **Orchestral**: Symphony, Concerto, Overture, Suite, Tone Poem
- **Chamber**: String Quartet, Piano Trio, String Quintet
- **Keyboard**: Sonata, Prelude, Fugue, Etude, Nocturne, Waltz
- **Vocal**: Mass, Requiem, Cantata, Oratorio, Opera, Motet
- And many more...

### Musical Key Extraction

Parses musical keys from titles:
- Recognizes: "D minor", "E♭ major", "F# minor", etc.
- Normalizes accidentals (♯, #, ♭, b)
- Handles various notations (major/maj/dur, minor/min/moll)

## Architecture

### Module Structure

```
etl/transform/
├── __init__.py           # Module exports
├── transformer.py        # Main transformation logic
├── parsers.py           # Text parsing utilities
├── data_quality.py      # Validation and quality metrics
├── run_transform.py     # CLI script
└── README.md           # This file
```

### Key Classes

- **MusicDataTransformer**: Main transformation pipeline
- **CatalogNumberParser**: Extract catalog numbers from titles
- **WorkTypeParser**: Identify work types
- **KeyParser**: Extract musical keys
- **PeriodClassifier**: Classify composers by historical period
- **DataQualityValidator**: Validate data and generate reports

## Current Limitations

1. **Album-Level Data**: The current implementation works with album/release group data from MusicBrainz rather than individual musical works (symphonies, concertos, etc.)

2. **No Track Data**: The `recordings` and `audio_features` tables are currently empty placeholders. Track-level extraction needs to be implemented in the ETL extract phase.

3. **Fuzzy Matching Not Implemented**: Linking Spotify tracks to MusicBrainz works will require fuzzy matching logic (planned for when track data is available).

## Future Enhancements

### When Track Data Becomes Available

1. **Update Extract Phase**:
   - Fetch individual tracks from Spotify albums
   - Retrieve audio features for each track
   - Link tracks to MusicBrainz works

2. **Transform Phase Extensions**:
   - Implement fuzzy matching to link Spotify tracks to MusicBrainz works
   - Aggregate audio features at the work level
   - Calculate popularity metrics
   - Populate recordings and audio_features tables

3. **Enhanced Parsing**:
   - Better movement/part extraction from titles
   - Instrumentation detection
   - Performer extraction

### Schema Extensions

Potential additions:
- **composers**: Image URLs, Wikipedia links, related composers
- **works**: Movement information, instrumentation, duration ranges
- **recordings**: Performer details, recording quality metadata
- **performer_credits**: Separate table for performer information

## Dependencies

```python
pandas>=1.5.0
pyarrow>=10.0.0
```

## Integration

### Loading Data in Python

```python
import pandas as pd

# Load composers
composers = pd.read_parquet('data/processed/composers.parquet')

# Load works
works = pd.read_parquet('data/processed/works.parquet')

# Join tables
data = works.merge(composers, on='composer_id', how='left')

# Filter by period
baroque_works = data[data['period'] == 'Baroque']
```

### Using with Recommendation System

The normalized schema supports various recommendation approaches:

1. **Collaborative Filtering**: Use recording popularity and user preferences
2. **Content-Based**: Use audio features, work types, keys, periods
3. **Hybrid**: Combine both approaches

Example features for ML:
- Composer period
- Work type
- Musical key
- Audio features (tempo, energy, valence, acousticness)
- Tags
- Popularity scores

## Troubleshooting

### Issue: No artists detected

**Solution**: Ensure `*_combined.json` files exist in input directory

### Issue: JSON serialization errors

**Solution**: All numeric types are now properly converted in the quality report

### Issue: Duplicate work IDs

**Cause**: Multiple albums may have the same MusicBrainz release group ID
**Impact**: Only the last occurrence is kept in the works table
**Solution**: Accept this limitation for now (albums aren't true "works" anyway)

## Contributing

When extending this module:

1. Add new parsers to `parsers.py`
2. Update schemas in docstrings and README
3. Add validation logic to `data_quality.py`
4. Update tests
5. Document changes in this README

## License

Part of the Classical Music Recommender project.
