# Classical Music Recommendation System

A production-grade content-based recommendation system for classical music works. This system uses advanced feature engineering and similarity computation to provide high-quality, diverse recommendations based on composer, period, work type, musical key, and tags.

## Table of Contents

- [Features](#features)
- [Architecture](#architecture)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage Examples](#usage-examples)
- [Configuration](#configuration)
- [API Reference](#api-reference)
- [Performance](#performance)
- [Testing](#testing)
- [Future Enhancements](#future-enhancements)

## Features

### Core Capabilities

- **Content-Based Recommendations**: Find similar works based on musical characteristics
- **Multi-Feature Engineering**: Combines composer, period, work type, key, tags, and catalog patterns
- **Diversity Control**: Avoid recommending too many similar works using MMR (Maximal Marginal Relevance)
- **Flexible Querying**: Search by title, filter by attributes, or get recommendations by ID
- **Smart Key Encoding**: Uses circle of fifths for musically-aware key similarity
- **Explanation Generation**: Provides human-readable reasons for recommendations
- **Production Ready**: Caching, metrics, health checks, async support

### Feature Engineering

The system extracts and weights the following features:

| Feature Group | Weight | Description |
|---------------|--------|-------------|
| Composer | 5.0 | Highest weight - composer similarity is most important |
| Period | 3.0 | Era/period (Baroque, Classical, Romantic, etc.) |
| Tags | 2.5 | TF-IDF of genres, instrumentation, moods |
| Work Type | 2.0 | Symphony, concerto, sonata, etc. |
| Key | 1.0 | Musical key with circle of fifths encoding |
| Catalog Pattern | 0.5 | Op., BWV, K., etc. |
| Composite | Variable | Combined features like "baroque_concerto" |
| Country | Variable | Composer's country of origin |

## Architecture

```
recommender/
├── config.py           # Configuration management with Pydantic
├── data_loader.py      # Data loading, validation, quality checks
├── features.py         # Feature extraction and engineering
├── recommender.py      # Core recommendation engine
├── evaluation.py       # Quality metrics and test cases
├── service.py          # Production API layer with monitoring
├── __init__.py         # Package exports
└── tests/              # Comprehensive unit tests
    ├── test_config.py
    ├── test_data_loader.py
    ├── test_features.py
    └── ...
```

### Data Flow

```
Parquet Files (works, composers, tags)
    ↓
DataLoader (validation + cleaning)
    ↓
MusicDataset (merged data)
    ↓
FeatureExtractor (engineering + encoding)
    ↓
FeatureMatrix (weighted vectors)
    ↓
MusicRecommender (similarity computation)
    ↓
Recommendations (ranked + explained)
```

## Installation

### Prerequisites

- Python 3.11+
- Virtual environment (recommended)

### Install Dependencies

```bash
# Activate your virtual environment
source venv/bin/activate  # On macOS/Linux
# or
venv\Scripts\activate  # On Windows

# Install requirements
pip install -r requirements.txt
```

### Verify Installation

```bash
# Run tests
pytest recommender/tests/ -v

# Check code coverage
pytest recommender/tests/ --cov=recommender --cov-report=html
```

## Quick Start

### Basic Usage

```python
from recommender.service import initialize_service, get_service

# Initialize the system (load data + build features)
# This happens once at application startup
initialize_service()

# Get the service instance
service = get_service()

# Get recommendations for a work by ID
recommendations = service.recommend_similar(
    work_id="some-work-id",
    n=10
)

# Print recommendations
for rec in recommendations:
    print(f"{rec['rank']}. {rec['title']} by {rec['composer']}")
    print(f"   Score: {rec['similarity_score']:.3f}")
    print(f"   {rec['explanation']}")
    print()
```

### Using the Core Engine Directly

```python
from recommender import MusicRecommender

# Create and load recommender
recommender = MusicRecommender()
recommender.load()

# Get similar works
recommendations = recommender.recommend_similar(
    work_id="work-123",
    n=10
)

# Get diverse recommendations
diverse_recs = recommender.recommend_diverse(
    work_id="work-123",
    n=10,
    diversity_weight=0.4  # Higher = more diverse
)
```

## Usage Examples

### Example 1: Search and Recommend

```python
service = get_service()

# Search for a work by title
recommendations = service.recommend_by_query(
    query="Brandenburg Concerto",
    n=5
)
```

### Example 2: Filter by Criteria

```python
# Browse baroque concertos
recommendations = service.recommend_by_filters(
    period="Baroque",
    work_type="concerto",
    n=10
)

# Browse Mozart symphonies
recommendations = service.recommend_by_filters(
    composer="Mozart",
    work_type="symphony",
    n=10
)
```

### Example 3: Diverse Recommendations

```python
# Avoid recommending 10 Bach works when seed is Bach
recommendations = service.recommend_similar(
    work_id="bach-work-123",
    n=10,
    diverse=True,
    diversity_weight=0.3  # Balance similarity and diversity
)
```

### Example 4: Get Work Information

```python
work_info = service.get_work_info("work-123")
print(f"Title: {work_info['title']}")
print(f"Composer: {work_info['composer']}")
print(f"Tags: {', '.join(work_info['tags'])}")
```

### Example 5: Health Checks and Monitoring

```python
# Check system health
health = service.health_check()
print(f"Status: {health['status']}")
print(f"Dataset size: {health['dataset_size']} works")

# Get metrics
metrics = service.get_metrics()
print(f"Total requests: {metrics['total_requests']}")
print(f"Avg latency: {metrics['avg_latency_ms']:.2f}ms")
print(f"Cache hit rate: {metrics['cache_hit_rate']:.1%}")
```

## Configuration

### Environment-Based Configuration

Configure the system via the `Config` class or environment variables:

```python
from recommender.config import Config, FeatureWeights

# Create custom configuration
config = Config(
    feature_weights=FeatureWeights(
        composer=6.0,  # Increase composer importance
        period=4.0,
        work_type=2.0
    ),
    random_seed=123
)

# Use custom config
from recommender.service import RecommenderService
service = RecommenderService(config)
service.initialize()
```

### Key Configuration Parameters

```python
# Feature weights (relative importance)
feature_weights:
    composer: 5.0          # Higher = more important
    period: 3.0
    work_type: 2.0
    key: 1.0
    tags: 2.5

# Recommendation settings
recommender:
    similarity_metric: "cosine"      # cosine, euclidean, manhattan
    diversity_weight: 0.3             # 0=pure similarity, 1=pure diversity
    max_same_composer_ratio: 0.5     # Max % from same composer
    enable_cache: true               # Cache features and similarities

# Feature engineering
feature_engineering:
    tfidf_max_features: 100          # Max TF-IDF features for tags
    use_circle_of_fifths: true       # Musically-aware key encoding
    create_composite_features: true  # Generate combined features

# Data handling
data_loader:
    validate_on_load: true
    handle_missing_composers: "create_unknown"  # drop, create_unknown, keep_null
    deduplicate_works: true
```

## API Reference

### RecommenderService

The main production API interface.

#### `initialize(force_rebuild: bool = False)`

Load data and initialize the system. Call once at startup.

#### `recommend_similar(work_id: str, n: int = 10, diverse: bool = False) -> List[Dict]`

Get recommendations for a specific work.

**Parameters:**
- `work_id`: Work identifier
- `n`: Number of recommendations
- `diverse`: Use diversity-aware ranking
- `diversity_weight`: Weight for diversity (only if `diverse=True`)

**Returns:** List of recommendation dictionaries with keys:
- `work_id`, `title`, `composer`, `work_type`, `key`, `period`
- `similarity_score`: Float similarity score
- `rank`: Position in ranking (1-indexed)
- `explanation`: Human-readable explanation
- `tags`: List of tags

#### `recommend_by_query(query: str, n: int = 10) -> List[Dict]`

Search for works and recommend similar ones.

#### `recommend_by_filters(composer, period, work_type, key, n) -> List[Dict]`

Browse recommendations by filtering criteria.

#### `get_work_info(work_id: str) -> Dict`

Get detailed metadata for a work.

#### `health_check() -> Dict`

Check system health and status.

#### `get_metrics() -> Dict`

Get performance metrics (requests, latency, cache hits, errors).

### MusicRecommender

Core recommendation engine (lower-level API).

#### `load(force_rebuild: bool = False)`

Load dataset and build features.

#### `recommend_similar(work_id: str, n: int = 10) -> List[Recommendation]`

Get similar works (basic similarity ranking).

#### `recommend_diverse(work_id: str, n: int = 10, diversity_weight: float = 0.3) -> List[Recommendation]`

Get diverse recommendations using MMR algorithm.

#### `recommend_by_query(query: str, n: int = 10) -> List[Recommendation]`

Search and recommend.

#### `recommend_by_filters(...) -> List[Recommendation]`

Filter-based browsing.

## Performance

### Benchmarks

On a dataset of ~500 works (current size):

| Operation | Latency | Notes |
|-----------|---------|-------|
| Initial load (cold) | ~2-5s | First-time feature building |
| Initial load (cached) | ~0.5s | Loading from cache |
| Recommendation query | <50ms | Pre-computed similarity matrix |
| Search query | <100ms | Includes search + recommendation |

### Scaling Considerations

**Current implementation** (in-memory):
- Handles up to ~10,000 works efficiently
- Memory usage: ~100MB for 10k works with full similarity matrix

**For larger datasets (>10,000 works)**:
1. Enable approximate nearest neighbors (FAISS)
2. Disable full similarity matrix pre-computation
3. Use on-demand similarity computation with caching

```python
config = Config()
config.recommender.use_approximate_neighbors = True
config.recommender.faiss_index_type = "IndexIVFFlat"
```

### Caching Strategy

- **Feature vectors**: Cached after first build (~MB)
- **Similarity matrix**: Cached for fast lookups (~N²)
- **Recommendations**: Not cached (computed on-demand)

Cache location: `data/cache/recommender/`

To rebuild cache:
```python
service.initialize(force_rebuild=True)
```

## Testing

### Run All Tests

```bash
pytest recommender/tests/ -v
```

### Run Specific Test Modules

```bash
pytest recommender/tests/test_config.py -v
pytest recommender/tests/test_features.py -v
```

### Test Coverage

```bash
pytest recommender/tests/ --cov=recommender --cov-report=html
open htmlcov/index.html  # View coverage report
```

### Manual Evaluation

```python
from recommender import MusicRecommender, RecommenderEvaluator

# Load recommender
recommender = MusicRecommender()
recommender.load()

# Create evaluator
evaluator = RecommenderEvaluator(recommender)

# Generate full evaluation report
report = evaluator.generate_full_report(n_samples=50, n_recommendations=10)
print(report)
```

**Evaluation metrics:**
- **Diversity**: How varied are recommendations?
- **Coverage**: What % of catalog gets recommended?
- **Similarity distribution**: Are scores well-distributed?
- **Test cases**: Predefined scenarios with expected results

## Future Enhancements

### Planned Features

1. **Collaborative Filtering**
   - Integrate user listening history
   - Hybrid content + collaborative recommendations

2. **Popularity Signals**
   - Recording count
   - Streaming statistics
   - Trending works

3. **Recording-Level Recommendations**
   - Recommend specific recordings, not just works
   - Consider performer, conductor, orchestra

4. **Playlist Generation**
   - Create cohesive multi-work playlists
   - Consider flow, mood progression, duration

5. **Multi-Lingual Support**
   - Work titles in multiple languages
   - Composer names in original languages

6. **Advanced Diversity**
   - Temporal diversity (different eras)
   - Instrumental diversity (vary instrumentation)
   - Mood/emotion diversity

7. **User Personalization**
   - Learn user preferences over time
   - Personalized feature weights
   - "More like this" vs "Explore new" modes

8. **Performance Optimizations**
   - FAISS integration for large-scale
   - GPU acceleration for similarity computation
   - Distributed caching (Redis)

### Design Decisions Documented

**Q: How to handle works with identical features?**
- **Decision**: Use random tie-breaking with fixed seed for reproducibility
- **Location**: Similarity computation uses stable sort

**Q: Should "same composer" be a hard boost or soft signal?**
- **Decision**: Soft signal via weighted features, with diversity controls
- **Rationale**: Allows flexibility - users can get "more Bach" or "explore beyond Bach"

**Q: Pre-compute all recommendations or compute on-demand?**
- **Decision**: Pre-compute similarity matrix, compute rankings on-demand
- **Rationale**: Balances memory usage with query latency

**Q: Handle multi-movement works?**
- **Current**: Treat each cataloged work as atomic unit
- **Future**: Could add movement-level granularity

## Contributing

This is a modular, extensible system. To add new features:

1. **New feature type**: Extend `FeatureExtractor._extract_XXX_features()`
2. **New similarity metric**: Add to `MusicRecommender._compute_similarity()`
3. **New recommendation mode**: Add method to `MusicRecommender`
4. **New evaluation metric**: Extend `RecommenderEvaluator`

All contributions should include:
- Type hints
- Docstrings with examples
- Unit tests (aim for >80% coverage)
- Update this README

## License

[Add your license here]

## Authors

[Add author information]
