# Classical Music Auto-Tagging System

Production-grade auto-tagging for classical music works using LLM inference with human review.

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Set API Key
Add to `.env`:
```bash
ANTHROPIC_API_KEY=sk-ant-...
```

### 3. Test on 10 Works
```bash
python tagging/test_on_10_works.py
```

### 4. Enhance MusicBrainz-Tagged Works (NEW!)
```bash
# Preview what will be enhanced
python tagging/test_enhance_mb.py

# Enhance works that already have MusicBrainz tags
python -m tagging.manage_tags enhance-mb --max-works 10

# Then enhance all
python -m tagging.manage_tags enhance-mb
```

### 5. Tag All Works
```bash
# Estimate cost
python -m tagging.manage_tags estimate

# Tag with review
python -m tagging.manage_tags tag --sample-review 20

# Analyze quality
python -m tagging.manage_tags analyze
```

## Features

✅ **7-category taxonomy**: mood, character, tempo, instrumentation, form, complexity, popularity
✅ **LLM-powered**: Uses Claude Sonnet 4 for intelligent tagging
✅ **Human review**: Interactive CLI for corrections
✅ **Learning**: Tracks patterns in corrections
✅ **Quality metrics**: Coverage, diversity, consistency
✅ **Cost controls**: Budget limits and estimates
✅ **Checkpoints**: Resume after interruptions

## Commands

| Command | Description |
|---------|-------------|
| `estimate` | Estimate API cost before tagging |
| `tag` | Auto-tag works with optional review |
| `enhance-mb` | Add auto-tagger tags to works that already have MusicBrainz tags |
| `review` | Review and correct auto-tagged works |
| `analyze` | Generate quality report |
| `export` | Export tags to file |
| `retag` | Retag specific works |
| `corrections` | Manage tag corrections |

## Documentation

See [AUTO_TAGGING_GUIDE.md](../docs/AUTO_TAGGING_GUIDE.md) for complete documentation.

## Architecture

```
tagging/
├── tagging_config.py    # Configuration
├── auto_tagger.py       # Core tagging logic
├── tag_reviewer.py      # Interactive review
├── tag_learner.py       # Learning system
├── tag_quality.py       # Quality metrics
├── manage_tags.py       # CLI tool
└── tests/               # Unit tests
```

## Tag Taxonomy

**67 tags across 7 categories:**

- **Mood** (15): melancholic, joyful, dramatic, peaceful, energetic, ...
- **Character** (10): lyrical, virtuosic, bold, delicate, powerful, ...
- **Tempo** (6): very-slow, slow, moderate, fast, very-fast, varied-tempo
- **Instrumentation** (10): solo-piano, full-orchestra, string-quartet, ...
- **Form** (8): sonata-form, fugue, rondo, multi-movement, ...
- **Complexity** (4): beginner-friendly, intermediate, advanced, virtuosic
- **Popularity** (5): famous, well-known, repertoire-staple, lesser-known, obscure

## Cost

**~$0.0045 per work** with Claude Sonnet 4

- 100 works: ~$0.45
- 500 works: ~$2.25
- 1,000 works: ~$4.50

## Examples

### Tag and Review
```bash
python -m tagging.manage_tags tag \
  --max-works 100 \
  --sample-review 15 \
  --auto-approve-threshold 0.85
```

### Enhance MusicBrainz-Tagged Works
```bash
# Add auto-tagger tags to works that already have MusicBrainz tags
python -m tagging.manage_tags enhance-mb

# Limit to 50 works
python -m tagging.manage_tags enhance-mb --max-works 50

# Skip confirmation prompt
python -m tagging.manage_tags enhance-mb --skip-review
```

### Analyze Corrections
```bash
python -m tagging.manage_tags corrections analyze
```

### Export Results
```bash
python -m tagging.manage_tags export work_tags_final.csv
```

## Testing

### Run Unit Tests
```bash
pytest tagging/tests/
```

### Run Integration Test
```bash
python tagging/test_on_10_works.py
```

## Workflow

1. **Estimate** → Check cost
2. **Tag** → Auto-tag with LLM
3. **Review** → Human corrections
4. **Learn** → Analyze patterns
5. **Analyze** → Quality report
6. **Export** → Save results

## Success Metrics

- ✅ Coverage: 100% of works
- ✅ Quality: 80%+ approval rate
- ✅ Efficiency: <$0.50 for 100 works
- ✅ Speed: <10 minutes for 100 works
- ✅ Consistency: Similar works get similar tags

## Troubleshooting

**API errors?** Check your `.env` file has valid `ANTHROPIC_API_KEY`

**JSON parse errors?** System retries 3x automatically

**Cost too high?** Adjust `max_cost_usd` in config or tag in smaller batches

**Low approval?** Review correction patterns and adjust prompts

## License

Part of Classical Music Recommender project.
