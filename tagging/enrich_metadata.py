"""
CLI script to enrich missing work_type and key metadata using LLM inference.

Usage:
    python -m tagging.enrich_metadata           # Dry-run: show what needs enrichment
    python -m tagging.enrich_metadata --run     # Run enrichment
    python -m tagging.enrich_metadata --resume  # Resume an interrupted run
"""

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def _load_data(project_root: Path):
    works_df = pd.read_parquet(project_root / "data" / "processed" / "works.parquet")
    composers_df = pd.read_parquet(project_root / "data" / "processed" / "composers.parquet")
    tags_path = project_root / "data" / "processed" / "work_tags_enhanced.parquet"
    tags_df = pd.read_parquet(tags_path) if tags_path.exists() else None
    return works_df, composers_df, tags_df


def _print_summary(works_df: pd.DataFrame, label: str = "Current state") -> None:
    total = len(works_df)
    missing_type = works_df["work_type"].apply(lambda v: not isinstance(v, str)).sum()
    missing_key = works_df["key"].apply(lambda v: not isinstance(v, str)).sum()
    needs_either = (
        works_df["work_type"].apply(lambda v: not isinstance(v, str)) |
        works_df["key"].apply(lambda v: not isinstance(v, str))
    ).sum()

    print(f"\n{label} ({total} works):")
    print(f"  Missing work_type : {missing_type:3d}  ({missing_type / total * 100:.1f}%)")
    print(f"  Missing key       : {missing_key:3d}  ({missing_key / total * 100:.1f}%)")
    print(f"  Works to enrich   : {needs_either:3d}")

    # ~400 input + ~50 output tokens per work (metadata only, not full tagging prompt)
    n = needs_either
    cost = (n * 400 / 1_000_000) * 3.0 + (n * 50 / 1_000_000) * 15.0
    minutes = (n * 1.0) / 60
    print(f"  Estimated cost    : ${cost:.3f} USD  (~{minutes:.1f} min at 1 req/s)")


def _clear_recommender_cache(project_root: Path) -> None:
    cache_dir = project_root / "data" / "cache" / "recommender"
    if not cache_dir.exists():
        return
    deleted = sum(1 for f in cache_dir.iterdir() if f.is_file() and f.unlink() is None)
    if deleted:
        print(f"\nCleared {deleted} stale cache file(s) — recommender will rebuild on next request.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Enrich missing work_type and key metadata using Claude"
    )
    parser.add_argument("--run", action="store_true", help="Execute enrichment (default is dry-run)")
    parser.add_argument("--resume", action="store_true", help="Resume an interrupted run")
    args = parser.parse_args()

    project_root = Path(__file__).parent.parent

    works_df, composers_df, tags_df = _load_data(project_root)
    _print_summary(works_df)

    if not args.run and not args.resume:
        print("\nDry-run only. Pass --run to execute, or --resume to continue an interrupted run.")
        return

    # Import here so dry-run doesn't require ANTHROPIC_API_KEY
    from tagging.metadata_enricher import WorkMetadataEnricher
    from tagging.tagging_config import TaggingConfig

    config = TaggingConfig()
    enricher = WorkMetadataEnricher(config)

    if args.resume:
        print("\nResuming from checkpoint…")
    else:
        enricher.clear_checkpoint()
        print("\nStarting enrichment…")

    enriched_df = enricher.enrich_batch(works_df, composers_df, tags_df)

    # Save both parquet and CSV to keep them in sync
    out_parquet = project_root / "data" / "processed" / "works.parquet"
    out_csv = project_root / "data" / "processed" / "works.csv"
    enriched_df.to_parquet(out_parquet, index=False)
    enriched_df.to_csv(out_csv, index=False)
    print(f"\nSaved enriched data → {out_parquet.name}, {out_csv.name}")

    _print_summary(enriched_df, label="After enrichment")

    stats = enricher.get_usage_stats()
    total_tokens = stats["total_input_tokens"] + stats["total_output_tokens"]
    print(
        f"\nAPI usage: {stats['total_requests']} requests, "
        f"{total_tokens:,} tokens, "
        f"~${stats['estimated_cost_usd']} USD"
    )
    if stats["failed_requests"]:
        print(f"  {stats['failed_requests']} request(s) failed — re-run with --resume to retry.")

    _clear_recommender_cache(project_root)
    print("\nDone. Restart the backend server (or wait for cache TTL) to use enriched data.")


if __name__ == "__main__":
    main()
