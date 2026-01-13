#!/usr/bin/env python3
"""
Quick test script to tag 10 works and validate results.

This script tests the auto-tagging system on a small sample.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import logging
import pandas as pd
from rich.console import Console
from rich.table import Table

from tagging.auto_tagger import ClassicalMusicAutoTagger
from tagging.tagging_config import TaggingConfig


logging.basicConfig(level=logging.INFO)
console = Console()


def main():
    """Test tagging on 10 works."""
    console.print("\n[bold cyan]Testing Auto-Tagger on 10 Works[/bold cyan]\n")

    try:
        # Initialize
        config = TaggingConfig()
        tagger = ClassicalMusicAutoTagger(config)

        # Load works
        works_df = pd.read_csv(config.works_file)
        console.print(f"Loaded {len(works_df)} total works")

        # Get 10 works with varying metadata quality
        test_works = works_df.head(10)

        console.print(f"\nTagging {len(test_works)} works...\n")

        # Estimate cost first
        estimate = tagger.estimate_cost(len(test_works))
        console.print("[yellow]Cost Estimate:[/yellow]")
        console.print(f"  Estimated cost: ${estimate['estimated_cost_usd']:.4f}")
        console.print(f"  Estimated time: {estimate['estimated_time_minutes']:.1f} minutes\n")

        # Tag the works
        works_list = test_works.to_dict('records')
        results_df = tagger.tag_batch(works_list, save_progress=False)

        # Display results
        console.print(f"\n[green]✓ Successfully tagged {len(test_works)} works[/green]")
        console.print(f"Generated {len(results_df)} tag associations\n")

        # Show usage stats
        stats = tagger.get_usage_stats()
        console.print("[yellow]API Usage:[/yellow]")
        console.print(f"  Total requests: {stats['total_requests']}")
        console.print(f"  Failed requests: {stats['failed_requests']}")
        console.print(f"  Success rate: {stats['success_rate']:.1%}")
        console.print(f"  Total tokens: {stats['total_tokens']:,}")
        console.print(f"  Actual cost: ${stats['total_cost_usd']:.4f}\n")

        # Load composers for display
        composers_df = None
        try:
            composers_df = pd.read_csv(config.composers_file)
        except:
            pass

        # Display sample results
        console.print("[bold cyan]Sample Results:[/bold cyan]\n")

        for _, work in test_works.head(5).iterrows():
            work_id = work['work_id']
            title = work['title']
            composer_id = work.get('composer_id')
            work_tags = results_df[results_df['work_id'] == work_id]['tag'].tolist()

            # Get composer name
            composer_name = "Unknown"
            if composers_df is not None and composer_id and pd.notna(composer_id):
                composer_row = composers_df[composers_df['composer_id'] == composer_id]
                if not composer_row.empty:
                    composer_name = composer_row.iloc[0]['name']
                    period = composer_row.iloc[0].get('period', '')
                    if period and pd.notna(period):
                        composer_name = f"{composer_name} ({period})"

            console.print(f"[bold]{title}[/bold]")
            console.print(f"  Composer: {composer_name}")
            console.print(f"  Tags ({len(work_tags)}): {', '.join(work_tags)}")
            console.print()

        # Quality metrics
        console.print("[bold cyan]Quality Metrics:[/bold cyan]")
        tags_per_work = results_df.groupby('work_id').size()
        console.print(f"  Average tags per work: {tags_per_work.mean():.1f}")
        console.print(f"  Min tags: {tags_per_work.min()}")
        console.print(f"  Max tags: {tags_per_work.max()}")
        console.print(f"  Unique tags used: {results_df['tag'].nunique()}")

        # Tag distribution
        console.print("\n[bold cyan]Tag Distribution:[/bold cyan]")
        top_tags = results_df['tag'].value_counts().head(10)

        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("Tag", style="cyan")
        table.add_column("Count", style="white", justify="right")

        for tag, count in top_tags.items():
            table.add_row(tag, str(count))

        console.print(table)

        console.print("\n[green]✓ Test completed successfully![/green]\n")

    except Exception as e:
        console.print(f"\n[red]Error: {e}[/red]")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
