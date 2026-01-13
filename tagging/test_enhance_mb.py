#!/usr/bin/env python3
"""
Test script to verify the enhance-mb functionality.

This script shows what works will be enhanced without actually running the tagger.
"""

import sys
from pathlib import Path

import pandas as pd
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

# Add project root to path
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from tagging.tagging_config import TaggingConfig
from tagging.auto_tagger import ClassicalMusicAutoTagger


def main():
    console = Console()

    console.print(Panel.fit(
        "[bold cyan]MusicBrainz Tag Enhancement - Preview[/bold cyan]",
        border_style="cyan"
    ))

    # Load config and data
    config = TaggingConfig()
    works_df = pd.read_csv(config.works_file)
    tags_df = pd.read_csv(config.existing_tags_file)

    # Find works to enhance
    mb_tagged_work_ids = set(
        tags_df[tags_df['source'] == 'musicbrainz']['work_id'].unique()
    )

    auto_tagged_work_ids = set(
        tags_df[tags_df['source'] == 'auto-tagger']['work_id'].unique()
    )

    works_to_enhance_ids = mb_tagged_work_ids - auto_tagged_work_ids
    works_to_enhance = works_df[works_df['work_id'].isin(works_to_enhance_ids)]

    # Display summary
    console.print(f"\n[bold yellow]Summary:[/bold yellow]")
    console.print(f"  Total works in database: {len(works_df)}")
    console.print(f"  Works with MusicBrainz tags: {len(mb_tagged_work_ids)}")
    console.print(f"  Works with auto-tagger tags: {len(auto_tagged_work_ids)}")
    console.print(f"  Works needing enhancement: {len(works_to_enhance_ids)}")

    # Calculate current tag stats for MB-tagged works
    mb_tags_df = tags_df[tags_df['work_id'].isin(mb_tagged_work_ids)]
    avg_tags_current = len(mb_tags_df) / len(mb_tagged_work_ids)
    console.print(f"\n[bold yellow]Current Tagging:[/bold yellow]")
    console.print(f"  Average tags per MB work: {avg_tags_current:.2f}")
    console.print(f"  Target tags per work: 5-10 (MusicBrainz + auto-tagger)")

    # Estimate cost
    tagger = ClassicalMusicAutoTagger(config)
    estimate = tagger.estimate_cost(len(works_to_enhance_ids))

    console.print(f"\n[bold yellow]Enhancement Estimate:[/bold yellow]")
    console.print(f"  Works to enhance: {len(works_to_enhance_ids)}")
    console.print(f"  Estimated cost: ${estimate['estimated_cost_usd']:.3f}")
    console.print(f"  Estimated time: {estimate['estimated_time_minutes']:.1f} minutes")
    console.print(f"  Within budget: {'✓ Yes' if estimate['within_budget'] else '✗ No'}")

    # Show sample works
    console.print(f"\n[bold cyan]Sample Works to Enhance (first 10):[/bold cyan]")

    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Title", style="cyan", width=40)
    table.add_column("Type", style="green", width=15)
    table.add_column("Current MB Tags", style="yellow", width=30)

    for _, work in works_to_enhance.head(10).iterrows():
        work_id = work['work_id']
        title = work['title'][:37] + "..." if len(work['title']) > 40 else work['title']
        work_type = work.get('work_type', '')[:12] + "..." if len(work.get('work_type', '')) > 15 else work.get('work_type', '')

        # Get current tags
        current_tags = tags_df[tags_df['work_id'] == work_id]['tag'].tolist()
        tags_str = ', '.join(current_tags) if current_tags else 'none'
        if len(tags_str) > 30:
            tags_str = tags_str[:27] + "..."

        table.add_row(title, work_type, tags_str)

    console.print(table)

    # Show what happens next
    console.print(f"\n[bold green]To enhance these works:[/bold green]")
    console.print("  1. Test on 10 works first:")
    console.print("     [cyan]python -m tagging.manage_tags enhance-mb --max-works 10[/cyan]")
    console.print("\n  2. Then enhance all:")
    console.print("     [cyan]python -m tagging.manage_tags enhance-mb[/cyan]")

    console.print(f"\n[bold yellow]Expected Outcome:[/bold yellow]")
    console.print(f"  • Each work will get 5-10 additional auto-tagger tags")
    console.print(f"  • MusicBrainz tags will be preserved")
    console.print(f"  • Total tags per work will increase from ~{avg_tags_current:.1f} to ~7-12")
    console.print(f"  • Results saved to: {config.output_file}")


if __name__ == "__main__":
    main()
