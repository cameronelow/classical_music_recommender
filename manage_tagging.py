#!/usr/bin/env python3
"""
Unified Classical Music Auto-Tagging CLI

This consolidated script provides all tagging functionality in one place:
- Tag untagged works
- Enhance existing MusicBrainz tags
- Find and tag playful/silly works
- Estimate costs
- Review and correct tags
- Analyze tag quality
- Clear checkpoints and retry failed works

Usage:
    python3 manage_tagging.py tag-all           # Tag all remaining untagged works
    python3 manage_tagging.py enhance-mb        # Enhance works with MB tags
    python3 manage_tagging.py find-playful      # Find works needing playful/silly tags
    python3 manage_tagging.py estimate          # Estimate costs
    python3 manage_tagging.py clear-checkpoint  # Clear checkpoint to retry failed works
    python3 manage_tagging.py stats             # Show current tagging statistics
    python3 manage_tagging.py analyze           # Analyze tag quality

For more options:
    python3 manage_tagging.py --help
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import pandas as pd
import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.prompt import Confirm

from tagging.auto_tagger import ClassicalMusicAutoTagger
from tagging.tagging_config import TaggingConfig
from tagging.tag_quality import TagQualityEvaluator
from tagging.tag_reviewer import TagReviewer

# Initialize CLI
app = typer.Typer(
    name="manage_tagging",
    help="Unified Classical Music Auto-Tagging System",
    add_completion=False
)
console = Console()


def load_data(config: TaggingConfig) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load works, existing tags, and enhanced tags."""
    # Load works
    if config.works_file.exists():
        works_df = pd.read_csv(config.works_file)
    else:
        console.print(f"[red]Error: Works file not found: {config.works_file}[/red]")
        raise typer.Exit(1)

    # Load existing MusicBrainz tags
    if config.existing_tags_file.exists():
        mb_tags_df = pd.read_csv(config.existing_tags_file)
    else:
        mb_tags_df = pd.DataFrame(columns=['work_id', 'tag', 'source', 'confidence', 'created_at'])

    # Load enhanced tags (MB + auto-tagger)
    enhanced_tags_file = Path("data/processed/work_tags_enhanced.parquet")
    if enhanced_tags_file.exists():
        enhanced_tags_df = pd.read_parquet(enhanced_tags_file)
    else:
        enhanced_tags_df = mb_tags_df.copy()

    return works_df, mb_tags_df, enhanced_tags_df


@app.command("stats")
def show_stats():
    """Show current tagging statistics and data quality."""
    try:
        config = TaggingConfig()
        works_df, mb_tags_df, enhanced_tags_df = load_data(config)

        # Calculate statistics
        total_works = len(works_df)
        works_with_mb = len(set(mb_tags_df['work_id'].unique()))
        works_with_any_tags = len(set(enhanced_tags_df['work_id'].unique()))
        works_with_auto_tags = len(set(enhanced_tags_df[enhanced_tags_df['source'] == 'auto-tagger']['work_id'].unique()))
        untagged_works = total_works - works_with_any_tags

        # Create statistics table
        table = Table(title="Tagging Statistics", show_header=True, header_style="bold cyan")
        table.add_column("Metric", style="cyan")
        table.add_column("Count", justify="right", style="yellow")
        table.add_column("Percentage", justify="right", style="green")

        table.add_row("Total Works", f"{total_works:,}", "100.0%")
        table.add_row("Works with MusicBrainz tags", f"{works_with_mb:,}", f"{works_with_mb/total_works*100:.1f}%")
        table.add_row("Works with auto-tagger tags", f"{works_with_auto_tags:,}", f"{works_with_auto_tags/total_works*100:.1f}%")
        table.add_row("Works with ANY tags", f"{works_with_any_tags:,}", f"{works_with_any_tags/total_works*100:.1f}%")
        table.add_row("[bold red]Untagged works[/bold red]", f"[bold red]{untagged_works:,}[/bold red]", f"[bold red]{untagged_works/total_works*100:.1f}%[/bold red]")

        console.print(table)

        # Tag distribution
        console.print("\n[bold]Tag Distribution:[/bold]")
        console.print(f"  MusicBrainz tags: {len(mb_tags_df)} entries")
        console.print(f"  Auto-tagger tags: {len(enhanced_tags_df[enhanced_tags_df['source'] == 'auto-tagger'])} entries")
        console.print(f"  Total enhanced tags: {len(enhanced_tags_df)} entries")

        # Check checkpoint status
        checkpoint_file = Path("data/corrections/tagging_checkpoint.json")
        if checkpoint_file.exists():
            import json
            with open(checkpoint_file) as f:
                checkpoint = json.load(f)
            console.print(f"\n[yellow]⚠ Checkpoint exists:[/yellow]")
            console.print(f"  Processed works: {len(checkpoint.get('processed_work_ids', []))}")
            console.print(f"  Completed: {checkpoint.get('completed', False)}")
            if 'stats' in checkpoint:
                stats = checkpoint['stats']
                console.print(f"  Last run success rate: {stats.get('success_rate', 0)*100:.1f}%")

    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)


@app.command("tag-all")
def tag_all_remaining(
    skip_confirm: bool = typer.Option(False, "--yes", "-y", help="Skip confirmation prompt"),
    resume: bool = typer.Option(False, "--resume", help="Resume from checkpoint"),
):
    """Tag all remaining untagged works."""
    try:
        config = TaggingConfig()
        works_df, mb_tags_df, enhanced_tags_df = load_data(config)

        console.print(Panel.fit(
            "[bold cyan]Tag All Remaining Untagged Works[/bold cyan]",
            border_style="cyan"
        ))

        # Find untagged works
        tagged_work_ids = set(enhanced_tags_df['work_id'].unique())
        untagged_works = works_df[~works_df['work_id'].isin(tagged_work_ids)]

        console.print(f"\n[bold]Summary:[/bold]")
        console.print(f"  Total works: {len(works_df)}")
        console.print(f"  Already tagged: {len(tagged_work_ids)}")
        console.print(f"  [yellow]Untagged works: {len(untagged_works)}[/yellow]\n")

        if len(untagged_works) == 0:
            console.print("[green]✓ All works are already tagged![/green]")
            return

        # Estimate cost
        tagger = ClassicalMusicAutoTagger(config)
        estimate = tagger.estimate_cost(len(untagged_works))

        console.print("[bold]Cost Estimate:[/bold]")
        console.print(f"  Estimated tokens: ~{estimate['estimated_tokens']:,}")
        console.print(f"  [yellow]Estimated cost: ${estimate['estimated_cost_usd']:.2f}[/yellow]")
        console.print(f"  Estimated time: {estimate['estimated_time_minutes']:.1f} minutes")
        console.print(f"  Budget limit: ${config.max_cost_usd:.2f}\n")

        if not estimate['within_budget']:
            console.print(f"[red]✗ Estimated cost exceeds budget limit![/red]")
            raise typer.Exit(1)

        # Confirm
        if not skip_confirm:
            if not Confirm.ask(f"Tag {len(untagged_works)} works?"):
                console.print("[yellow]Cancelled.[/yellow]")
                return

        # Clear checkpoint if not resuming
        if not resume:
            console.print("[yellow]Clearing checkpoint...[/yellow]")
            tagger.clear_checkpoint()

        # Tag works
        console.print("\n[bold green]Starting auto-tagging...[/bold green]\n")
        works_list = untagged_works.to_dict('records')
        new_tags = tagger.tag_batch(
            works=works_list,
            batch_size=None,  # Process all works (bug fix applied)
            save_progress=True
        )

        # Merge with existing tags
        all_tags_df = pd.concat([enhanced_tags_df, new_tags], ignore_index=True)

        # Save to enhanced files
        output_parquet = Path("data/processed/work_tags_enhanced.parquet")
        output_csv = Path("data/processed/work_tags_enhanced.csv")

        all_tags_df.to_parquet(output_parquet, index=False)
        all_tags_df.to_csv(output_csv, index=False)

        console.print(f"\n[bold green]✓ Complete![/bold green]")
        console.print(f"\nResults:")
        console.print(f"  New tags added: {len(new_tags)}")
        console.print(f"  Total tag entries: {len(all_tags_df)}")
        console.print(f"  Works now tagged: {all_tags_df['work_id'].nunique()}")
        console.print(f"\nSaved to:")
        console.print(f"  {output_parquet}")
        console.print(f"  {output_csv}")

        # Show API usage
        stats = tagger.get_usage_stats()
        console.print(f"\n[bold]API Usage:[/bold]")
        console.print(f"  Requests: {stats['total_requests']}")
        console.print(f"  Success rate: {stats['success_rate']:.1%}")
        console.print(f"  Total tokens: {stats['total_tokens']:,}")
        console.print(f"  [yellow]Actual cost: ${stats['total_cost_usd']:.3f}[/yellow]")

        console.print(f"\n[bold cyan]Next steps:[/bold cyan]")
        console.print("  1. Restart your backend server to load the new tags")
        console.print("  2. Run: python3 manage_tagging.py stats")
        console.print("  3. Test semantic search - you should see much better explanations!")

    except Exception as e:
        console.print(f"\n[red]Error: {e}[/red]")
        import traceback
        traceback.print_exc()
        raise typer.Exit(1)


@app.command("enhance-mb")
def enhance_musicbrainz_tags(
    skip_confirm: bool = typer.Option(False, "--yes", "-y", help="Skip confirmation prompt"),
):
    """Enhance works that already have MusicBrainz tags with additional auto-tagger tags."""
    try:
        config = TaggingConfig()
        works_df, mb_tags_df, enhanced_tags_df = load_data(config)

        console.print(Panel.fit(
            "[bold cyan]Enhance MusicBrainz Tagged Works[/bold cyan]",
            border_style="cyan"
        ))

        # Find works with MB tags but no auto-tagger tags
        mb_work_ids = set(mb_tags_df['work_id'].unique())
        auto_tagged_ids = set(enhanced_tags_df[enhanced_tags_df['source'] == 'auto-tagger']['work_id'].unique())
        to_enhance = mb_work_ids - auto_tagged_ids

        works_to_enhance = works_df[works_df['work_id'].isin(to_enhance)]

        console.print(f"\n[bold]Summary:[/bold]")
        console.print(f"  Works with MB tags: {len(mb_work_ids)}")
        console.print(f"  Already enhanced: {len(mb_work_ids & auto_tagged_ids)}")
        console.print(f"  [yellow]To enhance: {len(to_enhance)}[/yellow]\n")

        if len(to_enhance) == 0:
            console.print("[green]✓ All MB-tagged works are already enhanced![/green]")
            return

        # Estimate and confirm
        tagger = ClassicalMusicAutoTagger(config)
        estimate = tagger.estimate_cost(len(works_to_enhance))

        console.print("[bold]Cost Estimate:[/bold]")
        console.print(f"  [yellow]Estimated cost: ${estimate['estimated_cost_usd']:.2f}[/yellow]")
        console.print(f"  Estimated time: {estimate['estimated_time_minutes']:.1f} minutes\n")

        if not skip_confirm:
            if not Confirm.ask(f"Enhance {len(works_to_enhance)} works?"):
                console.print("[yellow]Cancelled.[/yellow]")
                return

        # Tag works
        console.print("\n[bold green]Enhancing works...[/bold green]\n")
        works_list = works_to_enhance.to_dict('records')
        new_tags = tagger.tag_batch(works_list, batch_size=None, save_progress=True)

        # Merge and save
        all_tags_df = pd.concat([enhanced_tags_df, new_tags], ignore_index=True)
        output_parquet = Path("data/processed/work_tags_enhanced.parquet")
        output_csv = Path("data/processed/work_tags_enhanced.csv")
        all_tags_df.to_parquet(output_parquet, index=False)
        all_tags_df.to_csv(output_csv, index=False)

        console.print(f"\n[bold green]✓ Complete![/bold green]")
        console.print(f"  Enhanced {new_tags['work_id'].nunique()} works with {len(new_tags)} new tags")

    except Exception as e:
        console.print(f"\n[red]Error: {e}[/red]")
        raise typer.Exit(1)


@app.command("estimate")
def estimate_cost():
    """Estimate cost and time for tagging remaining untagged works."""
    try:
        config = TaggingConfig()
        works_df, mb_tags_df, enhanced_tags_df = load_data(config)

        tagged_work_ids = set(enhanced_tags_df['work_id'].unique())
        untagged_works = works_df[~works_df['work_id'].isin(tagged_work_ids)]

        console.print(Panel.fit(
            "[bold cyan]Cost Estimation[/bold cyan]",
            border_style="cyan"
        ))

        console.print(f"\nUntagged works: {len(untagged_works)}")

        if len(untagged_works) == 0:
            console.print("[green]All works are tagged![/green]")
            return

        tagger = ClassicalMusicAutoTagger(config)
        estimate = tagger.estimate_cost(len(untagged_works))

        console.print(f"\n[bold yellow]Estimate:[/bold yellow]")
        console.print(f"  Tokens: ~{estimate['estimated_tokens']:,}")
        console.print(f"  Cost: ${estimate['estimated_cost_usd']:.3f}")
        console.print(f"  Time: {estimate['estimated_time_minutes']:.1f} minutes")
        console.print(f"  Budget: ${config.max_cost_usd:.2f}")

        if estimate['within_budget']:
            console.print(f"\n[green]✓ Within budget[/green]")
        else:
            console.print(f"\n[red]✗ Exceeds budget[/red]")

    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)


@app.command("clear-checkpoint")
def clear_checkpoint(
    force: bool = typer.Option(False, "--force", "-f", help="Force clear without confirmation")
):
    """Clear the checkpoint to retry failed works."""
    try:
        checkpoint_file = Path("data/corrections/tagging_checkpoint.json")

        if not checkpoint_file.exists():
            console.print("[yellow]No checkpoint file found.[/yellow]")
            return

        # Show checkpoint info
        import json
        with open(checkpoint_file) as f:
            checkpoint = json.load(f)

        console.print(Panel.fit(
            "[bold cyan]Clear Checkpoint[/bold cyan]",
            border_style="cyan"
        ))

        console.print(f"\nCurrent checkpoint:")
        console.print(f"  Processed works: {len(checkpoint.get('processed_work_ids', []))}")
        console.print(f"  Total works: {checkpoint.get('total_works', 0)}")
        console.print(f"  Completed: {checkpoint.get('completed', False)}")

        if 'stats' in checkpoint:
            stats = checkpoint['stats']
            console.print(f"\nLast run statistics:")
            console.print(f"  Attempted: {stats.get('attempted', 0)}")
            console.print(f"  Successful: {stats.get('successful', 0)}")
            console.print(f"  Failed: {stats.get('failed', 0)}")
            console.print(f"  Success rate: {stats.get('success_rate', 0)*100:.1f}%")

        console.print("\n[yellow]Clearing the checkpoint will allow all works to be retried.[/yellow]")

        if not force:
            if not Confirm.ask("Clear checkpoint?"):
                console.print("[yellow]Cancelled.[/yellow]")
                return

        checkpoint_file.unlink()
        console.print("[green]✓ Checkpoint cleared![/green]")
        console.print("\nYou can now re-run tagging with:")
        console.print("  python3 manage_tagging.py tag-all")

    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)


@app.command("find-playful")
def find_playful_works(
    apply_tags: bool = typer.Option(False, "--apply", "-a", help="Automatically apply suggested tags"),
):
    """Find works that should have playful/silly tags and optionally apply them."""
    try:
        config = TaggingConfig()
        works_df, mb_tags_df, enhanced_tags_df = load_data(config)

        console.print(Panel.fit(
            "[bold cyan]Find Playful/Silly Works[/bold cyan]",
            border_style="cyan"
        ))

        # Pattern matching for potentially playful works
        playful_patterns = {
            'scherzo': ['playful', 'energetic'],
            'humoreske': ['humorous', 'playful'],
            'humoresque': ['humorous', 'playful'],
            'children': ['lighthearted', 'playful'],
            'minuet': ['playful', 'elegant'],
            'polka': ['playful', 'cheerful'],
            'galop': ['playful', 'energetic'],
            'gigue': ['playful', 'energetic'],
            'toy': ['playful', 'whimsical'],
            'carnival': ['playful', 'whimsical'],
            'joke': ['humorous', 'playful'],
            'jest': ['humorous', 'playful'],
        }

        playful_mood_tags = ['playful', 'whimsical', 'humorous', 'cheerful', 'lighthearted']

        candidates = []

        for _, work in works_df.iterrows():
            work_id = work['work_id']
            title = work['title'].lower()

            # Get current tags
            current_tags = enhanced_tags_df[enhanced_tags_df['work_id'] == work_id]['tag'].tolist()
            has_playful = any(tag in current_tags for tag in playful_mood_tags)

            # Check patterns
            matched_pattern = None
            suggested_tags = []

            for pattern, tags in playful_patterns.items():
                if pattern in title:
                    matched_pattern = pattern
                    # Only suggest tags not already present
                    suggested_tags = [t for t in tags if t not in current_tags]
                    break

            if matched_pattern and (not has_playful or suggested_tags):
                candidates.append({
                    'work_id': work_id,
                    'title': work['title'],
                    'pattern': matched_pattern,
                    'has_playful': has_playful,
                    'current_tags': current_tags[:7],
                    'suggested_tags': suggested_tags
                })

        # Display results
        console.print(f"\n[bold]Found {len(candidates)} works that may need playful/silly tags:[/bold]\n")

        table = Table(show_header=True, header_style="bold cyan")
        table.add_column("Title", style="cyan", no_wrap=False, max_width=50)
        table.add_column("Pattern", style="yellow")
        table.add_column("Has Playful?", style="green")
        table.add_column("Suggested Tags", style="magenta")

        for candidate in candidates[:20]:  # Show first 20
            has_playful_icon = "✓" if candidate['has_playful'] else "✗"
            suggested = ", ".join(candidate['suggested_tags']) if candidate['suggested_tags'] else "Already tagged"
            table.add_row(
                candidate['title'][:50],
                candidate['pattern'],
                has_playful_icon,
                suggested
            )

        console.print(table)

        if len(candidates) > 20:
            console.print(f"\n[dim]... and {len(candidates) - 20} more[/dim]")

        # Summary
        needs_tags = [c for c in candidates if c['suggested_tags']]
        console.print(f"\n[bold]Summary:[/bold]")
        console.print(f"  Total candidates: {len(candidates)}")
        console.print(f"  Already have playful tags: {sum(1 for c in candidates if c['has_playful'])}")
        console.print(f"  [yellow]Need additional tags: {len(needs_tags)}[/yellow]")

        if apply_tags and needs_tags:
            console.print("\n[bold yellow]Note:[/bold yellow] Automatic tag application is not yet implemented.")
            console.print("These works will be properly tagged when you run:")
            console.print("  [cyan]python3 manage_tagging.py tag-all[/cyan]")
            console.print("\nThe auto-tagger has been updated with guidelines to recognize:")
            console.print("  • Scherzos → playful, energetic")
            console.print("  • Humoresques → humorous, playful")
            console.print("  • Children's music → lighthearted, playful")
            console.print("  • Dance forms → playful, elegant, cheerful")

        console.print("\n[bold cyan]Recommended action:[/bold cyan]")
        console.print("  Run the auto-tagger to apply proper mood tags:")
        console.print("  [cyan]python3 manage_tagging.py tag-all[/cyan]")

    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        import traceback
        traceback.print_exc()
        raise typer.Exit(1)


@app.command("analyze")
def analyze_quality():
    """Analyze tag quality and coverage."""
    try:
        config = TaggingConfig()
        works_df, mb_tags_df, enhanced_tags_df = load_data(config)

        console.print(Panel.fit(
            "[bold cyan]Tag Quality Analysis[/bold cyan]",
            border_style="cyan"
        ))

        # Tag coverage by source
        console.print("\n[bold]Tags by Source:[/bold]")
        source_counts = enhanced_tags_df['source'].value_counts()
        for source, count in source_counts.items():
            console.print(f"  {source}: {count} tags")

        # Average tags per work
        tags_per_work = enhanced_tags_df.groupby('work_id').size()
        console.print(f"\n[bold]Tags per Work:[/bold]")
        console.print(f"  Mean: {tags_per_work.mean():.1f}")
        console.print(f"  Median: {tags_per_work.median():.1f}")
        console.print(f"  Min: {tags_per_work.min()}")
        console.print(f"  Max: {tags_per_work.max()}")

        # Most common tags
        console.print(f"\n[bold]Top 20 Most Common Tags:[/bold]")
        top_tags = enhanced_tags_df['tag'].value_counts().head(20)
        for tag, count in top_tags.items():
            console.print(f"  {tag}: {count}")

    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)


if __name__ == "__main__":
    app()
