#!/usr/bin/env python3
"""
CLI tool for managing the classical music auto-tagging pipeline.

Provides commands for estimating costs, tagging works, reviewing tags,
analyzing quality, and exporting results.
"""

import sys
import logging
from pathlib import Path
from typing import Optional, List

import typer
import pandas as pd
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Confirm

# Add project root to path
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from tagging.auto_tagger import ClassicalMusicAutoTagger
from tagging.tag_reviewer import TagReviewer
from tagging.tag_learner import TagLearner
from tagging.tag_quality import TagQualityEvaluator
from tagging.tagging_config import TaggingConfig


# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize Typer app
app = typer.Typer(help="Classical Music Auto-Tagging System")
console = Console()


def load_works(config: TaggingConfig) -> pd.DataFrame:
    """Load works data."""
    if not config.works_file.exists():
        console.print(f"[red]Error: Works file not found at {config.works_file}[/red]")
        raise typer.Exit(1)

    return pd.read_csv(config.works_file)


def load_existing_tags(config: TaggingConfig) -> pd.DataFrame:
    """Load existing tags if available."""
    if config.existing_tags_file.exists():
        return pd.read_csv(config.existing_tags_file)
    return pd.DataFrame(columns=['work_id', 'tag', 'source'])


@app.command()
def estimate(
    works_file: Optional[Path] = typer.Option(None, help="Path to works CSV file"),
    max_works: Optional[int] = typer.Option(None, help="Limit number of works to estimate")
):
    """
    Estimate cost and time for auto-tagging.
    """
    try:
        config = TaggingConfig()
        if works_file:
            config.works_file = works_file

        console.print(Panel.fit(
            "[bold cyan]Auto-Tagging Cost Estimation[/bold cyan]",
            border_style="cyan"
        ))

        # Load works
        works_df = load_works(config)
        existing_tags_df = load_existing_tags(config)

        # Find untagged works
        tagged_work_ids = set(existing_tags_df['work_id'].unique())
        untagged_works = works_df[~works_df['work_id'].isin(tagged_work_ids)]

        num_to_tag = min(len(untagged_works), max_works) if max_works else len(untagged_works)

        console.print(f"\nTotal works: {len(works_df)}")
        console.print(f"Already tagged: {len(tagged_work_ids)}")
        console.print(f"Untagged works: {len(untagged_works)}")
        console.print(f"Works to tag: {num_to_tag}")

        # Estimate
        tagger = ClassicalMusicAutoTagger(config)
        estimate = tagger.estimate_cost(num_to_tag)

        console.print(f"\n[bold yellow]Cost Estimate:[/bold yellow]")
        console.print(f"  Estimated tokens: ~{estimate['estimated_tokens']:,}")
        console.print(f"  Estimated cost: ${estimate['estimated_cost_usd']:.3f}")
        console.print(f"  Estimated time: {estimate['estimated_time_minutes']:.1f} minutes")
        console.print(f"  Budget limit: ${estimate['max_budget_usd']:.2f}")

        if estimate['within_budget']:
            console.print(f"\n[green]✓ Within budget[/green]")
        else:
            console.print(f"\n[red]✗ Exceeds budget limit[/red]")

    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)


@app.command()
def tag(
    works_file: Optional[Path] = typer.Option(None, help="Path to works CSV file"),
    max_works: Optional[int] = typer.Option(None, help="Limit number of works to tag"),
    sample_review: int = typer.Option(10, help="Number of works to review as sample"),
    auto_approve_threshold: float = typer.Option(0.8, help="Auto-approve threshold"),
    skip_review: bool = typer.Option(False, help="Skip review and auto-approve all"),
    resume: bool = typer.Option(False, help="Resume from last checkpoint")
):
    """
    Auto-tag untagged works with optional human review.
    """
    try:
        config = TaggingConfig()
        if works_file:
            config.works_file = works_file

        config.auto_approve_threshold = auto_approve_threshold

        console.print(Panel.fit(
            "[bold cyan]Auto-Tagging Classical Music Works[/bold cyan]",
            border_style="cyan"
        ))

        # Load data
        works_df = load_works(config)
        existing_tags_df = load_existing_tags(config)

        # Find untagged works
        tagged_work_ids = set(existing_tags_df['work_id'].unique())
        untagged_works = works_df[~works_df['work_id'].isin(tagged_work_ids)]

        if max_works:
            untagged_works = untagged_works.head(max_works)

        console.print(f"\nFound {len(untagged_works)} untagged works")

        if len(untagged_works) == 0:
            console.print("[green]All works are already tagged![/green]")
            return

        # Estimate and confirm
        tagger = ClassicalMusicAutoTagger(config)
        estimate = tagger.estimate_cost(len(untagged_works))

        console.print(f"\nEstimated cost: ${estimate['estimated_cost_usd']:.3f}")
        console.print(f"Estimated time: {estimate['estimated_time_minutes']:.1f} minutes")

        if not estimate['within_budget']:
            console.print(f"[red]Error: Estimated cost exceeds budget limit[/red]")
            raise typer.Exit(1)

        if not skip_review:
            proceed = Confirm.ask("\nProceed with tagging?")
            if not proceed:
                console.print("[yellow]Tagging cancelled[/yellow]")
                return

        # Clear checkpoint if not resuming
        if not resume:
            tagger.clear_checkpoint()

        # Tag works
        console.print(f"\n[bold]Tagging {len(untagged_works)} works...[/bold]")

        works_list = untagged_works.to_dict('records')
        new_tags_df = tagger.tag_batch(works_list, save_progress=True)

        console.print(f"[green]✓ Tagged {len(new_tags_df)} tag associations[/green]")

        # Display usage stats
        stats = tagger.get_usage_stats()
        console.print(f"\n[bold yellow]API Usage:[/bold yellow]")
        console.print(f"  Total requests: {stats['total_requests']}")
        console.print(f"  Failed requests: {stats['failed_requests']}")
        console.print(f"  Success rate: {stats['success_rate']:.1%}")
        console.print(f"  Total tokens: {stats['total_tokens']:,}")
        console.print(f"  Actual cost: ${stats['total_cost_usd']:.3f}")

        # Review sample or skip
        if skip_review:
            console.print("\n[yellow]Skipping review - auto-approving all tags[/yellow]")
            final_tags_df = pd.concat([existing_tags_df, new_tags_df], ignore_index=True)
        else:
            console.print(f"\n[bold]Reviewing sample of {sample_review} works...[/bold]")

            reviewer = TagReviewer(config)
            review_results = reviewer.review_batch(
                auto_tagged_works=new_tags_df,
                works_metadata=works_df,
                sample_size=sample_review,
                mode='quick'
            )

            # Combine approved and corrected tags
            approved_ids = set(review_results['approved'])
            corrected_data = {c['work_id']: c['new_tags'] for c in review_results['corrected']}

            # Build final tags DataFrame
            final_new_tags = []
            for _, row in new_tags_df.iterrows():
                work_id = row['work_id']

                if work_id in review_results['rejected']:
                    continue  # Skip rejected works

                if work_id in corrected_data:
                    # Use corrected tags
                    if row['tag'] in corrected_data[work_id]:
                        final_new_tags.append(row)
                else:
                    # Use original tags
                    final_new_tags.append(row)

            # Add any new tags from corrections
            for correction in review_results['corrected']:
                work_id = correction['work_id']
                old_tags = set(correction['old_tags'])
                new_tags = set(correction['new_tags'])
                added_tags = new_tags - old_tags

                for tag in added_tags:
                    final_new_tags.append({
                        'work_id': work_id,
                        'tag': tag,
                        'source': 'human-review',
                        'confidence': 1.0
                    })

            final_new_tags_df = pd.DataFrame(final_new_tags)
            final_tags_df = pd.concat([existing_tags_df, final_new_tags_df], ignore_index=True)

        # Save results
        output_file = config.output_file
        final_tags_df.to_csv(output_file, index=False)
        console.print(f"\n[green]✓ Saved {len(final_tags_df)} tag associations to {output_file}[/green]")

    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        logger.exception("Tagging failed")
        raise typer.Exit(1)


@app.command()
def review(
    tags_file: Optional[Path] = typer.Option(None, help="Path to tags CSV file"),
    works_file: Optional[Path] = typer.Option(None, help="Path to works CSV file"),
    sample_size: int = typer.Option(20, help="Number of works to review"),
    mode: str = typer.Option('quick', help="Review mode: quick, full, confidence")
):
    """
    Review and correct auto-tagged works.
    """
    try:
        config = TaggingConfig()
        if tags_file:
            config.existing_tags_file = tags_file
        if works_file:
            config.works_file = works_file

        console.print(Panel.fit(
            "[bold cyan]Tag Review Session[/bold cyan]",
            border_style="cyan"
        ))

        # Load data
        works_df = load_works(config)
        tags_df = load_existing_tags(config)

        if len(tags_df) == 0:
            console.print("[red]No tags found to review[/red]")
            return

        # Filter to auto-tagged works
        auto_tagged = tags_df[tags_df['source'] == 'auto-tagger']

        if len(auto_tagged) == 0:
            console.print("[yellow]No auto-tagged works to review[/yellow]")
            return

        console.print(f"Found {auto_tagged['work_id'].nunique()} auto-tagged works")

        # Start review
        reviewer = TagReviewer(config)
        review_results = reviewer.review_batch(
            auto_tagged_works=auto_tagged,
            works_metadata=works_df,
            sample_size=sample_size,
            mode=mode
        )

        console.print(f"\n[green]Review session complete[/green]")

    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)


@app.command()
def analyze(
    tags_file: Optional[Path] = typer.Option(None, help="Path to tags CSV file"),
    works_file: Optional[Path] = typer.Option(None, help="Path to works CSV file"),
    show_stats: bool = typer.Option(True, help="Show quality statistics"),
    compare_original: bool = typer.Option(False, help="Compare with original tags")
):
    """
    Analyze tag quality and generate report.
    """
    try:
        config = TaggingConfig()
        if tags_file:
            config.output_file = tags_file
        else:
            # Try to load enhanced tags if they exist
            if config.output_file.exists():
                tags_file = config.output_file
            else:
                tags_file = config.existing_tags_file

        if works_file:
            config.works_file = works_file

        console.print(Panel.fit(
            "[bold cyan]Tag Quality Analysis[/bold cyan]",
            border_style="cyan"
        ))

        # Load data
        works_df = load_works(config)
        tags_df = pd.read_csv(tags_file)

        evaluator = TagQualityEvaluator(config)

        # Load old tags for comparison if requested
        old_tags_df = None
        if compare_original and config.existing_tags_file.exists():
            old_tags_df = pd.read_csv(config.existing_tags_file)

        # Generate report
        evaluator.generate_quality_report(
            tags_df=tags_df,
            works_df=works_df,
            old_tags_df=old_tags_df
        )

        # Analyze corrections if available
        learner = TagLearner(config)
        correction_stats = learner.get_correction_stats()

        if correction_stats['total_corrections'] > 0:
            console.print(f"\n[bold yellow]Correction Statistics:[/bold yellow]")
            console.print(f"  Total corrections: {correction_stats['total_corrections']}")
            console.print(f"  Avg tags removed: {correction_stats['avg_tags_removed']}")
            console.print(f"  Avg tags added: {correction_stats['avg_tags_added']}")

            # Show patterns
            analysis = learner.analyze_corrections()
            if analysis['suggestions']:
                console.print(f"\n[bold yellow]Improvement Suggestions:[/bold yellow]")
                for suggestion in analysis['suggestions'][:5]:
                    console.print(f"  • {suggestion}")

    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)


@app.command()
def export(
    output: Path = typer.Argument(..., help="Output file path"),
    tags_file: Optional[Path] = typer.Option(None, help="Path to tags CSV file"),
    format: str = typer.Option('csv', help="Output format: csv or jsonl")
):
    """
    Export enhanced tags to file.
    """
    try:
        config = TaggingConfig()
        if tags_file:
            source_file = tags_file
        elif config.output_file.exists():
            source_file = config.output_file
        else:
            source_file = config.existing_tags_file

        console.print(f"Exporting tags from {source_file}...")

        tags_df = pd.read_csv(source_file)

        if format == 'csv':
            tags_df.to_csv(output, index=False)
        elif format == 'jsonl':
            tags_df.to_json(output, orient='records', lines=True)
        else:
            console.print(f"[red]Unsupported format: {format}[/red]")
            raise typer.Exit(1)

        console.print(f"[green]✓ Exported {len(tags_df)} tag associations to {output}[/green]")

        # Show summary
        console.print(f"\nCoverage: {tags_df['work_id'].nunique()} works")
        console.print(f"Avg tags per work: {len(tags_df) / tags_df['work_id'].nunique():.1f}")

    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)


@app.command()
def retag(
    work_ids: str = typer.Argument(..., help="Comma-separated work IDs to retag"),
    works_file: Optional[Path] = typer.Option(None, help="Path to works CSV file")
):
    """
    Retag specific works.
    """
    try:
        config = TaggingConfig()
        if works_file:
            config.works_file = works_file

        work_id_list = [wid.strip() for wid in work_ids.split(',')]

        console.print(f"Retagging {len(work_id_list)} works...")

        # Load works
        works_df = load_works(config)
        works_to_tag = works_df[works_df['work_id'].isin(work_id_list)]

        if len(works_to_tag) == 0:
            console.print("[red]No matching works found[/red]")
            return

        # Tag works
        tagger = ClassicalMusicAutoTagger(config)
        works_list = works_to_tag.to_dict('records')
        new_tags_df = tagger.tag_batch(works_list, save_progress=False)

        console.print(f"[green]✓ Retagged {len(works_to_tag)} works[/green]")
        console.print(f"Generated {len(new_tags_df)} tag associations")

        # Display new tags
        for work_id in work_id_list:
            work_tags = new_tags_df[new_tags_df['work_id'] == work_id]
            if len(work_tags) > 0:
                work_title = works_to_tag[works_to_tag['work_id'] == work_id].iloc[0]['title']
                console.print(f"\n{work_title}:")
                for tag in work_tags['tag']:
                    console.print(f"  • {tag}")

    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)


@app.command()
def enhance_mb(
    works_file: Optional[Path] = typer.Option(None, help="Path to works CSV file"),
    max_works: Optional[int] = typer.Option(None, help="Limit number of works to enhance"),
    skip_review: bool = typer.Option(False, help="Skip review and auto-approve all"),
):
    """
    Enhance works that already have MusicBrainz tags with additional auto-tagger tags.
    """
    try:
        config = TaggingConfig()
        if works_file:
            config.works_file = works_file

        console.print(Panel.fit(
            "[bold cyan]Enhancing MusicBrainz-Tagged Works[/bold cyan]",
            border_style="cyan"
        ))

        # Load data
        works_df = load_works(config)
        existing_tags_df = load_existing_tags(config)

        # Find works that have MusicBrainz tags
        mb_tagged_work_ids = set(
            existing_tags_df[existing_tags_df['source'] == 'musicbrainz']['work_id'].unique()
        )

        # Filter to works that only have MB tags (no auto-tagger tags yet)
        auto_tagged_work_ids = set(
            existing_tags_df[existing_tags_df['source'] == 'auto-tagger']['work_id'].unique()
        )

        # Works with MB tags but no auto-tagger tags
        works_to_enhance_ids = mb_tagged_work_ids - auto_tagged_work_ids
        works_to_enhance = works_df[works_df['work_id'].isin(works_to_enhance_ids)]

        if max_works:
            works_to_enhance = works_to_enhance.head(max_works)

        console.print(f"\nTotal works with MusicBrainz tags: {len(mb_tagged_work_ids)}")
        console.print(f"Works needing enhancement: {len(works_to_enhance)}")

        if len(works_to_enhance) == 0:
            console.print("[green]All MusicBrainz-tagged works already have auto-tagger tags![/green]")
            return

        # Estimate and confirm
        tagger = ClassicalMusicAutoTagger(config)
        estimate = tagger.estimate_cost(len(works_to_enhance))

        console.print(f"\nEstimated cost: ${estimate['estimated_cost_usd']:.3f}")
        console.print(f"Estimated time: {estimate['estimated_time_minutes']:.1f} minutes")

        if not estimate['within_budget']:
            console.print(f"[red]Error: Estimated cost exceeds budget limit[/red]")
            raise typer.Exit(1)

        if not skip_review:
            proceed = Confirm.ask("\nProceed with enhancing?")
            if not proceed:
                console.print("[yellow]Enhancement cancelled[/yellow]")
                return

        # Tag works
        console.print(f"\n[bold]Adding auto-tagger tags to {len(works_to_enhance)} works...[/bold]")

        works_list = works_to_enhance.to_dict('records')
        new_tags_df = tagger.tag_batch(works_list, save_progress=True)

        console.print(f"[green]✓ Generated {len(new_tags_df)} new tag associations[/green]")

        # Display usage stats
        stats = tagger.get_usage_stats()
        console.print(f"\n[bold yellow]API Usage:[/bold yellow]")
        console.print(f"  Total requests: {stats['total_requests']}")
        console.print(f"  Failed requests: {stats['failed_requests']}")
        console.print(f"  Success rate: {stats['success_rate']:.1%}")
        console.print(f"  Total tokens: {stats['total_tokens']:,}")
        console.print(f"  Actual cost: ${stats['total_cost_usd']:.3f}")

        # Merge with existing tags
        console.print("\n[bold]Merging with existing tags...[/bold]")
        final_tags_df = pd.concat([existing_tags_df, new_tags_df], ignore_index=True)

        # Save results
        output_file = config.output_file
        final_tags_df.to_csv(output_file, index=False)
        console.print(f"\n[green]✓ Saved {len(final_tags_df)} tag associations to {output_file}[/green]")

        # Show summary stats
        console.print(f"\n[bold cyan]Summary:[/bold cyan]")
        console.print(f"  Total works tagged: {final_tags_df['work_id'].nunique()}")
        console.print(f"  Total tag associations: {len(final_tags_df)}")
        console.print(f"  Average tags per work: {len(final_tags_df) / final_tags_df['work_id'].nunique():.1f}")

    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        logger.exception("Enhancement failed")
        raise typer.Exit(1)


@app.command()
def corrections(
    action: str = typer.Argument(..., help="Action: list, analyze, export, clear"),
    output: Optional[Path] = typer.Option(None, help="Output file for export")
):
    """
    Manage tag corrections and learning.
    """
    try:
        config = TaggingConfig()
        learner = TagLearner(config)

        if action == 'list':
            corrections = learner.load_corrections()
            console.print(f"Total corrections: {len(corrections)}")

            for i, c in enumerate(corrections[-10:], 1):  # Show last 10
                console.print(f"\n{i}. {c['title']}")
                console.print(f"   Removed: {', '.join(c['removed']) if c['removed'] else 'none'}")
                console.print(f"   Added: {', '.join(c['added']) if c['added'] else 'none'}")

        elif action == 'analyze':
            analysis = learner.analyze_corrections()

            console.print(f"\nTotal corrections: {analysis['total_corrections']}")

            if analysis['most_rejected_tags']:
                console.print(f"\n[bold yellow]Most Rejected Tags:[/bold yellow]")
                for tag, count in analysis['most_rejected_tags'][:5]:
                    console.print(f"  {tag}: {count} times")

            if analysis['most_added_tags']:
                console.print(f"\n[bold yellow]Most Added Tags:[/bold yellow]")
                for tag, count in analysis['most_added_tags'][:5]:
                    console.print(f"  {tag}: {count} times")

            if analysis['suggestions']:
                console.print(f"\n[bold cyan]Suggestions:[/bold cyan]")
                for suggestion in analysis['suggestions']:
                    console.print(f"  • {suggestion}")

        elif action == 'export':
            if not output:
                console.print("[red]Error: --output required for export[/red]")
                return

            learner.export_corrections_to_csv(output)
            console.print(f"[green]✓ Exported corrections to {output}[/green]")

        elif action == 'clear':
            learner.clear_corrections()

        else:
            console.print(f"[red]Unknown action: {action}[/red]")

    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)


if __name__ == "__main__":
    app()
