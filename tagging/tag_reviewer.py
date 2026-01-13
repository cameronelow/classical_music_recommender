"""
Human-in-the-loop review system for auto-generated tags.

Provides an interactive CLI interface for reviewing, correcting,
and approving auto-tagged classical music works.
"""

import logging
from typing import Dict, List, Optional, Set
from datetime import datetime

import pandas as pd
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.prompt import Prompt, Confirm
from rich import print as rprint

from .auto_tagger import TAG_TAXONOMY
from .tag_learner import TagLearner
from .tagging_config import TaggingConfig


logger = logging.getLogger(__name__)


class TagReviewer:
    """
    Interactive review system for auto-tagged works.

    Allows human reviewers to approve, reject, or modify auto-generated tags.
    """

    def __init__(
        self,
        config: Optional[TaggingConfig] = None,
        learner: Optional[TagLearner] = None
    ):
        """
        Initialize the tag reviewer.

        Args:
            config: Configuration object
            learner: TagLearner instance for recording corrections
        """
        self.config = config or TaggingConfig()
        self.learner = learner or TagLearner(config=self.config)
        self.console = Console()

        # Load composers for display
        self.composers_df = self._load_composers()

    def _load_composers(self) -> Optional[pd.DataFrame]:
        """Load composer data if available."""
        try:
            if self.config.composers_file and self.config.composers_file.exists():
                return pd.read_csv(self.config.composers_file)
        except Exception as e:
            logger.warning(f"Could not load composers file: {e}")
        return None

    def _get_composer_name(self, composer_id: str) -> str:
        """Get composer name from ID."""
        if self.composers_df is not None and composer_id and pd.notna(composer_id):
            composer_row = self.composers_df[self.composers_df['composer_id'] == composer_id]
            if not composer_row.empty:
                composer = composer_row.iloc[0]
                name = composer['name']
                period = composer.get('period', '')
                if period and pd.notna(period):
                    return f"{name} ({period})"
                return name
        return "Unknown Composer"

    def review_batch(
        self,
        auto_tagged_works: pd.DataFrame,
        works_metadata: pd.DataFrame,
        sample_size: Optional[int] = None,
        mode: str = 'quick'
    ) -> Dict:
        """
        Interactive review of auto-tagged works.

        Args:
            auto_tagged_works: DataFrame with columns [work_id, tag, source, confidence]
            works_metadata: DataFrame with work metadata
            sample_size: Number of works to review (None = all)
            mode: Review mode ('quick', 'full', 'confidence')

        Returns:
            Dictionary with approval statistics and corrections
        """
        # Determine works to review
        unique_work_ids = auto_tagged_works['work_id'].unique()

        if mode == 'quick' and sample_size:
            # Random sample
            import random
            work_ids_to_review = random.sample(
                list(unique_work_ids),
                min(sample_size, len(unique_work_ids))
            )
        elif mode == 'confidence':
            # Review low-confidence works only
            low_conf = auto_tagged_works[
                auto_tagged_works['confidence'] < self.config.confidence_threshold
            ]
            work_ids_to_review = low_conf['work_id'].unique()
        else:
            # Full review
            work_ids_to_review = unique_work_ids

        results = {
            'approved': [],
            'corrected': [],
            'rejected': [],
            'skipped': [],
            'approval_rate': 0.0,
            'total_reviewed': 0
        }

        self.console.print(Panel.fit(
            f"[bold cyan]Tag Review Session[/bold cyan]\n"
            f"Total works to review: {len(work_ids_to_review)}\n"
            f"Mode: {mode}",
            border_style="cyan"
        ))

        for i, work_id in enumerate(work_ids_to_review):
            work_tags = auto_tagged_works[auto_tagged_works['work_id'] == work_id]
            work_meta = works_metadata[works_metadata['work_id'] == work_id].iloc[0]

            tags = work_tags['tag'].tolist()

            # Display work for review
            action = self.display_work_for_review(
                work=work_meta.to_dict(),
                tags=tags,
                work_number=i + 1,
                total_works=len(work_ids_to_review)
            )

            if action['action'] == 'approve':
                results['approved'].append(work_id)
                results['total_reviewed'] += 1

            elif action['action'] == 'correct':
                old_tags = tags.copy()
                new_tags = action['new_tags']

                results['corrected'].append({
                    'work_id': work_id,
                    'old_tags': old_tags,
                    'new_tags': new_tags
                })
                results['total_reviewed'] += 1

                # Record correction for learning
                self.learner.record_correction(
                    work=work_meta.to_dict(),
                    old_tags=old_tags,
                    new_tags=new_tags,
                    reason=action.get('reason', 'User correction')
                )

            elif action['action'] == 'reject':
                results['rejected'].append(work_id)
                results['total_reviewed'] += 1

            elif action['action'] == 'skip':
                results['skipped'].append(work_id)

            elif action['action'] == 'quit':
                self.console.print("[yellow]Review session interrupted by user[/yellow]")
                break

        # Calculate approval rate
        total_decisions = results['total_reviewed']
        if total_decisions > 0:
            approved_count = len(results['approved']) + len(results['corrected'])
            results['approval_rate'] = approved_count / total_decisions

        # Display summary
        self._display_review_summary(results)

        # Check if we should auto-approve remaining works
        if (
            mode == 'quick' and
            results['approval_rate'] >= self.config.auto_approve_threshold and
            len(results['skipped']) == 0
        ):
            remaining = set(unique_work_ids) - set(work_ids_to_review)
            if remaining:
                should_approve = Confirm.ask(
                    f"\n[green]Approval rate is {results['approval_rate']:.1%}[/green]. "
                    f"Auto-approve {len(remaining)} remaining works?"
                )
                if should_approve:
                    results['approved'].extend(remaining)
                    self.console.print(
                        f"[green]Auto-approved {len(remaining)} remaining works[/green]"
                    )

        return results

    def display_work_for_review(
        self,
        work: Dict,
        tags: List[str],
        work_number: int,
        total_works: int
    ) -> Dict:
        """
        Display a work and its tags for review.

        Args:
            work: Work metadata dictionary
            tags: List of auto-generated tags
            work_number: Current work number
            total_works: Total works in review session

        Returns:
            Dictionary with action and any modifications
        """
        self.console.print("\n" + "─" * 80)
        self.console.print(f"[bold]Work #{work_number} of {total_works}[/bold]")
        self.console.print("─" * 80)

        # Display work info
        title = work.get('title', 'Unknown Title')
        composer_id = work.get('composer_id')
        work_type = work.get('work_type', '')
        key = work.get('key', '')

        self.console.print(f"\n[bold cyan]Title:[/bold cyan] {title}")

        # Display composer info
        composer_name = self._get_composer_name(composer_id)
        if composer_name != 'Unknown Composer':
            self.console.print(f"[bold cyan]Composer:[/bold cyan] {composer_name}")
            if composer_id:
                self.console.print(f"[dim]  ID: {composer_id}[/dim]")

        # Display work type and key
        if work_type or key:
            type_key = f"Type: {work_type}" if work_type else ""
            key_info = f"Key: {key}" if key else ""
            separator = " | " if type_key and key_info else ""
            self.console.print(f"[bold cyan]{type_key}{separator}{key_info}[/bold cyan]")

        # Display tags
        self.console.print(f"\n[bold yellow]Auto-generated tags ({len(tags)}):[/bold yellow]")
        for tag in tags:
            # Find category
            category = self._find_tag_category(tag)
            self.console.print(f"  ✓ [green]{tag}[/green] [dim]({category})[/dim]")

        # Get user action
        self.console.print("\n[bold]Options:[/bold]")
        self.console.print("  [a] Approve all tags")
        self.console.print("  [r] Remove specific tags")
        self.console.print("  [+] Add new tags")
        self.console.print("  [e] Edit (remove + add)")
        self.console.print("  [x] Reject all and skip")
        self.console.print("  [s] Skip for later review")
        self.console.print("  [q] Quit review session")

        while True:
            choice = Prompt.ask(
                "\nYour choice",
                choices=['a', 'r', '+', 'e', 'x', 's', 'q'],
                default='a'
            )

            if choice == 'a':
                return {'action': 'approve'}

            elif choice == 'r':
                return self._handle_remove_tags(tags)

            elif choice == '+':
                return self._handle_add_tags(tags)

            elif choice == 'e':
                return self._handle_edit_tags(tags)

            elif choice == 'x':
                return {'action': 'reject'}

            elif choice == 's':
                return {'action': 'skip'}

            elif choice == 'q':
                return {'action': 'quit'}

    def _find_tag_category(self, tag: str) -> str:
        """Find which category a tag belongs to."""
        for category, tags in TAG_TAXONOMY.items():
            if tag in tags:
                return category
        return "unknown"

    def _handle_remove_tags(self, current_tags: List[str]) -> Dict:
        """Handle removing specific tags."""
        self.console.print("\n[yellow]Enter tag names to remove (comma-separated):[/yellow]")
        to_remove = Prompt.ask("Tags to remove")

        remove_list = [t.strip() for t in to_remove.split(',')]
        new_tags = [t for t in current_tags if t not in remove_list]

        self.console.print(f"[green]Removed {len(current_tags) - len(new_tags)} tags[/green]")

        reason = Prompt.ask("Reason for removal (optional)", default="")

        return {
            'action': 'correct',
            'new_tags': new_tags,
            'reason': reason or 'User removed tags'
        }

    def _handle_add_tags(self, current_tags: List[str]) -> Dict:
        """Handle adding new tags."""
        self._display_available_tags()

        self.console.print("\n[yellow]Enter tag names to add (comma-separated):[/yellow]")
        to_add = Prompt.ask("Tags to add")

        add_list = [t.strip() for t in to_add.split(',')]

        # Validate tags
        valid_tags = self._validate_tags(add_list)
        new_tags = current_tags + valid_tags

        self.console.print(f"[green]Added {len(valid_tags)} tags[/green]")

        reason = Prompt.ask("Reason for addition (optional)", default="")

        return {
            'action': 'correct',
            'new_tags': new_tags,
            'reason': reason or 'User added tags'
        }

    def _handle_edit_tags(self, current_tags: List[str]) -> Dict:
        """Handle editing tags (remove + add)."""
        # First remove
        remove_result = self._handle_remove_tags(current_tags)
        intermediate_tags = remove_result['new_tags']

        # Then add
        add_result = self._handle_add_tags(intermediate_tags)

        return add_result

    def _validate_tags(self, tags: List[str]) -> List[str]:
        """Validate that tags are in taxonomy."""
        valid_tags = []
        all_valid = set()
        for category_tags in TAG_TAXONOMY.values():
            all_valid.update(category_tags)

        for tag in tags:
            if tag in all_valid:
                valid_tags.append(tag)
            else:
                self.console.print(f"[red]Warning: '{tag}' is not in taxonomy, skipping[/red]")

        return valid_tags

    def _display_available_tags(self):
        """Display all available tags by category."""
        self.console.print("\n[bold cyan]Available Tags:[/bold cyan]")

        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("Category", style="cyan")
        table.add_column("Tags", style="white")

        for category, tags in TAG_TAXONOMY.items():
            table.add_row(category, ", ".join(tags))

        self.console.print(table)

    def _display_review_summary(self, results: Dict):
        """Display summary of review session."""
        self.console.print("\n" + "═" * 80)
        self.console.print("[bold cyan]Review Session Summary[/bold cyan]")
        self.console.print("═" * 80)

        total = results['total_reviewed']
        approved = len(results['approved'])
        corrected = len(results['corrected'])
        rejected = len(results['rejected'])
        skipped = len(results['skipped'])

        self.console.print(f"\nTotal reviewed: {total}")
        self.console.print(f"  [green]✓ Approved: {approved}[/green]")
        self.console.print(f"  [yellow]✎ Corrected: {corrected}[/yellow]")
        self.console.print(f"  [red]✗ Rejected: {rejected}[/red]")
        self.console.print(f"  [blue]○ Skipped: {skipped}[/blue]")

        if total > 0:
            approval_rate = results['approval_rate'] * 100
            color = "green" if approval_rate >= 80 else "yellow" if approval_rate >= 60 else "red"
            self.console.print(f"\n[{color}]Approval rate: {approval_rate:.1f}%[/{color}]")

    def quick_approve_all(self, work_ids: List[str]) -> Dict:
        """
        Quickly approve all works without review.

        Args:
            work_ids: List of work IDs to approve

        Returns:
            Results dictionary
        """
        return {
            'approved': work_ids,
            'corrected': [],
            'rejected': [],
            'skipped': [],
            'approval_rate': 1.0,
            'total_reviewed': len(work_ids)
        }
