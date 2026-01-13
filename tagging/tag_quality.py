"""
Tag quality evaluation and metrics.

Provides tools to measure and report on tagging quality.
"""

import logging
from typing import Dict, List, Optional, Tuple
from collections import Counter

import pandas as pd
import numpy as np
from rich.console import Console
from rich.table import Table

from .tagging_config import TaggingConfig


logger = logging.getLogger(__name__)


class TagQualityEvaluator:
    """
    Evaluator for tag quality metrics.

    Measures coverage, diversity, consistency, and other quality indicators.
    """

    def __init__(self, config: Optional[TaggingConfig] = None):
        """
        Initialize the quality evaluator.

        Args:
            config: Configuration object
        """
        self.config = config or TaggingConfig()
        self.console = Console()

    def evaluate_coverage(self, tags_df: pd.DataFrame, works_df: pd.DataFrame) -> Dict:
        """
        Evaluate tag coverage across works.

        Args:
            tags_df: DataFrame with tag associations
            works_df: DataFrame with all works

        Returns:
            Dictionary with coverage metrics
        """
        total_works = len(works_df)
        tagged_works = tags_df['work_id'].nunique()
        coverage_pct = (tagged_works / total_works * 100) if total_works > 0 else 0

        # Tag associations per work
        tags_per_work = tags_df.groupby('work_id').size()
        avg_tags = tags_per_work.mean() if len(tags_per_work) > 0 else 0

        return {
            'total_works': total_works,
            'tagged_works': tagged_works,
            'untagged_works': total_works - tagged_works,
            'coverage_pct': round(coverage_pct, 2),
            'total_tag_associations': len(tags_df),
            'avg_tags_per_work': round(avg_tags, 2),
            'min_tags_per_work': int(tags_per_work.min()) if len(tags_per_work) > 0 else 0,
            'max_tags_per_work': int(tags_per_work.max()) if len(tags_per_work) > 0 else 0
        }

    def evaluate_diversity(self, tags_df: pd.DataFrame) -> Dict:
        """
        Evaluate tag diversity.

        Checks if tags are well-distributed or dominated by a few tags.

        Args:
            tags_df: DataFrame with tag associations

        Returns:
            Dictionary with diversity metrics
        """
        tag_counts = tags_df['tag'].value_counts()
        total_unique_tags = len(tag_counts)

        # Calculate entropy (higher = more diverse)
        if len(tag_counts) > 0:
            probabilities = tag_counts / tag_counts.sum()
            entropy = -np.sum(probabilities * np.log2(probabilities))
            # Normalize by max entropy
            max_entropy = np.log2(len(tag_counts))
            normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0
        else:
            entropy = 0
            normalized_entropy = 0

        # Top tags concentration
        top_5_count = tag_counts.head(5).sum() if len(tag_counts) >= 5 else tag_counts.sum()
        total_count = tag_counts.sum()
        top_5_pct = (top_5_count / total_count * 100) if total_count > 0 else 0

        return {
            'unique_tags': total_unique_tags,
            'total_uses': int(total_count),
            'entropy': round(entropy, 3),
            'normalized_entropy': round(normalized_entropy, 3),
            'top_5_concentration_pct': round(top_5_pct, 2),
            'most_common_tags': tag_counts.head(10).to_dict(),
            'least_common_tags': tag_counts.tail(10).to_dict()
        }

    def evaluate_consistency(
        self,
        tags_df: pd.DataFrame,
        works_df: pd.DataFrame,
        composers_df: Optional[pd.DataFrame] = None
    ) -> Dict:
        """
        Evaluate tag consistency.

        Checks if similar works get similar tags.

        Args:
            tags_df: DataFrame with tag associations
            works_df: DataFrame with works
            composers_df: Optional DataFrame with composers

        Returns:
            Dictionary with consistency metrics
        """
        consistency_scores = {}

        # Consistency by work type
        if 'work_type' in works_df.columns:
            work_type_consistency = self._calculate_group_consistency(
                tags_df, works_df, 'work_type'
            )
            consistency_scores['by_work_type'] = work_type_consistency

        # Consistency by composer
        if composers_df is not None and 'composer_id' in works_df.columns:
            composer_consistency = self._calculate_group_consistency(
                tags_df, works_df, 'composer_id'
            )
            consistency_scores['by_composer'] = composer_consistency

        # Consistency by key
        if 'key' in works_df.columns:
            key_consistency = self._calculate_group_consistency(
                tags_df, works_df, 'key'
            )
            consistency_scores['by_key'] = key_consistency

        return consistency_scores

    def _calculate_group_consistency(
        self,
        tags_df: pd.DataFrame,
        works_df: pd.DataFrame,
        group_column: str
    ) -> Dict:
        """
        Calculate tag consistency within groups.

        Args:
            tags_df: Tag associations
            works_df: Works metadata
            group_column: Column to group by

        Returns:
            Dictionary with consistency scores per group
        """
        # Merge to get group information
        merged = tags_df.merge(
            works_df[['work_id', group_column]],
            on='work_id',
            how='left'
        )

        # Calculate consistency for each group
        group_scores = {}

        for group_value, group_data in merged.groupby(group_column):
            if pd.isna(group_value) or group_value == '':
                continue

            # Get works in this group
            work_ids = group_data['work_id'].unique()

            if len(work_ids) < 2:  # Need at least 2 works to measure consistency
                continue

            # Get tags for each work in group
            tags_by_work = {}
            for work_id in work_ids:
                work_tags = set(group_data[group_data['work_id'] == work_id]['tag'])
                tags_by_work[work_id] = work_tags

            # Calculate pairwise Jaccard similarity
            similarities = []
            work_list = list(tags_by_work.keys())

            for i in range(len(work_list)):
                for j in range(i + 1, len(work_list)):
                    tags_i = tags_by_work[work_list[i]]
                    tags_j = tags_by_work[work_list[j]]

                    if len(tags_i) == 0 and len(tags_j) == 0:
                        continue

                    # Jaccard similarity
                    intersection = len(tags_i & tags_j)
                    union = len(tags_i | tags_j)
                    similarity = intersection / union if union > 0 else 0
                    similarities.append(similarity)

            if similarities:
                avg_similarity = np.mean(similarities)
                group_scores[str(group_value)] = {
                    'works_count': len(work_ids),
                    'avg_similarity': round(avg_similarity, 3),
                    'consistency_score': round(avg_similarity, 3)
                }

        return group_scores

    def compare_before_after(
        self,
        old_tags: pd.DataFrame,
        new_tags: pd.DataFrame,
        works_df: pd.DataFrame
    ) -> Dict:
        """
        Compare tag quality before and after auto-tagging.

        Args:
            old_tags: Original tags DataFrame
            new_tags: New tags DataFrame
            works_df: Works DataFrame

        Returns:
            Dictionary with comparison metrics
        """
        old_coverage = self.evaluate_coverage(old_tags, works_df)
        new_coverage = self.evaluate_coverage(new_tags, works_df)

        old_diversity = self.evaluate_diversity(old_tags)
        new_diversity = self.evaluate_diversity(new_tags)

        return {
            'coverage': {
                'before': old_coverage,
                'after': new_coverage,
                'improvement': {
                    'coverage_pct': round(
                        new_coverage['coverage_pct'] - old_coverage['coverage_pct'], 2
                    ),
                    'avg_tags_per_work': round(
                        new_coverage['avg_tags_per_work'] - old_coverage['avg_tags_per_work'], 2
                    ),
                    'total_associations': (
                        new_coverage['total_tag_associations'] -
                        old_coverage['total_tag_associations']
                    )
                }
            },
            'diversity': {
                'before': old_diversity,
                'after': new_diversity,
                'improvement': {
                    'unique_tags': (
                        new_diversity['unique_tags'] - old_diversity['unique_tags']
                    ),
                    'normalized_entropy': round(
                        new_diversity['normalized_entropy'] - old_diversity['normalized_entropy'], 3
                    )
                }
            }
        }

    def generate_quality_report(
        self,
        tags_df: pd.DataFrame,
        works_df: pd.DataFrame,
        composers_df: Optional[pd.DataFrame] = None,
        old_tags_df: Optional[pd.DataFrame] = None
    ):
        """
        Generate and display a comprehensive quality report.

        Args:
            tags_df: Current tags DataFrame
            works_df: Works DataFrame
            composers_df: Optional composers DataFrame
            old_tags_df: Optional old tags for comparison
        """
        self.console.print("\n" + "═" * 80)
        self.console.print("[bold cyan]TAG QUALITY REPORT[/bold cyan]")
        self.console.print("═" * 80)

        # Coverage metrics
        coverage = self.evaluate_coverage(tags_df, works_df)
        self._print_coverage_section(coverage, old_tags_df, works_df)

        # Diversity metrics
        diversity = self.evaluate_diversity(tags_df)
        self._print_diversity_section(diversity)

        # Consistency metrics
        consistency = self.evaluate_consistency(tags_df, works_df, composers_df)
        self._print_consistency_section(consistency)

        # Overall assessment
        self._print_overall_assessment(coverage, diversity)

        self.console.print("═" * 80 + "\n")

    def _print_coverage_section(
        self,
        coverage: Dict,
        old_tags_df: Optional[pd.DataFrame],
        works_df: pd.DataFrame
    ):
        """Print coverage section of report."""
        self.console.print("\n[bold yellow]Coverage:[/bold yellow]")

        if old_tags_df is not None:
            old_coverage = self.evaluate_coverage(old_tags_df, works_df)
            self.console.print(
                f"  Before: {old_coverage['tagged_works']}/{old_coverage['total_works']} "
                f"works ({old_coverage['coverage_pct']:.1f}%)"
            )
            self.console.print(
                f"  After: {coverage['tagged_works']}/{coverage['total_works']} "
                f"works ({coverage['coverage_pct']:.1f}%) [green]✓[/green]"
            )
        else:
            self.console.print(
                f"  {coverage['tagged_works']}/{coverage['total_works']} "
                f"works ({coverage['coverage_pct']:.1f}%)"
            )

        self.console.print(f"\n[bold yellow]Tag Associations:[/bold yellow]")
        if old_tags_df is not None:
            old_count = len(old_tags_df)
            increase = coverage['total_tag_associations'] - old_count
            multiplier = coverage['total_tag_associations'] / old_count if old_count > 0 else 0
            self.console.print(f"  Before: {old_count:,} associations")
            self.console.print(
                f"  After: {coverage['total_tag_associations']:,} associations "
                f"({multiplier:.1f}x increase) [green]✓[/green]"
            )
        else:
            self.console.print(f"  Total: {coverage['total_tag_associations']:,} associations")

        self.console.print(f"\n[bold yellow]Average Tags per Work:[/bold yellow]")
        self.console.print(f"  {coverage['avg_tags_per_work']:.2f} tags/work")
        self.console.print(
            f"  Range: {coverage['min_tags_per_work']}-{coverage['max_tags_per_work']} tags"
        )

    def _print_diversity_section(self, diversity: Dict):
        """Print diversity section of report."""
        self.console.print(f"\n[bold yellow]Tag Diversity:[/bold yellow]")
        self.console.print(f"  Unique tags: {diversity['unique_tags']}")
        self.console.print(f"  Diversity score: {diversity['normalized_entropy']:.2f} / 1.00")

        self.console.print(f"\n[bold yellow]Most Common Tags:[/bold yellow]")
        for tag, count in list(diversity['most_common_tags'].items())[:5]:
            self.console.print(f"  {tag}: {count}")

    def _print_consistency_section(self, consistency: Dict):
        """Print consistency section of report."""
        self.console.print(f"\n[bold yellow]Consistency Scores:[/bold yellow]")

        if 'by_composer' in consistency and consistency['by_composer']:
            # Show top consistent composers
            composer_scores = consistency['by_composer']
            sorted_composers = sorted(
                composer_scores.items(),
                key=lambda x: x[1]['consistency_score'],
                reverse=True
            )[:5]

            for composer_id, metrics in sorted_composers:
                score = metrics['consistency_score']
                works_count = metrics['works_count']
                quality = "excellent" if score >= 0.8 else "good" if score >= 0.6 else "fair"
                self.console.print(
                    f"  {composer_id}: {score:.2f} ({quality}, {works_count} works)"
                )

    def _print_overall_assessment(self, coverage: Dict, diversity: Dict):
        """Print overall quality assessment."""
        self.console.print(f"\n[bold yellow]OVERALL QUALITY:[/bold yellow]", end=" ")

        # Simple scoring
        coverage_score = min(coverage['coverage_pct'] / 100, 1.0)
        diversity_score = diversity['normalized_entropy']
        avg_tags_score = min(coverage['avg_tags_per_work'] / 7, 1.0)  # Target 7 tags

        overall_score = (coverage_score + diversity_score + avg_tags_score) / 3

        if overall_score >= 0.8:
            self.console.print("[bold green]EXCELLENT ✓[/bold green]")
        elif overall_score >= 0.6:
            self.console.print("[bold yellow]GOOD[/bold yellow]")
        else:
            self.console.print("[bold red]NEEDS IMPROVEMENT[/bold red]")
