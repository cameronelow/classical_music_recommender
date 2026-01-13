"""
Learning system for tag corrections.

Tracks and analyzes human corrections to improve future auto-tagging.
"""

import json
import logging
from typing import Dict, List, Optional
from datetime import datetime
from pathlib import Path
from collections import Counter, defaultdict

import pandas as pd

from .tagging_config import TaggingConfig


logger = logging.getLogger(__name__)


class TagLearner:
    """
    System for learning from human tag corrections.

    Tracks corrections and identifies patterns to improve future tagging.
    """

    def __init__(self, config: Optional[TaggingConfig] = None):
        """
        Initialize the tag learner.

        Args:
            config: Configuration object
        """
        self.config = config or TaggingConfig()

        # Ensure corrections file directory exists
        self.config.corrections_file.parent.mkdir(parents=True, exist_ok=True)

    def record_correction(
        self,
        work: Dict,
        old_tags: List[str],
        new_tags: List[str],
        reason: str,
        reviewer: Optional[str] = None
    ):
        """
        Record a human correction for future learning.

        Args:
            work: Work metadata dictionary
            old_tags: Original auto-generated tags
            new_tags: Corrected tags
            reason: Reason for correction
            reviewer: Optional reviewer identifier
        """
        # Calculate what changed
        old_set = set(old_tags)
        new_set = set(new_tags)

        removed = list(old_set - new_set)
        added = list(new_set - old_set)

        correction = {
            'work_id': work.get('work_id'),
            'title': work.get('title'),
            'composer_id': work.get('composer_id'),
            'work_type': work.get('work_type'),
            'key': work.get('key'),
            'catalog_number': work.get('catalog_number'),
            'auto_tags': old_tags,
            'corrected_tags': new_tags,
            'removed': removed,
            'added': added,
            'reason': reason,
            'reviewer': reviewer or 'unknown',
            'timestamp': datetime.now().isoformat()
        }

        # Append to JSONL file
        try:
            with open(self.config.corrections_file, 'a') as f:
                f.write(json.dumps(correction) + '\n')
            logger.info(f"Recorded correction for work {work.get('work_id')}")
        except Exception as e:
            logger.error(f"Failed to record correction: {e}")

    def load_corrections(self) -> List[Dict]:
        """
        Load all recorded corrections.

        Returns:
            List of correction dictionaries
        """
        corrections = []

        if not self.config.corrections_file.exists():
            return corrections

        try:
            with open(self.config.corrections_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        corrections.append(json.loads(line))
        except Exception as e:
            logger.error(f"Failed to load corrections: {e}")

        return corrections

    def analyze_corrections(self) -> Dict:
        """
        Analyze patterns in corrections.

        Returns:
            Dictionary with analysis results including:
            - most_rejected_tags: Tags frequently removed
            - most_added_tags: Tags frequently added
            - common_patterns: Recurring correction patterns
            - suggestions: Actionable improvement suggestions
        """
        corrections = self.load_corrections()

        if not corrections:
            return {
                'total_corrections': 0,
                'most_rejected_tags': [],
                'most_added_tags': [],
                'common_patterns': [],
                'suggestions': []
            }

        # Count removed and added tags
        removed_counter = Counter()
        added_counter = Counter()

        # Track patterns by work type, composer, etc.
        patterns_by_work_type = defaultdict(lambda: {'removed': Counter(), 'added': Counter()})
        patterns_by_composer = defaultdict(lambda: {'removed': Counter(), 'added': Counter()})

        for correction in corrections:
            # Overall counts
            removed_counter.update(correction.get('removed', []))
            added_counter.update(correction.get('added', []))

            # By work type
            work_type = correction.get('work_type', 'unknown')
            patterns_by_work_type[work_type]['removed'].update(correction.get('removed', []))
            patterns_by_work_type[work_type]['added'].update(correction.get('added', []))

            # By composer
            composer = correction.get('composer_id', 'unknown')
            patterns_by_composer[composer]['removed'].update(correction.get('removed', []))
            patterns_by_composer[composer]['added'].update(correction.get('added', []))

        # Generate insights
        most_rejected = removed_counter.most_common(10)
        most_added = added_counter.most_common(10)

        # Find common patterns
        common_patterns = []

        # Pattern: Specific work types need specific tags
        for work_type, pattern_data in patterns_by_work_type.items():
            if work_type == 'unknown':
                continue

            added = pattern_data['added'].most_common(3)
            if added and added[0][1] >= 3:  # At least 3 occurrences
                common_patterns.append({
                    'type': 'work_type_missing_tag',
                    'work_type': work_type,
                    'tag': added[0][0],
                    'frequency': added[0][1],
                    'description': f"{work_type} works often need '{added[0][0]}' tag"
                })

        # Pattern: Tags that are always removed
        for tag, count in most_rejected:
            if count >= 5:  # Removed at least 5 times
                common_patterns.append({
                    'type': 'frequently_incorrect_tag',
                    'tag': tag,
                    'frequency': count,
                    'description': f"Tag '{tag}' is frequently incorrect (removed {count} times)"
                })

        # Generate suggestions
        suggestions = self._generate_suggestions(
            most_rejected,
            most_added,
            common_patterns
        )

        return {
            'total_corrections': len(corrections),
            'most_rejected_tags': most_rejected,
            'most_added_tags': most_added,
            'common_patterns': common_patterns,
            'suggestions': suggestions,
            'patterns_by_work_type': dict(patterns_by_work_type),
            'patterns_by_composer': dict(patterns_by_composer)
        }

    def _generate_suggestions(
        self,
        most_rejected: List[tuple],
        most_added: List[tuple],
        patterns: List[Dict]
    ) -> List[str]:
        """
        Generate actionable suggestions based on analysis.

        Args:
            most_rejected: Most frequently removed tags
            most_added: Most frequently added tags
            patterns: Common correction patterns

        Returns:
            List of suggestion strings
        """
        suggestions = []

        # Suggestion for frequently rejected tags
        if most_rejected and most_rejected[0][1] >= 5:
            tag, count = most_rejected[0]
            suggestions.append(
                f"Consider updating the prompt to be more careful about using '{tag}' "
                f"(removed {count} times)"
            )

        # Suggestion for frequently added tags
        if most_added and most_added[0][1] >= 5:
            tag, count = most_added[0]
            suggestions.append(
                f"The model often misses '{tag}' - consider adding examples to the prompt "
                f"(manually added {count} times)"
            )

        # Suggestions from patterns
        for pattern in patterns[:3]:  # Top 3 patterns
            if pattern['type'] == 'work_type_missing_tag':
                suggestions.append(
                    f"Add rule: {pattern['work_type']} works should usually include "
                    f"'{pattern['tag']}' tag"
                )

        if not suggestions:
            suggestions.append("Not enough corrections yet to generate specific suggestions")

        return suggestions

    def get_correction_stats(self) -> Dict:
        """
        Get statistics about corrections.

        Returns:
            Dictionary with correction statistics
        """
        corrections = self.load_corrections()

        if not corrections:
            return {
                'total_corrections': 0,
                'avg_tags_removed': 0,
                'avg_tags_added': 0,
                'total_tags_removed': 0,
                'total_tags_added': 0
            }

        total_removed = sum(len(c.get('removed', [])) for c in corrections)
        total_added = sum(len(c.get('added', [])) for c in corrections)

        return {
            'total_corrections': len(corrections),
            'avg_tags_removed': round(total_removed / len(corrections), 2),
            'avg_tags_added': round(total_added / len(corrections), 2),
            'total_tags_removed': total_removed,
            'total_tags_added': total_added,
            'first_correction': corrections[0].get('timestamp') if corrections else None,
            'last_correction': corrections[-1].get('timestamp') if corrections else None
        }

    def export_corrections_to_csv(self, output_file: Path):
        """
        Export corrections to CSV for analysis.

        Args:
            output_file: Path to output CSV file
        """
        corrections = self.load_corrections()

        if not corrections:
            logger.warning("No corrections to export")
            return

        # Flatten corrections for CSV
        rows = []
        for c in corrections:
            rows.append({
                'work_id': c.get('work_id'),
                'title': c.get('title'),
                'composer_id': c.get('composer_id'),
                'work_type': c.get('work_type'),
                'removed_tags': ', '.join(c.get('removed', [])),
                'added_tags': ', '.join(c.get('added', [])),
                'reason': c.get('reason'),
                'reviewer': c.get('reviewer'),
                'timestamp': c.get('timestamp')
            })

        df = pd.DataFrame(rows)
        df.to_csv(output_file, index=False)
        logger.info(f"Exported {len(rows)} corrections to {output_file}")

    def clear_corrections(self, force: bool = False):
        """
        Clear all recorded corrections (use with caution!).

        Args:
            force: If True, skip confirmation prompt
        """
        if not self.config.corrections_file.exists():
            logger.info("No corrections file to delete")
            return

        if force:
            self.config.corrections_file.unlink()
            logger.info("Corrections file deleted")
        else:
            # For interactive use, would need to import Confirm from rich
            logger.warning("Use force=True to delete corrections file")
            logger.info("File not deleted - use force=True parameter")
