#!/usr/bin/env python3
"""
Formats recommendation explanations in natural, conversational language.

This module provides the ExplanationFormatter class which transforms technical
recommendation data into human-readable, grammatically correct explanations.
"""

from typing import List, Dict, Any, Optional
import re


class ExplanationFormatter:
    """Formats recommendation explanations in natural, conversational language."""

    def format_semantic_explanation(
        self,
        reasons: List[str],
        work: Dict[str, Any],
        query: str,
        is_exact_match: bool
    ) -> str:
        """Format explanation for semantic search results.

        Args:
            reasons: List of reason strings from the recommender
            work: Work data dictionary
            query: Original search query
            is_exact_match: Whether reasons contain exact query matches

        Returns:
            Natural language explanation string
        """
        if not reasons:
            # Fallback when no reasons provided
            work_type = work.get('work_type', 'piece')
            return f"A {work_type} that matches the mood you're looking for"

        # Parse reasons into structured components
        components = self._parse_reason_components(reasons)

        # Build explanation using natural sentence structure
        return self._build_semantic_sentence(components, work, is_exact_match)

    def format_similarity_explanation(
        self,
        reasons: List[str],
        seed_work: Any,
        rec_work: Any
    ) -> str:
        """Format explanation for similarity-based recommendations.

        Args:
            reasons: List of reason strings (e.g., "same composer", "both symphonies")
            seed_work: Original seed work
            rec_work: Recommended work

        Returns:
            Natural language explanation string
        """
        if not reasons:
            return "Recommended based on overall musical similarity"

        # Parse similarity-based reasons
        components = self._parse_similarity_reasons(reasons, rec_work)

        # Build natural sentence
        return self._build_similarity_sentence(components, rec_work)

    def _parse_reason_components(self, reasons: List[str]) -> Dict[str, List[str]]:
        """Parse reasons into structured categories.

        Args:
            reasons: Raw reason strings

        Returns:
            Dictionary with keys: 'keys', 'moods', 'tags', 'period', 'work_type'
        """
        components = {
            'keys': [],
            'moods': [],
            'tags': [],
            'period': None,
            'work_type': None
        }

        for reason in reasons:
            # Extract key and moods: "E minor (sad, lonely)"
            key_mood_match = re.match(r'([A-G][#b]? (?:major|minor))\s*\(([^)]+)\)', reason)
            if key_mood_match:
                key = key_mood_match.group(1)
                moods_str = key_mood_match.group(2)
                moods = [m.strip() for m in moods_str.split(',')]
                components['keys'].append(key)
                components['moods'].extend(moods)
                continue

            # Extract standalone key: "E minor"
            if re.match(r'^[A-G][#b]? (?:major|minor)$', reason):
                components['keys'].append(reason)
                continue

            # Extract tags: "tags: dramatic, peaceful"
            if reason.startswith('tags:'):
                tags_str = reason[5:].strip()
                tags = [t.strip() for t in tags_str.split(',')]
                components['tags'].extend(tags)
                continue

            # Extract period: "Romantic period" or "Romantic era"
            period_match = re.match(r'(.+?)\s+(?:period|era)$', reason)
            if period_match:
                components['period'] = period_match.group(1)
                continue

            # Otherwise, treat as a tag/characteristic
            components['tags'].append(reason)

        return components

    def _parse_similarity_reasons(
        self,
        reasons: List[str],
        rec_work: Any
    ) -> Dict[str, Any]:
        """Parse similarity-based reasons into structured data.

        Args:
            reasons: Raw reason strings
            rec_work: Recommended work

        Returns:
            Dictionary with parsed components
        """
        components = {
            'same_composer': False,
            'same_work_type': None,
            'same_key': None,
            'same_period': None,
            'similar_style': None
        }

        for reason in reasons:
            if reason == 'same composer':
                components['same_composer'] = True
            elif reason.startswith('both ') and 's' in reason:
                # "both symphonies" -> extract work type
                work_type = reason[5:].rstrip('s')  # Remove "both " and trailing "s"
                components['same_work_type'] = work_type
            elif reason.startswith('both in '):
                # "both in E minor"
                key = reason[8:]
                components['same_key'] = key
            elif reason.startswith('both ') and reason != 'both ':
                # "both Romantic"
                period = reason[5:]
                components['same_period'] = period
            elif reason.startswith('similar style ('):
                # "similar style (lyrical)"
                style_match = re.match(r'similar style \(([^)]+)\)', reason)
                if style_match:
                    components['similar_style'] = style_match.group(1)

        return components

    def _build_semantic_sentence(
        self,
        components: Dict[str, List[str]],
        work: Dict[str, Any],
        is_exact_match: bool
    ) -> str:
        """Build a natural sentence from semantic search components.

        Args:
            components: Parsed reason components
            work: Work data
            is_exact_match: Whether this is an exact match

        Returns:
            Natural language sentence
        """
        parts = []

        # Get key and moods
        key = components['keys'][0] if components['keys'] else None
        moods = components['moods'][:2]  # Limit to 2 moods
        tags = components['tags'][:3]  # Limit to 3 tags
        period = components['period']
        work_type = components['work_type'] or work.get('work_type')

        # Build mood + key phrase if available
        mood_key_phrase = None
        if moods and key:
            mood_str = self._join_with_conjunctions(moods)
            mood_key_phrase = f"{mood_str} in {key}"
        elif moods:
            mood_str = self._join_with_conjunctions(moods)
            mood_key_phrase = mood_str
        elif key:
            mood_key_phrase = f"in {key}"

        # Choose sentence structure based on available components
        if period and work_type and mood_key_phrase:
            # Full context: "A Romantic nocturne that's melancholic and contemplative in E minor"
            article = "An" if work_type[0].lower() in 'aeiou' else "A"
            sentence = f"{article} {period} {work_type} that's {mood_key_phrase}"

            # Add tags if available
            if tags:
                tags_str = self._join_with_conjunctions(tags)
                sentence += f" with {tags_str} character"

            return sentence

        elif mood_key_phrase:
            # Start with mood/key
            if key and moods:
                # "Melancholic and contemplative in E minor"
                sentence = mood_key_phrase.capitalize()
            else:
                sentence = mood_key_phrase.capitalize()

            # Add tags
            if tags:
                tags_str = self._join_with_conjunctions(tags)
                sentence += f" with {tags_str} character"
            else:
                sentence = mood_key_phrase.capitalize()

            # Add period/work type context if available
            if period and work_type:
                sentence += f", a {period} {work_type}"
            elif work_type:
                article = "an" if work_type[0].lower() in 'aeiou' else "a"
                sentence += f", {article} {work_type}"

            return sentence

        elif tags:
            # Tags only: "Featuring dramatic and peaceful character"
            tags_str = self._join_with_conjunctions(tags)
            if period and work_type:
                article = "An" if work_type[0].lower() in 'aeiou' else "A"
                return f"{article} {period} {work_type} with {tags_str} character"
            else:
                return f"Featuring {tags_str} character"

        elif work_type:
            # Work type only
            article = "An" if work_type[0].lower() in 'aeiou' else "A"
            if period:
                return f"{article} {period} {work_type} that matches your vibe"
            else:
                return f"{article} {work_type} that matches the mood you're looking for"

        # Final fallback
        return "Recommended based on musical similarity to your search"

    def _build_similarity_sentence(
        self,
        components: Dict[str, Any],
        rec_work: Any
    ) -> str:
        """Build a natural sentence from similarity components.

        Args:
            components: Parsed similarity components
            rec_work: Recommended work

        Returns:
            Natural language sentence
        """
        shared_attrs = []

        # Collect shared attributes
        if components['same_period']:
            shared_attrs.append(components['same_period'])

        if components['same_work_type']:
            shared_attrs.append(components['same_work_type'])

        if components['same_key']:
            shared_attrs.append(f"in {components['same_key']}")

        if components['same_composer']:
            if shared_attrs:
                # "Like your selection, this is a Romantic symphony in E minor by the same composer"
                attrs_str = " ".join(shared_attrs)
                return f"Like your selection, this is a {attrs_str} by the same composer"
            else:
                return "From the same composer as your selection"

        if shared_attrs:
            # "Shares the same key (E minor) and Romantic-era style with your selection"
            if len(shared_attrs) == 1:
                return f"Like your selection, this is also {shared_attrs[0]}"
            else:
                # Build list of shared characteristics
                attrs_str = self._join_with_conjunctions(shared_attrs)
                return f"Shares {attrs_str} with your selection"

        if components['similar_style']:
            return f"Features similar {components['similar_style']} character to your selection"

        return "Recommended based on musical similarity to your selection"

    def _join_with_conjunctions(self, items: List[str]) -> str:
        """Join items with proper English conjunctions.

        Args:
            items: List of strings to join

        Returns:
            Properly joined string

        Examples:
            ['sad'] -> 'sad'
            ['sad', 'lonely'] -> 'sad and lonely'
            ['sad', 'lonely', 'dark'] -> 'sad, lonely, and dark'
        """
        if not items:
            return ""
        if len(items) == 1:
            return items[0]
        if len(items) == 2:
            return f"{items[0]} and {items[1]}"

        # 3 or more items: use Oxford comma
        return ", ".join(items[:-1]) + f", and {items[-1]}"
