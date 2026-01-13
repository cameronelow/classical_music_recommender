"""
Core auto-tagging module for classical music works.

Uses Claude API to intelligently tag works based on metadata alone.
"""

import json
import time
import logging
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from pathlib import Path

import pandas as pd
from anthropic import Anthropic

from .tagging_config import TaggingConfig


logger = logging.getLogger(__name__)


# Tag taxonomy - 7 categories for classical music
TAG_TAXONOMY = {
    "mood": [
        "melancholic", "joyful", "dramatic", "peaceful", "energetic",
        "introspective", "triumphant", "somber", "playful", "majestic",
        "tender", "passionate", "mysterious", "serene", "agitated",
        "tragic", "heroic", "nostalgic", "contemplative", "dark",
        "bright", "gentle", "intense", "spiritual", "romantic",
        "whimsical", "lighthearted", "humorous", "cheerful"
    ],
    "character": [
        "lyrical", "virtuosic", "bold", "delicate", "powerful",
        "intimate", "grand", "light", "expressive", "technical",
        "elegant", "refined", "rustic", "sophisticated", "ornate",
        "simple", "complex", "flowing", "angular", "graceful"
    ],
    "tempo": [
        "very-slow", "slow", "moderate", "fast", "very-fast", "varied-tempo"
    ],
    "instrumentation": [
        "solo-piano", "solo-violin", "solo-cello", "solo-voice",
        "string-quartet", "piano-trio", "string-quintet",
        "chamber-ensemble", "full-orchestra", "piano-and-orchestra",
        "voice-and-piano", "choir", "organ", "solo-winds", "brass-ensemble",
        "mixed-ensemble", "orchestral-strings", "woodwind-ensemble"
    ],
    "form": [
        "sonata-form", "theme-and-variations", "rondo", "fugue",
        "suite", "free-form", "single-movement", "multi-movement",
        "concerto-form", "symphony", "dance-form", "prelude", "etude"
    ],
    "complexity": [
        "beginner-friendly", "intermediate", "advanced", "virtuosic"
    ],
    "popularity": [
        "famous", "well-known", "repertoire-staple", "lesser-known", "obscure"
    ],
    "period": [
        "baroque", "classical", "romantic", "modern", "contemporary",
        "renaissance", "early-music", "20th-century"
    ]
}


class ClassicalMusicAutoTagger:
    """
    Auto-tagger for classical music works using LLM inference.

    Uses Claude to intelligently infer tags from work metadata including
    title, composer, period, work type, key, and catalog number.
    """

    def __init__(self, config: Optional[TaggingConfig] = None):
        """
        Initialize the auto-tagger.

        Args:
            config: Configuration object. If None, uses default config.
        """
        self.config = config or TaggingConfig()
        self.client = Anthropic(api_key=self.config.anthropic_api_key)

        # Token tracking
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_requests = 0
        self.failed_requests = 0

        # Load composers for enhanced context
        self.composers_df = self._load_composers()

    def _load_composers(self) -> Optional[pd.DataFrame]:
        """Load composer data if available."""
        try:
            if self.config.composers_file and self.config.composers_file.exists():
                return pd.read_csv(self.config.composers_file)
        except Exception as e:
            logger.warning(f"Could not load composers file: {e}")
        return None

    def _get_composer_info(self, composer_id: str) -> Optional[Dict]:
        """Get composer information from loaded data."""
        if self.composers_df is not None and composer_id and pd.notna(composer_id):
            composer_row = self.composers_df[self.composers_df['composer_id'] == composer_id]
            if not composer_row.empty:
                return composer_row.iloc[0].to_dict()
        return None

    def _build_tagging_prompt(self, work: Dict) -> str:
        """
        Build the prompt for Claude to tag a work.

        Args:
            work: Dictionary with work metadata

        Returns:
            Prompt string for Claude
        """
        # Extract work metadata
        title = work.get('title', 'Unknown Title')
        composer_id = work.get('composer_id')
        work_type = work.get('work_type', '')
        key = work.get('key', '')
        catalog_number = work.get('catalog_number', '')

        # Try to get composer info
        composer_info = self._get_composer_info(composer_id) if composer_id else None
        composer_name = composer_info.get('name', 'Unknown Composer') if composer_info else 'Unknown Composer'
        period = composer_info.get('period', '') if composer_info else ''
        birth_year = composer_info.get('birth_year') if composer_info else None
        death_year = composer_info.get('death_year') if composer_info else None

        # Build context
        context_parts = [f"Title: {title}"]

        if composer_name != 'Unknown Composer':
            composer_line = f"Composer: {composer_name}"
            if period:
                composer_line += f" ({period}"
                if birth_year and death_year:
                    composer_line += f", {birth_year}-{death_year}"
                elif birth_year:
                    composer_line += f", b. {birth_year}"
                composer_line += ")"
            context_parts.append(composer_line)

        if work_type:
            context_parts.append(f"Type: {work_type}")
        if key:
            context_parts.append(f"Key: {key}")
        if catalog_number:
            context_parts.append(f"Catalog: {catalog_number}")

        work_context = "\n".join(context_parts)

        # Build taxonomy reference
        taxonomy_str = json.dumps(TAG_TAXONOMY, indent=2)

        prompt = f"""You are an expert classical music scholar. Your task is to tag a classical music work with descriptive tags from a predefined taxonomy.

Work Information:
{work_context}

Tag Taxonomy (choose ONLY from these tags):
{taxonomy_str}

Instructions:
1. Select 5-10 tags that best describe this work
2. Consider the composer's style, the historical period, and typical characteristics of this work type
3. For well-known works, use your knowledge of their actual characteristics
4. For lesser-known works, infer from the composer, period, and work type
5. Choose tags that would help users discover similar works

Guidelines:
- MOOD: What emotional character does the work convey?
  * For SCHERZOS (means "joke" in Italian): Usually playful, lighthearted, or energetic unless explicitly dark
  * For HUMORESQUES/HUMORESKE: Always humorous and playful
  * For CHILDREN'S music: Typically lighthearted and playful
  * For MINUETS, POLKAS, GALOPADES: Generally playful and elegant
  * For WALTZES in major keys: Often cheerful and lighthearted
- CHARACTER: What is the musical personality/style?
- TEMPO: What is the typical tempo (or tempo variation)?
- INSTRUMENTATION: What instruments/ensemble is required?
- FORM: What musical form/structure does it follow?
- COMPLEXITY: What technical level is required to perform it?
- POPULARITY: How well-known is this work?
- PERIOD: What historical period or style does it belong to?

Return ONLY valid JSON in this exact format (no markdown, no explanations):
{{"tags": ["tag1", "tag2", "tag3", ...]}}

Ensure all tags are from the taxonomy provided above."""

        return prompt

    def tag_work(
        self,
        work: Dict,
        retry_count: int = 0
    ) -> Tuple[List[str], Optional[str]]:
        """
        Tag a single work using Claude API.

        Args:
            work: Dictionary with work metadata (title, composer_id, work_type, key, etc.)
            retry_count: Current retry attempt (internal use)

        Returns:
            Tuple of (list of tags, error message if failed)
        """
        try:
            prompt = self._build_tagging_prompt(work)

            # Call Claude API
            response = self.client.messages.create(
                model=self.config.model,
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )

            # Track token usage
            self.total_input_tokens += response.usage.input_tokens
            self.total_output_tokens += response.usage.output_tokens
            self.total_requests += 1

            # Parse response
            response_text = response.content[0].text.strip()

            # Remove markdown code blocks if present
            if response_text.startswith("```json"):
                response_text = response_text.replace("```json", "", 1)
            if response_text.startswith("```"):
                response_text = response_text.replace("```", "", 1)
            if response_text.endswith("```"):
                response_text = response_text.rsplit("```", 1)[0]

            response_text = response_text.strip()

            # Parse JSON
            try:
                result = json.loads(response_text)
                tags = result.get('tags', [])

                # Validate tags
                tags = self._validate_tags(tags)

                # Enforce tag count limits
                if len(tags) < self.config.min_tags_per_work:
                    logger.warning(
                        f"Work {work.get('work_id')} has only {len(tags)} tags, "
                        f"minimum is {self.config.min_tags_per_work}"
                    )
                if len(tags) > self.config.max_tags_per_work:
                    tags = tags[:self.config.max_tags_per_work]

                return tags, None

            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse JSON response: {response_text[:200]}")
                if retry_count < self.config.retry_attempts:
                    time.sleep(self.config.retry_delay)
                    return self.tag_work(work, retry_count + 1)
                else:
                    self.failed_requests += 1
                    return [], f"JSON parse error: {str(e)}"

        except Exception as e:
            logger.error(f"Error tagging work {work.get('work_id', 'unknown')}: {e}")
            if retry_count < self.config.retry_attempts:
                time.sleep(self.config.retry_delay)
                return self.tag_work(work, retry_count + 1)
            else:
                self.failed_requests += 1
                return [], str(e)

    def _validate_tags(self, tags: List[str]) -> List[str]:
        """
        Validate that tags are in the taxonomy.

        Args:
            tags: List of tag strings

        Returns:
            List of valid tags only
        """
        valid_tags = []
        all_valid_tags = set()
        for category_tags in TAG_TAXONOMY.values():
            all_valid_tags.update(category_tags)

        for tag in tags:
            if tag in all_valid_tags:
                valid_tags.append(tag)
            else:
                logger.warning(f"Invalid tag '{tag}' not in taxonomy, skipping")

        return valid_tags

    def tag_batch(
        self,
        works: List[Dict],
        batch_size: Optional[int] = None,
        save_progress: bool = True
    ) -> pd.DataFrame:
        """
        Tag multiple works with progress tracking and rate limiting.

        Args:
            works: List of work dictionaries
            batch_size: Number of works to process (None = all)
            save_progress: Whether to save checkpoints

        Returns:
            DataFrame with columns: work_id, tag, source, confidence
        """
        # Fix: If batch_size is explicitly None, process all works
        # Otherwise use provided batch_size or fall back to config default
        if batch_size is None:
            works_to_process = works
        else:
            effective_batch_size = batch_size if batch_size > 0 else self.config.batch_size
            works_to_process = works[:effective_batch_size]

        logger.info(f"Starting batch tagging of {len(works_to_process)} works")

        results = []
        checkpoint_data = self._load_checkpoint()
        processed_ids = set(checkpoint_data.get('processed_work_ids', []))

        for i, work in enumerate(works_to_process):
            work_id = work.get('work_id')

            # Skip already processed works
            if work_id in processed_ids:
                logger.debug(f"Skipping already processed work: {work_id}")
                continue

            logger.info(f"Tagging work {i + 1}/{len(works_to_process)}: {work.get('title', 'Unknown')}")

            # Tag the work
            tags, error = self.tag_work(work)

            if error:
                logger.error(f"Failed to tag work {work_id}: {error}")
                continue

            # Add to results
            for tag in tags:
                results.append({
                    'work_id': work_id,
                    'tag': tag,
                    'source': 'auto-tagger',
                    'confidence': 1.0,  # Default confidence
                    'created_at': datetime.now().isoformat()
                })

            # Fix: Only mark as processed if we successfully generated tags
            # This allows failed works to be retried in the next run
            if tags:
                processed_ids.add(work_id)
                logger.debug(f"Successfully tagged work {work_id} with {len(tags)} tags")
            else:
                logger.warning(f"No tags generated for work {work_id}, will retry in next run")

            # Save checkpoint periodically
            if save_progress and (i + 1) % self.config.save_checkpoint_every == 0:
                self._save_checkpoint({
                    'processed_work_ids': list(processed_ids),
                    'total_works': len(works_to_process),
                    'timestamp': datetime.now().isoformat()
                })
                logger.info(f"Checkpoint saved at {i + 1} works")

            # Rate limiting
            if i < len(works_to_process) - 1:  # Don't delay after last work
                time.sleep(self.config.delay_between_requests)

        # Calculate statistics
        num_attempted = len([w for w in works_to_process if w.get('work_id') not in checkpoint_data.get('processed_work_ids', [])])
        num_successful = len(processed_ids) - len(checkpoint_data.get('processed_work_ids', []))
        num_failed = num_attempted - num_successful

        # Final checkpoint
        if save_progress:
            self._save_checkpoint({
                'processed_work_ids': list(processed_ids),
                'total_works': len(works_to_process),
                'completed': True,
                'timestamp': datetime.now().isoformat(),
                'stats': {
                    'attempted': num_attempted,
                    'successful': num_successful,
                    'failed': num_failed,
                    'success_rate': num_successful / num_attempted if num_attempted > 0 else 0
                }
            })

        logger.info(f"Batch tagging complete. Tagged {len(results)} tag associations")
        logger.info(f"Success rate: {num_successful}/{num_attempted} works ({num_successful/num_attempted*100:.1f}%)" if num_attempted > 0 else "No new works processed")
        if num_failed > 0:
            logger.warning(f"{num_failed} works failed to generate tags and will be retried in next run")

        return pd.DataFrame(results)

    def _load_checkpoint(self) -> Dict:
        """Load checkpoint data if it exists."""
        try:
            if self.config.checkpoint_file.exists():
                with open(self.config.checkpoint_file, 'r') as f:
                    return json.load(f)
        except Exception as e:
            logger.warning(f"Could not load checkpoint: {e}")
        return {}

    def _save_checkpoint(self, data: Dict):
        """Save checkpoint data."""
        try:
            with open(self.config.checkpoint_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"Could not save checkpoint: {e}")

    def clear_checkpoint(self):
        """Clear saved checkpoint data."""
        try:
            if self.config.checkpoint_file.exists():
                self.config.checkpoint_file.unlink()
                logger.info("Checkpoint cleared")
        except Exception as e:
            logger.error(f"Could not clear checkpoint: {e}")

    def estimate_cost(self, num_works: int) -> Dict:
        """
        Estimate API cost before running.

        Args:
            num_works: Number of works to tag

        Returns:
            Dictionary with cost estimates
        """
        estimated_cost = self.config.estimate_total_cost(num_works)
        estimated_time_minutes = (num_works * self.config.delay_between_requests) / 60

        return {
            'total_works': num_works,
            'estimated_tokens': num_works * 700,  # ~500 input + ~200 output
            'estimated_cost_usd': round(estimated_cost, 3),
            'estimated_time_minutes': round(estimated_time_minutes, 1),
            'within_budget': estimated_cost <= self.config.max_cost_usd,
            'max_budget_usd': self.config.max_cost_usd
        }

    def get_usage_stats(self) -> Dict:
        """
        Get current token usage and cost statistics.

        Returns:
            Dictionary with usage stats
        """
        input_cost = (self.total_input_tokens / 1_000_000) * 3.0
        output_cost = (self.total_output_tokens / 1_000_000) * 15.0
        total_cost = input_cost + output_cost

        return {
            'total_requests': self.total_requests,
            'failed_requests': self.failed_requests,
            'success_rate': (
                (self.total_requests - self.failed_requests) / self.total_requests
                if self.total_requests > 0 else 0
            ),
            'total_input_tokens': self.total_input_tokens,
            'total_output_tokens': self.total_output_tokens,
            'total_tokens': self.total_input_tokens + self.total_output_tokens,
            'input_cost_usd': round(input_cost, 3),
            'output_cost_usd': round(output_cost, 3),
            'total_cost_usd': round(total_cost, 3)
        }
