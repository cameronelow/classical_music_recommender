"""
LLM-based metadata enrichment for classical music works.

Uses Claude to infer missing work_type and key metadata from titles and context.
Mirrors ClassicalMusicAutoTagger patterns: same rate limiting, retry logic,
checkpointing, and config reuse.
"""

import json
import re
import time
import logging
from typing import Dict, List, Optional, Tuple
from datetime import datetime

import pandas as pd
from anthropic import Anthropic

from .tagging_config import TaggingConfig


logger = logging.getLogger(__name__)


# Valid work types — mirrors WorkTypeParser.WORK_TYPES + KNOWN_TITLES in etl/transform/parsers.py
VALID_WORK_TYPES = sorted({
    "Ballade", "Cantata", "Capriccio", "Choral", "Concertino", "Concerto",
    "Dance", "Divertimento", "Etude", "Fantasia", "Fugue", "Gloria", "Hymn",
    "Impromptu", "Intermezzo", "March", "Mass", "Mazurka", "Motet",
    "Movement", "Nocturne", "Opera", "Oratorio", "Overture", "Passion",
    "Piano Quartet", "Piano Trio", "Polonaise", "Prelude", "Psalm",
    "Quartet", "Requiem", "Rhapsody", "Rondo", "Scherzo", "Serenade",
    "Song", "Sonata", "Stabat Mater", "String Quartet", "String Quintet",
    "Suite", "Symphony", "Te Deum", "Toccata", "Tone Poem", "Trio",
    "Variation", "Waltz",
})

# Regex to validate a key string returned by Claude (e.g. "D minor", "F♯ major", "Bb major")
_KEY_PATTERN = re.compile(
    r'^[A-G][#♯b♭]?\s*(major|minor)$',
    re.IGNORECASE
)


def _normalize_key(key: str) -> Optional[str]:
    """
    Validate and normalize a key string like 'D minor' or 'F# major'.
    Returns the normalized string or None if invalid.
    """
    if not isinstance(key, str):
        return None
    key = key.strip()
    # Normalize ASCII accidentals to unicode
    key = key.replace('#', '♯').replace('b ', '♭ ')
    # Check it matches expected format
    if _KEY_PATTERN.match(key):
        # Capitalize note, lowercase mode
        parts = key.split()
        return f"{parts[0][0].upper()}{parts[0][1:]} {parts[1].lower()}"
    return None


class WorkMetadataEnricher:
    """
    Enriches missing work_type and key fields in works.parquet using Claude.
    """

    CHECKPOINT_FILENAME = "metadata_enrichment_checkpoint.json"

    def __init__(self, config: Optional[TaggingConfig] = None):
        self.config = config or TaggingConfig()
        self.client = Anthropic(api_key=self.config.anthropic_api_key)

        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_requests = 0
        self.failed_requests = 0

        self._checkpoint_file = (
            self.config.checkpoint_file.parent / self.CHECKPOINT_FILENAME
        )

    def _build_prompt(
        self,
        title: str,
        composer_name: str,
        period: str,
        catalog_number: str,
        existing_tags: List[str],
        needs_work_type: bool,
        needs_key: bool,
    ) -> str:
        context_parts = [f"Title: {title}"]

        if isinstance(composer_name, str) and composer_name:
            line = f"Composer: {composer_name}"
            if isinstance(period, str) and period:
                line += f" ({period})"
            context_parts.append(line)

        if isinstance(catalog_number, str) and catalog_number:
            context_parts.append(f"Catalog: {catalog_number}")

        if existing_tags:
            context_parts.append(f"Tags: {', '.join(existing_tags[:10])}")

        context = "\n".join(context_parts)

        fields: List[str] = []
        if needs_work_type:
            types_str = ", ".join(sorted(VALID_WORK_TYPES))
            fields.append(
                f'"work_type": one of [{types_str}], or null if truly unknown'
            )
        if needs_key:
            fields.append(
                '"key": musical key in "X major" or "X minor" format '
                '(e.g. "D minor", "F♯ major", "B♭ major"), or null if truly unknown'
            )

        fields_str = "\n  ".join(fields)

        return (
            "You are an expert classical music scholar. "
            "Given the following work, infer the missing metadata.\n\n"
            f"Work:\n{context}\n\n"
            f"Return ONLY valid JSON with these fields:\n{{\n  {fields_str}\n}}\n\n"
            "Use your knowledge of well-known works. For lesser-known works, "
            "infer from the title, composer, and period. "
            "Return null only if there is genuinely no reasonable inference."
        )

    def enrich_work(
        self,
        title: str,
        composer_name: str,
        period: str,
        catalog_number: str,
        existing_tags: List[str],
        needs_work_type: bool,
        needs_key: bool,
        retry_count: int = 0,
    ) -> Tuple[Optional[str], Optional[str], Optional[str]]:
        """
        Infer missing work_type and/or key for one work.

        Returns:
            (work_type | None, key | None, error | None)
        """
        try:
            prompt = self._build_prompt(
                title, composer_name, period, catalog_number,
                existing_tags, needs_work_type, needs_key,
            )
            response = self.client.messages.create(
                model=self.config.model,
                max_tokens=80,
                temperature=0.1,
                messages=[{"role": "user", "content": prompt}],
            )

            self.total_input_tokens += response.usage.input_tokens
            self.total_output_tokens += response.usage.output_tokens
            self.total_requests += 1

            text = response.content[0].text.strip()
            # Strip markdown fences if present
            if text.startswith("```json"):
                text = text[7:]
            elif text.startswith("```"):
                text = text[3:]
            if text.endswith("```"):
                text = text[:-3]
            text = text.strip()

            result = json.loads(text)

            inferred_type: Optional[str] = None
            inferred_key: Optional[str] = None

            if needs_work_type:
                raw_type = result.get("work_type")
                if isinstance(raw_type, str) and raw_type in VALID_WORK_TYPES:
                    inferred_type = raw_type
                elif isinstance(raw_type, str):
                    logger.warning(f"work_type '{raw_type}' not in vocabulary, discarding")

            if needs_key:
                raw_key = result.get("key")
                if isinstance(raw_key, str):
                    inferred_key = _normalize_key(raw_key)
                    if inferred_key is None:
                        logger.warning(f"key '{raw_key}' failed format validation, discarding")

            return inferred_type, inferred_key, None

        except json.JSONDecodeError as e:
            if retry_count < self.config.retry_attempts:
                time.sleep(self.config.retry_delay)
                return self.enrich_work(
                    title, composer_name, period, catalog_number,
                    existing_tags, needs_work_type, needs_key, retry_count + 1,
                )
            self.failed_requests += 1
            return None, None, f"JSON parse error: {e}"

        except Exception as e:
            logger.error(f"Error enriching '{title}': {e}")
            if retry_count < self.config.retry_attempts:
                time.sleep(self.config.retry_delay)
                return self.enrich_work(
                    title, composer_name, period, catalog_number,
                    existing_tags, needs_work_type, needs_key, retry_count + 1,
                )
            self.failed_requests += 1
            return None, None, str(e)

    def enrich_batch(
        self,
        works_df: pd.DataFrame,
        composers_df: pd.DataFrame,
        tags_df: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """
        Fill in missing work_type and key for all qualifying works.

        Returns an updated copy of works_df.
        """
        # Build composer lookup
        composer_lookup: Dict[str, Dict] = {}
        for _, row in composers_df.iterrows():
            cid = row.get("composer_id")
            if isinstance(cid, str):
                composer_lookup[cid] = row.to_dict()

        # Build tags lookup
        tags_lookup: Dict[str, List[str]] = {}
        if tags_df is not None:
            for work_id, group in tags_df.groupby("work_id"):
                if isinstance(work_id, str):
                    tags_lookup[work_id] = group["tag"].tolist()

        needs_work_type_mask = works_df["work_type"].apply(lambda v: not isinstance(v, str))
        needs_key_mask = works_df["key"].apply(lambda v: not isinstance(v, str))
        to_enrich_idx = works_df[needs_work_type_mask | needs_key_mask].index

        logger.info(
            f"Works missing work_type: {needs_work_type_mask.sum()}, "
            f"missing key: {needs_key_mask.sum()}. "
            f"Total to enrich: {len(to_enrich_idx)}."
        )

        checkpoint = self._load_checkpoint()
        processed_ids = set(checkpoint.get("processed_work_ids", []))

        result_df = works_df.copy()

        for i, idx in enumerate(to_enrich_idx):
            row = result_df.loc[idx]
            work_id = row.get("work_id")

            if not isinstance(work_id, str) or work_id in processed_ids:
                continue

            title = row.get("title") if isinstance(row.get("title"), str) else "Unknown"
            logger.info(f"[{i + 1}/{len(to_enrich_idx)}] {title}")

            composer_id = row.get("composer_id")
            composer_info = (
                composer_lookup.get(composer_id)
                if isinstance(composer_id, str) else None
            )
            composer_name = composer_info.get("name", "") if composer_info else ""
            period = composer_info.get("period", "") if composer_info else ""
            catalog_number = row.get("catalog_number", "")
            existing_tags = tags_lookup.get(work_id, [])

            needs_type = not isinstance(row.get("work_type"), str)
            needs_key = not isinstance(row.get("key"), str)

            inferred_type, inferred_key, error = self.enrich_work(
                title, composer_name, period,
                catalog_number if isinstance(catalog_number, str) else "",
                existing_tags, needs_type, needs_key,
            )

            if error:
                logger.error(f"Failed to enrich {work_id}: {error}")
            else:
                if needs_type and isinstance(inferred_type, str):
                    result_df.at[idx, "work_type"] = inferred_type
                    logger.debug(f"  work_type → {inferred_type}")
                if needs_key and isinstance(inferred_key, str):
                    result_df.at[idx, "key"] = inferred_key
                    logger.debug(f"  key → {inferred_key}")
                processed_ids.add(work_id)

            if (i + 1) % self.config.save_checkpoint_every == 0:
                self._save_checkpoint({"processed_work_ids": list(processed_ids)})
                logger.info(f"Checkpoint saved ({i + 1} processed)")

            if i < len(to_enrich_idx) - 1:
                time.sleep(self.config.delay_between_requests)

        self._save_checkpoint({
            "processed_work_ids": list(processed_ids),
            "completed": True,
            "timestamp": datetime.now().isoformat(),
        })

        return result_df

    def _load_checkpoint(self) -> Dict:
        try:
            if self._checkpoint_file.exists():
                with open(self._checkpoint_file) as f:
                    return json.load(f)
        except Exception as e:
            logger.warning(f"Could not load checkpoint: {e}")
        return {}

    def _save_checkpoint(self, data: Dict) -> None:
        try:
            self._checkpoint_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self._checkpoint_file, "w") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"Could not save checkpoint: {e}")

    def clear_checkpoint(self) -> None:
        try:
            if self._checkpoint_file.exists():
                self._checkpoint_file.unlink()
                logger.info("Enrichment checkpoint cleared")
        except Exception as e:
            logger.error(f"Could not clear checkpoint: {e}")

    def get_usage_stats(self) -> Dict:
        input_cost = (self.total_input_tokens / 1_000_000) * 3.0
        output_cost = (self.total_output_tokens / 1_000_000) * 15.0
        return {
            "total_requests": self.total_requests,
            "failed_requests": self.failed_requests,
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "estimated_cost_usd": round(input_cost + output_cost, 4),
        }
