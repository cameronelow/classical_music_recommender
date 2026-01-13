"""
Configuration for the classical music tagging system.
"""

import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


@dataclass
class TaggingConfig:
    """Configuration for the auto-tagging system."""

    # API settings
    anthropic_api_key: str = field(default_factory=lambda: os.getenv("ANTHROPIC_API_KEY", ""))
    model: str = "claude-sonnet-4-20250514"
    temperature: float = 0.3
    max_tokens: int = 300

    # Rate limiting
    requests_per_minute: int = 60
    delay_between_requests: float = 1.0
    retry_attempts: int = 3
    retry_delay: float = 2.0

    # Batch settings
    batch_size: int = 50
    save_checkpoint_every: int = 10

    # Review settings
    default_sample_size: int = 10
    auto_approve_threshold: float = 0.8
    confidence_threshold: float = 0.7

    # Tag settings
    min_tags_per_work: int = 5
    max_tags_per_work: int = 10

    # File paths
    project_root: Path = field(default_factory=lambda: Path(__file__).parent.parent)
    works_file: Optional[Path] = None
    composers_file: Optional[Path] = None
    existing_tags_file: Optional[Path] = None
    output_file: Optional[Path] = None
    corrections_file: Optional[Path] = None
    checkpoint_file: Optional[Path] = None

    # Cost limits
    max_cost_usd: float = 5.0

    def __post_init__(self):
        """Initialize file paths if not provided."""
        if self.works_file is None:
            self.works_file = self.project_root / "data" / "processed" / "works.csv"

        if self.composers_file is None:
            self.composers_file = self.project_root / "data" / "processed" / "composers.csv"

        if self.existing_tags_file is None:
            self.existing_tags_file = self.project_root / "data" / "processed" / "work_tags.csv"

        if self.output_file is None:
            self.output_file = self.project_root / "data" / "processed" / "work_tags_enhanced.csv"

        if self.corrections_file is None:
            self.corrections_file = self.project_root / "data" / "corrections" / "tag_corrections.jsonl"

        if self.checkpoint_file is None:
            self.checkpoint_file = self.project_root / "data" / "corrections" / "tagging_checkpoint.json"

        # Ensure directories exist
        self.corrections_file.parent.mkdir(parents=True, exist_ok=True)

        # Validate API key
        if not self.anthropic_api_key:
            raise ValueError(
                "ANTHROPIC_API_KEY not found. Please set it in your .env file or environment."
            )

    @property
    def estimated_cost_per_work(self) -> float:
        """
        Estimate cost per work based on Claude Sonnet 4 pricing.

        Assuming ~500 input tokens (prompt + work metadata) and
        ~200 output tokens (JSON response).

        Claude Sonnet 4 pricing (as of 2025):
        - Input: $3 per million tokens
        - Output: $15 per million tokens
        """
        input_tokens = 500
        output_tokens = 200

        input_cost = (input_tokens / 1_000_000) * 3.0
        output_cost = (output_tokens / 1_000_000) * 15.0

        return input_cost + output_cost

    def estimate_total_cost(self, num_works: int) -> float:
        """Estimate total cost for tagging N works."""
        return self.estimated_cost_per_work * num_works

    def validate_cost(self, num_works: int) -> bool:
        """Check if estimated cost is within budget."""
        estimated = self.estimate_total_cost(num_works)
        return estimated <= self.max_cost_usd
