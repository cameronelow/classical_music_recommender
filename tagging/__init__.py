"""
Classical music auto-tagging system with human-in-the-loop review.

This package provides intelligent tagging of classical music works using
LLM inference with human review and correction capabilities.
"""

from .auto_tagger import ClassicalMusicAutoTagger, TAG_TAXONOMY
from .tag_reviewer import TagReviewer
from .tag_learner import TagLearner
from .tag_quality import TagQualityEvaluator
from .tagging_config import TaggingConfig

__all__ = [
    'ClassicalMusicAutoTagger',
    'TAG_TAXONOMY',
    'TagReviewer',
    'TagLearner',
    'TagQualityEvaluator',
    'TaggingConfig',
]
