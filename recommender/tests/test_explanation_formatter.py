#!/usr/bin/env python3
"""
Unit tests for ExplanationFormatter.

Tests natural language generation for recommendation explanations.
"""

import unittest
from recommender.explanation_formatter import ExplanationFormatter


class TestExplanationFormatter(unittest.TestCase):
    """Test cases for ExplanationFormatter class."""

    def setUp(self):
        """Set up test fixtures."""
        self.formatter = ExplanationFormatter()

    def test_join_with_conjunctions_empty(self):
        """Test joining empty list."""
        result = self.formatter._join_with_conjunctions([])
        self.assertEqual(result, "")

    def test_join_with_conjunctions_single(self):
        """Test joining single item."""
        result = self.formatter._join_with_conjunctions(['sad'])
        self.assertEqual(result, "sad")

    def test_join_with_conjunctions_two_items(self):
        """Test joining two items."""
        result = self.formatter._join_with_conjunctions(['sad', 'lonely'])
        self.assertEqual(result, "sad and lonely")

    def test_join_with_conjunctions_three_items(self):
        """Test joining three items with Oxford comma."""
        result = self.formatter._join_with_conjunctions(['sad', 'lonely', 'dark'])
        self.assertEqual(result, "sad, lonely, and dark")

    def test_join_with_conjunctions_four_items(self):
        """Test joining four items with Oxford comma."""
        result = self.formatter._join_with_conjunctions(['sad', 'lonely', 'dark', 'brooding'])
        self.assertEqual(result, "sad, lonely, dark, and brooding")

    def test_parse_reason_components_key_with_moods(self):
        """Test parsing key with moods in parentheses."""
        reasons = ["E minor (sad, lonely)"]
        components = self.formatter._parse_reason_components(reasons)

        self.assertIn("E minor", components['keys'])
        self.assertIn("sad", components['moods'])
        self.assertIn("lonely", components['moods'])

    def test_parse_reason_components_standalone_key(self):
        """Test parsing standalone key."""
        reasons = ["C major"]
        components = self.formatter._parse_reason_components(reasons)

        self.assertIn("C major", components['keys'])
        self.assertEqual(len(components['moods']), 0)

    def test_parse_reason_components_tags(self):
        """Test parsing tags."""
        reasons = ["tags: dramatic, peaceful"]
        components = self.formatter._parse_reason_components(reasons)

        self.assertIn("dramatic", components['tags'])
        self.assertIn("peaceful", components['tags'])

    def test_parse_reason_components_period(self):
        """Test parsing period."""
        reasons = ["Romantic period"]
        components = self.formatter._parse_reason_components(reasons)

        self.assertEqual(components['period'], "Romantic")

    def test_parse_reason_components_era(self):
        """Test parsing era."""
        reasons = ["Baroque era"]
        components = self.formatter._parse_reason_components(reasons)

        self.assertEqual(components['period'], "Baroque")

    def test_parse_reason_components_mixed(self):
        """Test parsing mixed reasons."""
        reasons = [
            "E minor (melancholic, contemplative)",
            "tags: dramatic, lyrical",
            "Romantic era"
        ]
        components = self.formatter._parse_reason_components(reasons)

        self.assertIn("E minor", components['keys'])
        self.assertIn("melancholic", components['moods'])
        self.assertIn("contemplative", components['moods'])
        self.assertIn("dramatic", components['tags'])
        self.assertIn("lyrical", components['tags'])
        self.assertEqual(components['period'], "Romantic")

    def test_parse_similarity_reasons_same_composer(self):
        """Test parsing same composer reason."""
        reasons = ["same composer"]
        components = self.formatter._parse_similarity_reasons(reasons, None)

        self.assertTrue(components['same_composer'])

    def test_parse_similarity_reasons_same_work_type(self):
        """Test parsing same work type reason."""
        reasons = ["both symphonies"]
        components = self.formatter._parse_similarity_reasons(reasons, None)

        self.assertEqual(components['same_work_type'], "symphonie")  # s removed

    def test_parse_similarity_reasons_same_key(self):
        """Test parsing same key reason."""
        reasons = ["both in E minor"]
        components = self.formatter._parse_similarity_reasons(reasons, None)

        self.assertEqual(components['same_key'], "E minor")

    def test_parse_similarity_reasons_same_period(self):
        """Test parsing same period reason."""
        reasons = ["both Romantic"]
        components = self.formatter._parse_similarity_reasons(reasons, None)

        self.assertEqual(components['same_period'], "Romantic")

    def test_parse_similarity_reasons_similar_style(self):
        """Test parsing similar style reason."""
        reasons = ["similar style (lyrical)"]
        components = self.formatter._parse_similarity_reasons(reasons, None)

        self.assertEqual(components['similar_style'], "lyrical")

    def test_semantic_explanation_with_key_and_moods(self):
        """Test semantic explanation with key and moods."""
        reasons = ["E minor (melancholic, contemplative)"]
        work = {"work_type": "nocturne"}

        result = self.formatter.format_semantic_explanation(
            reasons=reasons,
            work=work,
            query="moody and dark",
            is_exact_match=True
        )

        # Should be natural and conversational
        self.assertNotIn(";", result)  # No semicolons
        self.assertNotIn("This piece was recommended", result)  # No verbose preamble
        self.assertIn("minor", result.lower())

    def test_semantic_explanation_with_period_and_work_type(self):
        """Test semantic explanation with full context."""
        reasons = [
            "melancholic and contemplative (E minor)",
            "dramatic",
            "lyrical",
            "Romantic era"
        ]
        work = {"work_type": "nocturne"}

        result = self.formatter.format_semantic_explanation(
            reasons=reasons,
            work=work,
            query="moody",
            is_exact_match=False
        )

        # Should be a natural sentence
        self.assertNotIn(";", result)
        self.assertNotIn("tags:", result)
        # Should mention key characteristics
        self.assertTrue(
            any(word in result.lower() for word in ['romantic', 'nocturne', 'minor'])
        )

    def test_semantic_explanation_empty_reasons(self):
        """Test semantic explanation with no reasons."""
        work = {"work_type": "symphony"}

        result = self.formatter.format_semantic_explanation(
            reasons=[],
            work=work,
            query="energetic",
            is_exact_match=False
        )

        # Should provide a fallback
        self.assertIsInstance(result, str)
        self.assertGreater(len(result), 0)
        self.assertNotIn(";", result)

    def test_similarity_explanation_same_composer(self):
        """Test similarity explanation with same composer."""
        reasons = ["same composer", "both symphonies", "both in E minor"]

        result = self.formatter.format_similarity_explanation(
            reasons=reasons,
            seed_work=None,
            rec_work=None
        )

        # Should be natural
        self.assertNotIn(";", result)
        self.assertIn("composer", result.lower())
        # Should mention shared attributes
        self.assertTrue(
            any(word in result.lower() for word in ['same', 'like', 'shares'])
        )

    def test_similarity_explanation_shared_attributes(self):
        """Test similarity explanation with shared attributes only."""
        reasons = ["both symphonies", "both in E minor", "both Romantic"]

        result = self.formatter.format_similarity_explanation(
            reasons=reasons,
            seed_work=None,
            rec_work=None
        )

        # Should list shared characteristics naturally
        self.assertNotIn(";", result)
        self.assertNotIn("Recommended because:", result)  # No old format

    def test_similarity_explanation_similar_style(self):
        """Test similarity explanation with similar style."""
        reasons = ["similar style (lyrical)"]

        result = self.formatter.format_similarity_explanation(
            reasons=reasons,
            seed_work=None,
            rec_work=None
        )

        # Should mention the style
        self.assertIn("lyrical", result.lower())
        self.assertNotIn(";", result)

    def test_similarity_explanation_empty_reasons(self):
        """Test similarity explanation with no reasons."""
        result = self.formatter.format_similarity_explanation(
            reasons=[],
            seed_work=None,
            rec_work=None
        )

        # Should provide a meaningful fallback
        self.assertIsInstance(result, str)
        self.assertGreater(len(result), 0)
        self.assertIn("similar", result.lower())

    def test_no_semicolons_in_any_output(self):
        """Test that no explanations contain semicolons."""
        test_cases = [
            {
                'reasons': ["E minor (sad, lonely)", "dramatic", "lyrical", "Romantic era"],
                'work': {"work_type": "nocturne"},
                'query': "moody",
                'is_exact': False
            },
            {
                'reasons': ["C major (bright, cheerful)", "tags: energetic, playful"],
                'work': {"work_type": "concerto"},
                'query': "happy",
                'is_exact': True
            },
            {
                'reasons': ["melancholic and contemplative (D minor)", "peaceful"],
                'work': {"work_type": "prelude"},
                'query': "calm",
                'is_exact': False
            }
        ]

        for case in test_cases:
            result = self.formatter.format_semantic_explanation(
                reasons=case['reasons'],
                work=case['work'],
                query=case['query'],
                is_exact_match=case['is_exact']
            )
            self.assertNotIn(";", result, f"Semicolon found in: {result}")

    def test_no_technical_labels_in_output(self):
        """Test that technical labels like 'tags:' don't appear in output."""
        reasons = ["tags: dramatic, peaceful", "Romantic era"]
        work = {"work_type": "symphony"}

        result = self.formatter.format_semantic_explanation(
            reasons=reasons,
            work=work,
            query="peaceful",
            is_exact_match=True
        )

        self.assertNotIn("tags:", result)
        self.assertNotIn("Tags:", result)

    def test_grammatical_correctness(self):
        """Test that outputs are grammatically correct."""
        reasons = ["E minor (melancholic, contemplative)", "dramatic", "Romantic era"]
        work = {"work_type": "nocturne"}

        result = self.formatter.format_semantic_explanation(
            reasons=reasons,
            work=work,
            query="moody",
            is_exact_match=False
        )

        # Should start with capital letter
        self.assertTrue(result[0].isupper())

        # Should not have "it's:" followed by semicolons
        self.assertNotIn("it's:", result.lower())
        self.assertNotIn("it is:", result.lower())


if __name__ == '__main__':
    unittest.main()
