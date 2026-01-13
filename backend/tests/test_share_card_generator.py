"""
Unit tests for share card generator.
"""

import pytest
from PIL import Image
import io
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from share_card_generator import ShareCardGenerator
from design_tokens import DESIGN_TOKENS


class TestShareCardGenerator:
    """Test suite for ShareCardGenerator class."""

    @pytest.fixture
    def generator(self):
        """Create a generator instance for testing."""
        return ShareCardGenerator()

    @pytest.fixture
    def sample_work(self):
        """Sample work data for testing."""
        return {
            'title': 'Nocturne in E minor, Op. 72, No. 1',
            'composer': 'Frédéric Chopin',
            'period': 'Romantic',
            'work_type': 'Nocturne',
            'instrumentation': 'Piano',
        }

    def test_generator_initialization(self, generator):
        """Test that generator initializes correctly."""
        assert generator.width == DESIGN_TOKENS["dimensions"]["width"]
        assert generator.height == DESIGN_TOKENS["dimensions"]["height"]
        assert generator.design == DESIGN_TOKENS
        assert generator.fonts is not None

    def test_basic_card_generation(self, generator, sample_work):
        """Test basic card generation without album art."""
        image_bytes = generator.generate_card(
            query="Dark Academia",
            work=sample_work
        )

        # Verify it's valid PNG
        assert image_bytes[:8] == b'\x89PNG\r\n\x1a\n'

        # Verify image size
        img = Image.open(io.BytesIO(image_bytes))
        assert img.size == (1200, 630)
        assert img.mode == 'RGB'

    def test_card_with_user_name(self, generator, sample_work):
        """Test card generation with personalized user name."""
        image_bytes = generator.generate_card(
            query="Cozy Study Session",
            work=sample_work,
            user_name="Cameron"
        )

        assert image_bytes is not None
        assert len(image_bytes) > 0

        # Verify it's valid image
        img = Image.open(io.BytesIO(image_bytes))
        assert img.size == (1200, 630)

    def test_card_with_long_title(self, generator):
        """Test card handles long titles correctly."""
        long_work = {
            'title': 'Symphony No. 41 in C major, K. 551, "Jupiter" - Very Long Title That Should Be Truncated',
            'composer': 'Wolfgang Amadeus Mozart',
            'period': 'Classical',
            'work_type': 'Symphony',
        }

        image_bytes = generator.generate_card(
            query="Epic Classical",
            work=long_work
        )

        assert image_bytes is not None
        img = Image.open(io.BytesIO(image_bytes))
        assert img.size == (1200, 630)

    def test_card_with_minimal_work_data(self, generator):
        """Test card generation with minimal work data."""
        minimal_work = {
            'title': 'Some Piece',
            'composer': 'Unknown',
        }

        image_bytes = generator.generate_card(
            query="Mystery Vibe",
            work=minimal_work
        )

        assert image_bytes is not None
        img = Image.open(io.BytesIO(image_bytes))
        assert img.size == (1200, 630)

    def test_truncate_text(self, generator):
        """Test text truncation helper."""
        short_text = "Short"
        assert generator._truncate_text(short_text, 10) == "Short"

        long_text = "This is a very long text that needs truncation"
        truncated = generator._truncate_text(long_text, 20)
        assert len(truncated) <= 20
        assert truncated.endswith("...")

    def test_hex_to_rgb_conversion(self, generator):
        """Test hex to RGB color conversion."""
        from design_tokens import hex_to_rgb

        # Test with hash
        assert hex_to_rgb("#FFFFFF") == (255, 255, 255)
        assert hex_to_rgb("#000000") == (0, 0, 0)
        assert hex_to_rgb("#C4A87C") == (196, 168, 124)

        # Test without hash
        assert hex_to_rgb("FFFFFF") == (255, 255, 255)

    def test_image_to_bytes_format(self, generator, sample_work):
        """Test that generated image is in correct PNG format."""
        image_bytes = generator.generate_card(
            query="Test",
            work=sample_work
        )

        # PNG signature
        assert image_bytes[:8] == b'\x89PNG\r\n\x1a\n'

        # Verify it can be loaded
        img = Image.open(io.BytesIO(image_bytes))
        assert img.format == 'PNG'

    def test_different_instrumentations(self, generator):
        """Test emoji selection for different instrumentations."""
        test_cases = [
            {'instrumentation': 'Piano', 'expected_has': '🎹'},
            {'instrumentation': 'Violin', 'expected_has': '🎻'},
            {'instrumentation': 'Orchestra', 'expected_has': '🎺'},
            {'instrumentation': 'Vocal', 'expected_has': '🎤'},
        ]

        for case in test_cases:
            work = {
                'title': 'Test Piece',
                'composer': 'Test Composer',
                'instrumentation': case['instrumentation'],
            }

            image_bytes = generator.generate_card(
                query="Test",
                work=work
            )

            assert image_bytes is not None
            # Just verify it generates without errors
            img = Image.open(io.BytesIO(image_bytes))
            assert img.size == (1200, 630)

    def test_gradient_background(self, generator):
        """Test that gradient background is applied."""
        canvas = generator._create_canvas()
        generator._add_gradient_background(canvas)

        # Verify canvas is still correct size
        assert canvas.size == (1200, 630)

        # Get pixel colors at top and bottom
        top_pixel = canvas.getpixel((600, 0))
        bottom_pixel = canvas.getpixel((600, 629))

        # Top and bottom should be different (gradient effect)
        assert top_pixel != bottom_pixel

    def test_make_circular_image(self, generator):
        """Test circular image mask creation."""
        # Create a test square image
        test_img = Image.new('RGB', (100, 100), color='red')

        # Make it circular
        circular = generator._make_circular(test_img)

        # Should have alpha channel
        assert circular.mode == 'RGBA'
        assert circular.size == (100, 100)

        # Corners should be transparent
        corner_alpha = circular.getpixel((0, 0))[3]
        assert corner_alpha == 0  # Fully transparent

        # Center should be opaque
        center_alpha = circular.getpixel((50, 50))[3]
        assert center_alpha == 255  # Fully opaque

    def test_custom_design_tokens(self):
        """Test generator with custom design tokens."""
        custom_tokens = DESIGN_TOKENS.copy()
        custom_tokens["colors"]["accent_gold"] = "#FF0000"

        generator = ShareCardGenerator(design_tokens=custom_tokens)

        assert generator.design["colors"]["accent_gold"] == "#FF0000"

    def test_card_file_size(self, generator, sample_work):
        """Test that generated cards are reasonably sized."""
        image_bytes = generator.generate_card(
            query="Test",
            work=sample_work
        )

        # Should be between 50KB and 500KB (reasonable for PNG)
        size_kb = len(image_bytes) / 1024
        assert 50 < size_kb < 500, f"Image size {size_kb}KB is outside expected range"

    @pytest.mark.skip(reason="Requires internet connection")
    def test_card_with_album_art_url(self, generator, sample_work):
        """Test card generation with album art URL."""
        # This test requires internet and a valid image URL
        image_url = "https://via.placeholder.com/300"

        image_bytes = generator.generate_card(
            query="Test",
            work=sample_work,
            album_art_url=image_url
        )

        assert image_bytes is not None
        img = Image.open(io.BytesIO(image_bytes))
        assert img.size == (1200, 630)

    def test_download_image_with_invalid_url(self, generator):
        """Test that invalid album art URLs are handled gracefully."""
        result = generator._download_and_resize_image("invalid-url", 100)
        assert result is None

    def test_multiple_queries(self, generator, sample_work):
        """Test generating cards for multiple queries."""
        queries = [
            "Dark Academia",
            "Cozy Study",
            "Dramatic",
            "Peaceful Morning"
        ]

        for query in queries:
            image_bytes = generator.generate_card(
                query=query,
                work=sample_work
            )

            assert image_bytes is not None
            img = Image.open(io.BytesIO(image_bytes))
            assert img.size == (1200, 630)


class TestDesignTokens:
    """Test design tokens configuration."""

    def test_design_tokens_structure(self):
        """Test that design tokens have required structure."""
        assert "colors" in DESIGN_TOKENS
        assert "typography" in DESIGN_TOKENS
        assert "dimensions" in DESIGN_TOKENS
        assert "branding" in DESIGN_TOKENS

    def test_required_colors(self):
        """Test that all required colors are defined."""
        required_colors = [
            "background_start",
            "background_end",
            "accent_gold",
            "text_primary",
            "text_secondary",
            "card_bg",
            "card_border",
        ]

        for color in required_colors:
            assert color in DESIGN_TOKENS["colors"]

    def test_required_dimensions(self):
        """Test that required dimensions are defined."""
        assert DESIGN_TOKENS["dimensions"]["width"] == 1200
        assert DESIGN_TOKENS["dimensions"]["height"] == 630


# Run tests with: pytest backend/tests/test_share_card_generator.py -v
