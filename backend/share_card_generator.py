"""
Share Card Generator for Classical Music Recommender.
Generates beautiful 1200x630px social media cards.
"""

from PIL import Image, ImageDraw, ImageFont
from typing import Dict, Optional
import io
import requests
from pathlib import Path
from design_tokens import DESIGN_TOKENS, hex_to_rgb, interpolate_color


class ShareCardGenerator:
    """Generate beautiful share cards for social media."""

    def __init__(self, design_tokens: dict = None):
        """
        Initialize card generator with design system.

        Args:
            design_tokens: Optional custom design tokens (defaults to DESIGN_TOKENS)
        """
        self.design = design_tokens or DESIGN_TOKENS
        self.width = self.design["dimensions"]["width"]
        self.height = self.design["dimensions"]["height"]
        self.fonts = self._load_fonts()

    def generate_card(
        self,
        query: str,
        work: Dict,
        album_art_url: Optional[str] = None,
        user_name: Optional[str] = None
    ) -> bytes:
        """
        Generate share card as PNG bytes.

        Args:
            query: User's search query (e.g., "Dark Academia")
            work: Work dict with title, composer, period, work_type, instrumentation
            album_art_url: Optional URL to album art/composer portrait
            user_name: Optional user name to personalize

        Returns:
            PNG image as bytes
        """
        # Store query and user_name for access in _add_work_card
        self._current_query = query
        self._current_user_name = user_name

        # Create canvas with base background
        img = self._create_canvas()

        # Add gradient background
        self._add_gradient_background(img)

        # Create drawing context
        draw = ImageDraw.Draw(img)

        # Add header section (empty now, content moved to card)
        self._add_header(draw, query, user_name)

        # Add work card with all content (center)
        self._add_work_card(img, draw, work, album_art_url)

        # Add logo (if available)
        self._add_logo(img)

        # Convert to bytes
        return self._image_to_bytes(img)

    def _create_canvas(self) -> Image.Image:
        """Create base canvas with background color."""
        bg_color = hex_to_rgb(self.design["colors"]["background_start"])
        return Image.new('RGB', (self.width, self.height), bg_color)

    def _add_gradient_background(self, img: Image.Image):
        """Add gradient matching app: dark blue 0-60%, then fade to yellow 60-100%."""
        draw = ImageDraw.Draw(img)

        start_color = hex_to_rgb(self.design["colors"]["background_start"])
        mid_color = hex_to_rgb(self.design["colors"]["background_mid"])
        end_color = hex_to_rgb(self.design["colors"]["background_end"])

        for y in range(self.height):
            ratio = y / self.height

            if ratio <= 0.6:
                # First 60%: solid dark blue
                color = mid_color
            else:
                # Last 40%: fade from dark blue to yellow
                # Normalize ratio from 0.6-1.0 to 0.0-1.0
                fade_ratio = (ratio - 0.6) / 0.4
                color = interpolate_color(mid_color, end_color, fade_ratio)

            draw.line([(0, y), (self.width, y)], fill=color)

    def _add_header(self, draw: ImageDraw.Draw, query: str, user_name: Optional[str]):
        """Add title section at top - will be placed inside the card."""
        # Header will be drawn inside the card, not here
        pass

    def _add_work_card(
        self,
        img: Image.Image,
        draw: ImageDraw.Draw,
        work: Dict,
        album_art_url: Optional[str]
    ):
        """
        Add centered card with all content on cream background.

        Args:
            img: PIL Image to draw on
            draw: ImageDraw context
            work: Work data dictionary
            album_art_url: Optional URL to album artwork
        """
        # Make card larger to fit all content
        card_width = 700
        card_height = 450
        card_x = (self.width - card_width) // 2
        card_y = 90  # Start higher to center vertically

        # Draw card background with rounded corners (no border)
        self._draw_rounded_rectangle(
            draw,
            [card_x, card_y, card_x + card_width, card_y + card_height],
            radius=self.design["dimensions"]["corner_radius"],
            fill=hex_to_rgb(self.design["colors"]["card_bg"]),
            outline=hex_to_rgb(self.design["colors"]["card_bg"]),
            width=0
        )

        # Get query and user_name from the calling context
        # These will be passed through the method
        query = getattr(self, '_current_query', 'My Vibe')
        user_name = getattr(self, '_current_user_name', None)

        # Header emoji with optional personalization
        if user_name:
            header_text = f"🎵 {user_name}'s Classical Music Vibe 🎵"
        else:
            header_text = "🎵 My Classical Music Vibe 🎵"

        self._draw_centered_text(
            draw,
            header_text,
            y=card_y + 30,
            font=self.fonts['header'],
            fill=self.design["colors"]["text_secondary"]
        )

        # Query text (with quotes) - ensure it fits within card with padding
        query_text = f'"{query}"'
        # Check if query text fits within card (700px width - 80px padding = 620px max)
        bbox = draw.textbbox((0, 0), query_text, font=self.fonts['title'])
        text_width = bbox[2] - bbox[0]
        max_width = 620  # Card width minus padding

        # Truncate if too long
        while text_width > max_width and len(query) > 10:
            query = query[:-1]
            query_text = f'"{query.strip()}..."'
            bbox = draw.textbbox((0, 0), query_text, font=self.fonts['title'])
            text_width = bbox[2] - bbox[0]

        self._draw_centered_text(
            draw,
            query_text,
            y=card_y + 80,
            font=self.fonts['title'],
            fill=self.design["colors"]["text_primary"]
        )

        # Work title
        work_title = self._truncate_text(work.get('title', 'Untitled'), max_chars=50)
        self._draw_centered_text(
            draw,
            work_title,
            y=card_y + 180,
            font=self.fonts['work_title'],
            fill=self.design["colors"]["text_primary"]
        )

        # Composer name
        composer_text = f"by {work.get('composer', 'Unknown')}"
        self._draw_centered_text(
            draw,
            composer_text,
            y=card_y + 235,
            font=self.fonts['composer'],
            fill=self.design["colors"]["text_secondary"]
        )

        # Metadata with emoji
        self._add_metadata(draw, work, card_y, card_height)

        # Footer inside card
        self._add_footer_in_card(draw, card_y, card_height)

    def _add_music_icon(self, draw: ImageDraw.Draw, card_y: int, card_height: int):
        """Add decorative musical note icon in center of card."""
        # Large musical note emoji in center
        icon_text = "🎵"
        icon_y = card_y + 140  # Center position

        self._draw_centered_text(
            draw,
            icon_text,
            y=icon_y,
            font=self.fonts['icon'],  # Will add this font
            fill=hex_to_rgb(self.design["colors"]["accent_gold"])
        )

    def _add_metadata(self, draw: ImageDraw.Draw, work: Dict, card_y: int, card_height: int):
        """Add metadata line inside card with emoji."""
        metadata_parts = []

        # Add period if available
        if work.get('period'):
            metadata_parts.append(work['period'])

        # Add work type or instrumentation
        if work.get('work_type'):
            metadata_parts.append(work['work_type'])
        elif work.get('instrumentation'):
            # Truncate long instrumentation
            instr = str(work['instrumentation'])
            if len(instr) > 20:
                instr = instr[:17] + "..."
            metadata_parts.append(instr)

        if metadata_parts:
            metadata_text = " | ".join(metadata_parts)

            # Add emoji based on instrumentation
            instrumentation_lower = str(work.get('instrumentation', '')).lower()
            work_type_lower = str(work.get('work_type', '')).lower()

            if 'piano' in instrumentation_lower or 'piano' in work_type_lower:
                emoji = "🎹"
            elif 'violin' in instrumentation_lower or 'violin' in work_type_lower:
                emoji = "🎻"
            elif 'orchestra' in instrumentation_lower or 'symphony' in work_type_lower:
                emoji = "🎺"
            elif 'vocal' in instrumentation_lower or 'voice' in instrumentation_lower:
                emoji = "🎤"
            elif 'organ' in instrumentation_lower:
                emoji = "🎹"
            else:
                emoji = "🎵"

            metadata_text = f"{emoji} {metadata_text}"

            self._draw_centered_text(
                draw,
                metadata_text,
                y=card_y + 290,
                font=self.fonts['metadata'],
                fill=self.design["colors"]["text_tertiary"]
            )

    def _add_footer_in_card(self, draw: ImageDraw.Draw, card_y: int, card_height: int):
        """Add footer inside the card."""
        cta_text = self.design["branding"]["tagline"]
        self._draw_centered_text(
            draw,
            cta_text,
            y=card_y + card_height - 70,
            font=self.fonts['cta'],
            fill=self.design["colors"]["text_secondary"]
        )

        # Domain/app name
        url_text = self.design["branding"]["domain"]
        self._draw_centered_text(
            draw,
            url_text,
            y=card_y + card_height - 40,
            font=self.fonts['cta'],
            fill=self.design["colors"]["text_secondary"]
        )

    def _add_footer(self, draw: ImageDraw.Draw):
        """Footer is now inside the card - this method is deprecated."""
        pass

    def _add_logo(self, img: Image.Image):
        """Add small logo in bottom right corner if available."""
        logo_path = Path(__file__).parent / "assets" / "logo.png"

        if logo_path.exists():
            try:
                logo = Image.open(logo_path)
                logo = logo.convert('RGBA')
                logo.thumbnail((60, 60), Image.Resampling.LANCZOS)
                # Position in bottom right
                logo_x = self.width - logo.width - 30
                logo_y = self.height - logo.height - 20
                img.paste(logo, (logo_x, logo_y), logo)
            except Exception as e:
                # Logo loading failed, skip silently
                pass

    # ============================================================================
    # Helper Methods
    # ============================================================================

    def _load_fonts(self) -> dict:
        """Load all fonts with fallbacks to default."""
        fonts = {}
        font_dir = Path(__file__).parent / "fonts"

        font_configs = {
            'title': ('PlayfairDisplay-Bold.ttf', self.design["typography"]["title_size"]),
            'work_title': ('PlayfairDisplay-Bold.ttf', self.design["typography"]["work_title_size"]),
            'composer': ('Inter-Regular.ttf', self.design["typography"]["composer_size"]),
            'metadata': ('Inter-Regular.ttf', self.design["typography"]["metadata_size"]),
            'cta': ('Inter-Regular.ttf', self.design["typography"]["cta_size"]),
            'header': ('Inter-Regular.ttf', self.design["typography"]["header_size"]),
            'icon': ('Inter-Bold.ttf', self.design["typography"]["icon_size"]),
        }

        for key, (font_file, size) in font_configs.items():
            font_path = font_dir / font_file
            try:
                if font_path.exists():
                    fonts[key] = ImageFont.truetype(str(font_path), size)
                else:
                    # Try loading default PIL font
                    fonts[key] = ImageFont.load_default()
            except Exception:
                # Fallback to default font
                fonts[key] = ImageFont.load_default()

        return fonts

    def _draw_centered_text(
        self,
        draw: ImageDraw.Draw,
        text: str,
        y: int,
        font: ImageFont.FreeTypeFont,
        fill: str
    ):
        """Draw text centered horizontally at given y position."""
        bbox = draw.textbbox((0, 0), text, font=font)
        text_width = bbox[2] - bbox[0]
        x = (self.width - text_width) // 2
        draw.text((x, y), text, font=font, fill=fill)

    def _draw_rounded_rectangle(
        self,
        draw: ImageDraw.Draw,
        coords: list,
        radius: int,
        fill: tuple,
        outline: tuple,
        width: int
    ):
        """Draw rectangle with rounded corners."""
        draw.rounded_rectangle(
            coords,
            radius=radius,
            fill=fill,
            outline=outline,
            width=width
        )

    def _draw_circle(
        self,
        draw: ImageDraw.Draw,
        center: tuple,
        radius: int,
        fill: tuple,
        outline: tuple,
        width: int
    ):
        """Draw circle at given center with radius."""
        x, y = center
        draw.ellipse(
            [x - radius, y - radius, x + radius, y + radius],
            fill=fill,
            outline=outline,
            width=width
        )

    def _download_and_resize_image(self, url: str, size: int) -> Optional[Image.Image]:
        """
        Download image from URL and resize to square.

        Args:
            url: Image URL
            size: Target size (width and height)

        Returns:
            PIL Image or None if download fails
        """
        try:
            response = requests.get(url, timeout=5)
            response.raise_for_status()

            img = Image.open(io.BytesIO(response.content))
            img = img.convert('RGB')

            # Resize to square, maintaining aspect ratio
            img.thumbnail((size, size), Image.Resampling.LANCZOS)

            # If not square, crop to center square
            if img.size[0] != img.size[1]:
                # Center crop
                min_side = min(img.size)
                left = (img.size[0] - min_side) // 2
                top = (img.size[1] - min_side) // 2
                right = left + min_side
                bottom = top + min_side
                img = img.crop((left, top, right, bottom))

            # Resize to exact size
            img = img.resize((size, size), Image.Resampling.LANCZOS)

            return img
        except Exception as e:
            # Download or processing failed
            return None

    def _make_circular(self, img: Image.Image) -> Image.Image:
        """
        Convert square image to circular with transparent background.

        Args:
            img: Square PIL Image

        Returns:
            Circular image with alpha channel
        """
        size = img.size[0]

        # Create circular mask
        mask = Image.new('L', (size, size), 0)
        mask_draw = ImageDraw.Draw(mask)
        mask_draw.ellipse((0, 0, size, size), fill=255)

        # Create output with transparency
        output = Image.new('RGBA', (size, size), (0, 0, 0, 0))
        output.paste(img, (0, 0))
        output.putalpha(mask)

        return output

    def _truncate_text(self, text: str, max_chars: int) -> str:
        """Truncate text with ellipsis if too long."""
        if len(text) <= max_chars:
            return text
        return text[:max_chars - 3] + "..."

    def _image_to_bytes(self, img: Image.Image) -> bytes:
        """
        Convert PIL Image to PNG bytes.

        Args:
            img: PIL Image

        Returns:
            PNG image as bytes
        """
        img_bytes = io.BytesIO()
        img.save(img_bytes, format='PNG', optimize=True, quality=95)
        img_bytes.seek(0)
        return img_bytes.getvalue()


# ============================================================================
# Standalone testing function
# ============================================================================

def test_generator():
    """Test the share card generator with sample data."""
    generator = ShareCardGenerator()

    sample_work = {
        'title': 'Nocturne in E minor, Op. 72, No. 1',
        'composer': 'Frédéric Chopin',
        'period': 'Romantic',
        'work_type': 'Nocturne',
        'instrumentation': 'Piano',
    }

    image_bytes = generator.generate_card(
        query="Dark Academia",
        work=sample_work,
        user_name="Cameron"
    )

    # Save to file
    output_path = Path(__file__).parent / "test_share_card.png"
    with open(output_path, 'wb') as f:
        f.write(image_bytes)

    print(f"✓ Test card generated: {output_path}")
    print(f"  Size: {len(image_bytes) / 1024:.1f} KB")


if __name__ == "__main__":
    test_generator()
