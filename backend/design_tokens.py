"""
Design tokens for share card generator.
Matches the app's design system from Tailwind config.
"""

DESIGN_TOKENS = {
    "colors": {
        # Background gradient (matching app's exact gradient)
        # App uses: linear-gradient(to bottom, #1A3263 0%, #1A3263 60%, #FAB95B 100%)
        "background_start": "#1A3263",  # Dark blue (from app)
        "background_mid": "#1A3263",    # Dark blue holds until 60%
        "background_end": "#FAB95B",    # Warm yellow (from app)

        # Primary colors from app design
        "primary_dark_blue": "#1A3263",  # From tailwind config
        "accent_gold": "#FAB95B",        # Warm yellow accent
        "accent_yellow": "#FAB95B",      # Warm yellow from config

        # Card colors
        "card_bg": "#FFFFFF",            # White card background (stands out on gradient)
        "card_border": "#FAB95B",        # Warm yellow border

        # Text colors
        "text_primary": "#1A3263",       # Dark blue for main text (readable on white)
        "text_secondary": "#1A3263",     # Dark blue for secondary text
        "text_tertiary": "#666666",      # Muted gray for metadata
        "text_on_gradient": "#E8E2DB",   # Light cream for text directly on gradient
    },

    "typography": {
        # Font sizes (in pixels for PIL)
        "title_size": 56,         # Main query title
        "work_title_size": 36,    # Work title (reduced to fit better)
        "composer_size": 28,      # Composer name
        "metadata_size": 22,      # Period, instrumentation
        "cta_size": 18,           # Call to action text
        "header_size": 24,        # Header emoji text
        "icon_size": 72,          # Large decorative music icon
    },

    "spacing": {
        "card_padding": 40,
        "section_gap": 60,
        "text_line_height": 1.4,
    },

    "dimensions": {
        "width": 1200,
        "height": 630,
        "card_width": 600,
        "card_height": 320,  # Reduced to prevent overlap
        "album_art_size": 150,  # Reduced slightly to fit better
        "corner_radius": 20,
    },

    "branding": {
        "app_name": "Classical Vibes",
        "tagline": "Discover your perfect classical piece",
        "domain": "expressivo.com",  # Update with actual domain
    }
}


def hex_to_rgb(hex_color: str) -> tuple:
    """Convert hex color to RGB tuple."""
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))


def interpolate_color(color1: tuple, color2: tuple, ratio: float) -> tuple:
    """Interpolate between two RGB colors."""
    r = int(color1[0] * (1 - ratio) + color2[0] * ratio)
    g = int(color1[1] * (1 - ratio) + color2[1] * ratio)
    b = int(color1[2] * (1 - ratio) + color2[2] * ratio)
    return (r, g, b)
