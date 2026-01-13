"""
Share API endpoints for Classical Music Recommender.
Handles share card generation and share landing pages.
"""

from fastapi import APIRouter, Query, HTTPException, Request
from fastapi.responses import Response, HTMLResponse
from pathlib import Path
import hashlib
import logging
from typing import Optional
import sys
import os

# Add parent directory to import recommender
sys.path.insert(0, str(Path(__file__).parent.parent))

from share_card_generator import ShareCardGenerator
from recommender.service import get_service

logger = logging.getLogger(__name__)

# Create router
router = APIRouter()

# Initialize generator
generator = ShareCardGenerator()

# Cache directory for generated cards
CACHE_DIR = Path(__file__).parent / "cache" / "share_cards"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Base URL for share links - use environment variable for production
BASE_URL = os.getenv("BACKEND_URL", "http://localhost:8000")
FRONTEND_URL = os.getenv("FRONTEND_URL", "http://localhost:3001")


def get_work_by_id(work_id: str) -> Optional[dict]:
    """
    Retrieve work details from the recommender service.

    Args:
        work_id: Work identifier

    Returns:
        Work dictionary or None if not found
    """
    try:
        service = get_service()
        work_info = service.get_work_info(work_id)

        if work_info is None:
            return None

        # Ensure required fields exist
        work_dict = {
            'work_id': work_info.get('work_id', work_id),
            'title': work_info.get('title', 'Untitled'),
            'composer': work_info.get('composer', 'Unknown'),
            'work_type': work_info.get('work_type'),
            'period': work_info.get('period'),
            'key': work_info.get('key'),
            'instrumentation': work_info.get('instrumentation'),
        }

        return work_dict

    except ValueError:
        # Work not found
        logger.warning(f"Work not found: {work_id}")
        return None
    except Exception as e:
        logger.error(f"Error fetching work {work_id}: {e}")
        return None


@router.get("/api/share-card/{work_id}")
async def generate_share_card(
    work_id: str,
    query: str = Query(..., description="User's search query"),
    user_name: str = Query(None, description="Optional user name for personalization")
):
    """
    Generate dynamic share card image.

    Args:
        work_id: Work identifier
        query: Search query that led to this recommendation
        user_name: Optional user name for personalization

    Returns:
        PNG image with proper cache headers
    """
    # Generate cache key from parameters
    cache_key = hashlib.md5(
        f"{work_id}-{query}-{user_name or ''}".encode()
    ).hexdigest()
    cache_path = CACHE_DIR / f"{cache_key}.png"

    # Check cache first
    if cache_path.exists():
        logger.info(f"Serving cached share card: {cache_key}")
        with open(cache_path, 'rb') as f:
            image_bytes = f.read()
    else:
        # Generate new card
        logger.info(f"Generating new share card for work {work_id}")

        # Get work details
        work = get_work_by_id(work_id)
        if not work:
            raise HTTPException(status_code=404, detail=f"Work not found: {work_id}")

        # Generate card
        try:
            image_bytes = generator.generate_card(
                query=query,
                work=work,
                album_art_url=work.get('album_art_url'),  # Optional, may not exist
                user_name=user_name
            )

            # Save to cache
            with open(cache_path, 'wb') as f:
                f.write(image_bytes)

            logger.info(f"Share card generated and cached: {cache_key}")

        except Exception as e:
            logger.error(f"Error generating share card: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail="Failed to generate share card")

    # Return image with cache headers
    return Response(
        content=image_bytes,
        media_type="image/png",
        headers={
            "Cache-Control": "public, max-age=86400",  # 24 hour cache
            "Content-Disposition": f'inline; filename="share-{work_id}.png"'
        }
    )


@router.get("/share/{work_id}", response_class=HTMLResponse)
async def share_landing_page(
    request: Request,
    work_id: str,
    q: str = Query(None, alias="query", description="Search query"),
    ref: str = Query(None, description="Referrer user ID")
):
    """
    Landing page for shared links with Open Graph metadata.

    When shared on social media: displays beautiful card preview
    When clicked: redirects user to app

    Args:
        work_id: Work identifier
        q: Original search query
        ref: Optional referral tracking ID

    Returns:
        HTML page with meta tags and auto-redirect
    """
    # Get work details
    work = get_work_by_id(work_id)
    if not work:
        raise HTTPException(status_code=404, detail="Work not found")

    query = q or "classical music"

    # Generate share card URL
    share_card_url = f"{BASE_URL}/api/share-card/{work_id}?query={query}"
    if request.headers.get("user_name"):
        share_card_url += f"&user_name={request.headers.get('user_name')}"

    # Track referral (if analytics implemented)
    if ref:
        logger.info(f"Share referral: {ref} -> {work_id}")
        # TODO: Implement analytics tracking
        # track_referral(ref, work_id)

    # Build app redirect URL
    redirect_url = f"{FRONTEND_URL}/recommend?vibe={query}&work={work_id}"
    if ref:
        redirect_url += f"&ref={ref}"

    # HTML template with Open Graph tags
    html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{work['title']} by {work['composer']}</title>

    <!-- Open Graph / Facebook -->
    <meta property="og:type" content="music.song">
    <meta property="og:title" content='My "{query}" Classical Music Match'>
    <meta property="og:description" content='{work["title"]} by {work["composer"]} - {work.get("period", "")} period'>
    <meta property="og:image" content="{share_card_url}">
    <meta property="og:image:width" content="1200">
    <meta property="og:image:height" content="630">
    <meta property="og:url" content="{request.url}">
    <meta property="og:site_name" content="Classical Vibes">

    <!-- Twitter Card -->
    <meta name="twitter:card" content="summary_large_image">
    <meta name="twitter:title" content='My "{query}" Classical Music Match'>
    <meta name="twitter:description" content='{work["title"]} by {work["composer"]}'>
    <meta name="twitter:image" content="{share_card_url}">

    <!-- Auto-redirect to app after 2 seconds -->
    <meta http-equiv="refresh" content="2;url={redirect_url}">

    <style>
        body {{
            margin: 0;
            padding: 0;
            font-family: -apple-system, BlinkMacSystemFont, 'Inter', 'Segoe UI', sans-serif;
            background: linear-gradient(135deg, #1A1A1A 0%, #2D2D2D 100%);
            color: white;
            display: flex;
            justify-content: center;
            align-items: center;
            min-height: 100vh;
        }}
        .container {{
            text-align: center;
            padding: 40px;
            max-width: 600px;
        }}
        .spinner {{
            border: 3px solid rgba(255,255,255,0.1);
            border-top: 3px solid #C4A87C;
            border-radius: 50%;
            width: 40px;
            height: 40px;
            animation: spin 1s linear infinite;
            margin: 20px auto;
        }}
        @keyframes spin {{
            0% {{ transform: rotate(0deg); }}
            100% {{ transform: rotate(360deg); }}
        }}
        h1 {{
            color: #C4A87C;
            margin-bottom: 10px;
            font-size: 24px;
            font-weight: 600;
        }}
        p {{
            color: #E8E2DB;
            margin: 10px 0;
            font-size: 16px;
        }}
        .work-info {{
            margin-top: 30px;
            padding: 20px;
            background: rgba(255,255,255,0.05);
            border-radius: 12px;
            border: 1px solid rgba(196, 168, 124, 0.3);
        }}
        .work-title {{
            font-size: 20px;
            color: white;
            margin-bottom: 8px;
        }}
        .composer {{
            font-size: 16px;
            color: #C4A87C;
        }}
        a {{
            color: #C4A87C;
            text-decoration: none;
            font-weight: 500;
        }}
        a:hover {{
            text-decoration: underline;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🎵 Loading your classical music match...</h1>
        <div class="spinner"></div>

        <div class="work-info">
            <div class="work-title">{work['title']}</div>
            <div class="composer">by {work['composer']}</div>
            {f'<p style="color: #AAAAAA; font-size: 14px; margin-top: 10px;">{work.get("period", "")} • {work.get("work_type", "")}</p>' if work.get("period") or work.get("work_type") else ''}
        </div>

        <p style="margin-top: 30px;">Redirecting to app...</p>

        <p style="margin-top: 20px; font-size: 14px;">
            <a href="{redirect_url}">Click here if not redirected automatically</a>
        </p>
    </div>
</body>
</html>
"""

    return HTMLResponse(content=html_content)


# ============================================================================
# Cache Management
# ============================================================================

def cleanup_old_cache(days: int = 7):
    """
    Remove cache files older than specified days.

    Args:
        days: Number of days to keep cache files
    """
    import time

    cutoff = time.time() - (days * 24 * 60 * 60)
    removed_count = 0

    try:
        for cache_file in CACHE_DIR.glob("*.png"):
            if cache_file.stat().st_mtime < cutoff:
                cache_file.unlink()
                removed_count += 1

        if removed_count > 0:
            logger.info(f"Cleaned up {removed_count} old cache files")

    except Exception as e:
        logger.error(f"Error cleaning up cache: {e}")


# Optional: Pre-warm cache for popular works
async def pre_generate_popular_cards():
    """
    Pre-generate cards for popular work/query combinations.
    This can be run as a background task to improve performance.
    """
    POPULAR_QUERIES = ["moody", "studying", "relaxing", "dark academia", "cozy"]

    try:
        service = get_service()
        # Get some popular works (you could track this via analytics)
        # For now, just get first few works as example
        works_df = service.data_loader.works_df
        popular_works = works_df.head(10).to_dict('records')

        count = 0
        for work in popular_works:
            for query in POPULAR_QUERIES:
                try:
                    work_dict = {
                        'work_id': work.get('work_id', ''),
                        'title': work.get('title', ''),
                        'composer': work.get('composer', 'Unknown'),
                        'work_type': work.get('work_type'),
                        'period': work.get('period'),
                        'instrumentation': work.get('instrumentation'),
                    }

                    # Generate and cache
                    generator.generate_card(query, work_dict)
                    count += 1

                except Exception as e:
                    logger.warning(f"Failed to pre-generate card: {e}")
                    continue

        logger.info(f"Pre-generated {count} popular share cards")

    except Exception as e:
        logger.error(f"Error pre-generating cards: {e}")
