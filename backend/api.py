"""
FastAPI backend for the Classical Music Recommender.
Provides REST API endpoints for the frontend application.
"""

from fastapi import FastAPI, HTTPException, Header, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
from datetime import datetime
import sys
import os
from pathlib import Path
import logging
from dotenv import load_dotenv

# Load environment variables from .env file
env_path = Path(__file__).parent / '.env'
load_dotenv(env_path)

# Add parent directory to path to import recommender
sys.path.insert(0, str(Path(__file__).parent.parent))

from recommender.service import initialize_service, get_service
from backend.share_api import router as share_router
from backend.rate_limiter import RateLimitMiddleware, create_endpoint_limiter


# Security Headers Middleware
class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Add security headers to all responses."""

    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)

        # Security headers
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"

        # Content Security Policy
        csp = (
            "default-src 'self'; "
            "img-src 'self' data: https:; "
            "script-src 'self' 'unsafe-inline'; "
            "style-src 'self' 'unsafe-inline'; "
            "font-src 'self' data:; "
            "connect-src 'self'"
        )
        response.headers["Content-Security-Policy"] = csp

        return response


app = FastAPI(
    title="Classical Music Recommender API",
    docs_url="/docs" if os.getenv("ENVIRONMENT") != "production" else None,
    redoc_url="/redoc" if os.getenv("ENVIRONMENT") != "production" else None,
)

# Include share endpoints
app.include_router(share_router)

# Configure CORS - use environment variable for production
ALLOWED_ORIGINS = os.getenv(
    "ALLOWED_ORIGINS",
    "http://localhost:3000,http://127.0.0.1:3000,http://localhost:3001,http://127.0.0.1:3001"
).split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["Content-Type", "X-User-ID", "X-Session-ID"],
    max_age=600,  # Cache preflight requests for 10 minutes
)

# Add security headers middleware
app.add_middleware(SecurityHeadersMiddleware)

# Add rate limiting middleware
# Configurable via environment variable, defaults to 60 requests per minute
RATE_LIMIT = int(os.getenv("RATE_LIMIT", "60"))
app.add_middleware(RateLimitMiddleware, rate=RATE_LIMIT, period=60)

# Create stricter rate limiter for expensive search operations
# 20 requests per minute for semantic search
search_limiter = create_endpoint_limiter(rate=20, period=60)

# Initialize service on startup
@app.on_event("startup")
async def startup_event():
    """Initialize the recommender service."""
    import logging
    logging.basicConfig(level=logging.INFO, force=True)
    logger = logging.getLogger(__name__)
    logger.info("Starting recommender service initialization...")

    try:
        initialize_service()
        logger.info("Service initialization complete")

        # Verify semantic search loaded
        service = get_service()
        semantic_search = service.get_semantic_search()
        if semantic_search:
            logger.info("✓ Semantic search engine loaded successfully")
        else:
            logger.warning("✗ Semantic search engine NOT loaded - mood search will not work!")
    except (FileNotFoundError, RuntimeError) as e:
        if "Works file not found" in str(e) or "FileNotFoundError" in str(e):
            logger.warning(f"Data files not found: {e}")
            logger.warning("Recommender service will not be available. Run data processing first.")
            logger.info("API will start anyway for testing security features...")
        else:
            logger.error(f"Service initialization failed: {e}", exc_info=True)
            raise
    except Exception as e:
        logger.error(f"Service initialization failed: {e}", exc_info=True)
        raise

# Request/Response Models
class MoodSearchRequest(BaseModel):
    query: str
    n: int = 1

class ActivitySearchRequest(BaseModel):
    activity: str
    context: Optional[str] = None
    n: int = 1

class RecommendationResponse(BaseModel):
    work_id: str
    title: str
    composer: str
    work_type: Optional[str] = None
    key: Optional[str] = None
    similarity_score: float
    explanation: str
    spotify_url: Optional[str] = None

class SearchResponse(BaseModel):
    query: str
    recommendations: List[RecommendationResponse]

# Feedback Models
class FeedbackSubmitRequest(BaseModel):
    work_id: str
    rating: int  # -1 or 1
    comment: Optional[str] = None
    vibe: str

class FeedbackResponse(BaseModel):
    id: str
    user_id: str
    work_id: str
    rating: int
    comment: Optional[str] = None
    vibe: str
    created_at: str
    updated_at: str

class FeedbackStatsResponse(BaseModel):
    work_id: str
    thumbs_up_count: int
    thumbs_down_count: int
    total_feedbacks: int
    avg_rating: float
    user_feedback: Optional[FeedbackResponse] = None

# Saved Pieces Models
class SavedPieceRequest(BaseModel):
    work_id: str
    title: str
    composer: str
    composer_id: Optional[str] = None
    vibe: Optional[str] = None
    explanation: Optional[str] = None
    notes: Optional[str] = None

class SavedPieceResponse(BaseModel):
    id: str
    user_id: str
    work_id: str
    title: str
    composer: str
    composer_id: Optional[str] = None
    vibe: Optional[str] = None
    explanation: Optional[str] = None
    notes: Optional[str] = None
    saved_at: str

class SavedPieceListResponse(BaseModel):
    pieces: List[SavedPieceResponse]
    count: int

# Analytics Models
class AnalyticsEventRequest(BaseModel):
    event_type: str
    event_data: Optional[Dict[str, Any]] = {}
    page_url: Optional[str] = None
    referrer: Optional[str] = None
    user_agent: Optional[str] = None
    session_id: Optional[str] = None

class AnalyticsEventResponse(BaseModel):
    success: bool
    message: str

# Recommendation History Models (for internal use)
class RecommendationHistoryLog(BaseModel):
    work_id: str
    composer_id: Optional[str] = None
    query: str
    vibe: str
    rank: int
    relevance_score: float

# API Endpoints
@app.get("/")
async def root():
    """Health check endpoint."""
    return {"status": "ok", "message": "Classical Music Recommender API"}

@app.get("/api/health")
async def health_check():
    """Check service health."""
    service = get_service()
    health = service.health_check()
    return health

@app.get("/api/security/summary")
async def security_summary(hours: int = 24):
    """
    Get security monitoring summary.

    This endpoint should be protected in production (e.g., require admin auth).
    For now, it's only available in development mode.
    """
    if os.getenv("ENVIRONMENT") == "production":
        raise HTTPException(
            status_code=404,
            detail="Not found"
        )

    try:
        from security_monitor import get_security_summary
        summary = get_security_summary(hours=hours)
        return summary
    except Exception as e:
        logger.error(f"Error getting security summary: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/search/mood", response_model=SearchResponse)
async def search_by_mood(
    mood_request: MoodSearchRequest,
    request: Request,
    x_user_id: Optional[str] = Header(None, alias="X-User-ID"),
    x_session_id: Optional[str] = Header(None, alias="X-Session-ID")
):
    """Search for music by mood/vibe."""
    logger = logging.getLogger(__name__)

    # Apply stricter rate limiting for search endpoints
    is_allowed, rate_info = search_limiter.is_allowed(request)
    if not is_allowed:
        raise HTTPException(
            status_code=429,
            detail=f"Search rate limit exceeded. Please try again in {rate_info.get('retry_after', 0)} seconds."
        )

    service = get_service()

    try:
        results = service.search_by_mood(mood_request.query, n=mood_request.n)

        recommendations = [
            RecommendationResponse(
                work_id=rec.get('work_id', ''),
                title=rec.get('title', ''),
                composer=rec.get('composer', 'Unknown'),
                work_type=rec.get('work_type'),
                key=rec.get('key'),
                similarity_score=rec.get('similarity_score', 0.0),
                explanation=rec.get('explanation', ''),
                spotify_url=None  # TODO: Add Spotify integration
            )
            for rec in results
        ]

        # Log recommendation history to database (async, non-blocking)
        try:
            from supabase import create_client

            supabase_url = os.getenv("SUPABASE_URL")
            supabase_key = os.getenv("SUPABASE_SERVICE_KEY")

            if supabase_url and supabase_key:
                supabase = create_client(supabase_url, supabase_key)

                # Prepare history records for all recommendations
                history_records = []
                for idx, rec in enumerate(results):
                    history_records.append({
                        "user_id": x_user_id,
                        "session_id": x_session_id,
                        "work_id": rec.get('work_id', ''),
                        "composer_id": rec.get('composer_id'),  # If available
                        "query": mood_request.query,
                        "vibe": mood_request.query,  # Use query as vibe for now
                        "rank": idx + 1,
                        "relevance_score": rec.get('similarity_score', 0.0)
                    })

                # Insert all records in batch
                if history_records:
                    supabase.table("recommendation_history").insert(history_records).execute()
        except Exception as log_error:
            # Don't fail the request if logging fails
            logger.warning(f"Failed to log recommendation history: {log_error}")

        return SearchResponse(
            query=mood_request.query,
            recommendations=recommendations
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/search/activity", response_model=SearchResponse)
async def search_by_activity(
    activity_request: ActivitySearchRequest,
    request: Request
):
    """Search for music by activity."""
    # Apply stricter rate limiting for search endpoints
    is_allowed, rate_info = search_limiter.is_allowed(request)
    if not is_allowed:
        raise HTTPException(
            status_code=429,
            detail=f"Search rate limit exceeded. Please try again in {rate_info.get('retry_after', 0)} seconds."
        )

    service = get_service()

    try:
        results = service.search_by_activity(
            activity_request.activity,
            context=activity_request.context,
            n=activity_request.n
        )

        recommendations = [
            RecommendationResponse(
                work_id=rec.get('work_id', ''),
                title=rec.get('title', ''),
                composer=rec.get('composer', 'Unknown'),
                work_type=rec.get('work_type'),
                key=rec.get('key'),
                similarity_score=rec.get('similarity_score', 0.0),
                explanation=rec.get('explanation', ''),
                spotify_url=None
            )
            for rec in results
        ]

        return SearchResponse(
            query=f"{activity_request.activity} {activity_request.context or ''}".strip(),
            recommendations=recommendations
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/recommend/similar/{work_id}")
async def get_similar_works(work_id: str, n: int = 5):
    """Get similar works to a given work."""
    service = get_service()

    try:
        results = service.recommend_similar(work_id, n=n)

        recommendations = [
            RecommendationResponse(
                work_id=rec.get('work_id', ''),
                title=rec.get('title', ''),
                composer=rec.get('composer', 'Unknown'),
                work_type=rec.get('work_type'),
                key=rec.get('key'),
                similarity_score=rec.get('similarity_score', 0.0),
                explanation=rec.get('explanation', ''),
                spotify_url=None
            )
            for rec in results
        ]

        return {"recommendations": recommendations}
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Saved Pieces Endpoints
@app.get("/api/saved-pieces", response_model=SavedPieceListResponse)
async def get_saved_pieces(
    x_user_id: Optional[str] = Header(None, alias="X-User-ID")
):
    """Get all saved pieces for the authenticated user."""
    logger = logging.getLogger(__name__)

    if not x_user_id:
        raise HTTPException(status_code=401, detail="User not authenticated")

    try:
        from supabase import create_client

        supabase_url = os.getenv("SUPABASE_URL")
        supabase_key = os.getenv("SUPABASE_SERVICE_KEY")

        if not supabase_url or not supabase_key:
            raise HTTPException(
                status_code=500,
                detail="Supabase configuration missing"
            )

        supabase = create_client(supabase_url, supabase_key)

        response = supabase.table("saved_pieces").select("*").eq(
            "user_id", x_user_id
        ).order("saved_at", desc=True).execute()

        pieces = [
            SavedPieceResponse(
                id=piece["id"],
                user_id=piece["user_id"],
                work_id=piece["work_id"],
                title=piece["title"],
                composer=piece["composer"],
                composer_id=piece.get("composer_id"),
                vibe=piece.get("vibe"),
                explanation=piece.get("explanation"),
                notes=piece.get("notes"),
                saved_at=piece["saved_at"]
            )
            for piece in response.data
        ]

        return SavedPieceListResponse(
            pieces=pieces,
            count=len(pieces)
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching saved pieces: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/saved-pieces", response_model=SavedPieceResponse)
async def save_piece(
    request: SavedPieceRequest,
    x_user_id: Optional[str] = Header(None, alias="X-User-ID")
):
    """Save a piece to user's favorites."""
    logger = logging.getLogger(__name__)

    if not x_user_id:
        raise HTTPException(status_code=401, detail="User not authenticated")

    try:
        from supabase import create_client

        supabase_url = os.getenv("SUPABASE_URL")
        supabase_key = os.getenv("SUPABASE_SERVICE_KEY")

        if not supabase_url or not supabase_key:
            raise HTTPException(
                status_code=500,
                detail="Supabase configuration missing"
            )

        supabase = create_client(supabase_url, supabase_key)

        piece_data = {
            "user_id": x_user_id,
            "work_id": request.work_id,
            "title": request.title,
            "composer": request.composer,
            "composer_id": request.composer_id,
            "vibe": request.vibe,
            "explanation": request.explanation,
            "notes": request.notes,
        }

        response = supabase.table("saved_pieces").upsert(
            piece_data,
            on_conflict="user_id,work_id"
        ).execute()

        if not response.data:
            raise HTTPException(status_code=500, detail="Failed to save piece")

        saved = response.data[0]

        return SavedPieceResponse(
            id=saved["id"],
            user_id=saved["user_id"],
            work_id=saved["work_id"],
            title=saved["title"],
            composer=saved["composer"],
            composer_id=saved.get("composer_id"),
            vibe=saved.get("vibe"),
            explanation=saved.get("explanation"),
            notes=saved.get("notes"),
            saved_at=saved["saved_at"]
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error saving piece: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/api/saved-pieces/{work_id}")
async def delete_saved_piece(
    work_id: str,
    x_user_id: Optional[str] = Header(None, alias="X-User-ID")
):
    """Remove a piece from user's saved favorites."""
    logger = logging.getLogger(__name__)

    if not x_user_id:
        raise HTTPException(status_code=401, detail="User not authenticated")

    try:
        from supabase import create_client

        supabase_url = os.getenv("SUPABASE_URL")
        supabase_key = os.getenv("SUPABASE_SERVICE_KEY")

        if not supabase_url or not supabase_key:
            raise HTTPException(
                status_code=500,
                detail="Supabase configuration missing"
            )

        supabase = create_client(supabase_url, supabase_key)

        response = supabase.table("saved_pieces").delete().eq(
            "user_id", x_user_id
        ).eq("work_id", work_id).execute()

        return {"success": True, "message": "Piece removed from saved"}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting saved piece: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# Feedback Endpoints
@app.post("/api/feedback", response_model=FeedbackResponse)
async def submit_feedback(
    request: FeedbackSubmitRequest,
    x_user_id: Optional[str] = Header(None, alias="X-User-ID")
):
    """
    Submit or update feedback for a recommendation.
    Uses Supabase for storage with upsert behavior.
    """
    logger = logging.getLogger(__name__)

    if not x_user_id:
        raise HTTPException(status_code=401, detail="User not authenticated")

    if request.rating not in [-1, 1]:
        raise HTTPException(status_code=400, detail="Rating must be -1 or 1")

    try:
        from supabase import create_client

        supabase_url = os.getenv("SUPABASE_URL")
        supabase_key = os.getenv("SUPABASE_SERVICE_KEY")

        if not supabase_url or not supabase_key:
            raise HTTPException(
                status_code=500,
                detail="Supabase configuration missing. Please set SUPABASE_URL and SUPABASE_SERVICE_KEY environment variables."
            )

        supabase = create_client(supabase_url, supabase_key)

        feedback_data = {
            "user_id": x_user_id,
            "work_id": request.work_id,
            "rating": request.rating,
            "comment": request.comment,
            "vibe": request.vibe,
        }

        response = supabase.table("recommendation_feedback").upsert(
            feedback_data,
            on_conflict="user_id,work_id,vibe"
        ).execute()

        if not response.data:
            raise HTTPException(status_code=500, detail="Failed to save feedback")

        feedback = response.data[0]

        return FeedbackResponse(
            id=feedback["id"],
            user_id=feedback["user_id"],
            work_id=feedback["work_id"],
            rating=feedback["rating"],
            comment=feedback.get("comment"),
            vibe=feedback["vibe"],
            created_at=feedback["created_at"],
            updated_at=feedback["updated_at"]
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error submitting feedback: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/feedback/{work_id}", response_model=FeedbackStatsResponse)
async def get_feedback(
    work_id: str,
    vibe: str = Query(..., description="The vibe/search query"),
    x_user_id: Optional[str] = Header(None, alias="X-User-ID")
):
    """
    Get feedback stats for a work and the current user's feedback if it exists.
    """
    logger = logging.getLogger(__name__)

    try:
        from supabase import create_client

        supabase_url = os.getenv("SUPABASE_URL")
        supabase_key = os.getenv("SUPABASE_SERVICE_KEY")

        if not supabase_url or not supabase_key:
            raise HTTPException(
                status_code=500,
                detail="Supabase configuration missing"
            )

        supabase = create_client(supabase_url, supabase_key)

        # Get all feedback for this work
        response = supabase.table("recommendation_feedback").select("*").eq(
            "work_id", work_id
        ).execute()

        feedbacks = response.data

        # Calculate stats
        thumbs_up = sum(1 for f in feedbacks if f["rating"] == 1)
        thumbs_down = sum(1 for f in feedbacks if f["rating"] == -1)
        total = len(feedbacks)
        avg_rating = sum(f["rating"] for f in feedbacks) / total if total > 0 else 0.0

        # Get user's specific feedback for this work + vibe combo
        user_feedback = None
        if x_user_id:
            user_response = supabase.table("recommendation_feedback").select("*").eq(
                "user_id", x_user_id
            ).eq("work_id", work_id).eq("vibe", vibe).execute()

            if user_response.data:
                uf = user_response.data[0]
                user_feedback = FeedbackResponse(
                    id=uf["id"],
                    user_id=uf["user_id"],
                    work_id=uf["work_id"],
                    rating=uf["rating"],
                    comment=uf.get("comment"),
                    vibe=uf["vibe"],
                    created_at=uf["created_at"],
                    updated_at=uf["updated_at"]
                )

        return FeedbackStatsResponse(
            work_id=work_id,
            thumbs_up_count=thumbs_up,
            thumbs_down_count=thumbs_down,
            total_feedbacks=total,
            avg_rating=avg_rating,
            user_feedback=user_feedback
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching feedback: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/api/feedback/{work_id}")
async def delete_feedback(
    work_id: str,
    vibe: str = Query(..., description="The vibe/search query"),
    x_user_id: Optional[str] = Header(None, alias="X-User-ID")
):
    """Delete user's feedback for a specific work + vibe."""
    logger = logging.getLogger(__name__)

    if not x_user_id:
        raise HTTPException(status_code=401, detail="User not authenticated")

    try:
        from supabase import create_client

        supabase_url = os.getenv("SUPABASE_URL")
        supabase_key = os.getenv("SUPABASE_SERVICE_KEY")

        if not supabase_url or not supabase_key:
            raise HTTPException(
                status_code=500,
                detail="Supabase configuration missing"
            )

        supabase = create_client(supabase_url, supabase_key)

        response = supabase.table("recommendation_feedback").delete().eq(
            "user_id", x_user_id
        ).eq("work_id", work_id).eq("vibe", vibe).execute()

        return {"success": True, "message": "Feedback deleted"}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting feedback: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# Analytics Endpoints
@app.post("/api/analytics", response_model=AnalyticsEventResponse)
async def log_analytics_event(
    request: AnalyticsEventRequest,
    x_user_id: Optional[str] = Header(None, alias="X-User-ID")
):
    """
    Log an analytics event.
    Allows both authenticated and anonymous tracking.
    """
    logger = logging.getLogger(__name__)

    try:
        from supabase import create_client

        supabase_url = os.getenv("SUPABASE_URL")
        supabase_key = os.getenv("SUPABASE_SERVICE_KEY")

        if not supabase_url or not supabase_key:
            raise HTTPException(
                status_code=500,
                detail="Supabase configuration missing"
            )

        supabase = create_client(supabase_url, supabase_key)

        event_data = {
            "user_id": x_user_id,  # Can be None for anonymous
            "session_id": request.session_id,
            "event_type": request.event_type,
            "event_data": request.event_data or {},
            "page_url": request.page_url,
            "referrer": request.referrer,
            "user_agent": request.user_agent,
        }

        response = supabase.table("analytics_events").insert(event_data).execute()

        if not response.data:
            raise HTTPException(status_code=500, detail="Failed to log analytics event")

        return AnalyticsEventResponse(
            success=True,
            message="Analytics event logged successfully"
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error logging analytics event: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
