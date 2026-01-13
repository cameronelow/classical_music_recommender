# Classical Music Recommender - Backend API

FastAPI backend that provides REST API endpoints for the classical music recommender frontend.

## Features

- RESTful API for mood-based and activity-based music search
- Integration with the existing Python recommender engine
- CORS enabled for frontend development
- Health check endpoints

## Installation

```bash
pip install -r requirements.txt
```

## Running the Server

```bash
# From the backend directory
python api.py

# Or using uvicorn directly
uvicorn api:app --reload --host 0.0.0.0 --port 8000
```

The API will be available at `http://localhost:8000`

API documentation (Swagger UI): `http://localhost:8000/docs`

## API Endpoints

### Health Check
- `GET /` - Basic health check
- `GET /api/health` - Detailed service health

### Music Search
- `POST /api/search/mood` - Search by mood/vibe
  ```json
  {
    "query": "relaxing and peaceful",
    "n": 1
  }
  ```

- `POST /api/search/activity` - Search by activity
  ```json
  {
    "activity": "studying",
    "context": "need to focus",
    "n": 1
  }
  ```

### Recommendations
- `GET /api/recommend/similar/{work_id}?n=5` - Get similar works

## Response Format

All search endpoints return:
```json
{
  "query": "user's search query",
  "recommendations": [
    {
      "work_id": "unique-id",
      "title": "Piece Title",
      "composer": "Composer Name",
      "work_type": "Symphony",
      "key": "C major",
      "similarity_score": 0.92,
      "explanation": "Why this was recommended",
      "spotify_url": null
    }
  ]
}
```

## Integration with Frontend

The backend is designed to work with the Next.js frontend on `http://localhost:3000`.

Ensure the recommender service has data available in `data/processed/` before starting the API server.
