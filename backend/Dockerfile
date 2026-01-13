FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements from root
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install production ASGI server
RUN pip install --no-cache-dir gunicorn

# Copy data directory (needed for recommender engine)
COPY data ./data

# Copy recommender module (needed for imports)
COPY recommender ./recommender

# Copy backend code
COPY backend ./backend

# Set Python path to include parent directory for imports
ENV PYTHONPATH=/app

# Set working directory to backend where api.py lives
WORKDIR /app/backend

# Create cache directory
RUN mkdir -p cache

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
  CMD python -c "import requests; requests.get('http://localhost:8000/api/health')" || exit 1

# Production startup command with Gunicorn + Uvicorn workers
# 4 workers for production, timeout increased for ML model loading
CMD ["gunicorn", "api:app", "--workers", "4", "--worker-class", "uvicorn.workers.UvicornWorker", "--bind", "0.0.0.0:8000", "--timeout", "120"]
