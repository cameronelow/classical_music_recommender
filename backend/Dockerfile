FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements from root (shared dependencies)
COPY ../requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Install production ASGI server
RUN pip install --no-cache-dir gunicorn

# Copy backend application code
COPY . /app/

# Copy data directory from parent (needed for recommender engine)
COPY ../data /app/data

# Copy recommender module from parent (needed for import)
COPY ../recommender /app/recommender

# Create cache directory
RUN mkdir -p /app/cache

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
  CMD python -c "import requests; requests.get('http://localhost:8000/api/health')"

# Production startup command with Gunicorn + Uvicorn workers
# 4 workers for production, timeout increased for ML model loading
CMD ["gunicorn", "api:app", "--workers", "4", "--worker-class", "uvicorn.workers.UvicornWorker", "--bind", "0.0.0.0:8000", "--timeout", "120"]
