FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy and install requirements
COPY requirements.txt .
# Install PyTorch CPU-only version first (much smaller, reduces memory)
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
# Install remaining requirements
RUN pip install --no-cache-dir -r requirements.txt gunicorn

# Copy all necessary directories
COPY data ./data
COPY recommender ./recommender
COPY backend ./backend

# Set Python path for imports
ENV PYTHONPATH=/app

# Stay in /app as working directory so relative paths work
# The api.py is in backend/, so we'll specify it in CMD
WORKDIR /app

# Create cache directory in backend
RUN mkdir -p backend/cache

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
  CMD python -c "import requests; requests.get('http://localhost:8000/api/health')" || exit 1

# Start gunicorn with uvicorn workers
# Use only 1 worker for Render free/starter tier to reduce memory usage
# Run from /app directory and specify backend.api:app module path
CMD ["gunicorn", "backend.api:app", "--workers", "1", "--worker-class", "uvicorn.workers.UvicornWorker", "--bind", "0.0.0.0:8000", "--timeout", "120", "--max-requests", "1000", "--max-requests-jitter", "50"]
