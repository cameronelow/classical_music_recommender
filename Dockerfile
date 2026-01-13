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

# Set working directory to backend
WORKDIR /app/backend

# Create cache directory
RUN mkdir -p cache

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
  CMD python -c "import requests; requests.get('http://localhost:8000/api/health')" || exit 1

# Start gunicorn with uvicorn workers
CMD ["gunicorn", "api:app", "--workers", "4", "--worker-class", "uvicorn.workers.UvicornWorker", "--bind", "0.0.0.0:8000", "--timeout", "120"]
