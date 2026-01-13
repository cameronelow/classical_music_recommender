#!/bin/bash

echo "Starting Classical Music Recommender Backend API..."
echo "API will be available at http://localhost:8000"
echo "API docs at http://localhost:8000/docs"
echo ""

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    source venv/bin/activate
fi

# Run from project root so paths resolve correctly
python3 backend/api.py
