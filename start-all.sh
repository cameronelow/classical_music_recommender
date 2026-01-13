#!/bin/bash

# Script to start both frontend and backend servers
# Usage: ./start-all.sh

echo "🎵 Starting Classical Music Recommender App..."
echo ""

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "❌ Virtual environment not found. Creating one..."
    python3 -m venv venv
    source venv/bin/activate
    pip install -r backend/requirements.txt
else
    source venv/bin/activate
fi

# Check if backend dependencies are installed
if ! python -c "import fastapi" 2>/dev/null; then
    echo "📦 Installing backend dependencies..."
    pip install -r backend/requirements.txt
fi

# Check if frontend dependencies are installed
if [ ! -d "frontend/node_modules" ]; then
    echo "📦 Installing frontend dependencies..."
    cd frontend && npm install && cd ..
fi

echo ""
echo "✨ Starting servers..."
echo "   Backend:  http://localhost:8000"
echo "   Frontend: http://localhost:3000"
echo "   API Docs: http://localhost:8000/docs"
echo ""
echo "Press Ctrl+C to stop both servers"
echo ""

# Start backend in background (run from project root for correct paths)
python3 backend/api.py &
BACKEND_PID=$!

# Wait a moment for backend to start
sleep 2

# Start frontend in background
cd frontend
npm run dev &
FRONTEND_PID=$!
cd ..

# Function to cleanup on exit
cleanup() {
    echo ""
    echo "🛑 Stopping servers..."
    kill $BACKEND_PID 2>/dev/null
    kill $FRONTEND_PID 2>/dev/null
    exit 0
}

# Trap Ctrl+C and call cleanup
trap cleanup INT TERM

# Wait for both processes
wait
