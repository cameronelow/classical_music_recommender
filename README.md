# Espressivo: A Classical Music Recommender

A web application to discover classical music pieces based on your mood.

> *Note: "Classical" here goes beyond the classical period and extends to other eras and styles (Baroque, Romantic, Modern, etc.)*

## ✨ Features

- 🎵 **Mood-Based Discovery**: Find classical pieces that match your current vibe
- 🤖 **AI-Powered Recommendations**: Semantic search using advanced NLP
- 💾 **Save Favorites**: Create your personal collection of discoveries
- 🎨 **Beautiful UI**: Pixel-perfect design with gradient backgrounds
- 📱 **Fully Responsive**: Works seamlessly on desktop and mobile
- 🔐 **User Accounts**: Sign up to save your personalized recs

## Data Source

This project uses the **MusicBrainz API** to extract comprehensive metadata about classical music composers, works, and recordings.

## Features

- Extract classical music metadata from MusicBrainz
- Process and transform data into normalized tables
- Track composers, works (albums/release groups), and associated metadata
- Support for multiple composers and batch processing

## 🚀 Quick Start

### Prerequisites

- **Node.js** 18.17.0 or higher
- **Python** 3.8 or higher
- **npm** or **yarn**

### 1. Install Dependencies

```bash
# Install Python dependencies (recommender engine + backend)
pip install -r requirements.txt

# Install frontend dependencies
cd frontend
npm install
cd ..
```

### 2. Set Up Data

Ensure you have processed music data in `data/processed/`. If not, run the ETL pipeline:

```bash
# Extract data from MusicBrainz
cd etl/extract
python3 run_extract.py --artists "Bach" "Mozart" "Beethoven" "Chopin"

# Transform into normalized tables
cd ../transform
python3 run_transform.py --artists Bach Mozart Beethoven Chopin
cd ../..
```

### 3. Start the Application

#### Option A: Using the startup scripts (easiest)

```bash
# Terminal 1: Start the backend API
./start-backend.sh

# Terminal 2: Start the frontend
./start-frontend.sh
```

#### Option B: Manual startup

```bash
# Terminal 1: Start the backend API
cd backend
python3 api.py

# Terminal 2: Start the frontend
cd frontend
npm run dev
```

### 4. Open the App

Visit [http://localhost:3000](http://localhost:3000) in your browser!

- **Frontend**: http://localhost:3000
- **Backend API**: http://localhost:8000
- **API Docs**: http://localhost:8000/docs

## 📁 Project Structure

```
classical_music_recommender/
├── frontend/             # Next.js 14 web application
│   ├── app/             # Pages (login, signup, recommend, etc.)
│   ├── components/      # React components
│   ├── lib/             # API client, state management
│   └── public/          # Static assets
├── backend/             # FastAPI backend
│   ├── api.py          # REST API endpoints
│   └── requirements.txt
├── recommender/         # Core recommendation engine
│   ├── engine.py       # Recommendation algorithms
│   ├── semantic.py     # Semantic search with NLP
│   └── service.py      # Service layer
├── data/
│   ├── raw/            # Raw extracted data
│   ├── processed/      # Transformed normalized tables
│   └── embeddings/     # Semantic embeddings
├── etl/
│   ├── extract/        # Data extraction from MusicBrainz
│   └── transform/      # Data transformation pipeline
├── tagging/            # AI-powered music tagging
└── docs/               # Documentation
```

## Data Pipeline

### Extraction Phase
- Connects to MusicBrainz API
- Extracts composer metadata (birth/death years, period, country)
- Fetches works/release groups with tags
- Saves raw data in JSON and Parquet formats

### Transformation Phase
- Parses catalog numbers (BWV, Op., K., etc.)
- Classifies musical periods (Baroque, Classical, Romantic, etc.)
- Extracts work types (Symphony, Concerto, Sonata, etc.)
- Generates normalized tables:
  - `composers.parquet` - Composer information
  - `works.parquet` - Musical works/albums
  - `work_tags.parquet` - Genre and style tags

## Output Data Schema

### Composers Table
- composer_id, name, sort_name
- birth_year, death_year, country
- period (Baroque, Classical, Romantic, etc.)
- annotation, tags
- total_works_count

### Works Table
- work_id, composer_id
- title, work_type (Symphony, Concerto, etc.)
- catalog_number (BWV 1001, Op. 27, etc.)
- key (D minor, C major, etc.)
- tags, recording_count

### Work Tags Table
- work_id, tag, source

## Rate Limiting

The system respects MusicBrainz API rate limits (1 request per second by default). Adjust in `.env`:

```bash
MUSICBRAINZ_RATE_LIMIT=1.0  # seconds between requests
```

## 🎨 Tech Stack

### Frontend
- **Framework**: Next.js 14 with App Router
- **Language**: TypeScript
- **Styling**: Tailwind CSS
- **State**: Zustand
- **UI**: Custom components with Inter font

### Backend
- **API**: FastAPI (Python)
- **ML**: sentence-transformers for semantic search
- **Data**: pandas, scikit-learn

### Design
- **Colors**:
  - Dark Blue (#1A3263)
  - Warm Yellow (#FAB95B)
  - Light Cream (#E8E2DB)
- **Typography**: Inter font family
- **Layout**: Gradient backgrounds, card-based design

## 📖 Documentation

- **[Quick Start Guide](QUICK_START.md)** - Get running in 3 steps
- **[Semantic Search Guide](SEMANTIC_SEARCH.md)** - AI-powered mood-based search
- **[Auto-Tagging Guide](AUTO_TAGGING.md)** - Enrich your dataset with AI tags
- **[Features Overview](FEATURES.md)** - UI screens and functionality
- **[Backend API Docs](http://localhost:8000/docs)** - Interactive API documentation (when running)

## 🎯 How It Works

1. **User Input**: Enter your mood or vibe (e.g., "relaxing", "energetic", "melancholic")
2. **Semantic Search**: The system uses NLP to understand your mood
3. **AI Matching**: Advanced algorithms find pieces that match your emotional state
4. **Recommendation**: Get a beautiful classical piece with context and explanation
5. **Save & Enjoy**: Save your favorites and explore more

## 🔄 Development Workflow

```bash
# Run backend tests
cd recommender
python3 -m pytest

# Run example recommendations
python3 run_recommender_example.py

# Build frontend for production
cd frontend
npm run build
npm start
```

## 🛠️ Configuration

### Environment Variables

Copy `.env.example` to `.env` and fill in your values:

```bash
cp .env.example .env
cp frontend/.env.example frontend/.env.local
```

Edit `.env` with your configuration:
```bash
MUSICBRAINZ_EMAIL=your_email@example.com
ANTHROPIC_API_KEY=your_key_here  # For auto-tagging feature
```

Edit `frontend/.env.local`:
```bash
NEXT_PUBLIC_API_URL=http://localhost:8000
NEXT_PUBLIC_SUPABASE_URL=your_supabase_url
NEXT_PUBLIC_SUPABASE_ANON_KEY=your_supabase_key
```

## 📊 Data Pipeline

### 1. Extraction
- Connects to MusicBrainz API
- Extracts composer and work metadata
- Respects rate limits (1 req/sec)

### 2. Transformation
- Parses catalog numbers (BWV, Op., K.)
- Classifies periods (Baroque, Classical, Romantic)
- Generates normalized tables

### 3. Semantic Enhancement
- Creates rich descriptions for each piece
- Generates embeddings using sentence-transformers
- Enables mood-based search


## 📝 License

This project uses data from MusicBrainz, which is licensed under CC0 1.0.

## 🙏 Acknowledgments

- **MusicBrainz** for comprehensive music metadata
- **sentence-transformers** for semantic search capabilities
- **Next.js** and **FastAPI** for excellent developer experience
