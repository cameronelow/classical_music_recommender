# Classical Music Recommender - Frontend

A beautiful, modern web interface for discovering classical music based on your mood and vibe.

## Features

- **Mood-based Search**: Find classical pieces that match your current vibe
- **User Authentication**: Sign up and log in to save your favorite pieces
- **Saved Pieces**: Keep track of recommendations you love
- **Responsive Design**: Works beautifully on desktop, tablet, and mobile
- **Spotify Integration**: Open recommended pieces directly in Spotify

## Tech Stack

- **Framework**: Next.js 14 with App Router
- **Language**: TypeScript
- **Styling**: Tailwind CSS
- **State Management**: Zustand
- **API Integration**: REST API to Python backend

## Getting Started

### Prerequisites

- Node.js 18.17.0 or higher
- npm or yarn

### Installation

1. Install dependencies:
```bash
npm install
```

2. Create a `.env` file:
```bash
cp .env.example .env
```

3. Update the `.env` file with your API URL (defaults to `http://localhost:8000`)

### Development

Run the development server:
```bash
npm run dev
```

Open [http://localhost:3000](http://localhost:3000) in your browser.

### Build

Build for production:
```bash
npm run build
```

Start production server:
```bash
npm start
```

## Project Structure

```
frontend/
├── app/                    # Next.js app directory
│   ├── login/             # Login page
│   ├── signup/            # Sign up page
│   ├── recommend/         # Recommendation result page
│   ├── saved/             # Saved pieces page
│   ├── profile/           # Profile page
│   ├── layout.tsx         # Root layout
│   ├── page.tsx           # Home page
│   └── globals.css        # Global styles
├── components/            # React components
│   ├── ui/               # Reusable UI components
│   │   ├── Button.tsx
│   │   ├── Input.tsx
│   │   └── Card.tsx
│   ├── Header.tsx        # App header
│   ├── LandingPage.tsx   # Landing page component
│   └── LoadingScreen.tsx # Loading state component
├── lib/                  # Utilities
│   ├── api.ts           # API client
│   └── store.ts         # Zustand state management
└── public/              # Static assets

```

## Design

The UI follows a pixel-perfect implementation of the Figma designs with:

- **Color Palette**:
  - `#1A3263` - Dark blue (text, buttons, gradient background)
  - `#FAB95B` - Warm yellow (gradient background)
  - `#E8E2DB` - Light cream (cards, light text, buttons)

- **Typography**: Inter font family
- **Gradient Background**: Dark blue to warm yellow
- **Rounded Cards**: Soft shadows for depth
- **Responsive**: Mobile-first approach

## API Integration

The frontend connects to the Python backend recommender service. Key endpoints:

- `POST /api/search/mood` - Search by mood/vibe
- `POST /api/search/activity` - Search by activity
- `GET /api/recommend/similar/:workId` - Get similar works

See [lib/api.ts](lib/api.ts) for full API documentation.

## State Management

Using Zustand for lightweight state management:

- **Auth Store**: User authentication state
- **Saved Pieces Store**: Persisted saved recommendations

Both stores use localStorage persistence.
