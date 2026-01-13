// API client for classical music recommender backend

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'

export interface Recommendation {
  work_id: string
  title: string
  composer: string
  work_type?: string
  key?: string
  similarity_score: number
  explanation: string
  spotify_url?: string
}

export interface RecommendationResponse {
  recommendations: Recommendation[]
  query: string
}

export async function searchByMood(query: string, n: number = 1): Promise<RecommendationResponse> {
  try {
    const response = await fetch(`${API_BASE_URL}/api/search/mood`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ query, n }),
    })

    if (!response.ok) {
      throw new Error('Failed to fetch recommendations')
    }

    return await response.json()
  } catch (error) {
    console.error('API Error:', error)
    // Return mock data for development
    return {
      query,
      recommendations: [
        {
          work_id: 'mock-1',
          title: 'Nocturne in E-flat major, Op. 9, No. 2',
          composer: 'Frédéric Chopin',
          work_type: 'Nocturne',
          key: 'E-flat major',
          similarity_score: 0.92,
          explanation: `This piece perfectly matches your "${query}" vibe with its gentle, flowing melody and introspective character.`,
          spotify_url: 'https://open.spotify.com/track/example',
        },
      ],
    }
  }
}

export async function searchByActivity(
  activity: string,
  context?: string,
  n: number = 1
): Promise<RecommendationResponse> {
  try {
    const response = await fetch(`${API_BASE_URL}/api/search/activity`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ activity, context, n }),
    })

    if (!response.ok) {
      throw new Error('Failed to fetch recommendations')
    }

    return await response.json()
  } catch (error) {
    console.error('API Error:', error)
    throw error
  }
}

export async function getSimilarWorks(workId: string, n: number = 5): Promise<Recommendation[]> {
  try {
    const response = await fetch(`${API_BASE_URL}/api/recommend/similar/${workId}?n=${n}`)

    if (!response.ok) {
      throw new Error('Failed to fetch similar works')
    }

    const data = await response.json()
    return data.recommendations
  } catch (error) {
    console.error('API Error:', error)
    throw error
  }
}

// Feedback API

export interface FeedbackResponse {
  id: string
  user_id: string
  work_id: string
  rating: number
  comment?: string
  vibe: string
  created_at: string
  updated_at: string
}

export interface FeedbackStats {
  work_id: string
  thumbs_up_count: number
  thumbs_down_count: number
  total_feedbacks: number
  avg_rating: number
  user_feedback?: FeedbackResponse
}

export async function submitFeedback(
  workId: string,
  vibe: string,
  rating: number,
  comment?: string,
  userId?: string
): Promise<FeedbackResponse> {
  try {
    const headers: Record<string, string> = {
      'Content-Type': 'application/json',
    }

    if (userId) {
      headers['X-User-ID'] = userId
    }

    const response = await fetch(`${API_BASE_URL}/api/feedback`, {
      method: 'POST',
      headers,
      body: JSON.stringify({
        work_id: workId,
        rating,
        comment,
        vibe,
      }),
    })

    if (!response.ok) {
      throw new Error('Failed to submit feedback')
    }

    return await response.json()
  } catch (error) {
    console.error('API Error:', error)
    throw error
  }
}

export async function getFeedbackStats(
  workId: string,
  vibe: string,
  userId?: string
): Promise<FeedbackStats> {
  try {
    const headers: Record<string, string> = {}

    if (userId) {
      headers['X-User-ID'] = userId
    }

    const response = await fetch(
      `${API_BASE_URL}/api/feedback/${workId}?vibe=${encodeURIComponent(vibe)}`,
      { headers }
    )

    if (!response.ok) {
      throw new Error('Failed to fetch feedback stats')
    }

    return await response.json()
  } catch (error) {
    console.error('API Error:', error)
    throw error
  }
}

export async function deleteFeedback(
  workId: string,
  vibe: string,
  userId?: string
): Promise<void> {
  try {
    const headers: Record<string, string> = {}

    if (userId) {
      headers['X-User-ID'] = userId
    }

    const response = await fetch(
      `${API_BASE_URL}/api/feedback/${workId}?vibe=${encodeURIComponent(vibe)}`,
      {
        method: 'DELETE',
        headers,
      }
    )

    if (!response.ok) {
      throw new Error('Failed to delete feedback')
    }
  } catch (error) {
    console.error('API Error:', error)
    throw error
  }
}

// Saved Pieces API

export interface SavedPiece {
  id: string
  user_id: string
  work_id: string
  title: string
  composer: string
  composer_id?: string
  vibe?: string
  explanation?: string
  notes?: string
  saved_at: string
}

export interface SavedPiecesResponse {
  pieces: SavedPiece[]
  count: number
}

export async function getSavedPieces(userId: string): Promise<SavedPiecesResponse> {
  try {
    const response = await fetch(`${API_BASE_URL}/api/saved-pieces`, {
      headers: {
        'X-User-ID': userId,
      },
    })

    if (!response.ok) {
      throw new Error('Failed to fetch saved pieces')
    }

    return await response.json()
  } catch (error) {
    console.error('API Error:', error)
    throw error
  }
}

export async function savePiece(
  piece: {
    work_id: string
    title: string
    composer: string
    composer_id?: string
    vibe?: string
    explanation?: string
    notes?: string
  },
  userId: string
): Promise<SavedPiece> {
  try {
    const response = await fetch(`${API_BASE_URL}/api/saved-pieces`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'X-User-ID': userId,
      },
      body: JSON.stringify(piece),
    })

    if (!response.ok) {
      throw new Error('Failed to save piece')
    }

    return await response.json()
  } catch (error) {
    console.error('API Error:', error)
    throw error
  }
}

export async function unsavePiece(workId: string, userId: string): Promise<void> {
  try {
    const response = await fetch(`${API_BASE_URL}/api/saved-pieces/${workId}`, {
      method: 'DELETE',
      headers: {
        'X-User-ID': userId,
      },
    })

    if (!response.ok) {
      throw new Error('Failed to unsave piece')
    }
  } catch (error) {
    console.error('API Error:', error)
    throw error
  }
}

// Analytics API

export interface AnalyticsEvent {
  event_type: string
  event_data?: Record<string, any>
  page_url?: string
  referrer?: string
  user_agent?: string
  session_id?: string
}

export async function trackEvent(
  event: AnalyticsEvent,
  userId?: string
): Promise<void> {
  try {
    const headers: Record<string, string> = {
      'Content-Type': 'application/json',
    }

    if (userId) {
      headers['X-User-ID'] = userId
    }

    const response = await fetch(`${API_BASE_URL}/api/analytics`, {
      method: 'POST',
      headers,
      body: JSON.stringify(event),
    })

    if (!response.ok) {
      // Don't throw - analytics failures shouldn't break the app
      console.warn('Failed to track analytics event')
    }
  } catch (error) {
    // Silently fail - analytics shouldn't break user experience
    console.warn('Analytics error:', error)
  }
}
