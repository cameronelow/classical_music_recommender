'use client'

import { useEffect, useState, Suspense } from 'react'
import { useRouter, useSearchParams } from 'next/navigation'
import { useAuthStore, useSavedPiecesStore } from '@/lib/store'
import { searchByMood, type Recommendation } from '@/lib/api'
import { trackSearch, trackRecommendationView, trackSave } from '@/lib/analytics'
import { trackTiming } from '@/lib/performance'
import Header from '@/components/Header'
import LoadingScreen from '@/components/LoadingScreen'
import Card from '@/components/ui/Card'
import Button from '@/components/ui/Button'
import { FeedbackButton } from '@/components/FeedbackButton'

function RecommendContent() {
  const router = useRouter()
  const searchParams = useSearchParams()
  const vibe = searchParams.get('vibe') || ''

  const { isAuthenticated, user } = useAuthStore()
  const { addPiece, isPieceSaved, removePiece, isLoading: isSaving } = useSavedPiecesStore()

  const [recommendation, setRecommendation] = useState<Recommendation | null>(null)
  const [isLoading, setIsLoading] = useState(true)
  const [error, setError] = useState('')
  const [saveError, setSaveError] = useState('')

  useEffect(() => {
    async function fetchRecommendation() {
      if (!vibe) {
        router.push('/')
        return
      }

      const startTime = Date.now()

      try {
        setIsLoading(true)

        // Fetch top 5 recommendations for variety
        const response = await searchByMood(vibe, 5)

        if (response.recommendations && response.recommendations.length > 0) {
          // Randomly select from top results to avoid repetition
          const randomIndex = Math.floor(Math.random() * response.recommendations.length)
          const selectedRecommendation = response.recommendations[randomIndex]
          setRecommendation(selectedRecommendation)

          // Track analytics
          trackSearch(vibe, response.recommendations.length)
          trackRecommendationView(selectedRecommendation.work_id, vibe)
          trackTiming('recommendation_fetch', startTime)
        } else {
          setError('No recommendations found. Try a different vibe!')
          trackSearch(vibe, 0)
        }
      } catch (err) {
        setError('Failed to fetch recommendation. Please try again.')
        trackSearch(vibe, 0)
      } finally {
        setIsLoading(false)
      }
    }

    fetchRecommendation()
  }, [vibe, router])

  const handleSave = async () => {
    if (!recommendation || !user) return

    const startTime = Date.now()
    setSaveError('')

    try {
      if (isPieceSaved(recommendation.work_id)) {
        await removePiece(recommendation.work_id, user.id)
        trackSave(recommendation.work_id, 'unsaved')
      } else {
        await addPiece({
          work_id: recommendation.work_id,
          title: recommendation.title,
          composer: recommendation.composer,
          composer_id: undefined, // Add if available in recommendation
          vibe,
          explanation: recommendation.explanation,
        }, user.id)
        trackSave(recommendation.work_id, 'saved')
      }

      trackTiming('save_action', startTime)
    } catch (error: any) {
      setSaveError(error.message || 'Failed to save piece')
      console.error('Error saving piece:', error)
    }
  }

  const handleOpenSpotify = () => {
    if (!recommendation) return

    const searchQuery = encodeURIComponent(`${recommendation.composer} ${recommendation.title}`)
    const spotifySearchUrl = `https://open.spotify.com/search/${searchQuery}`
    window.open(spotifySearchUrl, '_blank', 'noopener,noreferrer')
  }

  if (isLoading) {
    return <LoadingScreen />
  }

  if (error || !recommendation) {
    return (
      <div className="min-h-screen flex flex-col">
        <Header />
        <div className="flex-1 flex items-center justify-center px-4">
          <Card className="max-w-md text-center">
            <p className="text-xl text-dark-blue mb-4">{error || 'No recommendation found'}</p>
            <Button variant="dark" onClick={() => router.push('/')}>
              Try Again
            </Button>
          </Card>
        </div>
      </div>
    )
  }

  const isSaved = isPieceSaved(recommendation.work_id)

  return (
    <div className="min-h-screen flex flex-col">
      <Header />

      <main className="flex-1 flex items-center justify-center px-4">
        <Card className="w-full max-w-2xl text-center">
          <h1 className="text-display font-bold text-dark-blue mb-2">
            {recommendation.title}
          </h1>
          <h2 className="text-heading-xl text-dark-blue mb-6">
            by {recommendation.composer}
          </h2>

          <div className="my-8">
            <p className="text-dark-blue text-body-lg mb-2">
              Your vibe was: <span className="font-semibold">{vibe}</span>
            </p>
            <p className="text-dark-blue text-body">
              <span className="font-medium">{recommendation.explanation}</span>
            </p>
          </div>

          <div className="flex flex-col sm:flex-row gap-3 justify-center mt-8">
            <Button variant="dark" onClick={handleOpenSpotify} className="w-full sm:w-auto">
              Open in Spotify
            </Button>

            {isAuthenticated ? (
              <>
                <Button
                  variant="dark"
                  onClick={handleSave}
                  className="w-full sm:w-auto"
                  disabled={isSaving}
                >
                  {isSaving ? 'Saving...' : isSaved ? 'Unsave' : 'Save'}
                </Button>
                <Button variant="dark" onClick={() => router.push('/saved')} className="w-full sm:w-auto">
                  View Saved Pieces
                </Button>
              </>
            ) : (
              <Button
                variant="dark"
                onClick={() => {
                  const currentPath = `/recommend?vibe=${encodeURIComponent(vibe)}`
                  router.push(`/login?redirectTo=${encodeURIComponent(currentPath)}`)
                }}
                className="w-full sm:w-auto"
              >
                Sign In to Save
              </Button>
            )}
          </div>

          {saveError && (
            <div className="mt-4">
              <p className="text-red-600 text-sm">{saveError}</p>
            </div>
          )}

          {isAuthenticated && (
            <div className="mt-6">
              <FeedbackButton
                workId={recommendation.work_id}
                vibe={vibe}
                showComment={true}
              />
            </div>
          )}

          <div className="mt-8">
            <Button variant="secondary" onClick={() => router.push('/')}>
              Find Another Piece
            </Button>
          </div>
        </Card>
      </main>
    </div>
  )
}

export default function RecommendPage() {
  return (
    <Suspense fallback={<LoadingScreen />}>
      <RecommendContent />
    </Suspense>
  )
}
