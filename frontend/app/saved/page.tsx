'use client'

import { useRouter } from 'next/navigation'
import { useSavedPiecesStore, useAuthStore } from '@/lib/store'
import { useState, useEffect } from 'react'
import Header from '@/components/Header'
import Card from '@/components/ui/Card'
import Button from '@/components/ui/Button'
import { FeedbackButton } from '@/components/FeedbackButton'
import LoadingScreen from '@/components/LoadingScreen'

type SavedPiece = {
  id: string
  work_id: string
  title: string
  composer: string
  vibe?: string
  explanation?: string
}

export default function SavedPiecesPage() {
  const router = useRouter()
  const { isAuthenticated, user } = useAuthStore()
  const { pieces, removePiece, addPiece, fetchPieces, isLoading, error } = useSavedPiecesStore()
  const [recentlyRemoved, setRecentlyRemoved] = useState<SavedPiece | null>(null)
  const [authChecked, setAuthChecked] = useState(false)

  // Fetch saved pieces on mount
  useEffect(() => {
    // Wait a tick for auth to hydrate from localStorage/Supabase
    const timer = setTimeout(() => {
      setAuthChecked(true)
      if (!isAuthenticated) {
        router.push('/login')
        return
      }

      if (user) {
        fetchPieces(user.id)
      }
    }, 100)

    return () => clearTimeout(timer)
  }, [isAuthenticated, user, router, fetchPieces])

  const handleUnsave = async (piece: SavedPiece) => {
    if (!user) return

    try {
      await removePiece(piece.work_id, user.id)
      setRecentlyRemoved(piece)
    } catch (error) {
      console.error('Failed to unsave piece:', error)
    }
  }

  const handleUndo = async () => {
    if (!recentlyRemoved || !user) return

    try {
      await addPiece({
        work_id: recentlyRemoved.work_id,
        title: recentlyRemoved.title,
        composer: recentlyRemoved.composer,
        vibe: recentlyRemoved.vibe,
        explanation: recentlyRemoved.explanation,
      }, user.id)
      setRecentlyRemoved(null)
    } catch (error) {
      console.error('Failed to undo:', error)
    }
  }

  const handleOpenSpotify = (piece: SavedPiece) => {
    const searchQuery = encodeURIComponent(`${piece.composer} ${piece.title}`)
    const spotifySearchUrl = `https://open.spotify.com/search/${searchQuery}`
    window.open(spotifySearchUrl, '_blank', 'noopener,noreferrer')
  }

  if (!authChecked || isLoading) {
    return <LoadingScreen />
  }

  return (
    <div className="min-h-screen flex flex-col">
      <Header />

      <main className="flex-1 px-4 py-8">
        <div className="max-w-6xl mx-auto">
          <h1 className="text-2xl sm:text-3xl md:text-display text-light-cream text-center mb-8 sm:mb-12">
            Saved Pieces
          </h1>

          {error && (
            <div className="mb-6 flex justify-center">
              <Card className="max-w-md">
                <p className="text-red-600">Error loading saved pieces: {error}</p>
              </Card>
            </div>
          )}

          {recentlyRemoved && (
            <div className="mb-6 flex justify-center">
              <Card className="max-w-md">
                <div className="flex items-center justify-between gap-4">
                  <p className="text-dark-blue">
                    Removed &quot;{recentlyRemoved.title}&quot;
                  </p>
                  <Button variant="dark" onClick={handleUndo}>
                    Undo
                  </Button>
                </div>
              </Card>
            </div>
          )}

          {pieces.length === 0 ? (
            <div className="flex flex-col items-center justify-center py-20">
              <Card className="max-w-md text-center">
                <div className="mb-6">
                  <span className="text-6xl" role="img" aria-label="Music note">
                    🎵
                  </span>
                </div>
                <h2 className="text-lg sm:text-xl md:text-heading-xl font-bold text-dark-blue mb-4">
                  Your Collection Awaits
                </h2>
                <p className="text-body text-dark-blue mb-8">
                  Start building your personal classical music library. Every piece you discover can be saved here for later.
                </p>
                <Button variant="dark" onClick={() => router.push('/')}>
                  Discover Your First Piece
                </Button>
              </Card>
            </div>
          ) : (
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4 sm:gap-6">
              {pieces.map((piece) => (
                <Card key={piece.id} className="flex flex-col">
                  <h3 className="text-lg sm:text-xl font-bold text-dark-blue mb-2">
                    {piece.title}
                  </h3>
                  <p className="text-base sm:text-lg text-dark-blue mb-4">
                    by {piece.composer}
                  </p>
                  <div className="flex-1">
                    {piece.vibe && (
                      <p className="text-sm text-dark-blue mb-2">
                        Your vibe was: <span className="font-semibold">{piece.vibe}</span>
                      </p>
                    )}
                  </div>
                  <div className="mt-4 flex flex-col gap-2">
                    <Button variant="dark" onClick={() => handleOpenSpotify(piece)} className="w-full">
                      Open in Spotify
                    </Button>
                    <Button variant="dark" onClick={() => handleUnsave(piece)} className="w-full">
                      Unsave
                    </Button>

                    {piece.vibe && (
                      <FeedbackButton
                        workId={piece.work_id}
                        vibe={piece.vibe}
                        showComment={true}
                      />
                    )}
                  </div>
                </Card>
              ))}
            </div>
          )}
        </div>
      </main>
    </div>
  )
}
