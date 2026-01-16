'use client'

import { useState } from 'react'
import { useRouter } from 'next/navigation'
import Header from './Header'
import Button from './ui/Button'
import Input from './ui/Input'

export default function LandingPage() {
  const router = useRouter()
  const [vibe, setVibe] = useState('')
  const [isLoading, setIsLoading] = useState(false)

  const handleFindPiece = async () => {
    if (!vibe.trim()) return

    setIsLoading(true)
    // Navigate to loading screen, which will then fetch and show recommendation
    router.push(`/recommend?vibe=${encodeURIComponent(vibe)}`)
  }

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter') {
      handleFindPiece()
    }
  }

  return (
    <div className="min-h-screen flex flex-col">
      <Header />

      <main id="main-content" className="flex-1 flex flex-col items-center justify-center px-4">
        <div className="flex flex-col items-center text-center max-w-xl w-full">
          <h1 className="text-light-cream text-3xl sm:text-5xl md:text-hero mb-8 sm:mb-12">
            What&apos;s your vibe today?
          </h1>

          <div className="space-y-4 w-full">
            <Input
              type="text"
              placeholder="I'm feeling..."
              value={vibe}
              onChange={(e) => setVibe(e.target.value)}
              onKeyDown={handleKeyDown}
              className="text-body-lg py-3"
              aria-label="Enter your mood or vibe"
            />

            <Button
              variant="landing"
              onClick={handleFindPiece}
              disabled={!vibe.trim() || isLoading}
            >
              {isLoading ? "Thinking..." : "Find My Piece"}
            </Button>
          </div>
        </div>
      </main>
    </div>
  );
}
