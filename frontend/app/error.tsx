'use client'

import { useEffect } from 'react'
import Header from '@/components/Header'
import Card from '@/components/ui/Card'
import Button from '@/components/ui/Button'

export default function Error({
  error,
  reset,
}: {
  error: Error & { digest?: string }
  reset: () => void
}) {
  useEffect(() => {
    // Log error to error reporting service
    console.error('Application error:', error)
  }, [error])

  return (
    <div className="min-h-screen flex flex-col">
      <Header />

      <main className="flex-1 flex items-center justify-center px-4">
        <Card className="max-w-md text-center">
          <h1 className="text-display font-bold text-dark-blue mb-4">
            Something went wrong
          </h1>
          <p className="text-body text-dark-blue mb-8">
            We encountered an unexpected error. Please try again or return home.
          </p>
          <div className="flex flex-col sm:flex-row gap-3 justify-center">
            <Button variant="dark" onClick={reset}>
              Try Again
            </Button>
            <Button variant="secondary" onClick={() => window.location.href = '/'}>
              Return Home
            </Button>
          </div>
        </Card>
      </main>
    </div>
  )
}
