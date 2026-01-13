'use client'

import { useEffect, Suspense } from 'react'
import { useRouter, useSearchParams } from 'next/navigation'
import { useAuthStore } from '@/lib/store'
import Header from '@/components/Header'
import LoadingScreen from '@/components/LoadingScreen'
import Card from '@/components/ui/Card'

function ShareContent() {
  const router = useRouter()
  const searchParams = useSearchParams()
  const { isAuthenticated } = useAuthStore()

  const title = searchParams.get('title') || ''
  const composer = searchParams.get('composer') || ''
  const vibe = searchParams.get('vibe') || ''
  const from = searchParams.get('from') || 'recommend'

  useEffect(() => {
    if (!isAuthenticated) {
      router.push('/login')
    }
  }, [isAuthenticated, router])

  useEffect(() => {
    if (!title || !composer || !vibe) {
      router.push('/')
    }
  }, [title, composer, vibe, router])

  const shareText = `My Classical Vibe Today...\n\n"${title}" by ${composer}\n\nVibe: ${vibe}\n\nDiscover your classical music match at Espressivo`
  const shareUrl = typeof window !== 'undefined' ? window.location.origin : ''

  const handleShare = (platform: string) => {
    const encodedText = encodeURIComponent(shareText)
    const encodedUrl = encodeURIComponent(shareUrl)

    let shareLink = ''

    switch (platform) {
      case 'twitter':
        shareLink = `https://twitter.com/intent/tweet?text=${encodedText}`
        break
      case 'facebook':
        shareLink = `https://www.facebook.com/sharer/sharer.php?u=${encodedUrl}&quote=${encodedText}`
        break
      case 'linkedin':
        shareLink = `https://www.linkedin.com/sharing/share-offsite/?url=${encodedUrl}`
        break
      case 'whatsapp':
        shareLink = `https://wa.me/?text=${encodedText}`
        break
      case 'instagram':
        // Instagram doesn't support direct web sharing, so we'll copy to clipboard
        navigator.clipboard.writeText(shareText).then(() => {
          alert('Text copied to clipboard! You can now paste it in your Instagram story or post.')
        })
        return
      default:
        return
    }

    window.open(shareLink, '_blank', 'noopener,noreferrer,width=600,height=400')
  }

  if (!isAuthenticated) {
    return <LoadingScreen />
  }

  return (
    <div className="min-h-screen flex flex-col">
      <Header />

      <div className="flex-1 flex items-center justify-center px-4 -mt-20">
        <Card className="w-full max-w-md text-center">
          <h1 className="text-2xl md:text-3xl font-bold text-dark-blue mb-6">
            My Classical Vibe Today...
          </h1>

          <div className="mb-8">
            <p className="text-lg md:text-xl font-semibold text-dark-blue mb-1">
              {title}
            </p>
            <p className="text-md md:text-lg text-dark-blue mb-4">
              by {composer}
            </p>
            <p className="text-sm text-dark-blue">
              Vibe: <span className="font-medium">{vibe}</span>
            </p>
          </div>

          <div className="grid grid-cols-5 gap-3 mb-8">
            <button
              onClick={() => handleShare('twitter')}
              className="aspect-square bg-dark-blue rounded-lg hover:opacity-80 transition-opacity flex items-center justify-center"
              aria-label="Share on Twitter"
            >
              <svg className="w-6 h-6 text-light-cream" fill="currentColor" viewBox="0 0 24 24">
                <path d="M18.244 2.25h3.308l-7.227 8.26 8.502 11.24H16.17l-5.214-6.817L4.99 21.75H1.68l7.73-8.835L1.254 2.25H8.08l4.713 6.231zm-1.161 17.52h1.833L7.084 4.126H5.117z" />
              </svg>
            </button>

            <button
              onClick={() => handleShare('facebook')}
              className="aspect-square bg-dark-blue rounded-lg hover:opacity-80 transition-opacity flex items-center justify-center"
              aria-label="Share on Facebook"
            >
              <svg className="w-6 h-6 text-light-cream" fill="currentColor" viewBox="0 0 24 24">
                <path d="M9.101 23.691v-7.98H6.627v-3.667h2.474v-1.58c0-4.085 1.848-5.978 5.858-5.978.401 0 .955.042 1.468.103a8.68 8.68 0 0 1 1.141.195v3.325a8.623 8.623 0 0 0-.653-.036 26.805 26.805 0 0 0-.733-.009c-.707 0-1.259.096-1.675.309a1.686 1.686 0 0 0-.679.622c-.258.42-.374.995-.374 1.752v1.297h3.919l-.386 2.103-.287 1.564h-3.246v8.245C19.396 23.238 24 18.179 24 12.044c0-6.627-5.373-12-12-12s-12 5.373-12 12c0 5.628 3.874 10.35 9.101 11.647Z" />
              </svg>
            </button>

            <button
              onClick={() => handleShare('linkedin')}
              className="aspect-square bg-dark-blue rounded-lg hover:opacity-80 transition-opacity flex items-center justify-center"
              aria-label="Share on LinkedIn"
            >
              <svg className="w-6 h-6 text-light-cream" fill="currentColor" viewBox="0 0 24 24">
                <path d="M20.447 20.452h-3.554v-5.569c0-1.328-.027-3.037-1.852-3.037-1.853 0-2.136 1.445-2.136 2.939v5.667H9.351V9h3.414v1.561h.046c.477-.9 1.637-1.85 3.37-1.85 3.601 0 4.267 2.37 4.267 5.455v6.286zM5.337 7.433c-1.144 0-2.063-.926-2.063-2.065 0-1.138.92-2.063 2.063-2.063 1.14 0 2.064.925 2.064 2.063 0 1.139-.925 2.065-2.064 2.065zm1.782 13.019H3.555V9h3.564v11.452zM22.225 0H1.771C.792 0 0 .774 0 1.729v20.542C0 23.227.792 24 1.771 24h20.451C23.2 24 24 23.227 24 22.271V1.729C24 .774 23.2 0 22.222 0h.003z" />
              </svg>
            </button>

            <button
              onClick={() => handleShare('whatsapp')}
              className="aspect-square bg-dark-blue rounded-lg hover:opacity-80 transition-opacity flex items-center justify-center"
              aria-label="Share on WhatsApp"
            >
              <svg className="w-6 h-6 text-light-cream" fill="currentColor" viewBox="0 0 24 24">
                <path d="M17.472 14.382c-.297-.149-1.758-.867-2.03-.967-.273-.099-.471-.148-.67.15-.197.297-.767.966-.94 1.164-.173.199-.347.223-.644.075-.297-.15-1.255-.463-2.39-1.475-.883-.788-1.48-1.761-1.653-2.059-.173-.297-.018-.458.13-.606.134-.133.298-.347.446-.52.149-.174.198-.298.298-.497.099-.198.05-.371-.025-.52-.075-.149-.669-1.612-.916-2.207-.242-.579-.487-.5-.669-.51-.173-.008-.371-.01-.57-.01-.198 0-.52.074-.792.372-.272.297-1.04 1.016-1.04 2.479 0 1.462 1.065 2.875 1.213 3.074.149.198 2.096 3.2 5.077 4.487.709.306 1.262.489 1.694.625.712.227 1.36.195 1.871.118.571-.085 1.758-.719 2.006-1.413.248-.694.248-1.289.173-1.413-.074-.124-.272-.198-.57-.347m-5.421 7.403h-.004a9.87 9.87 0 01-5.031-1.378l-.361-.214-3.741.982.998-3.648-.235-.374a9.86 9.86 0 01-1.51-5.26c.001-5.45 4.436-9.884 9.888-9.884 2.64 0 5.122 1.03 6.988 2.898a9.825 9.825 0 012.893 6.994c-.003 5.45-4.437 9.884-9.885 9.884m8.413-18.297A11.815 11.815 0 0012.05 0C5.495 0 .16 5.335.157 11.892c0 2.096.547 4.142 1.588 5.945L.057 24l6.305-1.654a11.882 11.882 0 005.683 1.448h.005c6.554 0 11.89-5.335 11.893-11.893a11.821 11.821 0 00-3.48-8.413Z" />
              </svg>
            </button>

            <button
              onClick={() => handleShare('instagram')}
              className="aspect-square bg-dark-blue rounded-lg hover:opacity-80 transition-opacity flex items-center justify-center"
              aria-label="Share on Instagram"
            >
              <svg className="w-6 h-6 text-light-cream" fill="currentColor" viewBox="0 0 24 24">
                <path d="M12 2.163c3.204 0 3.584.012 4.85.07 3.252.148 4.771 1.691 4.919 4.919.058 1.265.069 1.645.069 4.849 0 3.205-.012 3.584-.069 4.849-.149 3.225-1.664 4.771-4.919 4.919-1.266.058-1.644.07-4.85.07-3.204 0-3.584-.012-4.849-.07-3.26-.149-4.771-1.699-4.919-4.92-.058-1.265-.07-1.644-.07-4.849 0-3.204.013-3.583.07-4.849.149-3.227 1.664-4.771 4.919-4.919 1.266-.057 1.645-.069 4.849-.069zm0-2.163c-3.259 0-3.667.014-4.947.072-4.358.2-6.78 2.618-6.98 6.98-.059 1.281-.073 1.689-.073 4.948 0 3.259.014 3.668.072 4.948.2 4.358 2.618 6.78 6.98 6.98 1.281.058 1.689.072 4.948.072 3.259 0 3.668-.014 4.948-.072 4.354-.2 6.782-2.618 6.979-6.98.059-1.28.073-1.689.073-4.948 0-3.259-.014-3.667-.072-4.947-.196-4.354-2.617-6.78-6.979-6.98-1.281-.059-1.69-.073-4.949-.073zm0 5.838c-3.403 0-6.162 2.759-6.162 6.162s2.759 6.163 6.162 6.163 6.162-2.759 6.162-6.163c0-3.403-2.759-6.162-6.162-6.162zm0 10.162c-2.209 0-4-1.79-4-4 0-2.209 1.791-4 4-4s4 1.791 4 4c0 2.21-1.791 4-4 4zm6.406-11.845c-.796 0-1.441.645-1.441 1.44s.645 1.44 1.441 1.44c.795 0 1.439-.645 1.439-1.44s-.644-1.44-1.439-1.44z" />
              </svg>
            </button>
          </div>

          <button
            onClick={() => router.back()}
            className="text-dark-blue hover:underline text-sm"
          >
            {from === 'saved' ? 'Back to Saved Pieces' : 'Back to Recommendation'}
          </button>
        </Card>
      </div>
    </div>
  )
}

export default function SharePage() {
  return (
    <Suspense fallback={<LoadingScreen />}>
      <ShareContent />
    </Suspense>
  )
}
