import type { Metadata } from 'next'
import { Actor } from 'next/font/google'
import './globals.css'
import AuthProvider from '@/components/AuthProvider'
import SkipToContent from '@/components/SkipToContent'
import WebVitals from '@/components/WebVitals'

const actor = Actor({ weight: '400', subsets: ['latin'] })

export const metadata: Metadata = {
  title: 'CLASSICAL VIBE - Classical Music Recommender',
  description: 'Discover classical music tailored to your vibe with AI-powered recommendations',
  keywords: ['classical music', 'music recommendation', 'AI', 'mood-based music'],
  authors: [{ name: 'CLASSICAL VIBE' }],
  openGraph: {
    title: 'CLASSICAL VIBE - Classical Music Recommender',
    description: 'Discover classical music tailored to your vibe',
    type: 'website',
  },
}

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <html lang="en">
      <body className={actor.className}>
        <SkipToContent />
        <WebVitals />
        <AuthProvider>{children}</AuthProvider>
      </body>
    </html>
  )
}
