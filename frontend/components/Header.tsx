'use client'

import { useEffect, useState } from 'react'
import Link from 'next/link'
import { usePathname, useRouter } from 'next/navigation'
import { useAuthStore } from '@/lib/store'
import Button from './ui/Button'

export default function Header() {
  const pathname = usePathname()
  const router = useRouter()
  const { isAuthenticated, logout } = useAuthStore()
  const [mounted, setMounted] = useState(false)

  // Handle hydration mismatch
  useEffect(() => {
    setMounted(true)
  }, [])

  const handleSignOut = () => {
    logout()
    router.push('/')
  }

  // Don't show header on auth pages
  if (pathname === '/login' || pathname === '/signup') {
    return null
  }

  return (
    <header className="w-full py-4 px-16 relative z-10">
      <div className="w-full flex justify-center items-center relative min-h-header">
        <Link
          href="/"
          className="text-light-cream text-hero transition-opacity hover:opacity-90"
          aria-label="Espressivo - Home"
        >
          ESPRESSIVO
        </Link>

        <div className="flex gap-3 absolute right-0">
          {mounted && isAuthenticated ? (
            <>
              <Link href="/profile">
                <Button variant="header">Profile</Button>
              </Link>
              <Button variant="header" onClick={handleSignOut}>Sign Out</Button>
            </>
          ) : mounted ? (
            <>
              <Link href="/login">
                <Button variant="header">Log In</Button>
              </Link>
              <Link href="/signup">
                <Button variant="header">Sign Up</Button>
              </Link>
            </>
          ) : (
            // Placeholder to prevent layout shift
            <div className="w-48" />
          )}
        </div>
      </div>
    </header>
  )
}
