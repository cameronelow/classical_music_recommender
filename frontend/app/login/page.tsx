'use client'

import { useState } from 'react'
import { useRouter, useSearchParams } from 'next/navigation'
import Link from 'next/link'
import { useAuthStore } from '@/lib/store'
import Button from '@/components/ui/Button'
import Input from '@/components/ui/Input'
import Card from '@/components/ui/Card'

export default function LoginPage() {
  const router = useRouter()
  const searchParams = useSearchParams()
  const login = useAuthStore((state) => state.login)
  const [email, setEmail] = useState('')
  const [password, setPassword] = useState('')
  const [error, setError] = useState('')

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    setError('')

    try {
      await login(email, password)
      const redirectTo = searchParams.get('redirectTo')
      router.push(redirectTo || '/')
    } catch (err: any) {
      setError(err.message || 'Invalid email or password')
    }
  }

  return (
    <div className="min-h-screen flex flex-col">
      <header className="w-full py-2 px-16">
        <div className="w-full flex justify-center items-center relative min-h-header">
          <Link href="/" className="text-light-cream text-hero transition-opacity hover:opacity-90">
            ESPRESSIVO
          </Link>
          <div className="flex gap-3 absolute right-0">
            <Link href="/login">
              <Button variant="header">Log In</Button>
            </Link>
            <Link href="/signup">
              <Button variant="header">Sign Up</Button>
            </Link>
          </div>
        </div>
      </header>

      <main id="main-content" className="flex-1 flex items-center justify-center px-4">
        <Card className="w-full max-w-md">
          <h1 className="text-display font-bold text-dark-blue mb-6 text-center">
            Sign In
          </h1>

          <form onSubmit={handleSubmit} className="space-y-4">
            <Input
              type="email"
              label="Email"
              placeholder="Enter your email"
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              required
            />

            <Input
              type="password"
              label="Password"
              placeholder="Enter your password"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              required
            />

            {error && (
              <p className="text-red-600 text-sm">{error}</p>
            )}

            <Button
              type="submit"
              variant="dark"
              className="w-full mt-6"
            >
              Sign In
            </Button>
          </form>

          <div className="mt-4 text-center text-body-sm text-dark-blue space-y-2">
            <p>
              <Link href="/forgot-password" className="hover:underline">
                Forgot password?
              </Link>
            </p>
            <p>
              Don't have an account?{' '}
              <Link
                href={`/signup${searchParams.get('redirectTo') ? `?redirectTo=${encodeURIComponent(searchParams.get('redirectTo')!)}` : ''}`}
                className="hover:underline font-medium"
              >
                Sign up
              </Link>
            </p>
          </div>
        </Card>
      </main>
    </div>
  )
}
