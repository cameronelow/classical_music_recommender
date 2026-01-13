'use client'

import { Suspense, useState, useEffect } from 'react'
import { useRouter, useSearchParams } from 'next/navigation'
import Link from 'next/link'
import { useAuthStore } from '@/lib/store'
import Button from '@/components/ui/Button'
import Input from '@/components/ui/Input'
import Card from '@/components/ui/Card'

function SignUpForm() {
  const router = useRouter()
  const searchParams = useSearchParams()
  const signup = useAuthStore((state) => state.signup)
  const [name, setName] = useState('')
  const [email, setEmail] = useState('')
  const [password, setPassword] = useState('')
  const [confirmPassword, setConfirmPassword] = useState('')
  const [error, setError] = useState('')
  const [passwordError, setPasswordError] = useState('')
  const [confirmPasswordError, setConfirmPasswordError] = useState('')
  const [touched, setTouched] = useState({
    password: false,
    confirmPassword: false,
  })

  // Real-time password validation
  useEffect(() => {
    if (touched.password && password) {
      if (password.length < 6) {
        setPasswordError('Password must be at least 6 characters')
      } else {
        setPasswordError('')
      }
    }
  }, [password, touched.password])

  // Real-time confirm password validation
  useEffect(() => {
    if (touched.confirmPassword && confirmPassword) {
      if (confirmPassword !== password) {
        setConfirmPasswordError('Passwords do not match')
      } else {
        setConfirmPasswordError('')
      }
    }
  }, [confirmPassword, password, touched.confirmPassword])

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    setError('')

    // Final validation
    if (password.length < 6) {
      setPasswordError('Password must be at least 6 characters')
      return
    }

    if (password !== confirmPassword) {
      setConfirmPasswordError('Passwords do not match')
      return
    }

    try {
      await signup(name, email, password, confirmPassword)
      const redirectTo = searchParams.get('redirectTo')
      router.push(redirectTo || '/')
    } catch (err: any) {
      setError(err.message || 'Sign up failed')
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
            Sign Up
          </h1>

          <form onSubmit={handleSubmit} className="space-y-4">
            <Input
              type="text"
              label="Name"
              placeholder="Enter your name"
              value={name}
              onChange={(e) => setName(e.target.value)}
              required
            />

            <Input
              type="email"
              label="Email"
              placeholder="Enter your email"
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              required
            />

            <div>
              <Input
                type="password"
                label="Password"
                placeholder="Create a password (min. 6 characters)"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                onBlur={() => setTouched({ ...touched, password: true })}
                required
              />
              {passwordError && (
                <p className="text-red-600 text-body-sm mt-1 transition-opacity">{passwordError}</p>
              )}
            </div>

            <div>
              <Input
                type="password"
                label="Confirm Password"
                placeholder="Confirm your password"
                value={confirmPassword}
                onChange={(e) => setConfirmPassword(e.target.value)}
                onBlur={() => setTouched({ ...touched, confirmPassword: true })}
                required
              />
              {confirmPasswordError && (
                <p className="text-red-600 text-body-sm mt-1 transition-opacity">{confirmPasswordError}</p>
              )}
            </div>

            {error && (
              <p className="text-red-600 text-body-sm">{error}</p>
            )}

            <Button
              type="submit"
              variant="dark"
              className="w-full mt-6"
              disabled={!!passwordError || !!confirmPasswordError}
            >
              Create Account
            </Button>
          </form>

          <p className="mt-4 text-center text-body-sm text-dark-blue">
            Already have an account?{' '}
            <Link
              href={`/login${searchParams.get('redirectTo') ? `?redirectTo=${encodeURIComponent(searchParams.get('redirectTo')!)}` : ''}`}
              className="hover:underline font-medium"
            >
              Sign in
            </Link>
          </p>
        </Card>
      </main>
    </div>
  )
}

export default function SignUpPage() {
  return (
    <Suspense fallback={<div className="min-h-screen flex items-center justify-center">Loading...</div>}>
      <SignUpForm />
    </Suspense>
  )
}
