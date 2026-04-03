'use client'

import { useState } from 'react'
import Link from 'next/link'
import { createClient } from '@/lib/supabase/client'
import Button from '@/components/ui/Button'
import Input from '@/components/ui/Input'
import Card from '@/components/ui/Card'

export default function ForgotPasswordPage() {
  const [email, setEmail] = useState('')
  const [error, setError] = useState('')
  const [submitted, setSubmitted] = useState(false)

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    setError('')

    const supabase = createClient()
    const { error } = await supabase.auth.resetPasswordForEmail(email, {
      redirectTo: `${window.location.origin}/reset-password`,
    })

    if (error) {
      setError(error.message)
    } else {
      setSubmitted(true)
    }
  }

  return (
    <div className="min-h-screen flex flex-col">
      <header className="w-full py-2 px-4 sm:px-8 md:px-16">
        <div className="w-full flex justify-center items-center relative min-h-header">
          <Link href="/" className="text-light-cream text-2xl sm:text-4xl md:text-hero transition-opacity hover:opacity-90 text-center">
            CLASSICAL VIBE
          </Link>
          <div className="flex gap-2 sm:gap-3 absolute right-0">
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
          {submitted ? (
            <div className="text-center space-y-4">
              <h1 className="text-2xl sm:text-3xl md:text-display font-bold text-dark-blue">
                Check your email
              </h1>
              <p className="text-body-sm text-dark-blue">
                If an account exists for <strong>{email}</strong>, we sent a password reset link. Check your inbox and spam folder.
              </p>
              <Link href="/login" className="block mt-4 hover:underline text-body-sm text-dark-blue">
                Back to Sign In
              </Link>
            </div>
          ) : (
            <>
              <h1 className="text-2xl sm:text-3xl md:text-display font-bold text-dark-blue mb-2 text-center">
                Forgot password?
              </h1>
              <p className="text-body-sm text-dark-blue text-center mb-6">
                Enter your email and we'll send you a reset link.
              </p>

              <form onSubmit={handleSubmit} className="space-y-4">
                <Input
                  type="email"
                  label="Email"
                  placeholder="Enter your email"
                  value={email}
                  onChange={(e) => setEmail(e.target.value)}
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
                  Send reset link
                </Button>
              </form>

              <div className="mt-4 text-center text-body-sm text-dark-blue">
                <Link href="/login" className="hover:underline">
                  Back to Sign In
                </Link>
              </div>
            </>
          )}
        </Card>
      </main>
    </div>
  )
}
