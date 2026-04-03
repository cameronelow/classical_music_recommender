'use client'

import { useState } from 'react'
import { useRouter } from 'next/navigation'
import Link from 'next/link'
import { createClient } from '@/lib/supabase/client'
import Button from '@/components/ui/Button'
import Input from '@/components/ui/Input'
import Card from '@/components/ui/Card'

export default function ResetPasswordPage() {
  const router = useRouter()
  const [password, setPassword] = useState('')
  const [confirmPassword, setConfirmPassword] = useState('')
  const [error, setError] = useState('')
  const [success, setSuccess] = useState(false)

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    setError('')

    if (password !== confirmPassword) {
      setError('Passwords do not match')
      return
    }

    if (password.length < 6) {
      setError('Password must be at least 6 characters')
      return
    }

    const supabase = createClient()
    const { error } = await supabase.auth.updateUser({ password })

    if (error) {
      setError(error.message)
    } else {
      setSuccess(true)
      setTimeout(() => router.push('/login'), 2000)
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
          {success ? (
            <div className="text-center space-y-4">
              <h1 className="text-2xl sm:text-3xl md:text-display font-bold text-dark-blue">
                Password updated
              </h1>
              <p className="text-body-sm text-dark-blue">
                Your password has been reset. Redirecting to sign in...
              </p>
            </div>
          ) : (
            <>
              <h1 className="text-2xl sm:text-3xl md:text-display font-bold text-dark-blue mb-6 text-center">
                Set new password
              </h1>

              <form onSubmit={handleSubmit} className="space-y-4">
                <Input
                  type="password"
                  label="New password"
                  placeholder="Enter new password"
                  value={password}
                  onChange={(e) => setPassword(e.target.value)}
                  required
                />

                <Input
                  type="password"
                  label="Confirm new password"
                  placeholder="Confirm new password"
                  value={confirmPassword}
                  onChange={(e) => setConfirmPassword(e.target.value)}
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
                  Update password
                </Button>
              </form>
            </>
          )}
        </Card>
      </main>
    </div>
  )
}
