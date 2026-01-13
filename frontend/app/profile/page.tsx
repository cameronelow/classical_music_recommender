'use client'

import { useRouter } from 'next/navigation'
import { useAuthStore } from '@/lib/store'
import Header from '@/components/Header'
import Card from '@/components/ui/Card'
import Button from '@/components/ui/Button'

export default function ProfilePage() {
  const router = useRouter()
  const { user, logout } = useAuthStore()

  const handleLogout = () => {
    logout()
    router.push('/')
  }

  const handleDeleteAccount = () => {
    if (confirm('Are you sure you want to delete your account? This action cannot be undone.')) {
      logout()
      router.push('/')
    }
  }

  return (
    <div className="min-h-screen flex flex-col">
      <Header />

      <main className="flex-1 flex items-center justify-center px-4">
        <Card className="w-full max-w-md">
          <h1 className="text-display font-bold text-dark-blue mb-8 text-center">
            Hi, {user?.name || 'there'}!
          </h1>

          <div className="space-y-3">
            <Button
              variant="dark"
              className="w-full"
              onClick={() => router.push('/profile/edit')}
            >
              Edit Profile
            </Button>

            <Button
              variant="dark"
              className="w-full"
              onClick={() => router.push('/saved')}
            >
              View Saved Pieces
            </Button>

            <Button
              variant="dark"
              className="w-full"
              onClick={handleLogout}
            >
              Sign Out
            </Button>

            <Button
              variant="dark"
              className="w-full bg-red-600 hover:bg-red-700"
              onClick={handleDeleteAccount}
            >
              Delete Account
            </Button>
          </div>
        </Card>
      </main>
    </div>
  )
}
