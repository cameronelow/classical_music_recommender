'use client'

import { useState, useEffect } from 'react'
import { useRouter } from 'next/navigation'
import { useAuthStore } from '@/lib/store'
import Header from '@/components/Header'
import Card from '@/components/ui/Card'
import Button from '@/components/ui/Button'
import Input from '@/components/ui/Input'

export default function EditProfilePage() {
  const router = useRouter()
  const { isAuthenticated, user, updateProfile, updatePassword } = useAuthStore()

  // Profile form state
  const [name, setName] = useState('')
  const [profileError, setProfileError] = useState('')
  const [profileSuccess, setProfileSuccess] = useState('')
  const [isUpdatingProfile, setIsUpdatingProfile] = useState(false)

  // Password form state
  const [currentPassword, setCurrentPassword] = useState('')
  const [newPassword, setNewPassword] = useState('')
  const [confirmPassword, setConfirmPassword] = useState('')
  const [passwordError, setPasswordError] = useState('')
  const [passwordSuccess, setPasswordSuccess] = useState('')
  const [isUpdatingPassword, setIsUpdatingPassword] = useState(false)

  useEffect(() => {
    if (!isAuthenticated) {
      router.push('/login')
    } else if (user) {
      setName(user.name)
    }
  }, [isAuthenticated, user, router])

  const handleProfileSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    setProfileError('')
    setProfileSuccess('')
    setIsUpdatingProfile(true)

    try {
      await updateProfile(name)
      setProfileSuccess('Profile updated successfully!')
    } catch (err: any) {
      setProfileError(err.message || 'Failed to update profile')
    } finally {
      setIsUpdatingProfile(false)
    }
  }

  const handlePasswordSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    setPasswordError('')
    setPasswordSuccess('')
    setIsUpdatingPassword(true)

    try {
      await updatePassword(currentPassword, newPassword, confirmPassword)
      setPasswordSuccess('Password updated successfully!')
      setCurrentPassword('')
      setNewPassword('')
      setConfirmPassword('')
    } catch (err: any) {
      setPasswordError(err.message || 'Failed to update password')
    } finally {
      setIsUpdatingPassword(false)
    }
  }

  if (!isAuthenticated || !user) {
    return null
  }

  return (
    <div className="min-h-screen flex flex-col">
      <Header />

      <div className="flex-1 flex items-center justify-center px-4 py-12">
        <div className="w-full max-w-2xl space-y-6">
          <Card>
            <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4 mb-6">
              <h1 className="text-2xl sm:text-3xl font-bold text-dark-blue">Edit Profile</h1>
              <Button
                variant="secondary"
                onClick={() => router.push('/profile')}
                className="w-full sm:w-auto"
              >
                Back to Profile
              </Button>
            </div>

            <form onSubmit={handleProfileSubmit} className="space-y-4">
              <h2 className="text-xl font-semibold text-dark-blue">Personal Information</h2>

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
                value={user.email}
                disabled
                className="bg-gray-100 cursor-not-allowed"
              />
              <p className="text-xs text-gray-600 -mt-2">
                Email cannot be changed
              </p>

              {profileError && (
                <p className="text-red-600 text-sm">{profileError}</p>
              )}

              {profileSuccess && (
                <p className="text-green-600 text-sm">{profileSuccess}</p>
              )}

              <Button
                type="submit"
                variant="dark"
                className="w-full mt-6"
                disabled={isUpdatingProfile}
              >
                {isUpdatingProfile ? 'Updating...' : 'Update Profile'}
              </Button>
            </form>
          </Card>

          <Card>
            <form onSubmit={handlePasswordSubmit} className="space-y-4">
              <h2 className="text-xl font-semibold text-dark-blue">Change Password</h2>

              <Input
                type="password"
                label="Current Password"
                placeholder="Enter your current password"
                value={currentPassword}
                onChange={(e) => setCurrentPassword(e.target.value)}
                required
              />

              <Input
                type="password"
                label="New Password"
                placeholder="Enter your new password"
                value={newPassword}
                onChange={(e) => setNewPassword(e.target.value)}
                required
                minLength={6}
              />

              <Input
                type="password"
                label="Confirm New Password"
                placeholder="Confirm your new password"
                value={confirmPassword}
                onChange={(e) => setConfirmPassword(e.target.value)}
                required
                minLength={6}
              />

              {passwordError && (
                <p className="text-red-600 text-sm">{passwordError}</p>
              )}

              {passwordSuccess && (
                <p className="text-green-600 text-sm">{passwordSuccess}</p>
              )}

              <Button
                type="submit"
                variant="dark"
                className="w-full mt-6"
                disabled={isUpdatingPassword}
              >
                {isUpdatingPassword ? 'Updating...' : 'Update Password'}
              </Button>
            </form>
          </Card>
        </div>
      </div>
    </div>
  )
}
