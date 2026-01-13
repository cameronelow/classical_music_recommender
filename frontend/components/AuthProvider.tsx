'use client'

import { useEffect } from 'react'
import { createClient } from '@/lib/supabase/client'
import { useAuthStore, useSavedPiecesStore } from '@/lib/store'
import { autoMigrateSavedPieces, needsMigration } from '@/lib/migrate-saved-pieces'

export default function AuthProvider({ children }: { children: React.ReactNode }) {
  const setUser = useAuthStore((state) => state.setUser)
  const fetchPieces = useSavedPiecesStore((state) => state.fetchPieces)

  useEffect(() => {
    const supabase = createClient()

    // Helper function to handle user session and migration
    const handleUserSession = async (userId: string, email: string, name: string) => {
      setUser({ id: userId, email, name })

      // Auto-migrate localStorage data to Supabase if needed
      if (needsMigration()) {
        console.log('📦 Migrating saved pieces from localStorage to Supabase...')
        try {
          await autoMigrateSavedPieces(userId)
          console.log('✓ Migration complete')
        } catch (error) {
          console.error('✗ Migration failed:', error)
        }
      }

      // Fetch saved pieces from database
      try {
        await fetchPieces(userId)
      } catch (error) {
        console.error('Failed to fetch saved pieces:', error)
      }
    }

    // Check active sessions and sets the user
    supabase.auth.getSession().then(({ data: { session } }) => {
      if (session?.user) {
        const name = session.user.user_metadata?.name || session.user.email?.split('@')[0] || ''
        handleUserSession(session.user.id, session.user.email!, name)
      } else {
        setUser(null)
      }
    })

    // Listen for changes on auth state (sign in, sign out, etc.)
    const {
      data: { subscription },
    } = supabase.auth.onAuthStateChange((_event, session) => {
      if (session?.user) {
        const name = session.user.user_metadata?.name || session.user.email?.split('@')[0] || ''
        handleUserSession(session.user.id, session.user.email!, name)
      } else {
        setUser(null)
      }
    })

    return () => subscription.unsubscribe()
  }, [setUser, fetchPieces])

  return <>{children}</>
}
