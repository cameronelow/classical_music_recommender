import { create } from 'zustand'
import { persist } from 'zustand/middleware'
import { createClient } from './supabase/client'
import { getSavedPieces, savePiece as savePieceAPI, unsavePiece as unsavePieceAPI } from './api'

interface User {
  id: string
  email: string
  name: string
}

interface AuthState {
  user: User | null
  isAuthenticated: boolean
  login: (email: string, password: string) => Promise<void>
  signup: (name: string, email: string, password: string, confirmPassword: string) => Promise<void>
  logout: () => Promise<void>
  setUser: (user: User | null) => void
  updateProfile: (name: string) => Promise<void>
  updatePassword: (currentPassword: string, newPassword: string, confirmPassword: string) => Promise<void>
}

export const useAuthStore = create<AuthState>()(
  persist(
    (set) => ({
      user: null,
      isAuthenticated: false,
      login: async (email: string, password: string) => {
        const supabase = createClient()
        const { data, error } = await supabase.auth.signInWithPassword({
          email,
          password,
        })

        if (error) {
          throw error
        }

        if (data.user) {
          const user: User = {
            id: data.user.id,
            email: data.user.email!,
            name: data.user.user_metadata?.name || data.user.email?.split('@')[0] || '',
          }
          set({ user, isAuthenticated: true })
        }
      },
      signup: async (name: string, email: string, password: string, confirmPassword: string) => {
        if (password !== confirmPassword) {
          throw new Error('Passwords do not match')
        }

        const supabase = createClient()
        const { data, error } = await supabase.auth.signUp({
          email,
          password,
          options: {
            data: {
              name,
            },
          },
        })

        if (error) {
          throw error
        }

        if (data.user) {
          const user: User = {
            id: data.user.id,
            email: data.user.email!,
            name,
          }
          set({ user, isAuthenticated: true })
        }
      },
      logout: async () => {
        const supabase = createClient()
        await supabase.auth.signOut()
        set({ user: null, isAuthenticated: false })
        // Clear saved pieces when logging out
        useSavedPiecesStore.getState().clearPieces()
      },
      setUser: (user: User | null) => {
        set({ user, isAuthenticated: !!user })
      },
      updateProfile: async (name: string) => {
        const supabase = createClient()
        const { data, error } = await supabase.auth.updateUser({
          data: {
            name,
          },
        })

        if (error) {
          throw error
        }

        if (data.user) {
          const user: User = {
            id: data.user.id,
            email: data.user.email!,
            name,
          }
          set({ user })
        }
      },
      updatePassword: async (currentPassword: string, newPassword: string, confirmPassword: string) => {
        if (newPassword !== confirmPassword) {
          throw new Error('Passwords do not match')
        }

        const supabase = createClient()

        // Verify current password by attempting to sign in
        const { data: { user } } = await supabase.auth.getUser()
        if (user?.email) {
          const { error: signInError } = await supabase.auth.signInWithPassword({
            email: user.email,
            password: currentPassword,
          })

          if (signInError) {
            throw new Error('Current password is incorrect')
          }
        }

        // Update to new password
        const { error } = await supabase.auth.updateUser({
          password: newPassword,
        })

        if (error) {
          throw error
        }
      },
    }),
    {
      name: 'auth-storage',
    }
  )
)

interface SavedPiece {
  id: string
  work_id: string
  title: string
  composer: string
  composer_id?: string
  vibe?: string
  explanation?: string
  notes?: string
}

interface SavedPiecesState {
  pieces: SavedPiece[]
  isLoading: boolean
  error: string | null
  currentUserId: string | null
  fetchPieces: (userId: string) => Promise<void>
  addPiece: (piece: Omit<SavedPiece, 'id'>, userId: string) => Promise<void>
  removePiece: (workId: string, userId: string) => Promise<void>
  isPieceSaved: (workId: string) => boolean
  clearPieces: () => void
}

export const useSavedPiecesStore = create<SavedPiecesState>()((set, get) => ({
  pieces: [],
  isLoading: false,
  error: null,
  currentUserId: null,

  // Fetch saved pieces from Supabase
  fetchPieces: async (userId: string) => {
    // Always clear pieces before fetching to avoid duplicates
    set({ pieces: [], isLoading: true, error: null, currentUserId: userId })

    try {
      const response = await getSavedPieces(userId)
      set({ pieces: response.pieces, isLoading: false })
    } catch (error: any) {
      set({ error: error.message, isLoading: false })
      console.error('Failed to fetch saved pieces:', error)
    }
  },

  // Add a piece (syncs with Supabase)
  addPiece: async (piece: Omit<SavedPiece, 'id'>, userId: string) => {
    set({ isLoading: true, error: null })
    try {
      const savedPiece = await savePieceAPI(
        {
          work_id: piece.work_id,
          title: piece.title,
          composer: piece.composer,
          composer_id: piece.composer_id,
          vibe: piece.vibe,
          explanation: piece.explanation,
          notes: piece.notes,
        },
        userId
      )

      // Update local state with the saved piece from backend
      // Check for duplicates before adding
      set((state) => {
        const exists = state.pieces.some(p => p.work_id === savedPiece.work_id)
        if (exists) {
          return { isLoading: false }
        }
        return {
          pieces: [...state.pieces, savedPiece],
          isLoading: false,
        }
      })
    } catch (error: any) {
      set({ error: error.message, isLoading: false })
      console.error('Failed to save piece:', error)
      throw error
    }
  },

  // Remove a piece (syncs with Supabase)
  removePiece: async (workId: string, userId: string) => {
    set({ isLoading: true, error: null })
    try {
      await unsavePieceAPI(workId, userId)

      // Update local state
      set((state) => ({
        pieces: state.pieces.filter((p) => p.work_id !== workId),
        isLoading: false,
      }))
    } catch (error: any) {
      set({ error: error.message, isLoading: false })
      console.error('Failed to unsave piece:', error)
      throw error
    }
  },

  // Check if a piece is saved (by work_id)
  isPieceSaved: (workId: string) => {
    return get().pieces.some((p) => p.work_id === workId)
  },

  // Clear all pieces (for logout)
  clearPieces: () => {
    set({ pieces: [], error: null, currentUserId: null })
  },
}))

interface Feedback {
  workId: string
  vibe: string
  rating: number
  comment?: string
  timestamp: number
}

interface FeedbackState {
  feedbacks: Record<string, Feedback>
  setFeedback: (workId: string, vibe: string, rating: number, comment?: string) => void
  removeFeedback: (workId: string, vibe: string) => void
  getFeedback: (workId: string, vibe: string) => Feedback | null
}

export const useFeedbackStore = create<FeedbackState>()(
  persist(
    (set, get) => ({
      feedbacks: {},
      setFeedback: (workId: string, vibe: string, rating: number, comment?: string) => {
        const key = `${workId}_${vibe}`
        set((state) => ({
          feedbacks: {
            ...state.feedbacks,
            [key]: { workId, vibe, rating, comment, timestamp: Date.now() }
          }
        }))
      },
      removeFeedback: (workId: string, vibe: string) => {
        const key = `${workId}_${vibe}`
        set((state) => {
          const { [key]: removed, ...rest } = state.feedbacks
          return { feedbacks: rest }
        })
      },
      getFeedback: (workId: string, vibe: string) => {
        const key = `${workId}_${vibe}`
        return get().feedbacks[key] || null
      },
    }),
    {
      name: 'feedback-storage',
    }
  )
)
