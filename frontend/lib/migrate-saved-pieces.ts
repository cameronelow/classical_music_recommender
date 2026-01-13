/**
 * Data Migration Script - localStorage to Supabase
 *
 * This script migrates saved pieces from localStorage to Supabase database.
 * Run this once after implementing the new backend endpoints.
 *
 * Usage:
 * - Import and call this function on user login/app load
 * - It will automatically detect localStorage data and migrate it
 * - Safe to run multiple times - uses upsert to avoid duplicates
 */

import { savePiece } from './api'

interface LegacySavedPiece {
  id: string
  title: string
  composer: string
  vibe: string
  explanation: string
}

interface LegacyStorage {
  state: {
    pieces: LegacySavedPiece[]
  }
  version: number
}

export async function migrateSavedPiecesToSupabase(userId: string): Promise<{
  migrated: number
  skipped: number
  errors: number
}> {
  const stats = {
    migrated: 0,
    skipped: 0,
    errors: 0,
  }

  try {
    // Get localStorage data
    const storageKey = 'saved-pieces-storage'
    const rawData = localStorage.getItem(storageKey)

    if (!rawData) {
      console.log('No localStorage data to migrate')
      return stats
    }

    const storage: LegacyStorage = JSON.parse(rawData)
    const pieces = storage.state?.pieces || []

    if (pieces.length === 0) {
      console.log('No saved pieces to migrate')
      return stats
    }

    console.log(`Found ${pieces.length} saved pieces in localStorage`)

    // Migrate each piece
    for (const piece of pieces) {
      try {
        // The legacy format used 'id' as work_id
        await savePiece(
          {
            work_id: piece.id,
            title: piece.title,
            composer: piece.composer,
            vibe: piece.vibe,
            explanation: piece.explanation,
          },
          userId
        )
        stats.migrated++
        console.log(`✓ Migrated: ${piece.title} by ${piece.composer}`)
      } catch (error: any) {
        // If it already exists, that's fine (upsert behavior)
        if (error.message?.includes('already exists') || error.message?.includes('duplicate')) {
          stats.skipped++
          console.log(`- Skipped (already exists): ${piece.title}`)
        } else {
          stats.errors++
          console.error(`✗ Error migrating ${piece.title}:`, error)
        }
      }
    }

    // If migration was successful, optionally clear localStorage
    // Uncomment the line below to remove localStorage data after migration
    // localStorage.removeItem(storageKey)

    console.log('Migration complete:', stats)
    return stats

  } catch (error) {
    console.error('Migration failed:', error)
    throw error
  }
}

/**
 * Check if migration is needed
 */
export function needsMigration(): boolean {
  const storageKey = 'saved-pieces-storage'
  const rawData = localStorage.getItem(storageKey)

  if (!rawData) {
    return false
  }

  try {
    const storage: LegacyStorage = JSON.parse(rawData)
    const pieces = storage.state?.pieces || []
    return pieces.length > 0
  } catch {
    return false
  }
}

/**
 * Auto-migration hook - call this on app initialization
 *
 * Example usage in a React component:
 *
 * ```typescript
 * useEffect(() => {
 *   const user = useAuthStore.getState().user
 *   if (user && needsMigration()) {
 *     autoMigrateSavedPieces(user.id)
 *       .then(() => {
 *         // Refresh saved pieces from database
 *         useSavedPiecesStore.getState().fetchPieces(user.id)
 *       })
 *   }
 * }, [])
 * ```
 */
export async function autoMigrateSavedPieces(userId: string): Promise<void> {
  if (!needsMigration()) {
    console.log('No migration needed')
    return
  }

  console.log('Starting auto-migration...')
  const stats = await migrateSavedPiecesToSupabase(userId)

  if (stats.migrated > 0 || stats.skipped > 0) {
    console.log(`Auto-migration complete: ${stats.migrated} migrated, ${stats.skipped} skipped, ${stats.errors} errors`)
  }
}
