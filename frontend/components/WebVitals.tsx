'use client'

import { useEffect } from 'react'
import { trackWebVitals } from '@/lib/performance'

export default function WebVitals() {
  useEffect(() => {
    trackWebVitals()
  }, [])

  return null
}
