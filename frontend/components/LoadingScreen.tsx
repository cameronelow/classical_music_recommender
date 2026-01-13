'use client'

import Header from './Header'
import SkeletonLoader from './SkeletonLoader'

export default function LoadingScreen() {
  return (
    <div className="min-h-screen flex flex-col">
      <Header />

      <main id="main-content" className="flex-1 flex items-center justify-center px-4">
        <SkeletonLoader />
      </main>
    </div>
  )
}
