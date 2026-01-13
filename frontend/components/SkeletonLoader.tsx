export default function SkeletonLoader() {
  return (
    <div className="w-full max-w-2xl mx-auto animate-pulse">
      <div className="bg-light-cream bg-opacity-20 rounded-lg p-8">
        {/* Title skeleton */}
        <div className="h-12 bg-light-cream bg-opacity-30 rounded-button mb-4 w-3/4 mx-auto"></div>

        {/* Composer skeleton */}
        <div className="h-8 bg-light-cream bg-opacity-30 rounded-button mb-8 w-1/2 mx-auto"></div>

        {/* Vibe text skeleton */}
        <div className="h-6 bg-light-cream bg-opacity-30 rounded-button mb-3 w-2/3 mx-auto"></div>

        {/* Explanation skeleton */}
        <div className="space-y-2 mb-8">
          <div className="h-4 bg-light-cream bg-opacity-30 rounded-button w-full"></div>
          <div className="h-4 bg-light-cream bg-opacity-30 rounded-button w-5/6 mx-auto"></div>
          <div className="h-4 bg-light-cream bg-opacity-30 rounded-button w-4/5 mx-auto"></div>
        </div>

        {/* Buttons skeleton */}
        <div className="flex flex-col sm:flex-row gap-3 justify-center">
          <div className="h-12 bg-light-cream bg-opacity-30 rounded-button w-full sm:w-40"></div>
          <div className="h-12 bg-light-cream bg-opacity-30 rounded-button w-full sm:w-40"></div>
        </div>
      </div>
    </div>
  )
}
