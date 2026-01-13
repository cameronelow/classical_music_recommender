import Link from 'next/link'
import Header from '@/components/Header'
import Card from '@/components/ui/Card'
import Button from '@/components/ui/Button'

export default function NotFound() {
  return (
    <div className="min-h-screen flex flex-col">
      <Header />

      <main className="flex-1 flex items-center justify-center px-4">
        <Card className="max-w-md text-center">
          <h1 className="text-display font-bold text-dark-blue mb-4">
            404
          </h1>
          <p className="text-heading-lg text-dark-blue mb-2">
            Page Not Found
          </p>
          <p className="text-body text-dark-blue mb-8">
            The page you&apos;re looking for doesn&apos;t exist or has been moved.
          </p>
          <Link href="/">
            <Button variant="dark">
              Return Home
            </Button>
          </Link>
        </Card>
      </main>
    </div>
  )
}
