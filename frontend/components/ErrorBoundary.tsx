'use client'

import React, { Component, ReactNode } from 'react'
import Header from './Header'
import Card from './ui/Card'
import Button from './ui/Button'

interface Props {
  children: ReactNode
  fallback?: ReactNode
}

interface State {
  hasError: boolean
  error?: Error
}

export class ErrorBoundary extends Component<Props, State> {
  constructor(props: Props) {
    super(props)
    this.state = { hasError: false }
  }

  static getDerivedStateFromError(error: Error): State {
    return { hasError: true, error }
  }

  componentDidCatch(error: Error, errorInfo: React.ErrorInfo) {
    console.error('ErrorBoundary caught error:', error, errorInfo)
  }

  render() {
    if (this.state.hasError) {
      if (this.props.fallback) {
        return this.props.fallback
      }

      return (
        <div className="min-h-screen flex flex-col">
          <Header />

          <main className="flex-1 flex items-center justify-center px-4">
            <Card className="max-w-md text-center">
              <h1 className="text-display font-bold text-dark-blue mb-4">
                Oops!
              </h1>
              <p className="text-body text-dark-blue mb-8">
                Something went wrong while displaying this component.
              </p>
              <Button
                variant="dark"
                onClick={() => this.setState({ hasError: false })}
              >
                Try Again
              </Button>
            </Card>
          </main>
        </div>
      )
    }

    return this.props.children
  }
}
