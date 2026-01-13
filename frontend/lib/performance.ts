/**
 * Performance Monitoring Utility
 *
 * Tracks Web Vitals and custom performance metrics
 */

export interface PerformanceMetric {
  name: string
  value: number
  rating: 'good' | 'needs-improvement' | 'poor'
  timestamp: number
}

class PerformanceMonitor {
  private metrics: PerformanceMetric[] = []
  private enabled: boolean = typeof window !== 'undefined'

  /**
   * Track Core Web Vitals
   */
  trackWebVitals() {
    if (!this.enabled) return

    // Use Next.js web-vitals
    if (typeof window !== 'undefined') {
      import('web-vitals').then((webVitals) => {
        const { onCLS, onFCP, onLCP, onTTFB, onINP } = webVitals

        onCLS(this.handleMetric.bind(this))
        onFCP(this.handleMetric.bind(this))
        onLCP(this.handleMetric.bind(this))
        onTTFB(this.handleMetric.bind(this))

        // onINP is the new metric replacing onFID
        if (onINP) {
          onINP(this.handleMetric.bind(this))
        }

        // onFID is deprecated but keep for backwards compatibility if it exists
        if ('onFID' in webVitals) {
          (webVitals as any).onFID(this.handleMetric.bind(this))
        }
      }).catch((error) => {
        // Silently fail if web-vitals fails to load
        console.warn('Failed to load web-vitals:', error)
      })
    }
  }

  /**
   * Handle Web Vital metric
   */
  private handleMetric(metric: any) {
    const performanceMetric: PerformanceMetric = {
      name: metric.name,
      value: metric.value,
      rating: metric.rating,
      timestamp: Date.now(),
    }

    this.metrics.push(performanceMetric)

    // Log in development
    if (process.env.NODE_ENV === 'development') {
      console.log(`⚡ ${metric.name}:`, {
        value: metric.value,
        rating: metric.rating,
      })
    }

    // Send to analytics
    this.sendToAnalytics(performanceMetric)
  }

  /**
   * Track custom timing
   */
  trackTiming(name: string, startTime: number) {
    const duration = Date.now() - startTime
    const rating = this.getRatingForCustomMetric(name, duration)

    const metric: PerformanceMetric = {
      name,
      value: duration,
      rating,
      timestamp: Date.now(),
    }

    this.metrics.push(metric)

    if (process.env.NODE_ENV === 'development') {
      console.log(`⏱️  ${name}: ${duration}ms (${rating})`)
    }

    this.sendToAnalytics(metric)
  }

  /**
   * Track API request performance
   */
  trackAPICall(endpoint: string, duration: number, status: number) {
    const metric: PerformanceMetric = {
      name: `api_${endpoint.replace(/\//g, '_')}`,
      value: duration,
      rating: duration < 1000 ? 'good' : duration < 3000 ? 'needs-improvement' : 'poor',
      timestamp: Date.now(),
    }

    this.metrics.push(metric)

    if (process.env.NODE_ENV === 'development') {
      console.log(`🌐 API ${endpoint}: ${duration}ms (${status})`)
    }

    // Send to analytics with additional context
    this.sendToAnalytics({
      ...metric,
      status,
    } as any)
  }

  /**
   * Get rating for custom metrics
   */
  private getRatingForCustomMetric(name: string, value: number): 'good' | 'needs-improvement' | 'poor' {
    // Define thresholds for different metrics
    const thresholds: Record<string, { good: number; poor: number }> = {
      recommendation_fetch: { good: 1000, poor: 3000 },
      search_fetch: { good: 800, poor: 2000 },
      save_action: { good: 300, poor: 1000 },
      default: { good: 1000, poor: 3000 },
    }

    const threshold = thresholds[name] || thresholds.default

    if (value <= threshold.good) return 'good'
    if (value <= threshold.poor) return 'needs-improvement'
    return 'poor'
  }

  /**
   * Send metrics to analytics services
   */
  private sendToAnalytics(metric: PerformanceMetric | any) {
    // Google Analytics 4
    if (typeof window !== 'undefined' && (window as any).gtag) {
      (window as any).gtag('event', 'web_vitals', {
        event_category: 'Performance',
        event_label: metric.name,
        value: Math.round(metric.value),
        rating: metric.rating,
        non_interaction: true,
      })
    }

    // Custom analytics endpoint
    if (process.env.NEXT_PUBLIC_ANALYTICS_ENDPOINT) {
      fetch(process.env.NEXT_PUBLIC_ANALYTICS_ENDPOINT, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(metric),
      }).catch(() => {
        // Silently fail - don't impact user experience
      })
    }
  }

  /**
   * Get all collected metrics
   */
  getMetrics(): PerformanceMetric[] {
    return [...this.metrics]
  }

  /**
   * Get metrics summary
   */
  getSummary() {
    const summary: Record<string, { good: number; needsImprovement: number; poor: number; avg: number }> = {}

    this.metrics.forEach((metric) => {
      if (!summary[metric.name]) {
        summary[metric.name] = { good: 0, needsImprovement: 0, poor: 0, avg: 0 }
      }

      if (metric.rating === 'good') summary[metric.name].good++
      else if (metric.rating === 'needs-improvement') summary[metric.name].needsImprovement++
      else summary[metric.name].poor++
    })

    // Calculate averages
    Object.keys(summary).forEach((name) => {
      const values = this.metrics.filter((m) => m.name === name).map((m) => m.value)
      summary[name].avg = values.reduce((a, b) => a + b, 0) / values.length
    })

    return summary
  }

  /**
   * Clear all metrics
   */
  clear() {
    this.metrics = []
  }
}

// Export singleton
export const performanceMonitor = new PerformanceMonitor()

// Convenience functions
export const trackTiming = (name: string, startTime: number) =>
  performanceMonitor.trackTiming(name, startTime)

export const trackAPICall = (endpoint: string, duration: number, status: number) =>
  performanceMonitor.trackAPICall(endpoint, duration, status)

export const trackWebVitals = () =>
  performanceMonitor.trackWebVitals()
