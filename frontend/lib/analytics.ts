/**
 * Analytics Tracking Utility for Classical Music Recommender
 *
 * This module provides a centralized way to track user events.
 * Currently logs to console, but can be easily extended to send to
 * analytics services like Google Analytics, Plausible, or custom endpoints.
 */

export type SharePlatform = 'native' | 'copy-link' | 'download-card' | 'twitter' | 'facebook' | 'instagram';
export type ShareStatus = 'initiated' | 'success' | 'failed';

export interface ShareEvent {
  event: 'share';
  platform: SharePlatform;
  status: ShareStatus;
  workId?: string;
  query?: string;
  timestamp: number;
}

export interface SearchEvent {
  event: 'search';
  query: string;
  resultCount: number;
  timestamp: number;
}

export interface RecommendationEvent {
  event: 'recommendation_viewed';
  workId: string;
  query: string;
  timestamp: number;
}

export interface SaveEvent {
  event: 'piece_saved' | 'piece_unsaved';
  workId: string;
  timestamp: number;
}

export type AnalyticsEvent = ShareEvent | SearchEvent | RecommendationEvent | SaveEvent;

class Analytics {
  private enabled: boolean = true;
  private debugMode: boolean = process.env.NODE_ENV === 'development';

  /**
   * Track a share event
   */
  trackShare(platform: SharePlatform, status: ShareStatus, metadata?: { workId?: string; query?: string }) {
    const event: ShareEvent = {
      event: 'share',
      platform,
      status,
      workId: metadata?.workId,
      query: metadata?.query,
      timestamp: Date.now(),
    };

    this.track(event);

    // Send to analytics services
    this.sendToServices('share', {
      platform,
      status,
      ...metadata,
    });
  }

  /**
   * Track a search event
   */
  trackSearch(query: string, resultCount: number) {
    const event: SearchEvent = {
      event: 'search',
      query,
      resultCount,
      timestamp: Date.now(),
    };

    this.track(event);

    this.sendToServices('search', {
      query,
      result_count: resultCount,
    });
  }

  /**
   * Track when a recommendation is viewed
   */
  trackRecommendationView(workId: string, query: string) {
    const event: RecommendationEvent = {
      event: 'recommendation_viewed',
      workId,
      query,
      timestamp: Date.now(),
    };

    this.track(event);

    this.sendToServices('recommendation_viewed', {
      work_id: workId,
      query,
    });
  }

  /**
   * Track when a piece is saved/unsaved
   */
  trackSave(workId: string, action: 'saved' | 'unsaved') {
    const event: SaveEvent = {
      event: action === 'saved' ? 'piece_saved' : 'piece_unsaved',
      workId,
      timestamp: Date.now(),
    };

    this.track(event);

    this.sendToServices(event.event, {
      work_id: workId,
    });
  }

  /**
   * Core tracking method
   */
  private track(event: AnalyticsEvent) {
    if (!this.enabled) return;

    // Log to console in debug mode
    if (this.debugMode) {
      console.log('📊 Analytics:', event);
    }

    // Store in local storage for basic analytics (optional)
    this.storeLocally(event);
  }

  /**
   * Send events to external analytics services
   */
  private sendToServices(eventName: string, properties: Record<string, any>) {
    // Google Analytics 4 (gtag)
    if (typeof window !== 'undefined' && (window as any).gtag) {
      (window as any).gtag('event', eventName, properties);
    }

    // Plausible Analytics
    if (typeof window !== 'undefined' && (window as any).plausible) {
      (window as any).plausible(eventName, { props: properties });
    }

    // Custom analytics endpoint (implement if needed)
    // this.sendToCustomEndpoint(eventName, properties);
  }

  /**
   * Store events locally for basic analytics
   */
  private storeLocally(event: AnalyticsEvent) {
    try {
      const key = 'classical_analytics_events';
      const stored = localStorage.getItem(key);
      const events = stored ? JSON.parse(stored) : [];

      // Keep only last 100 events to avoid bloat
      events.push(event);
      if (events.length > 100) {
        events.shift();
      }

      localStorage.setItem(key, JSON.stringify(events));
    } catch (error) {
      // localStorage might be disabled or full
      console.warn('Failed to store analytics locally:', error);
    }
  }

  /**
   * Get locally stored events (for debugging or basic analytics)
   */
  getLocalEvents(): AnalyticsEvent[] {
    try {
      const key = 'classical_analytics_events';
      const stored = localStorage.getItem(key);
      return stored ? JSON.parse(stored) : [];
    } catch (error) {
      return [];
    }
  }

  /**
   * Clear locally stored events
   */
  clearLocalEvents() {
    try {
      localStorage.removeItem('classical_analytics_events');
    } catch (error) {
      console.warn('Failed to clear analytics:', error);
    }
  }

  /**
   * Enable/disable tracking
   */
  setEnabled(enabled: boolean) {
    this.enabled = enabled;
  }

  /**
   * Check if tracking is enabled
   */
  isEnabled(): boolean {
    return this.enabled;
  }
}

// Export singleton instance
export const analytics = new Analytics();

// Convenience functions
export const trackShare = (platform: SharePlatform, status: ShareStatus, metadata?: { workId?: string; query?: string }) =>
  analytics.trackShare(platform, status, metadata);

export const trackSearch = (query: string, resultCount: number) =>
  analytics.trackSearch(query, resultCount);

export const trackRecommendationView = (workId: string, query: string) =>
  analytics.trackRecommendationView(workId, query);

export const trackSave = (workId: string, action: 'saved' | 'unsaved') =>
  analytics.trackSave(workId, action);
