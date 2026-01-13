'use client';

import React, { useState, useEffect } from 'react';
import { useAuthStore } from '@/lib/store';
import { submitFeedback, deleteFeedback as apiDeleteFeedback } from '@/lib/api';
import styles from './FeedbackButton.module.css';

interface FeedbackButtonProps {
  workId: string;
  vibe: string;
  onFeedbackChange?: (rating: number | null) => void;
  showComment?: boolean;
}

export const FeedbackButton: React.FC<FeedbackButtonProps> = ({
  workId,
  vibe,
  onFeedbackChange,
  showComment = true
}) => {
  const { user } = useAuthStore();
  const [rating, setRating] = useState<number | null>(null);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [showCommentBox, setShowCommentBox] = useState(false);
  const [comment, setComment] = useState('');

  useEffect(() => {
    loadExistingFeedback();
  }, [workId, vibe]);

  const loadExistingFeedback = () => {
    const stored = localStorage.getItem(`feedback_${workId}_${vibe}`);
    if (stored) {
      try {
        const data = JSON.parse(stored);
        setRating(data.rating);
        setComment(data.comment || '');
      } catch (e) {
        console.error('Failed to load feedback from localStorage', e);
      }
    }
  };

  const handleThumbsUp = async () => {
    const newRating = rating === 1 ? null : 1;
    await submitFeedbackAction(newRating);
  };

  const handleThumbsDown = async () => {
    const newRating = rating === -1 ? null : -1;
    await submitFeedbackAction(newRating);
  };

  const submitFeedbackAction = async (newRating: number | null) => {
    setIsSubmitting(true);

    try {
      if (newRating === null) {
        await deleteFeedbackAction();
        setRating(null);
        setComment('');
        setShowCommentBox(false);
        localStorage.removeItem(`feedback_${workId}_${vibe}`);
      } else {
        if (user?.id) {
          await submitFeedback(workId, vibe, newRating, comment, user.id);
        }

        setRating(newRating);
        localStorage.setItem(
          `feedback_${workId}_${vibe}`,
          JSON.stringify({ rating: newRating, comment, timestamp: Date.now() })
        );
      }

      onFeedbackChange?.(newRating);
    } catch (error) {
      console.error('Failed to submit feedback:', error);
    } finally {
      setIsSubmitting(false);
    }
  };

  const handleCommentSubmit = async () => {
    if (rating === null) return;

    setIsSubmitting(true);
    try {
      if (user?.id) {
        await submitFeedback(workId, vibe, rating, comment, user.id);
      }

      localStorage.setItem(
        `feedback_${workId}_${vibe}`,
        JSON.stringify({ rating, comment, timestamp: Date.now() })
      );

      setShowCommentBox(false);
    } catch (error) {
      console.error('Failed to submit comment:', error);
    } finally {
      setIsSubmitting(false);
    }
  };

  const deleteFeedbackAction = async () => {
    if (user?.id) {
      await apiDeleteFeedback(workId, vibe, user.id);
    }
  };

  const ThumbsUpIcon = ({ filled }: { filled: boolean }) => (
    <svg
      width="20"
      height="20"
      viewBox="0 0 24 24"
      fill={filled ? "currentColor" : "none"}
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
    >
      <path d="M14 9V5a3 3 0 0 0-3-3l-4 9v11h11.28a2 2 0 0 0 2-1.7l1.38-9a2 2 0 0 0-2-2.3zM7 22H4a2 2 0 0 1-2-2v-7a2 2 0 0 1 2-2h3"></path>
    </svg>
  );

  const ThumbsDownIcon = ({ filled }: { filled: boolean }) => (
    <svg
      width="20"
      height="20"
      viewBox="0 0 24 24"
      fill={filled ? "currentColor" : "none"}
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
    >
      <path d="M10 15v4a3 3 0 0 0 3 3l4-9V2H5.72a2 2 0 0 0-2 1.7l-1.38 9a2 2 0 0 0 2 2.3zm7-13h2.67A2.31 2.31 0 0 1 22 4v7a2.31 2.31 0 0 1-2.33 2H17"></path>
    </svg>
  );

  return (
    <div className={styles.feedbackContainer}>
      <div className={styles.feedbackButtons}>
        <span className={styles.label}>Was this a good match?</span>

        <div className={styles.buttonGroup}>
          <button
            onClick={handleThumbsUp}
            disabled={isSubmitting}
            className={`${styles.thumbButton} ${rating === 1 ? styles.active : ''}`}
            aria-label="Thumbs up"
            title="This was a good match"
          >
            <ThumbsUpIcon filled={rating === 1} />
          </button>

          <button
            onClick={handleThumbsDown}
            disabled={isSubmitting}
            className={`${styles.thumbButton} ${rating === -1 ? styles.active : ''}`}
            aria-label="Thumbs down"
            title="This wasn't a good match"
          >
            <ThumbsDownIcon filled={rating === -1} />
          </button>
        </div>

        {showComment && rating !== null && (
          <button
            onClick={() => setShowCommentBox(!showCommentBox)}
            className={styles.commentToggle}
          >
            {comment ? 'Edit comment' : 'Add comment'}
          </button>
        )}
      </div>

      {showCommentBox && rating !== null && (
        <div className={styles.commentBox}>
          <textarea
            value={comment}
            onChange={(e) => setComment(e.target.value)}
            placeholder="Tell us more about your experience (optional)"
            className={styles.textarea}
            rows={3}
            maxLength={500}
          />
          <div className={styles.commentActions}>
            <span className={styles.charCount}>{comment.length}/500</span>
            <button
              onClick={handleCommentSubmit}
              className={styles.submitComment}
              disabled={isSubmitting}
            >
              {isSubmitting ? 'Saving...' : 'Save'}
            </button>
          </div>
        </div>
      )}
    </div>
  );
};
