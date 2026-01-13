'use client';

import React, { useState } from 'react';
import styles from './ShareButton.module.css';
import { trackShare } from '@/lib/analytics';

interface ShareButtonProps {
  query: string;
  work: {
    work_id: string;
    title: string;
    composer: string;
    work_type?: string;
    period?: string;
  };
  userName?: string;
}

export const ShareButton: React.FC<ShareButtonProps> = ({ query, work, userName }) => {
  const [isSharing, setIsSharing] = useState(false);
  const [showModal, setShowModal] = useState(false);
  const [copied, setCopied] = useState(false);

  // Generate share URLs
  // Backend URL for social media sharing (has Open Graph tags)
  const backendUrl = 'http://localhost:8000';
  const shareUrl = `${backendUrl}/share/${work.work_id}?query=${encodeURIComponent(query)}`;
  const shareCardUrl = `${backendUrl}/api/share-card/${work.work_id}?query=${encodeURIComponent(query)}${userName ? `&user_name=${encodeURIComponent(userName)}` : ''}`;

  const handleShare = async () => {
    setIsSharing(true);

    // Always show the custom modal with social media options and share card preview
    // This provides a consistent experience across all platforms and shows the card design
    setShowModal(true);
    setIsSharing(false);
  };

  const copyLink = async () => {
    try {
      await navigator.clipboard.writeText(shareUrl);
      setCopied(true);
      trackShare('copy-link', 'success', { workId: work.work_id, query });
      setTimeout(() => setCopied(false), 2000);
    } catch (err) {
      console.error('Copy failed:', err);
      // Fallback for older browsers
      fallbackCopyText(shareUrl);
    }
  };

  const fallbackCopyText = (text: string) => {
    const textArea = document.createElement('textarea');
    textArea.value = text;
    textArea.style.position = 'fixed';
    textArea.style.left = '-999999px';
    document.body.appendChild(textArea);
    textArea.select();
    try {
      document.execCommand('copy');
      setCopied(true);
      trackShare('copy-link', 'success', { workId: work.work_id, query });
      setTimeout(() => setCopied(false), 2000);
    } catch (err) {
      console.error('Fallback copy failed:', err);
    }
    document.body.removeChild(textArea);
  };

  const downloadCard = () => {
    const link = document.createElement('a');
    link.href = shareCardUrl;
    link.download = `${work.title.replace(/[^a-z0-9]/gi, '_')}_share.png`;
    link.click();
    trackShare('download-card', 'success', { workId: work.work_id, query });
  };

  const shareToTwitter = () => {
    const text = `I searched for "${query}" and discovered the perfect piece 🎵\n\n${work.title} by ${work.composer}\n\nThe AI really understood my vibe ✨`;
    const url = `https://twitter.com/intent/tweet?text=${encodeURIComponent(text)}&url=${encodeURIComponent(shareUrl)}`;
    window.open(url, '_blank', 'width=550,height=420');
    trackShare('twitter', 'initiated', { workId: work.work_id, query });
  };

  const shareToFacebook = () => {
    const url = `https://www.facebook.com/sharer/sharer.php?u=${encodeURIComponent(shareUrl)}`;
    window.open(url, '_blank', 'width=550,height=420');
    trackShare('facebook', 'initiated', { workId: work.work_id, query });
  };

  const shareToInstagram = async () => {
    // Instagram doesn't have a direct web share API like Twitter/Facebook
    // The best approach is to download the image and prompt the user to share it
    // We'll download the card and copy the share URL for the user to paste in their caption

    // First download the card
    downloadCard();

    // Then copy the link
    await copyLink();

    // Show helpful instructions
    const message = `✨ Share card downloaded!\n\n` +
      `📸 Next steps:\n` +
      `1. Open Instagram app\n` +
      `2. Create a new post\n` +
      `3. Upload the downloaded image\n` +
      `4. Paste the link (already copied!) in your caption\n\n` +
      `🎵 Share your classical music discovery!`;

    alert(message);
    trackShare('instagram', 'initiated', { workId: work.work_id, query });
  };

  const ShareIcon = () => (
    <svg
      width="18"
      height="18"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
    >
      <circle cx="18" cy="5" r="3"></circle>
      <circle cx="6" cy="12" r="3"></circle>
      <circle cx="18" cy="19" r="3"></circle>
      <line x1="8.59" y1="13.51" x2="15.42" y2="17.49"></line>
      <line x1="15.41" y1="6.51" x2="8.59" y2="10.49"></line>
    </svg>
  );

  const CheckIcon = () => (
    <svg
      width="18"
      height="18"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
    >
      <polyline points="20 6 9 17 4 12"></polyline>
    </svg>
  );

  const XIcon = () => (
    <svg
      width="20"
      height="20"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
    >
      <line x1="18" y1="6" x2="6" y2="18"></line>
      <line x1="6" y1="6" x2="18" y2="18"></line>
    </svg>
  );

  return (
    <>
      <button
        onClick={handleShare}
        disabled={isSharing}
        className={styles.shareButton}
      >
        <ShareIcon />
        <span>{isSharing ? 'Sharing...' : 'Share'}</span>
      </button>

      {showModal && (
        <div className={styles.modalOverlay} onClick={() => setShowModal(false)}>
          <div className={styles.modal} onClick={(e) => e.stopPropagation()}>
            <div className={styles.modalHeader}>
              <h3>Share Your Discovery</h3>
              <button onClick={() => setShowModal(false)} className={styles.closeButton}>
                <XIcon />
              </button>
            </div>

            <div className={styles.sharePreview}>
              <img src={shareCardUrl} alt="Share card preview" loading="lazy" />
            </div>

            <div className={styles.shareOptions}>
              <button onClick={shareToTwitter} className={`${styles.shareOption} ${styles.twitter}`}>
                <span className={styles.emoji}>🐦</span>
                <span>Share on Twitter</span>
              </button>

              <button onClick={shareToFacebook} className={`${styles.shareOption} ${styles.facebook}`}>
                <span className={styles.emoji}>📘</span>
                <span>Share on Facebook</span>
              </button>

              <button onClick={shareToInstagram} className={`${styles.shareOption} ${styles.instagram}`}>
                <span className={styles.emoji}>📸</span>
                <span>Share on Instagram</span>
              </button>

              <button onClick={downloadCard} className={`${styles.shareOption} ${styles.download}`}>
                <span className={styles.emoji}>📥</span>
                <span>Download Card</span>
              </button>

              <button onClick={copyLink} className={`${styles.shareOption} ${styles.copy}`}>
                {copied ? <CheckIcon /> : <span className={styles.emoji}>🔗</span>}
                <span>{copied ? 'Copied!' : 'Copy Link'}</span>
              </button>
            </div>
          </div>
        </div>
      )}
    </>
  );
};
