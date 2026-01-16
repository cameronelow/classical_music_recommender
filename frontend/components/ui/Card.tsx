import React from 'react'

interface CardProps {
  children: React.ReactNode
  className?: string
}

export default function Card({ children, className = '' }: CardProps) {
  return (
    <div className={`bg-light-cream rounded-lg shadow-lg p-4 sm:p-6 md:p-8 ${className}`}>
      {children}
    </div>
  )
}
