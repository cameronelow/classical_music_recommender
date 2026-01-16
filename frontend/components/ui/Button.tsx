import React from 'react'

interface ButtonProps extends React.ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: 'primary' | 'secondary' | 'dark' | 'header' | 'landing'
  children: React.ReactNode
}

export default function Button({
  variant = 'primary',
  children,
  className = '',
  ...props
}: ButtonProps) {
  const baseStyles = 'px-6 py-2 rounded-button font-medium transition-all duration-200 hover:scale-105 active:scale-95 disabled:opacity-60 disabled:cursor-not-allowed disabled:hover:scale-100'

  const variantStyles = {
    primary: 'bg-light-cream text-dark-blue hover:bg-opacity-90',
    secondary: 'bg-transparent border-2 border-light-cream text-light-cream hover:bg-light-cream hover:bg-opacity-10',
    dark: 'bg-dark-blue text-light-cream hover:bg-opacity-90',
    header: 'bg-light-cream text-dark-blue hover:bg-opacity-90 w-20 sm:w-28 md:w-button-header h-10 sm:h-12 md:h-button-header rounded-button-lg text-xs sm:text-sm md:text-base',
    landing: 'bg-button-dark text-light-cream hover:bg-opacity-90 w-full sm:w-button-landing h-12 sm:h-button-landing text-base sm:text-lg md:text-heading-lg',
  }

  return (
    <button
      className={`${baseStyles} ${variantStyles[variant]} ${className}`}
      {...props}
    >
      {children}
    </button>
  )
}
