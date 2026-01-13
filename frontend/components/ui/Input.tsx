import React from 'react'

interface InputProps extends React.InputHTMLAttributes<HTMLInputElement> {
  label?: string
}

export default function Input({ label, className = '', ...props }: InputProps) {
  return (
    <div className="w-full">
      {label && (
        <label className="block text-dark-blue text-sm font-medium mb-2">
          {label}
        </label>
      )}
      <input
        className={`w-full px-4 py-2 bg-white border border-gray-300 rounded focus:outline-none focus:ring-2 focus:ring-dark-blue focus:border-transparent ${className}`}
        {...props}
      />
    </div>
  )
}
