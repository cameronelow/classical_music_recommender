import type { Config } from 'tailwindcss'

const config: Config = {
  content: [
    './pages/**/*.{js,ts,jsx,tsx,mdx}',
    './components/**/*.{js,ts,jsx,tsx,mdx}',
    './app/**/*.{js,ts,jsx,tsx,mdx}',
  ],
  theme: {
    extend: {
      colors: {
        'dark-blue': '#1A3263',
        'warm-yellow': '#FAB95B',
        'light-cream': '#E8E2DB',
        'button-dark': '#2C4A7A',
      },
      fontFamily: {
        sans: ['Actor', 'sans-serif'],
      },
      fontSize: {
        // Typography scale with responsive defaults
        'hero': ['4.625rem', { lineHeight: '1.1', letterSpacing: '0.05em' }], // 74px (desktop)
        'display': ['3rem', { lineHeight: '1.2', letterSpacing: '0.02em' }], // 48px (desktop)
        'heading-xl': ['2rem', { lineHeight: '1.3' }], // 32px
        'heading-lg': ['1.5rem', { lineHeight: '1.4' }], // 24px
        'heading-md': ['1.25rem', { lineHeight: '1.4' }], // 20px
        'body-lg': ['1.125rem', { lineHeight: '1.5' }], // 18px
        'body': ['1rem', { lineHeight: '1.5' }], // 16px
        'body-sm': ['0.875rem', { lineHeight: '1.5' }], // 14px
        // Mobile-optimized sizes
        'hero-mobile': ['2.5rem', { lineHeight: '1.1', letterSpacing: '0.05em' }], // 40px
        'display-mobile': ['2rem', { lineHeight: '1.2', letterSpacing: '0.02em' }], // 32px
      },
      spacing: {
        // Consistent spacing scale
        '18': '4.5rem', // 72px
        '22': '5.5rem', // 88px
      },
      borderRadius: {
        'button': '0.5rem', // 8px
        'button-lg': '1.5625rem', // 25px
      },
      minHeight: {
        'header': '5rem', // 80px
      },
      width: {
        'button-header': '8.5625rem', // 137px
        'button-landing': '16.125rem', // 258px
      },
      height: {
        'button-header': '3.3125rem', // 53px
        'button-landing': '3.125rem', // 50px
      },
    },
  },
  plugins: [],
}
export default config
