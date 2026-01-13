import { test, expect } from '@playwright/test'

test.describe('Landing Page', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/')
  })

  test('has correct title', async ({ page }) => {
    await expect(page).toHaveTitle(/CLASSICAL VIBE/)
  })

  test('displays main heading', async ({ page }) => {
    const heading = page.getByRole('heading', { name: /What's your vibe today\?/i })
    await expect(heading).toBeVisible()
  })

  test('has input field and button', async ({ page }) => {
    const input = page.getByPlaceholderText(/I'm feeling.../i)
    const button = page.getByRole('button', { name: /Find My Piece/i })

    await expect(input).toBeVisible()
    await expect(button).toBeVisible()
  })

  test('button is disabled when input is empty', async ({ page }) => {
    const button = page.getByRole('button', { name: /Find My Piece/i })
    await expect(button).toBeDisabled()
  })

  test('button is enabled when input has value', async ({ page }) => {
    const input = page.getByPlaceholderText(/I'm feeling.../i)
    const button = page.getByRole('button', { name: /Find My Piece/i })

    await input.fill('happy')
    await expect(button).toBeEnabled()
  })

  test('navigates to recommend page on button click', async ({ page }) => {
    const input = page.getByPlaceholderText(/I'm feeling.../i)
    const button = page.getByRole('button', { name: /Find My Piece/i })

    await input.fill('peaceful')
    await button.click()

    await expect(page).toHaveURL(/\/recommend\?vibe=peaceful/)
  })

  test('navigates on Enter key press', async ({ page }) => {
    const input = page.getByPlaceholderText(/I'm feeling.../i)

    await input.fill('energetic')
    await input.press('Enter')

    await expect(page).toHaveURL(/\/recommend\?vibe=energetic/)
  })

  test('shows header with navigation', async ({ page }) => {
    const logo = page.getByText('CLASSICAL VIBE').first()
    const loginButton = page.getByRole('link', { name: /Log In/i })
    const signupButton = page.getByRole('link', { name: /Sign Up/i })

    await expect(logo).toBeVisible()
    await expect(loginButton).toBeVisible()
    await expect(signupButton).toBeVisible()
  })

  test('skip to content link works', async ({ page }) => {
    // Focus skip link with keyboard
    await page.keyboard.press('Tab')

    const skipLink = page.getByText(/Skip to main content/i)
    await expect(skipLink).toBeFocused()

    // Click and verify main content is focused
    await skipLink.click()
    const mainContent = page.locator('#main-content')
    await expect(mainContent).toBeFocused()
  })
})

test.describe('Landing Page Accessibility', () => {
  test('has proper ARIA labels', async ({ page }) => {
    await page.goto('/')

    const input = page.getByPlaceholderText(/I'm feeling.../i)
    await expect(input).toHaveAttribute('aria-label', 'Enter your mood or vibe')
  })

  test('is keyboard navigable', async ({ page }) => {
    await page.goto('/')

    // Tab through interactive elements
    await page.keyboard.press('Tab') // Skip link
    await page.keyboard.press('Tab') // Logo
    await page.keyboard.press('Tab') // Login button
    await page.keyboard.press('Tab') // Signup button
    await page.keyboard.press('Tab') // Input field

    const input = page.getByPlaceholderText(/I'm feeling.../i)
    await expect(input).toBeFocused()

    await input.fill('calm')
    await page.keyboard.press('Tab') // Find My Piece button

    const button = page.getByRole('button', { name: /Find My Piece/i })
    await expect(button).toBeFocused()
  })
})
