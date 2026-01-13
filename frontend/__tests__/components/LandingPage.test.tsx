import { render, screen, fireEvent, waitFor } from '@testing-library/react'
import { useRouter } from 'next/navigation'
import LandingPage from '@/components/LandingPage'

jest.mock('next/navigation')

describe('LandingPage Component', () => {
  const mockPush = jest.fn()

  beforeEach(() => {
    jest.clearAllMocks()
    ;(useRouter as jest.Mock).mockReturnValue({
      push: mockPush,
    })
  })

  it('renders the main heading', () => {
    render(<LandingPage />)
    expect(screen.getByText("What's your vibe today?")).toBeInTheDocument()
  })

  it('renders the input field with correct placeholder', () => {
    render(<LandingPage />)
    expect(screen.getByPlaceholderText("I'm feeling...")).toBeInTheDocument()
  })

  it('renders the Find My Piece button', () => {
    render(<LandingPage />)
    expect(screen.getByText('Find My Piece')).toBeInTheDocument()
  })

  it('disables button when input is empty', () => {
    render(<LandingPage />)
    const button = screen.getByText('Find My Piece')
    expect(button).toBeDisabled()
  })

  it('enables button when input has value', () => {
    render(<LandingPage />)
    const input = screen.getByPlaceholderText("I'm feeling...")
    const button = screen.getByText('Find My Piece')

    fireEvent.change(input, { target: { value: 'happy' } })
    expect(button).not.toBeDisabled()
  })

  it('navigates to recommend page with vibe on button click', async () => {
    render(<LandingPage />)
    const input = screen.getByPlaceholderText("I'm feeling...")
    const button = screen.getByText('Find My Piece')

    fireEvent.change(input, { target: { value: 'melancholic' } })
    fireEvent.click(button)

    await waitFor(() => {
      expect(mockPush).toHaveBeenCalledWith('/recommend?vibe=melancholic')
    })
  })

  it('navigates on Enter key press', async () => {
    render(<LandingPage />)
    const input = screen.getByPlaceholderText("I'm feeling...")

    fireEvent.change(input, { target: { value: 'energetic' } })
    fireEvent.keyDown(input, { key: 'Enter', code: 'Enter' })

    await waitFor(() => {
      expect(mockPush).toHaveBeenCalledWith('/recommend?vibe=energetic')
    })
  })

  it('does not navigate when Enter is pressed with empty input', () => {
    render(<LandingPage />)
    const input = screen.getByPlaceholderText("I'm feeling...")

    fireEvent.keyDown(input, { key: 'Enter', code: 'Enter' })
    expect(mockPush).not.toHaveBeenCalled()
  })

  it('shows loading state after submission', async () => {
    render(<LandingPage />)
    const input = screen.getByPlaceholderText("I'm feeling...")
    const button = screen.getByText('Find My Piece')

    fireEvent.change(input, { target: { value: 'peaceful' } })
    fireEvent.click(button)

    expect(screen.getByText('Thinking...')).toBeInTheDocument()
  })

  it('trims whitespace from input', async () => {
    render(<LandingPage />)
    const input = screen.getByPlaceholderText("I'm feeling...")
    const button = screen.getByText('Find My Piece')

    fireEvent.change(input, { target: { value: '  calm  ' } })
    fireEvent.click(button)

    await waitFor(() => {
      expect(mockPush).toHaveBeenCalledWith('/recommend?vibe=calm')
    })
  })

  it('has proper accessibility attributes', () => {
    render(<LandingPage />)
    const input = screen.getByPlaceholderText("I'm feeling...")
    expect(input).toHaveAttribute('aria-label', 'Enter your mood or vibe')
  })
})
