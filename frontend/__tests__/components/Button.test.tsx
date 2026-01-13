import { render, screen, fireEvent } from '@testing-library/react'
import Button from '@/components/ui/Button'

describe('Button Component', () => {
  it('renders children correctly', () => {
    render(<Button>Click me</Button>)
    expect(screen.getByText('Click me')).toBeInTheDocument()
  })

  it('applies primary variant styles by default', () => {
    render(<Button>Primary</Button>)
    const button = screen.getByText('Primary')
    expect(button).toHaveClass('bg-light-cream', 'text-dark-blue')
  })

  it('applies dark variant styles', () => {
    render(<Button variant="dark">Dark</Button>)
    const button = screen.getByText('Dark')
    expect(button).toHaveClass('bg-dark-blue', 'text-light-cream')
  })

  it('applies header variant styles with correct dimensions', () => {
    render(<Button variant="header">Header</Button>)
    const button = screen.getByText('Header')
    expect(button).toHaveClass('w-button-header', 'h-button-header', 'rounded-button-lg')
  })

  it('handles onClick events', () => {
    const handleClick = jest.fn()
    render(<Button onClick={handleClick}>Click</Button>)
    fireEvent.click(screen.getByText('Click'))
    expect(handleClick).toHaveBeenCalledTimes(1)
  })

  it('disables button when disabled prop is true', () => {
    render(<Button disabled>Disabled</Button>)
    const button = screen.getByText('Disabled')
    expect(button).toBeDisabled()
    expect(button).toHaveClass('disabled:opacity-60', 'disabled:cursor-not-allowed')
  })

  it('prevents click events when disabled', () => {
    const handleClick = jest.fn()
    render(<Button disabled onClick={handleClick}>Disabled</Button>)
    fireEvent.click(screen.getByText('Disabled'))
    expect(handleClick).not.toHaveBeenCalled()
  })

  it('applies custom className', () => {
    render(<Button className="custom-class">Custom</Button>)
    expect(screen.getByText('Custom')).toHaveClass('custom-class')
  })

  it('forwards other HTML button props', () => {
    render(<Button type="submit" data-testid="submit-btn">Submit</Button>)
    const button = screen.getByTestId('submit-btn')
    expect(button).toHaveAttribute('type', 'submit')
  })
})
