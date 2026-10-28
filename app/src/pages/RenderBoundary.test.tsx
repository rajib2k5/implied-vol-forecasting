import { afterEach, describe, expect, it, vi } from 'vitest'
import { render, screen } from '@testing-library/react'
import { RenderBoundary } from './RenderBoundary'

function Boom(): never {
  throw new Error('no GL context')
}

afterEach(() => {
  vi.restoreAllMocks()
})

describe('RenderBoundary', () => {
  it('renders its children when nothing throws', () => {
    render(
      <RenderBoundary fallback={<p>fallback</p>}>
        <p>child</p>
      </RenderBoundary>,
    )
    expect(screen.getByText('child')).toBeInTheDocument()
    expect(screen.queryByText('fallback')).toBeNull()
  })
  it('renders the fallback instead of crashing when a child throws during render', () => {
    vi.spyOn(console, 'error').mockImplementation(() => {}) // React logs the caught error
    render(
      <RenderBoundary fallback={<p>fallback</p>}>
        <Boom />
      </RenderBoundary>,
    )
    expect(screen.getByText('fallback')).toBeInTheDocument()
  })
})
