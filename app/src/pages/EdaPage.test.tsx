import { afterEach, describe, expect, it, vi } from 'vitest'
import { render, screen } from '@testing-library/react'
import eda from '../../public/data/synthetic/eda.json'
import { serve } from '../test/committedData'
import EdaPage from './EdaPage'

afterEach(() => vi.unstubAllGlobals())

describe('EdaPage on the committed synthetic file', () => {
  it('draws one ACF chart per series with its bars and white-noise band', async () => {
    render((serve(), <EdaPage />))
    await screen.findByText('synthetic data')
    for (const s of eda.payload.acf) {
      const figure = screen.getByRole('figure', { name: new RegExp(`^ACF of ${s.label.replace(/[|()]/g, '\\$&')}`) })
      expect(figure.querySelectorAll('[data-testid="viz-bar"]')).toHaveLength(s.acf.filter((v) => v !== null).length)
      expect(figure.querySelector('[data-testid="viz-band"]')).not.toBeNull()
      expect(figure).toHaveTextContent(`dashed: ±${s.band95} (95% white-noise band)`)
    }
  })
  it('shows each sample half-life with its interval', async () => {
    render((serve(), <EdaPage />))
    await screen.findByText('synthetic data')
    for (const h of eda.payload.half_life) {
      expect(screen.getByText(`${h.half_life_sessions} sessions`)).toBeInTheDocument()
      expect(screen.getByText(new RegExp(`95% CI ${h.half_life_ci95[0]} to ${h.half_life_ci95[1]}; phi ${h.phi}`))).toBeInTheDocument()
    }
  })
  it('greys out when the file is missing', async () => {
    render((serve({}), <EdaPage />))
    expect(await screen.findByRole('alert')).toHaveTextContent('eda.json: HTTP 404')
  })
})
