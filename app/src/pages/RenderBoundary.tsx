import { Component, type ReactNode } from 'react'

/**
 * Shows `fallback` instead of crashing the page when a child throws while rendering.
 * Needed around the WebGL canvas: creating a GL context can fail on machines without a usable
 * GPU driver even when feature detection said yes, and a lazy chunk can fail to download.
 */
export class RenderBoundary extends Component<{ fallback: ReactNode; children: ReactNode }, { failed: boolean }> {
  state = { failed: false }

  static getDerivedStateFromError() {
    return { failed: true }
  }

  render() {
    return this.state.failed ? this.props.fallback : this.props.children
  }
}
