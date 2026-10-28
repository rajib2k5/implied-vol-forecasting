/**
 * WebGL capability check, kept separate from the 3D component so that the decision
 * "can we draw the 3D view at all?" is testable without a GPU (jsdom has no WebGL).
 */

/** True when a WebGL (1 or 2) context can be created on a throwaway canvas. Never throws. */
export function detectWebGL(doc: Pick<Document, 'createElement'> = document): boolean {
  try {
    const canvas = doc.createElement('canvas')
    const gl = canvas.getContext('webgl2') ?? canvas.getContext('webgl')
    return gl != null
  } catch {
    return false
  }
}
