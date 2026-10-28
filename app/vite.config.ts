/// <reference types="vitest/config" />
import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// base './' so the built bundle works when served from any sub-path (static host, nginx, container)
export default defineConfig({
  base: './',
  plugins: [react()],
  build: {
    // The lazy 3D chunk (three.js + React Three Fiber) is ~940 kB minified. Warn above 1000 kB so
    // that growth of that chunk, or any creep of the main chunk towards it, still gets noticed.
    chunkSizeWarningLimit: 1000,
  },
  test: {
    environment: 'jsdom',
    globals: true,
    setupFiles: ['./src/test/setup.ts'],
    coverage: {
      provider: 'v8',
      include: ['src/lib/**'],
      thresholds: { lines: 85, functions: 85, branches: 75 },
    },
  },
})
