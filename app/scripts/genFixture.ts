/**
 * Regenerates the committed synthetic fixture from the seeded generator.
 *   npm run gen:fixture      (needs Node >= 22.18, which runs .ts files directly)
 * A unit test fails if the committed JSON drifts from what this produces.
 */
import { writeFileSync } from 'node:fs'
import { generateTermStructure } from '../src/lib/termStructure.ts'

const out = new URL('../src/data/term-structure.synthetic.json', import.meta.url)
writeFileSync(out, JSON.stringify(generateTermStructure(), null, 2) + '\n')
console.log(`wrote ${out.pathname}`)
