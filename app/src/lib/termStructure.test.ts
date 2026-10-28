import { describe, expect, it } from 'vitest'
import fixture from '../data/term-structure.synthetic.json'
import {
  BACKWARDATION_RGB,
  CONTANGO_RGB,
  FLAT_TOLERANCE,
  NEUTRAL_RGB,
  SCHEMA_VERSION,
  TENORS,
  TENOR_DAYS,
  buildRibbonGeometry,
  businessDates,
  classifyCurve,
  curveLevel,
  generateTermStructure,
  parseTermStructure,
  regimeCounts,
  regimeSummary,
  rgbToHex,
  slopeColour,
  termSlope,
  type TermStructureData,
} from './termStructure'

/** A tiny valid payload to mutate in the parser tests. */
const valid = (): Record<string, unknown> => ({
  schema_version: SCHEMA_VERSION,
  data_mode: 'synthetic',
  synthetic: true,
  tenors: ['A', 'B'],
  dates: ['2024-01-02', '2024-01-03'],
  levels: [
    [10, 12],
    [11, 13],
  ],
})

function grid(levels: number[][]): TermStructureData {
  return {
    schema_version: SCHEMA_VERSION,
    data_mode: 'synthetic',
    synthetic: true,
    description: '',
    tenors: levels[0].map((_, i) => `T${i}`),
    dates: levels.map((_, i) => `2024-01-${String(i + 1).padStart(2, '0')}`),
    levels,
  }
}

describe('businessDates', () => {
  it('lists consecutive weekdays, skipping the weekend', () => {
    // 2024-01-02 was a Tuesday: Tue 2, Wed 3, Thu 4, Fri 5, then Sat 6 and Sun 7 are skipped -> Mon 8.
    expect(businessDates('2024-01-02', 5)).toEqual(['2024-01-02', '2024-01-03', '2024-01-04', '2024-01-05', '2024-01-08'])
  })
  it('starts on the next weekday when the start is a weekend day', () => {
    // 2024-01-06 is a Saturday -> first date is Monday the 8th.
    expect(businessDates('2024-01-06', 2)).toEqual(['2024-01-08', '2024-01-09'])
  })
  it('rolls over month ends and leap days', () => {
    // 2024 is a leap year: Wed Feb 28, Thu Feb 29, Fri Mar 1, (weekend), Mon Mar 4.
    expect(businessDates('2024-02-28', 4)).toEqual(['2024-02-28', '2024-02-29', '2024-03-01', '2024-03-04'])
  })
  it('returns nothing for a count of zero', () => {
    expect(businessDates('2024-01-02', 0)).toEqual([])
  })
  it('rejects impossible dates, malformed strings and bad counts', () => {
    expect(() => businessDates('2024-02-30', 1)).toThrow('invalid start date')
    expect(() => businessDates('Jan 2 2024', 1)).toThrow('invalid start date')
    expect(() => businessDates('2024-01-02', -1)).toThrow('invalid count')
    expect(() => businessDates('2024-01-02', 1.5)).toThrow('invalid count')
  })
})

describe('curveLevel', () => {
  it('equals spot at tau = 0 and relaxes to the long-run level far out', () => {
    expect(curveLevel(26, 20, 0, 60)).toBe(26) // 20 + 6 * exp(0) = 26
    expect(curveLevel(26, 20, Infinity, 60)).toBe(20) // exp(-inf) = 0
  })
  it('hand-worked: one e-folding out, the gap to long-run has shrunk to 1/e', () => {
    // tau = reversionDays = 60: 20 + (26 - 20) * e^-1 = 20 + 6 * 0.367879 = 22.207277
    expect(curveLevel(26, 20, 60, 60)).toBeCloseTo(22.207277, 5)
    // spot below long-run: 20 + (10 - 20) * e^-1 = 20 - 3.678794 = 16.321206 (upward-sloping = contango)
    expect(curveLevel(10, 20, 60, 60)).toBeCloseTo(16.321206, 5)
  })
  it('rejects a non-positive reversion horizon', () => {
    expect(() => curveLevel(15, 20, 30, 0)).toThrow('reversionDays')
    expect(() => curveLevel(15, 20, 30, -5)).toThrow('reversionDays')
  })
})

describe('generateTermStructure', () => {
  it('is reproducible from the seed and differs between seeds', () => {
    expect(generateTermStructure({ seed: 5 })).toEqual(generateTermStructure({ seed: 5 }))
    expect(generateTermStructure({ seed: 5 }).levels).not.toEqual(generateTermStructure({ seed: 6 }).levels)
  })
  it('has one row per date and one column per tenor, with business-day dates', () => {
    const d = generateTermStructure({ nDates: 7, startDate: '2024-01-02', shockDay: 3, seed: 77 })
    expect(d.tenors).toEqual([...TENORS])
    expect(d.levels).toHaveLength(7)
    for (const row of d.levels) expect(row).toHaveLength(TENOR_DAYS.length)
    expect(d.dates).toEqual(businessDates('2024-01-02', 7))
    expect(d.synthetic).toBe(true)
    expect(d.data_mode).toBe('synthetic')
    expect(d.generator).toEqual({ name: 'ivf-synthetic-term-structure', seed: 77 })
  })
  it('without noise, shocks or randomness every date sits on the analytic curve from spot = spotMean', () => {
    // spot stays at 15 (log-spot = ln 15, pull = 0, sigma = 0). level(tau) = 20 + (15 - 20) * exp(-tau / 60):
    //   9D : 20 - 5 * 0.860708 = 15.69646  -> 15.70
    //   30D: 20 - 5 * 0.606531 = 16.96735  -> 16.97
    //   3M : 20 - 5 * 0.219442 = 18.90279  -> 18.90   (91 days)
    //   6M : 20 - 5 * 0.048155 = 19.75923  -> 19.76   (182 days)
    //   1Y : 20 - 5 * 0.002281 = 19.98860  -> 19.99   (365 days)
    const d = generateTermStructure({ nDates: 3, sigma: 0, noiseSd: 0, shockDay: null })
    for (const row of d.levels) expect(row).toEqual([15.7, 16.97, 18.9, 19.76, 19.99])
  })
  it('a scripted spike pins spot to shockLevel, inverts the curve, then decays by the OU pull', () => {
    const d = generateTermStructure({ nDates: 4, sigma: 0, noiseSd: 0, shockDay: 2, shockLevel: 30 })
    // Day 2: spot = 30. level(tau) = 20 + 10 * exp(-tau / 60): 9D 28.61 ... 1Y 20.02 -> slope 20.02 - 28.61 < 0.
    expect(d.levels[2]).toEqual([28.61, 26.07, 22.19, 20.48, 20.02])
    expect(classifyCurve(d.levels[2])).toBe('backwardation')
    // Day 3: ln(spot) = ln 30 + 0.1 * (ln 15 - ln 30) = ln 30 - 0.1 ln 2, spot = 30 * 2^-0.1 = 27.99099.
    //   9D: 20 + 7.99099 * 0.860708 = 26.8777 -> 26.88
    expect(d.levels[3][0]).toBe(26.88)
    expect(d.levels[3]).toEqual([26.88, 24.85, 21.75, 20.38, 20.02])
    // Day 0: spot 15 < 20, so contango.
    expect(classifyCurve(d.levels[0])).toBe('contango')
  })
  it('with the default shock, every seed contains both a contango and a backwardation stretch', () => {
    for (let seed = 1; seed <= 40; seed++) {
      const c = regimeCounts(generateTermStructure({ seed }))
      expect(c.contango, `seed ${seed}`).toBeGreaterThan(0)
      expect(c.backwardation, `seed ${seed}`).toBeGreaterThan(0)
    }
  })
  it('only produces finite, positive levels rounded to 2 decimals', () => {
    for (let seed = 1; seed <= 20; seed++) {
      for (const v of generateTermStructure({ seed }).levels.flat()) {
        expect(v).toBeGreaterThan(0)
        expect(Math.round(v * 100) / 100).toBe(v)
      }
    }
  })
  it('rejects invalid options', () => {
    expect(() => generateTermStructure({ nDates: 0 })).toThrow('nDates')
    expect(() => generateTermStructure({ nDates: 2.5 })).toThrow('nDates')
    expect(() => generateTermStructure({ longRun: 0 })).toThrow('longRun')
    expect(() => generateTermStructure({ spotMean: -1 })).toThrow('longRun')
    expect(() => generateTermStructure({ kappa: 1.5 })).toThrow('kappa')
    expect(() => generateTermStructure({ sigma: -0.1 })).toThrow('sigma')
    expect(() => generateTermStructure({ noiseSd: -1 })).toThrow('sigma')
    expect(() => generateTermStructure({ nDates: 5, shockDay: 5 })).toThrow('shockDay')
    expect(() => generateTermStructure({ shockDay: -1 })).toThrow('shockDay')
    expect(() => generateTermStructure({ shockDay: 1.5 })).toThrow('shockDay')
    expect(() => generateTermStructure({ shockLevel: 0 })).toThrow('shockLevel')
  })
})

describe('committed fixture', () => {
  it('is exactly what the seeded generator produces (run `npm run gen:fixture` if this fails)', () => {
    expect(fixture).toEqual(generateTermStructure())
  })
  it('passes the parser and is labelled synthetic', () => {
    const parsed = parseTermStructure(fixture)
    expect(parsed.synthetic).toBe(true)
    // The deploy guard refuses a data file without this field, so the fixture must carry it.
    expect(parsed.data_mode).toBe('synthetic')
    expect(fixture.data_mode).toBe('synthetic')
    expect(parsed.dates).toHaveLength(60)
    expect(parsed.tenors).toEqual(['9D', '30D', '3M', '6M', '1Y'])
  })
})

describe('parseTermStructure', () => {
  it('accepts a minimal valid payload and defaults the description', () => {
    const p = parseTermStructure(valid())
    expect(p.tenors).toEqual(['A', 'B'])
    expect(p.description).toBe('')
    expect(p.generator).toBeUndefined()
  })
  it('carries data_mode through, for both of its two values', () => {
    expect(parseTermStructure(valid()).data_mode).toBe('synthetic')
    // A real-mode payload is representable (the dev server may serve one later); only the guard keeps it out of builds.
    expect(parseTermStructure({ ...valid(), data_mode: 'real', synthetic: false })).toMatchObject({ data_mode: 'real', synthetic: false })
  })
  it('keeps a well-formed generator block and drops a malformed one', () => {
    expect(parseTermStructure({ ...valid(), generator: { name: 'g', seed: 3 } }).generator).toEqual({ name: 'g', seed: 3 })
    expect(parseTermStructure({ ...valid(), generator: { name: 'g' } }).generator).toBeUndefined()
  })
  it.each([
    ['null', null, 'expected an object'],
    ['a string', 'x', 'expected an object'],
    ['wrong schema_version', { ...valid(), schema_version: 2 }, 'schema_version'],
    ['missing schema_version', { ...valid(), schema_version: undefined }, 'schema_version'],
    ['non-boolean synthetic', { ...valid(), synthetic: 'yes' }, 'synthetic'],
    ['missing data_mode', { ...valid(), data_mode: undefined }, 'data_mode must be'],
    ['unknown data_mode', { ...valid(), data_mode: 'demo' }, 'data_mode must be'],
    ['data_mode in the wrong case', { ...valid(), data_mode: 'Synthetic' }, 'data_mode must be'],
    ['synthetic flag true but data_mode real', { ...valid(), data_mode: 'real', synthetic: true }, 'contradicts'],
    ['synthetic flag false but data_mode synthetic', { ...valid(), synthetic: false }, 'contradicts'],
    ['empty tenors', { ...valid(), tenors: [] }, 'tenors'],
    ['non-string tenor', { ...valid(), tenors: ['A', 2] }, 'tenors'],
    ['no dates', { ...valid(), dates: [], levels: [] }, 'dates'],
    ['malformed date', { ...valid(), dates: ['2024-01-02', '03/01/2024'] }, 'ISO dates'],
    ['impossible date', { ...valid(), dates: ['2024-01-02', '2024-02-30'] }, 'ISO dates'],
    ['dates out of order', { ...valid(), dates: ['2024-01-03', '2024-01-02'] }, 'strictly ascending'],
    ['duplicate dates', { ...valid(), dates: ['2024-01-02', '2024-01-02'] }, 'strictly ascending'],
    ['too few level rows', { ...valid(), levels: [[10, 12]] }, 'one row per date'],
    ['too many level rows', { ...valid(), levels: [[10, 12], [11, 13], [12, 14]] }, 'one row per date'],
    ['levels not an array', { ...valid(), levels: 'x' }, 'one row per date'],
    ['short level row', { ...valid(), levels: [[10, 12], [11]] }, 'levels[1]'],
    ['row not an array', { ...valid(), levels: [[10, 12], 'x'] }, 'levels[1]'],
    ['NaN level', { ...valid(), levels: [[10, NaN], [11, 13]] }, 'levels[0][1]'],
    ['Infinity level', { ...valid(), levels: [[10, 12], [Infinity, 13]] }, 'levels[1][0]'],
    ['zero level', { ...valid(), levels: [[10, 12], [11, 0]] }, 'levels[1][1]'],
    ['negative level', { ...valid(), levels: [[-1, 12], [11, 13]] }, 'levels[0][0]'],
    ['string level', { ...valid(), levels: [[10, '12'], [11, 13]] }, 'levels[0][1]'],
  ])('rejects %s', (_name, payload, message) => {
    expect(() => parseTermStructure(payload)).toThrow(message)
  })
})

describe('termSlope and classifyCurve', () => {
  it('slope is the long end minus the short end', () => {
    expect(termSlope([10, 12, 15])).toBe(5) // 15 - 10
    expect(termSlope([12, 9])).toBe(-3) // 9 - 12
  })
  it('needs at least two tenors', () => {
    expect(() => termSlope([10])).toThrow('at least two tenors')
    expect(() => termSlope([])).toThrow('at least two tenors')
  })
  it('classifies by slope sign, treating |slope| <= tolerance as flat (boundary inclusive)', () => {
    expect(classifyCurve([1, 1.75], 0.5)).toBe('contango') // slope 0.75 > 0.5
    expect(classifyCurve([1.75, 1], 0.5)).toBe('backwardation') // slope -0.75 < -0.5
    expect(classifyCurve([1, 1.5], 0.5)).toBe('flat') // slope exactly 0.5: not strictly greater
    expect(classifyCurve([1.5, 1], 0.5)).toBe('flat') // slope exactly -0.5
    expect(classifyCurve([1, 1.5], 0.25)).toBe('contango')
  })
  it('uses FLAT_TOLERANCE by default', () => {
    expect(FLAT_TOLERANCE).toBeGreaterThan(0)
    expect(classifyCurve([10, 10 + 2 * FLAT_TOLERANCE])).toBe('contango')
    expect(classifyCurve([10, 10 - 2 * FLAT_TOLERANCE])).toBe('backwardation')
    expect(classifyCurve([10, 10])).toBe('flat')
  })
})

describe('regimeCounts and regimeSummary', () => {
  const g = grid([
    [10, 12], // +2 contango
    [12, 10], // -2 backwardation
    [10, 10], // 0 flat
    [9, 14], // +5 contango
  ])
  it('counts each shape', () => {
    expect(regimeCounts(g)).toEqual({ contango: 2, backwardation: 1, flat: 1 })
  })
  it('summarises them in one sentence', () => {
    expect(regimeSummary(g)).toBe('4 dates: 2 in contango, 1 in backwardation, 1 flat')
  })
})

/** Compare colours channel by channel: the lerp is not bit-exact (0.78 + 1 * (0.23 - 0.78) = 0.2299999...). */
function expectRgb(actual: readonly number[], expected: readonly number[]) {
  expect(actual).toHaveLength(3)
  expected.forEach((v, i) => expect(actual[i]).toBeCloseTo(v, 10))
}

describe('slopeColour', () => {
  it('is neutral grey for a flat curve and saturates at +/- scale', () => {
    expectRgb(slopeColour(0, 4), NEUTRAL_RGB)
    expectRgb(slopeColour(4, 4), CONTANGO_RGB)
    expectRgb(slopeColour(-4, 4), BACKWARDATION_RGB)
  })
  it('clamps beyond the scale', () => {
    expectRgb(slopeColour(40, 4), CONTANGO_RGB)
    expectRgb(slopeColour(-40, 4), BACKWARDATION_RGB)
  })
  it('interpolates linearly from neutral', () => {
    // slope +2 of scale 4 -> halfway from grey (0.78, 0.78, 0.78) to blue (0.31, 0.56, 0.85):
    //   r = (0.78 + 0.31) / 2 = 0.545, g = (0.78 + 0.56) / 2 = 0.67, b = (0.78 + 0.85) / 2 = 0.815
    const half = slopeColour(2, 4)
    expect(half[0]).toBeCloseTo(0.545, 10)
    expect(half[1]).toBeCloseTo(0.67, 10)
    expect(half[2]).toBeCloseTo(0.815, 10)
    // slope -1 of scale 4 -> a quarter of the way from grey to orange (0.88, 0.50, 0.23):
    //   r = 0.78 + 0.25 * (0.88 - 0.78) = 0.805, g = 0.78 + 0.25 * (0.50 - 0.78) = 0.71, b = 0.78 + 0.25 * (0.23 - 0.78) = 0.6425
    const quarter = slopeColour(-1, 4)
    expect(quarter[0]).toBeCloseTo(0.805, 10)
    expect(quarter[1]).toBeCloseTo(0.71, 10)
    expect(quarter[2]).toBeCloseTo(0.6425, 10)
  })
  it('rejects a non-positive scale', () => {
    expect(() => slopeColour(1, 0)).toThrow('scale')
    expect(() => slopeColour(1, -2)).toThrow('scale')
    expect(() => slopeColour(1, NaN)).toThrow('scale')
  })
})

describe('rgbToHex', () => {
  it('rounds each channel to 0..255 and pads to two hex digits', () => {
    expect(rgbToHex([1, 0, 0.5])).toBe('#ff0080') // 0.5 * 255 = 127.5 -> 128 = 0x80
    expect(rgbToHex([0, 0, 0])).toBe('#000000')
    expect(rgbToHex([0.2, 0.2, 0.2])).toBe('#333333') // 0.2 * 255 = 51 = 0x33
  })
  it('clamps out-of-range channels', () => {
    expect(rgbToHex([2, -1, 0.2])).toBe('#ff0033')
  })
  it('turns the palette constants into the colours quoted in their comments', () => {
    expect(rgbToHex(CONTANGO_RGB)).toBe('#4f8fd9') // 79, 143 (142.8), 217 (216.75)
    expect(rgbToHex(BACKWARDATION_RGB)).toBe('#e0803b') // 224 (224.4), 128 (127.5), 59 (58.65)
  })
})

describe('buildRibbonGeometry', () => {
  // 2 dates x 2 tenors. min = 10, max = 14, range = 4. width = depth = 2, height = 1, slopeScale = 4.
  const small = grid([
    [10, 14],
    [12, 10],
  ])
  const opts = { width: 2, depth: 2, height: 1, slopeScale: 4 }

  it('places vertices: tenor along x, date along z, level scaled into [0, height] along y', () => {
    const g = buildRibbonGeometry(small, opts)
    // vertex = date * nTenors + tenor. x = -1 + 2 * t/(nT-1), z = -1 + 2 * d/(nD-1), y = (level - 10) / 4.
    //   v0 (d0,t0): (-1, (10-10)/4 = 0,   -1)
    //   v1 (d0,t1): ( 1, (14-10)/4 = 1,   -1)
    //   v2 (d1,t0): (-1, (12-10)/4 = 0.5,  1)
    //   v3 (d1,t1): ( 1, (10-10)/4 = 0,    1)
    expect(Array.from(g.positions)).toEqual([-1, 0, -1, 1, 1, -1, -1, 0.5, 1, 1, 0, 1])
    expect(g.vertexCount).toBe(4)
    expect(g.levelRange).toEqual({ min: 10, max: 14 })
  })
  it('builds two counter-clockwise triangles per grid cell', () => {
    const g = buildRibbonGeometry(small, opts)
    // a = 0, b = a + 1 = 1, c = a + nT = 2, e = c + 1 = 3  ->  (a, c, b), (b, c, e)
    expect(Array.from(g.indices)).toEqual([0, 2, 1, 1, 2, 3])
    expect(g.triangleCount).toBe(2)
  })
  it('colours each date by its slope: saturated blue for +4/4, half-way to orange for -2/4', () => {
    const g = buildRibbonGeometry(small, opts)
    expect(g.slopes).toEqual([4, -2]) // 14 - 10, 10 - 12
    const c = Array.from(g.colors)
    // d0: slope/scale = 1 -> exactly the contango colour, for both its vertices.
    // d1: slope/scale = -0.5 -> grey + 0.5 * (orange - grey) = ((0.78+0.88)/2, (0.78+0.50)/2, (0.78+0.23)/2) = (0.83, 0.64, 0.505)
    const expected = [...CONTANGO_RGB, ...CONTANGO_RGB, 0.83, 0.64, 0.505, 0.83, 0.64, 0.505]
    expect(c).toHaveLength(12)
    c.forEach((v, i) => expect(v).toBeCloseTo(expected[i], 6))
  })
  it('faces every triangle up (+y) on a flat grid, which has zero level range and must not give NaN', () => {
    const flat = grid([
      [10, 10],
      [10, 10],
    ])
    const g = buildRibbonGeometry(flat, opts)
    expect(Array.from(g.positions).every(Number.isFinite)).toBe(true)
    expect(Array.from(g.positions).filter((_, i) => i % 3 === 1)).toEqual([0, 0, 0, 0]) // all on the floor
    const p = (v: number) => [g.positions[v * 3], g.positions[v * 3 + 1], g.positions[v * 3 + 2]]
    for (let t = 0; t < g.triangleCount; t++) {
      const [a, b, c] = [g.indices[t * 3], g.indices[t * 3 + 1], g.indices[t * 3 + 2]].map(p)
      // normal.y = (b - a) x (c - a) component along y = (bz-az)*(cx-ax) - (bx-ax)*(cz-az)
      const ny = (b[2] - a[2]) * (c[0] - a[0]) - (b[0] - a[0]) * (c[2] - a[2])
      // e.g. triangle (0, 2, 1): a=(-1,0,-1) b=(-1,0,1) c=(1,0,-1): (1-(-1)) * (1-(-1)) - 0 * 0 = 4 > 0
      expect(ny, `triangle ${t}`).toBeGreaterThan(0)
    }
  })
  it('indexes a 3 x 3 grid correctly, including the last cell', () => {
    const three = grid([
      [1, 2, 3],
      [2, 3, 4],
      [3, 4, 5],
    ])
    const g = buildRibbonGeometry(three, opts)
    expect(g.vertexCount).toBe(9)
    expect(g.triangleCount).toBe(8) // (3-1) * (3-1) cells * 2
    expect(g.indices).toHaveLength(24)
    // Last cell (d=1, t=1): a = 1*3 + 1 = 4, b = 5, c = 7, e = 8 -> [4, 7, 5, 5, 7, 8], stored at 3 * 6 = 18.
    expect(Array.from(g.indices.slice(18, 24))).toEqual([4, 7, 5, 5, 7, 8])
    expect(Math.max(...g.indices)).toBe(8)
  })
  it('stretches the committed fixture to exactly [0, height] in y and to +/- half the extents in x and z', () => {
    const g = buildRibbonGeometry(parseTermStructure(fixture))
    expect(g.vertexCount).toBe(60 * 5)
    expect(g.triangleCount).toBe(59 * 4 * 2)
    const xs = Array.from(g.positions).filter((_, i) => i % 3 === 0)
    const ys = Array.from(g.positions).filter((_, i) => i % 3 === 1)
    const zs = Array.from(g.positions).filter((_, i) => i % 3 === 2)
    expect(Math.min(...ys)).toBe(0)
    expect(Math.max(...ys)).toBeCloseTo(3, 5) // default height 3
    expect(Math.min(...xs)).toBe(-4) // default width 8
    expect(Math.max(...xs)).toBe(4)
    expect(Math.min(...zs)).toBe(-3) // default depth 6
    expect(Math.max(...zs)).toBe(3)
  })
  it('rejects non-positive options and grids too small to form a surface', () => {
    expect(() => buildRibbonGeometry(small, { width: 0 })).toThrow('finite and positive')
    expect(() => buildRibbonGeometry(small, { height: -1 })).toThrow('finite and positive')
    expect(() => buildRibbonGeometry(small, { slopeScale: NaN })).toThrow('finite and positive')
    expect(() => buildRibbonGeometry(grid([[10, 12]]))).toThrow('at least 2 dates')
    expect(() => buildRibbonGeometry(grid([[10], [11]]))).toThrow('at least 2 dates and 2 tenors')
  })
})
