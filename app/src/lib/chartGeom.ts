/** Pure geometry helpers shared by the result charts (no DOM, no I/O). */

/** The CSS colour of a series slot: `--series-N`, or the ink colour for a null slot. */
export const strokeFor = (slot: number | null) => (slot === null ? 'var(--ink)' : `var(--series-${slot})`)

/** Split a series into runs of non-null points, so a gap is drawn as a gap. */
export function segments(values: (number | null)[]): { start: number; points: number[] }[] {
  const out: { start: number; points: number[] }[] = []
  values.forEach((v, i) => {
    if (v === null || !Number.isFinite(v)) return
    const last = out[out.length - 1]
    if (last && last.start + last.points.length === i) last.points.push(v)
    else out.push({ start: i, points: [v] })
  })
  return out
}

/** Push label y positions apart so no two are closer than `gap`, keeping their order. */
export function spreadLabels(ys: number[], gap: number): number[] {
  const order = ys.map((y, i) => ({ y, i })).sort((a, b) => a.y - b.y)
  for (let k = 1; k < order.length; k++) order[k].y = Math.max(order[k].y, order[k - 1].y + gap)
  const out = new Array<number>(ys.length)
  order.forEach(({ y, i }) => (out[i] = y))
  return out
}
