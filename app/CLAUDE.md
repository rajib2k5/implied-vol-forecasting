# Implied Vol Forecasting: app, project guide for Claude

Forecasting implied volatility (VIX and its term structure) with ARIMA, HAR, GARCH-family and LSTM
models, evaluated walk-forward and turned into a VXX/SVXY trading backtest. This directory is the
React front end; the Python package that does the computation lives elsewhere in the repository.

React 19 + TypeScript + Vite + React Three Fiber (v9) / drei (v10) + vitest.

## Commands

```bash
npm run dev          # local dev server (do not leave it running in a session)
npm run verify       # lint + tests + coverage thresholds + production build + deploy guard: the gate
npm run coverage     # tests with coverage table
npm run guard:dist   # deploy guard (ADR 0001 D2): fail if dist/ or the build input could ship real data; run after build
npm run gen:fixture  # rewrite src/data/term-structure.synthetic.json from the seeded generator
```

## Architecture invariants

- The app never computes results. Python computes; the app draws versioned JSON plus small pure UI
  transforms in `src/lib/`.
- `src/lib/` holds pure, unit-tested code: no I/O, no wall clock, no unseeded randomness, in domain
  code **or tests**. Randomness only via the seeded `mulberry32` in `src/lib/rng.ts`.
  Documented UI exceptions, which may use the DOM, timers or browser storage but never feed a
  computed result: the shell, chart and diagram components, and `src/lib/storage.ts` with
  `src/lib/theme.ts` (theme persistence in `localStorage`, with an in-memory fallback). There are
  no usage counters or other analytics: nothing in the app records what a visitor does.
- three.js is imported only in `src/pages/RibbonCanvas.tsx`, which is loaded with `React.lazy`.
  Anything that decides what the 3D view looks like (positions, indices, colours) is a pure function in
  `src/lib/termStructure.ts` and is tested there. Keep the WebGL check and the fallback: the page must
  not crash without WebGL.
- Synthetic data is always labelled synthetic in the UI. Never commit vendor data (CBOE, FRED,
  yfinance); CBOE's terms are personal and non-commercial. The only real-data-derived numbers that may
  ever be committed are the closed publication set P of ADR 0001 D2 (unitless, rounded, whole periods,
  no regime cells; listed in `DATA.md`). Never a daily series, a per-date error or an equity curve.
- Real data never ships (ADR 0001 D2). `scripts/guardDist.mjs` (`npm run guard:dist`) checks the whole
  app directory (not `node_modules/`, `coverage/`, `.git/` at the top level) and fails on: any symlink
  anywhere; the marker `"data_mode":"real"` (any spacing or quoting) in any file, because a bundler
  strips it from imported JSON so `dist/` alone cannot show it; any `.json` under `src/`, `public/` or
  `dist/` that lacks a top-level `"data_mode": "synthetic"`; any non-`.json` file in `src/data/`,
  `public/data/` or `dist/data/`; `public/data/real/` or `dist/data/real/` existing; an empty or
  missing `dist/`. So every JSON data file you add needs `"data_mode": "synthetic"`, and a doc or
  test that quotes the marker must be a `*.md` or `*.test.*` file. `verify` runs it after the build
  and the Dockerfile runs it as a build step. Never weaken it. It is a tripwire for accidents:
  `DATA.md` lists what it cannot catch. Real JSON belongs in
  `$VOL_DATA_DIR/implied-vol-forecasting/app-data/real/`, which only the dev server reads.
- The container runs as a non-root user (`nginxinc/nginx-unprivileged`, port 8080). `nginx.conf` sets
  `server_tokens off`, `X-Content-Type-Options`, `Referrer-Policy` and a Content-Security-Policy
  (`default-src 'self'`, no third-party origins, no inline scripts). Anything that needs another
  origin, a worker or `eval` has to change that policy on purpose. `.dockerignore` excludes
  `public/data/real` and `.env*`.
- Coverage thresholds (85% lines/functions, 75% branches on `src/lib/**`) are enforced in
  `vite.config.ts` and must never be weakened. Tests are never deleted or `.skip`ped to go green.
- Dependencies are pinned exactly in `package.json` (no `^` or `~`). Install with `npm ci`. If you
  add one, use `npm install --save-exact` and check peer ranges (React Three Fiber 9 needs
  React 19, drei 10 needs fiber 9). Note: with npm 10.9.x a *fresh* dependency resolution (no
  lockfile) crashed in arborist for vitest 4.x; installs from the existing lockfile work, and npm 11
  resolved it.
- The UI renders numbers computed by the core; tests recompute expectations from the core or from
  hand-worked oracles, never a hard-coded copy of the UI's own output.

## Domain invariants (tested)

- The committed fixture equals the seeded generator's output and carries `"data_mode": "synthetic"` (the
  deploy guard and `parseTermStructure` both require it); every seed yields both a contango and a
  backwardation stretch; levels are finite, positive and rounded to 2 decimals.
- Ribbon geometry: vertex count = dates x tenors; two counter-clockwise (seen from +y) triangles per
  grid cell; y spans exactly [0, height]; colours come from each date's slope (long end minus short end).
- Exactly one milestone is `current`; earlier ones are `done`, later ones `planned`. Update
  `MILESTONES` in `src/lib/project.ts` when a milestone finishes.
- Pipeline diagram data: every edge references an existing node and flows to a later layer, no
  orphans, labels fit their boxes; the fold sequence fits the scaler on the training window before any
  forecast; the milestone process cannot reach "push" without the adversarial review.
- No third-party data source is marked as committed to the repository.

## Testing rules

- Expected values are hand-worked with the arithmetic in a comment, or come from an independent
  implementation or an external reference. Never copy the code under test into the test.
- Boundary cases are mandatory: empty, single element, exactly at a limit, NaN/Infinity, extremes.
- Mutation-check new tests: break the code, watch the test fail, restore. A test that passes both ways
  is not a test.
- Never write a test that greps source text; assert behaviour.
- jsdom cannot render WebGL: mock the canvas module and test the fallback paths.
