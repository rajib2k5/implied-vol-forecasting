# Implied Vol Forecasting: app

The visual front end of the implied-volatility forecasting project: forecasting implied volatility
(VIX and its term structure) with ARIMA, HAR, GARCH-family and LSTM models, evaluated walk-forward and
turned into a VXX/SVXY trading backtest.

The Python package does all the computation. This app only visualises what the pipeline exports
(versioned JSON), plus small tested transforms such as the 3D ribbon geometry. It is built with
React 19, TypeScript, Vite and React Three Fiber.

> Not financial advice. This is an educational project; backtests are hypothetical and past
> performance does not predict future results.

## Pages

| Page | What it shows |
|---|---|
| About | The project, the original requirement, what it teaches (HAR baseline, GARCH on returns not on the IV level, QLIKE, Diebold-Mariano, Mincer-Zarnowitz, identical walk-forward splits, a holdout from 2024-07-01 that keeps the Aug-2024 and Apr-2025 spikes), the milestone roadmap, data sources and licensing. |
| Architecture | The end-to-end pipeline as an animated diagram, one walk-forward fold as a sequence diagram, and the per-milestone process as an activity diagram. A text list gives the exact data dependencies. |
| Term structure 3D | A placeholder 3D ribbon of a **synthetic** term structure (5 tenors by 60 dates), coloured by contango (blue) and backwardation (orange). Falls back to a flat heatmap when WebGL is unavailable. |

## Run

Requires Node 22 (and 22.18 or newer for `npm run gen:fixture`).

```bash
npm ci
npm run dev        # local dev server
npm run verify     # lint + tests + coverage thresholds + production build + deploy guard: the gate
npm run coverage   # tests with a coverage table
npm run build      # type-check and production build into dist/
npm run guard:dist # fail if dist/ or the build input could ship real market data (run after build)
npm run preview    # serve dist/ locally
npm run gen:fixture  # regenerate src/data/term-structure.synthetic.json
```

## Layout

```
src/lib/        pure, unit-tested code: term-structure maths and generator, pipeline/fold/process
                content and validators, project facts, WebGL detection, seeded rng, plus the shared
                shell, charts and diagram components
src/pages/      About, Architecture and 3D pages; RibbonCanvas.tsx is the only file that imports three.js
src/data/       the committed synthetic fixture
scripts/        genFixture.ts, which regenerates the fixture; guardDist.mjs, the deploy guard (and its tests)
```

## Data

The app ships a synthetic fixture only. CBOE data is not redistributed (personal, non-commercial
terms); the Python pipeline fetches vendor data into a local cache outside the repository.
`npm run verify` ends with a deploy guard that fails if real data could ship. See
[DATA.md](DATA.md).

## 3D and WebGL

The 3D code (three.js, `@react-three/fiber`, `@react-three/drei`) lives in a separate chunk that is
loaded only when the 3D page is shown on a browser with WebGL. Without WebGL, or if the renderer
fails to start, the page shows the same numbers as a flat heatmap with an explanation. Geometry
(vertex positions, triangle indices, contango/backwardation colours) is computed by a pure,
unit-tested function in `src/lib/termStructure.ts`; the component only uploads it to the GPU.

## Tests

- `src/lib/*.test.ts(x)`: unit tests on the pure code, with expected values worked out by hand
  (the arithmetic is in a comment next to each assertion) or taken from an independent
  implementation.
- `scripts/guardDist.test.mjs`: the deploy guard, run against temporary folders that hold a clean build,
  a planted real-data marker anywhere in the app, JSON without a synthetic `data_mode`, symlinks and a
  forbidden `public/data/real/`.
- `src/pages/*.test.tsx`, `src/App.test.tsx`: component and integration tests. jsdom cannot run
  WebGL, so the canvas module is replaced by a stand-in and the fallback paths are tested directly.
- Coverage thresholds on `src/lib/**` (85% lines and functions, 75% branches) are enforced in
  `vite.config.ts`.

## Dependencies

Every dependency in `package.json` is pinned to an exact version. React Three Fiber 9 is the line
that supports React 19, and `@react-three/drei` 10 matches it. Install from the lockfile with
`npm ci`.

## Docker

```bash
docker build -t implied-vol-forecasting-app .
docker run -p 8080:8080 implied-vol-forecasting-app
```

Multi-stage build: Node builds the bundle, `npm run guard:dist` runs on it (the image build fails if
real market data could ship), and `nginxinc/nginx-unprivileged` serves it on port 8080 as a non-root
user. nginx adds `X-Content-Type-Options`, `Referrer-Policy` and a Content-Security-Policy, and does
not announce its version. `.dockerignore` keeps `public/data/real`, `.env*`, `dist`, `coverage` and
`node_modules` out of the build context.

## Limitations

Milestone 0: the app shell, the About and Architecture pages and a 3D placeholder driven by
synthetic data. Real data, model results and the backtest arrive with milestones 1 to 6 (see the
roadmap on the About page).
