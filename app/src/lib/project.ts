/**
 * Facts about the project that the About page shows: pitch, original requirement, teaching
 * points, milestone roadmap, data sources. Kept as plain data so invariants (exactly one
 * current milestone, no redistributed vendor data, ...) can be tested instead of trusted.
 */

export const APP_SLUG = 'implied-vol-forecasting-app'
export const PROJECT_TITLE = 'Implied Vol Forecasting'
export const PROJECT_PITCH =
  'Forecasting implied volatility (VIX and its term structure) with ARIMA, HAR, GARCH-family and LSTM models, evaluated walk-forward and turned into a VXX/SVXY trading backtest.'

export const PROJECT_INTRO =
  'A learning project built in milestones. A Python package does all the computation; this app only visualises what the pipeline exports, so every number you see can be traced back to tested code.'

export const REQUIREMENT =
  'Develop and compare models that forecast implied volatility (ARIMA, GARCH, and machine learning such as LSTM). ' +
  'Start with exploratory analysis of volatility clustering and persistence, evaluate forecasts with RMSE, MAE and MAPE, ' +
  'and backtest trading strategies driven by the forecasts.'

/* ------------------------------------------------------------ teaching */

export interface TeachingPoint {
  id: string
  title: string
  body: string
}

export const TEACHING_POINTS: TeachingPoint[] = [
  {
    id: 'har',
    title: 'HAR is the baseline to beat',
    body: 'The heterogeneous autoregressive model (Corsi, 2009) regresses tomorrow\'s volatility on its daily, weekly and monthly averages. It is simple and hard to beat, so every fancier model is compared with it, not just with a random walk.',
  },
  {
    id: 'garch-category',
    title: 'GARCH belongs on returns, not on the IV level',
    body: 'GARCH models the conditional variance of a return series. Implied volatility is a level, an option-implied quote of future volatility, not a return, so fitting GARCH to the IV level is a category error. Here GARCH, GJR and EGARCH are fitted to the underlying\'s returns and their 21-day variance forecasts are compared with IV; ARIMA-GARCH is applied to log-IV changes.',
  },
  {
    id: 'qlike',
    title: 'QLIKE next to RMSE, MAE and MAPE',
    body: 'RMSE, MAE and MAPE depend on the level of volatility. QLIKE is a loss designed for variance forecasts: it punishes under-forecasting more than over-forecasting, and it stays valid when the realised-volatility target is a noisy proxy (Patton, 2011).',
  },
  {
    id: 'dm-mz',
    title: 'Diebold-Mariano and Mincer-Zarnowitz',
    body: 'Diebold-Mariano asks whether two models\' losses really differ or the gap is luck. Mincer-Zarnowitz regresses outcomes on forecasts: an unbiased forecast has intercept 0 and slope 1.',
  },
  {
    id: 'splits',
    title: 'Identical walk-forward splits, no look-ahead',
    body: 'Every model is trained and scored on the same walk-forward folds, and scalers are fitted on the training window only, so no information from the future leaks into a forecast.',
  },
  {
    id: 'holdout',
    title: 'A holdout with real stress',
    body: 'Everything from the holdout start (proposed: 2024-07-01, about 28 months) is held out from tuning. A strict trailing 24 months would start in late October 2024 and miss the August 2024 spike, so the fixed start keeps both the August 2024 and April 2025 spikes in the final comparison. Horizons are 1, 5 and 21 trading days.',
  },
]

/* ----------------------------------------------------------- milestones */

export type MilestoneStatus = 'done' | 'current' | 'planned'

export interface Milestone {
  id: number
  title: string
  summary: string
  status: MilestoneStatus
}

export const MILESTONES: Milestone[] = [
  { id: 0, title: 'Scaffold and design', summary: 'Repository, design documents, diagrams and this app shell with its About, Architecture and 3D placeholder pages.', status: 'current' },
  { id: 1, title: 'Data and EDA', summary: 'Load the CBOE volatility indices and underlyings; study clustering, persistence, AR(1) half-life, mean reversion and the IV-RV spread.', status: 'planned' },
  { id: 2, title: 'Baselines', summary: 'Random walk, historical mean, ARIMA on log-IV, and HAR.', status: 'planned' },
  { id: 3, title: 'GARCH done right', summary: 'GARCH, GJR and EGARCH on underlying returns for 21-day forecasts; ARIMA-GARCH on log-IV changes.', status: 'planned' },
  { id: 4, title: 'LSTM', summary: 'A PyTorch LSTM on the same walk-forward splits, with scaling fitted on train only, early stopping and several seeds.', status: 'planned' },
  { id: 5, title: 'Evaluation', summary: 'RMSE, MAE, MAPE and QLIKE at horizons 1, 5 and 21; Diebold-Mariano, Mincer-Zarnowitz and a regime breakdown.', status: 'planned' },
  { id: 6, title: 'Trading backtest', summary: 'A forecast-driven VXX/SVXY strategy with costs, compared with always-long and always-short volatility, with PSR and DSR.', status: 'planned' },
]

/** Problems with a roadmap, as readable strings; empty means valid. */
export function validateMilestones(ms: readonly Milestone[]): string[] {
  const problems: string[] = []
  ms.forEach((m, i) => {
    if (m.id !== i) problems.push(`milestone at position ${i} has id ${m.id}; ids must run 0, 1, 2, ...`)
    if (m.title.trim() === '' || m.summary.trim() === '') problems.push(`milestone ${m.id} needs a title and a summary`)
  })
  const current = ms.filter((m) => m.status === 'current')
  if (current.length !== 1) {
    problems.push(`exactly one milestone must be current, found ${current.length}`)
    return problems
  }
  const cur = current[0].id
  for (const m of ms) {
    if (m.id < cur && m.status !== 'done') problems.push(`milestone ${m.id} comes before the current one and must be done`)
    if (m.id > cur && m.status !== 'planned') problems.push(`milestone ${m.id} comes after the current one and must be planned`)
  }
  return problems
}

/* ---------------------------------------------------------- data sources */

export interface DataSource {
  id: string
  name: string
  provides: string
  /** Where the licence / terms question stands, in plain words. */
  terms: string
  /** True only if the repository itself contains this source's data. */
  committedToRepo: boolean
}

export const DATA_SOURCES: DataSource[] = [
  {
    id: 'cboe',
    name: 'CBOE volatility index history (CSV)',
    provides: 'VIX, VIX9D, VIX3M, VIX6M, VIX1Y and related indices: the implied-volatility series and its term structure.',
    terms: 'CBOE\'s terms restrict use to personal, non-commercial purposes. The data is fetched by the Python loaders into a local cache outside the repository and is not redistributed here.',
    committedToRepo: false,
  },
  {
    id: 'fred',
    name: 'FRED volatility series (VIXCLS and relatives)',
    provides: 'A fallback daily source for the same indices.',
    terms: 'Fetched at run time and cached locally; not committed to this repository.',
    committedToRepo: false,
  },
  {
    id: 'yfinance',
    name: 'Yahoo Finance via the yfinance library',
    provides: 'Underlying prices (for realised volatility and GARCH) and the VXX and SVXY ETPs used as the tradable proxy.',
    terms: 'Fetched at run time for personal research; not committed. VXX and SVXY have structural breaks the backtest must handle: VXX series resets, and SVXY\'s February 2018 leverage cut from -1x to -0.5x.',
    committedToRepo: false,
  },
  {
    id: 'synthetic',
    name: 'Synthetic fixture (ships with this app)',
    provides: 'A generated 5-tenor x 60-date term structure that drives the 3D placeholder.',
    terms: 'Produced by a seeded generator in this repository; it contains no vendor data and is labelled synthetic wherever it is shown.',
    committedToRepo: true,
  },
]

/** Problems with the data-source list; the policy is that only the synthetic fixture may be committed. */
export function validateDataSources(sources: readonly DataSource[]): string[] {
  const problems: string[] = []
  const seen = new Set<string>()
  for (const s of sources) {
    if (seen.has(s.id)) problems.push(`duplicate data source id: ${s.id}`)
    seen.add(s.id)
    if (s.id !== 'synthetic' && s.committedToRepo) problems.push(`${s.id}: third-party data must not be committed to the repository`)
    if (s.id === 'synthetic' && !s.committedToRepo) problems.push('the synthetic fixture is expected to ship with the app')
    if (s.terms.trim() === '') problems.push(`${s.id}: terms must be stated`)
  }
  return problems
}

export const DISCLAIMER =
  'Not financial advice. This is an educational project: nothing here is a recommendation or an offer to trade, backtests are hypothetical, and past performance does not predict future results.'
