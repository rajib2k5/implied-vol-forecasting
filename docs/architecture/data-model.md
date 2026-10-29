# Data model (milestones 0-6)

The decisions behind this model are in
[ADR 0001](../adr/0001-python-core-versioned-json-app.md). This page is the reference: every stored
table and every JSON file the app reads. The machine-readable forms are:
- the ERD spec, [`specs/erd.yaml`](specs/erd.yaml), drawn in Mermaid as [`erd.mmd`](erd.mmd);
- the JSON Schemas in [`schemas/app-data/`](../../schemas/app-data/).

## Conventions

| Item | Rule |
|---|---|
| `ROOT` | `$VOL_DATA_DIR/implied-vol-forecasting`, where `VOL_DATA_DIR` defaults to `~/data/vol-projects`. Synthetic runs use `ROOT/synthetic/…` with the identical layout. |
| Types | `date` = Arrow `date32` (a session date, exchange-local, with no time zone). `ts` = `timestamp[us, UTC]`. `str`, `bool`. `i8`/`i16`/`i32`/`i64`. `f64`. |
| Keys | **PK** is the primary key (composite where listed). **FK** `entity.column` gives the target. Every processed table carries `run_id`, an FK to `run.run_id`. The ERD draws only the main `run` edges. |
| Null | A column is NOT NULL unless marked `null`. Floats never hold NaN on disk: a missing value is null. |
| Units | **Vol points** are annualised % (VIX units, e.g. 18.25). **Variance** is annualised decimal variance, `(vol points / 100)²`. Returns are decimal log returns unless named `*_ret` (simple returns). |
| Committed | ✅ in git, ❌ never in git, 🧪 committed for synthetic data only |

## Layer map

| Layer | Location | Format | Committed |
|---|---|---|---|
| Config | `configs/*.yaml` | YAML | ✅ |
| Raw | `ROOT/raw/{source}/{symbol}/{sha256}.csv` and `ROOT/raw/provenance.jsonl` | CSV bytes as fetched, plus append-only JSONL | ❌ |
| Interim | `ROOT/interim/{table}.parquet`, plus `ROOT/interim/build_report.json` | Parquet, plus one JSON report (amended M1) | ❌ |
| Processed | `ROOT/processed/runs.parquet` and `ROOT/processed/run_id=<id>/{table}.parquet` | Parquet, hive-partitioned | ❌ |
| Local EDA | `ROOT/eda/<run_id>/eda_report.json` (amended M1) | JSON, read locally only | ❌ |
| Reports | `reports/*.agg.csv` | CSV. Real data, ADR D2's publication set P only, clean tree. Provenance stays local in `ROOT/reports-provenance/` (amended M0, review E). | ✅ |
| App data | `app/public/data/synthetic/*.json` and `ROOT/app-data/real/*.json` | JSON (contract below) | 🧪 |

## 1. Config entities (committed YAML)

**`protocol`** is stored in `configs/protocol.yaml` (real data) and `configs/protocol.synthetic.yaml`.

| Column | Type | Key | Notes |
|---|---|---|---|
| protocol_hash | str(64) | PK | SHA-256 of the canonical sorted-key JSON of the file (derived, not typed in) |
| train_start | date | | 1990-01-02 for real data (to verify in M1) |
| dev_start | date | | 2000-01-03 |
| holdout_start | date | | Proposed 2024-07-01 (ADR O1) |
| data_end | date | | Frozen in M1. `null` in the file until `ivf.data check --freeze-data-end` writes it, once; it never moves a frozen value (ADR 0001 rule R1) |
| horizons | i16[] | | `[1, 5, 21]` |
| fold_len | i16 | | 63 sessions |
| min_cell_n | i16 | | 20 |
| vix_regime_edges | f64[2] | | `[15, 25]` |
| seed | i64 | | Root of every `SeedSequence` |
| lstm_members | i16 | | The number of seeds averaged into `lstm_log`. Not a key of `configs/protocol.yaml` yet (M4); `ivf.protocol` rejects an unknown key, and adding it changes `protocol_hash` |
| cost_bps | f64 | | Cost per unit of turnover (M6). Not a key of `configs/protocol.yaml` yet; adding it changes `protocol_hash` |
| race_context_sessions | i16 | | 252: how many dev sessions before the holdout the model race shows |
| holdout_looks | i16 | | Incremented whenever a hyper-parameter changes after the holdout has been scored |

**`target_def`** is a list in `configs/protocol.yaml`.

| Column | Type | Key | Notes |
|---|---|---|---|
| target_id | str | PK | `vix_level`, `rv21` |
| label | str | | |
| units | str | | Always `vol_points` |
| horizons | i16[] | | `vix_level` uses `[1,5,21]`; `rv21` uses `[21]` |
| definition | str | | The formula from ADR 4.2 |

**`model_def`** is stored in `configs/models.yaml`.

| Column | Type | Key | Notes |
|---|---|---|---|
| model_id | str | PK | `^[a-z][a-z0-9_]*$` |
| label | str | | Display name |
| family | str | | One of `naive`, `arima`, `har`, `garch`, `calibrated`, `ml`, `implied` |
| fit_space | str | | One of `level`, `log_level`, `returns`, `variance` |
| multistep | str | | One of `direct`, `iterated`, `none` |
| targets | str[] | | FKs to `target_def.target_id` |
| milestone | i16 | | 2, 3 or 4 |
| retired | bool | | Never delete a model that has been tried. Retire it instead, so it still counts toward DSR trials and audit. |

The initial registry below is illustrative; M2-M4 finalise it.

| model_id | targets | Note |
|---|---|---|
| `rw` | vix_level, rv21 | `ŷ = y(t)` |
| `hist_mean` | vix_level | The training-window mean, frozen at the cutoff |
| `arima_log` | vix_level | ARIMA on ln V, with lognormal-mean back-transform |
| `har_log` | vix_level | Corsi HAR on ln V, direct per horizon |
| `har_rv` | rv21 | Corsi HAR on RV |
| `arima_garch_log` | vix_level | ARIMA mean with GARCH errors on Δln V. The point forecast is close to `arima_log`; the intervals differ. |
| `garch`, `gjr`, `egarch` | rv21, vix_level | Fitted on SPX returns. Uncalibrated on `vix_level`, to show the VRP gap. |
| `garch_cal`, `gjr_cal`, `egarch_cal` | vix_level | `v̂ = a_h + b_h·G_t(h)` |
| `vix_implied` | rv21 | `v̂ = (V_t/100)²` |
| `lstm_log` | vix_level | The mean over `lstm_members` seeds. Members are stored in `forecast_member`. |

**`strategy_def`** is stored in `configs/strategies.yaml` (M6).

| Column | Type | Key | Notes |
|---|---|---|---|
| strategy_id | str | PK | |
| label | str | | |
| kind | str | | `forecast` or `benchmark` (always-long VXX, always-short via SVXY, cash) |
| model_id | str | FK `model_def.model_id`, null | |
| target_id | str | FK `target_def.target_id`, null | |
| horizon | i16 | null | |
| rule | str | | e.g. `sign(ŷ − V_t) with band` |
| params_json | str | | |
| retired | bool | | Never delete a tried variant: `n_trials` for DSR counts every row. |

**`iv_index_meta`** is stored in `configs/iv_indices.yaml`.

| Column | Type | Key | Notes |
|---|---|---|---|
| symbol | str | PK | VIX, VIX9D, VIX3M, VIX6M, VIX1Y, VVIX (M1), VXN, RVX, VXD, GVZ, OVX, VXEEM, VXAPL, VXAZN, VXGOG, VXGS, VXIBM |
| name | str | | |
| tenor_days | i16 | null | Set for term-structure members only: 9, 30, 93, 184, 366, as Cboe's methodologies state them (VIX: the VIX methodology; VIX9D, VIX3M, VIX6M, VIX1Y: "Selected SPX Target Expected Volatility Term Indices Methodology" v7.0, pp. 5-6). Amended M1, review A: was 365 for VIX1Y, with 9, 93, 184 called unverified |
| underlying | str | FK `instrument_meta.ticker`, null | VIX→^GSPC, VVIX→null (VIX is not an `instrument_meta` ticker), VXN→^NDX, RVX→^RUT, VXD→^DJI, GVZ→GLD, OVX→USO, VXEEM→EEM, VXAPL→AAPL, … |
| group | str | | One of `term_structure`, `equity_index`, `commodity`, `emerging`, `single_stock`, `vol_of_vol` (amended M1: VVIX, an index whose underlying is VIX itself) |
| cboe_url | str | | `https://cdn.cboe.com/api/global/us_indices/daily_prices/{SYM}_History.csv` |
| fred_series | str | null | e.g. `VIXCLS` (the fallback) |

**`instrument_meta`** and **`tradable_segment`** are stored in `configs/instruments.yaml`.

| Entity | Column | Type | Key | Notes |
|---|---|---|---|---|
| instrument_meta | ticker | str | PK | ^GSPC, ^NDX, …, VXX, SVXY |
| | role | str | | `underlying` or `tradable` |
| | name | str | | |
| tradable_segment | ticker | str | PK, FK `instrument_meta.ticker` | |
| | valid_from | date | PK | |
| | valid_to | date | null | |
| | leverage | f64 | | Exposure to VIX futures: VXX is +1; SVXY is −1 before its 2018 cut and −0.5 after (dates verified in M6) |
| | usable | bool | | Backtests run only inside usable segments |
| | note | str | | The VXX series switch, reverse splits, and so on |

## 2. Raw layer (❌)

**`raw_file`.** These are the physical files, one per distinct content. The table is virtual: it is
`raw_fetch` grouped by `sha256`.

| Column | Type | Key | Notes |
|---|---|---|---|
| sha256 | str(64) | PK | Hash of the bytes. It is also the file name. |
| source | str | | One of `cboe`, `fred`, `yfinance`, `synthetic` |
| symbol | str | | A Cboe symbol, FRED series or ticker |
| raw_path | str | | Relative to `ROOT` |
| bytes | i64 | | |

**`raw_fetch`** is stored in `ROOT/raw/provenance.jsonl`: one JSON object per line, append-only.

| Column | Type | Key | Notes |
|---|---|---|---|
| fetch_id | str(32) | PK | uuid4 hex |
| sha256 | str(64) | FK `raw_file.sha256` | |
| source, symbol | str | | |
| url | str | | As requested |
| final_url | str | | After redirects. Cboe answers with a 307 to `cdn-api.cboe.com`. |
| fetched_at | ts | | |
| http_status | i16 | null | Null for yfinance |
| bytes | i64 | | |
| raw_path | str | | |
| changed | bool | | False if this content was already stored |
| tool_version | str | | e.g. `ivf 0.1.0`, `yfinance 1.7.0` |

yfinance returns a DataFrame, not bytes. It is fetched with `auto_adjust=False`, so both `Close` and
`Adj Close` are kept. The DataFrame is then serialised to CSV with a fixed float format and hashed.

## 3. Interim layer (❌, `ROOT/interim/`)

| Table | Column | Type | Key | Notes |
|---|---|---|---|---|
| **iv_index_daily** | symbol | str | PK, FK `iv_index_meta.symbol` | |
| | date | date | PK | |
| | open, high, low | f64 | null | Some Cboe files are close-only |
| | close | f64 | | > 0, in vol points |
| | source | str | | `cboe`, `synthetic` (the seeded generator, under `ROOT/synthetic/`; amended M1), or `fred` (fallback, which only fills dates missing from Cboe; M1 does not fetch it) |
| | raw_sha256 | str | FK `raw_file.sha256` | |
| **price_daily** | ticker | str | PK, FK `instrument_meta.ticker` | |
| | date | date | PK | |
| | open, high, low, close | f64 | | Unadjusted |
| | adj_close | f64 | | Split- and dividend-adjusted. Used for VXX/SVXY returns. |
| | volume | i64 | null | |
| | raw_sha256 | str | FK `raw_file.sha256` | |

**`build_report`** (amended M1) is stored in `ROOT/interim/build_report.json`, beside the two tables. It is local and never committed. `ivf.data.build_interim` writes it, `ivf.data.check` reads it (fetch times and duplicate counts), and the real export reads its raw hashes for `run_id`. It is one object, `{"files": [...]}`, with one element per raw file read.

| Field of an element | Type | Key | Notes |
|---|---|---|---|
| table | str | | `iv_index_daily` or `price_daily` |
| source, symbol | str | | |
| raw_sha256 | str | FK `raw_file.sha256` | |
| fetched_at | str | | RFC 3339 UTC, the fetch this snapshot came from |
| rows | int | | |
| first_date, last_date | str | | `YYYY-MM-DD` |
| duplicates_dropped | int | | Exact repeated rows, kept once |
| duplicate_dates | str[] | | |

## 4. Processed layer (❌, `ROOT/processed/`)

**`run`** is stored in `processed/runs.parquet` and copied to the local `ROOT/reports-provenance/<run_id>.json`. It is never committed: the `run_id` and the raw sha256 list are real-data-derived and outside P (amended M0, review E).

| Column | Type | Key | Notes |
|---|---|---|---|
| run_id | str(12) | PK | `sha256(protocol_hash, source_commit, source_dirty, data_mode, sorted raw sha256s)[:12]` |
| protocol_hash | str(64) | FK `protocol.protocol_hash` | |
| data_mode | str | | `synthetic` or `real` |
| source_commit | str(40) | | |
| source_dirty | bool | | |
| started_at, finished_at | ts | | `finished_at` is null while the run is in progress |
| ivf_version, python_version, platform | str | | |

**`run_input`**: PK `(run_id, raw_sha256)`. FKs to `run.run_id` and `raw_file.sha256`. Its `source`
and `symbol` are copied in for readability. A run has 1..* inputs; a raw file is used by 0..* runs.

The remaining tables live under `processed/run_id=<id>/`. Each has `run_id` as the first PK column,
which is not repeated in the tables below.

| Table (milestone) | PK after run_id | Columns (type) | FKs and notes |
|---|---|---|---|
| **calendar** (M1) | date | `t_idx` i32 (0-based, unique), `period` str (`burn_in`, `dev`, `holdout`), `fold_id` i32 null | 𝒯 is the NYSE session calendar from `exchange_calendars`, from `train_start` to `data_end`, plus the scheduled sessions needed to date the last targets. It is computed from dates, never from data presence (amended M0, ADR 4.1). `data_check` asserts that VIX and ^GSPC have a close on every session. |
| **feature** (M1) | date | `log_vix` f64; `har_w` f64 null (mean ln V over t−4..t; empty for the first 4 sessions); `har_m` f64 null (t−21..t; empty for the first 21 sessions); `r_spx` f64 null (empty on the first date); `rv21_back` f64 null (vol points; empty for the first 21 sessions); `vov21` f64 null (sd of Δln V over 21 sessions; empty for the first 21 sessions); `vix_vix3m` f64 null (empty before VIX3M); `iv_rv_spread` f64 null (variance units; empty for the first 21 sessions). Three more ratios were added in M1, each the shorter tenor over the longer, so above 1 means that segment of the curve is inverted, and each null where either index has no close: `vix9d_vix` f64 null (VIX9D / VIX), `vix3m_vix6m` f64 null (VIX3M / VIX6M), `vix6m_vix1y` f64 null (VIX6M / VIX1Y). Incomplete windows at the start of the training history are left empty rather than excluded as burn-in, because the feature table starts at the first training session (amended M0, review B and D: matches the ERD). | FK `calendar(date)`. Trailing windows only (rule L1). `rv21_back(t)` is the same number as `target_value(rv21, t)`: one series, two roles. |
| **target_value** (M1) | target_id, date | `y` f64 (vol points), `v` f64 (variance) | FK `target_def`, `calendar` |
| **regime_label** (M1) | date, scheme | `regime` str, `driver` f64 (V, or VIX/VIX3M) | `scheme` is `vix_level` (`low`, `mid`, `high`) or `term_structure` (`contango`, `backwardation`). Fixed thresholds (rule L5). |
| **eda_stat** (M1) | symbol, stat, sample | `value` f64, `n` i32 | FK `iv_index_meta(symbol)` (amended M0, review D). `stat` is one of `ar1_phi_log`, `half_life_sessions`, `adf_p`, `kpss_p` and `iv_rv_spread_mean` (one row per sample), and `acf_log_lag1`, `acf_log_lag5`, `acf_log_lag21` and `vov21_mean` (`pre_holdout` only): exactly what `ivf.data.process.eda_stat_rows` writes (amended M1, review B: this list named `mean` and `sd`, which nothing writes, and ended in "…"). `sample` is `pre_holdout`, `burn_in` or `dev` in M1 (amended M1, review A: M1's EDA computes nothing on or after `holdout_start`; it was `full`, `dev` or `holdout`, and that "dev" included burn-in). A `holdout` sample arrives only with the scored holdout run. Only `half_life_sessions` is exported, to `reports/half_life.agg.csv` (D2 P; amended M0, review D). |
| **data_check** (M1) | check_id, symbol | `n_affected` i32, `detail` str | M1 writes `missing_in_cboe` (one row per index), `missing_in_yf` (per ticker), `not_a_session`, `not_settled_at_fetch`, `duplicate_date`, `core_gap` (symbol `VIX+^GSPC`) and `train_start_is_first_vix`. `nonpositive_close` is not a check: `build_interim` refuses such a file. `fred_filled` is not written: M1 does not fetch FRED. Local only, not committed (D2 P; amended M0, review D; amended M1). |
| **fold** (M2) | fold_id | `cutoff_date` date, `last_origin` date, `period` str (`dev`, `holdout`), `n_sessions` i16 | FK `calendar(cutoff_date)`. The first origin equals the cutoff. |
| **fit** (M2) | model_id, target_id, fold_id, member | `cutoff_date` date, `n_train` i32, `train_first_date` date null, `train_last_target_date` date null, `converged` bool, `fallback` str (`none`, `previous_fold`, `rw_log`), `params_json` str, `hyperparams_json` str, `seed` i64 null, `device` str, `fit_seconds` f64, `loglik` f64 null, `aic` f64 null, `epochs_run` i16 null | FK `fold`, `model_def`. `member` is 0 for deterministic models and 1..K for LSTM seeds. **Witness for rule L2:** `train_last_target_date ≤ cutoff_date`, tested. **Amended M2:** `target_id` joins the key, because one model is fitted per implied-vol index; the two training dates are null for `rw`, which fits nothing; `rw_log` is `arima_log`'s first-fold fallback (ARIMA(0,1,0), closed form); `params_json` of `arima_log` holds the chosen order and every candidate's AIC, `hyperparams_json` the grid and the d rule (`d0` by default; the KPSS p-value only under `d_rule: kpss`; amended M2, review A). |
| **forecast** (M2) | model_id, target_id, origin_date, horizon | `target_date` date, `fold_id` i32, `yhat` f64 (vol points), `v_hat` f64 (variance), `var_method` str (`native`, `plugin`), `pi80_lo` f64 null, `pi80_hi` f64 null, `n_members` i16 | FKs: `model_def`; `fold`; `calendar(origin_date)`; `target_value(target_id, target_date)`, which has no matching row while the target is unobserved (the row is produced but not scored). `yhat = 100·√v_hat` on every row. `plugin`: `yhat` is the conditional mean. `native`: `v_hat` is the conditional mean of v, and `yhat` is the RMS vol (ADR 4.3; amended M0). `target_date = 𝒯[t_idx(origin)+horizon]` (tested). The same scored origin set holds for every model per (target, horizon, period). **Amended M2:** also `symbol` str, `cutoff_date` date (the fit's), `yhat_naive` f64 (`exp(mu)`, the lognormal median, for log-space models; equal to `yhat` otherwise), `mu_log` f64 null and `s2_log` f64 null (the forecast mean and error variance of ln y), and `fallback` str (copied from the fit). `yhat` is `exp(mu_log + s2_log/2)`, the lognormal mean, and is what is scored. `target_id` is `vix_level` for VIX and `<symbol>_level` (e.g. `vix3m_level`) for the other implied-vol indices, which are secondary targets defined by `configs/models.yaml`, not the protocol. `pi80_lo`/`pi80_hi` are `exp(mu ∓ 1.2816·√s2)` for log-space models. |
| **forecast_member** (M4) | model_id, target_id, origin_date, horizon, member | `fold_id` i32, `yhat` f64, `v_hat` f64 | FK `forecast` (the PK prefix). FK `fit(model_id, fold_id, member)`, which resolves through the stored fold id column (equal to the parent forecast's). (amended M0, review B: `fold_id` added, because the FK to `fit` could not resolve without it) |
| **metric** (M2/M5) | model_id, target_id, horizon, period, scheme, regime, forecast | `n` i32, `suppressed` bool, `bias`, `rmse`, `mae`, `mape` (fraction), `qlike`: all f64 null | `scheme`/`regime` are `all`/`all` for the unconditional row. **Amended M2:** `forecast` is `mean` (the scored `yhat`) or `median` (`yhat_naive`, log-space models only, to show which back-transform each loss rewards); M2 writes `period = dev` and `scheme = all` only, scoring dev origins whose target is dated before `holdout_start`. Values are null when `n < min_cell_n`. Exported only as ratios to the random walk, to `reports/forecast_ratios.agg.csv` (D2 P; amended M0, review D).  FK `target_def` (amended M0, review D). |
| **dm_test** (M5) | target_id, horizon, period, scheme, regime, loss, model_a, model_b | `n` i32, `mean_d` f64, `dm_stat` f64, `p_value` f64, `nw_lags` i16 | `loss` is `sq_error` or `qlike`. Pairs are unordered and stored once, with `model_a < model_b` (lexicographic), and `d = L_a − L_b`. `nw_lags` is the truncation lag h−1. The long-run variance is rectangular (DM 1995) with the HLN correction, and Bartlett only as a fallback when the rectangular estimate is ≤ 0 (ADR 4.8; amended M0). M5 adds an `lrv_method` column (`rectangular` or `bartlett_fallback`), and the ERD spec gets it then. Two FKs to `model_def`; FK `target_def` (amended M0, review D). Exported only as the HLN statistic and p-value against the benchmark, to `reports/dm_tests.agg.csv` (D2 P; amended M0, review D). |
| **mz_regression** (M5) | model_id, target_id, horizon, period | `space` str (`level`, `variance`), `n` i32, `alpha`, `alpha_se`, `beta`, `beta_se`, `r2`, `wald_stat`, `wald_p`: all f64; `nw_lags` i16 | FK `model_def`, FK `target_def` (amended M0, review D). Exported only as pass/fail booleans, to `reports/mz_tests.agg.csv` (D2 P; amended M0, review D). |
| **backtest_daily** (M6) | strategy_id, date | `origin_date` date (the decision origin, 2 sessions earlier: rule L7), `signal` i8 (+1 long vol, −1 short vol, 0 flat), `w_vxx`, `w_svxy`, `gross_ret`, `cost`, `net_ret` (simple returns), `equity` (starts at 1.0), `drawdown` (≤ 0), `turnover`: all f64 | FK `strategy_def`, `calendar(date)`. Returns come from `price_daily.adj_close`, only inside usable `tradable_segment` rows. |
| **strategy_summary** (M6) | strategy_id, period | `n_days` i32, `ann_return`, `ann_vol`, `sharpe_ann`, `sr_daily`, `skew`, `kurt_pearson` (normal = 3), `psr0` (PSR against SR* = 0), `e_max_sr` (per period), `dsr`, `max_drawdown`, `turnover_ann`, `cost_bps`: all f64; `n_trials` i32 | FK `strategy_def`. PSR and DSR come from `sharpe-observatory/scripts/metrics.py`. `n_trials` is the number of non-benchmark `strategy_def` rows ever run on that period. `ann_return` is arithmetic (mean simple daily return × 252). Exported only as `sharpe_ann`, `psr0`, `dsr`, `n_trials` and `turnover_ann`, to `reports/strategy_summary.agg.csv` (D2 P; amended M0, review D). |

**`eda_report`** (amended M1) is stored in `ROOT/eda/<run_id>/eda_report.json`, not under `processed/`. `ivf.data.process` writes it beside the processed tables. It holds every EDA number the `eda_stat` rows are taken from, in one object: `run_id`, `data_mode`, `data_end`, then the sections `sample` (the first and last session used, `holdout_start`, and the size of each sample; amended M1, review A), `clustering`, `persistence`, `half_life`, `unit_root` (with the tested gap-free segment), `iv_rv_spread`, `term_structure` and `vol_of_vol` (with the E6 bootstrap p-value, its number of paths and its block length `e6_block`; amended M1, review B; and the fitted AR(1) `e6_ar1_phi`, the verdict `e6_verdict`, "yes" or "not shown", and its `e6_reason`; amended M1, final check). Local only; on real data it is never committed (ADR D2).

### Cardinality

| From | To | From:To | Via |
|---|---|---|---|
| raw_file | raw_fetch | 1 : 1..* | `sha256` |
| raw_file | iv_index_daily / price_daily | 1 : 0..* | `raw_sha256` |
| raw_file | build_report | 1 : 0..* | `raw_sha256` (amended M1) |
| iv_index_meta | iv_index_daily | 1 : 0..* | `symbol` |
| iv_index_meta | eda_stat | 1 : 0..* | `symbol` (amended M0, review D: stated here to match the ERD) |
| instrument_meta | price_daily / tradable_segment | 1 : 0..* | `ticker` |
| instrument_meta | iv_index_meta | 0..1 : 0..* | `underlying` |
| protocol | run | 1 : 0..* | `protocol_hash` |
| run | run_input | 1 : 1..* | `run_id` |
| raw_file | run_input | 1 : 0..* | `raw_sha256` |
| run | calendar / fold | 1 : 1..* | `run_id` |
| run | eda_stat / data_check | 1 : 0..* | `run_id` (amended M0, review D: stated here to match the ERD) |
| run | eda_report | 1 : 0..1 | `run_id` (amended M1) |
| build_report | data_check | 1 : 0..* | Lineage only: the `duplicate_date` counts and the fetch times behind `not_settled_at_fetch` (amended M1) |
| calendar | feature | 1 : 0..1 | `date` |
| calendar | target_value / regime_label | 1 : 0..* | `date` |
| target_def | target_value | 1 : 0..* | `target_id` |
| target_def | metric / dm_test / mz_regression | 1 : 0..* | `target_id` (amended M0, review D: stated here to match the ERD) |
| calendar | fold | 1 : 0..1 | `cutoff_date` |
| fold | fit | 1 : 1..* | `fold_id` |
| model_def | fit / forecast / metric / mz_regression | 1 : 0..* | `model_id` |
| model_def | dm_test | 1 : 0..* (twice) | `model_a`, `model_b` |
| fold | forecast | 1 : 1..* | `fold_id` |
| calendar | forecast | 1 : 0..* | `origin_date` |
| target_value | forecast | 0..1 : 0..* | `(target_id, target_date)`. A forecast has no `target_value` while its target is unobserved: the row is produced but not scored, which the truncation test relies on. (amended M0, review B: was 1 : 0..*) |
| forecast | forecast_member | 1 : 0..* | The forecast PK |
| fit | forecast_member | 1 : 0..* | `(model_id, fold_id, member)` |
| forecast | metric | 0..* : 1..* | Aggregation. A forecast falls in the `all` row plus one row per scheme. |
| model_def | strategy_def | 0..1 : 0..* | `model_id` |
| strategy_def | backtest_daily / strategy_summary | 1 : 1..* | `strategy_id` |
| price_daily | backtest_daily | 1..* : 0..* | Returns of VXX and SVXY on `date` |

### Committed reports (`reports/`, ✅)

ADR D2's **publication set P** is the closed list of real-data-derived numbers that may ever be
committed (amended M0, review D; it replaces the M0 rules and review B's R1-R4). Every committed
report is a `reports/*.agg.csv` file holding exactly the columns below.
- Real data only, from a run with `source_dirty = false`.
- One release per frozen `data_end`, with no corrections. A wrong value is withdrawn, never replaced.
- Every value is unitless. Nothing in vol points or prices, no moments, no coefficients, no returns,
  no regime cells.
- Rows are whole periods only: `period ∈ {dev, holdout}`.
- Model rows are only the **in-P rows**, those that do not change when the VIX path is rescaled
  with public SPX held fixed (amended M0, review E):
  - `vix_level`: `hist_mean`, `arima_log`, `har_log`, `arima_garch_log`, `garch_cal`, `gjr_cal`,
    `egarch_cal`;
  - `rv21`: `har_rv`, `garch`, `gjr`, `egarch`;
  - `rw` appears in `mz_tests.agg.csv` only;
  - `lstm_log` only if M4 shows it scale-free (ADR O10).
  The uncalibrated `garch`, `gjr` and `egarch` on `vix_level`, and `vix_implied` on `rv21`, are
  scale-tied and stay local (ADR O8).
- The **benchmark** for every ratio and DM test is `rw`: ŷ = V_t for `vix_level`, and
  ŷ = `rv21_back(t)` for `rv21` (amended M0, review E).
- The local tables (`metric`, `dm_test`, `mz_regression`, `strategy_summary`, `eda_stat`,
  `data_check`) keep every column. Only this export is restricted.

| File | Key columns | Value columns (rounding) | Computed from |
|---|---|---|---|
| `forecast_ratios.agg.csv` | `period`, `target_id`, `horizon`, `model_id` | `n` (exact); `rel_rmse`, `rel_qlike`, `rel_mae`, `rel_mape` (ratio to the random walk's, 3 s.f.) | `metric` (`all`/`all` rows). `rel_mae` and `rel_mape` are flagged as not consistent for the mean (ADR 4.3). |
| `dm_tests.agg.csv` | `period`, `target_id`, `horizon`, `loss`, `model_id` (against the benchmark) | `n` (exact); `dm_stat_hln` (2 dp); `p_value` (2 dp) | `dm_test`, benchmark pairs only |
| `mz_tests.agg.csv` | `period`, `target_id`, `horizon`, `model_id` | `n` (exact); `beta_eq_1_pass`, `joint_pass` (booleans, 5%) | `mz_regression` (pass means not rejected) |
| `strategy_summary.agg.csv` | `period`, `strategy_id` | `n_days` (exact); `sharpe_ann`, `psr0`, `dsr` (2 dp); `n_trials` (exact); `turnover_ann` (2 s.f.) | `strategy_summary` |
| `half_life.agg.csv` | `period`, `symbol` | `n` (exact); `half_life_sessions` (2 s.f.) | `eda_stat` (`half_life_sessions` of ln(index)) |

Not committed (local only): `data_check` counts. Report data problems in prose, qualitatively.

**Provenance is not committed (amended M0, review E).** The `run` row, its `run_input` sha256 list
and the protocol go to the local `ROOT/reports-provenance/<run_id>.json`. Raw-file hashes and the
`run_id` built from them are real-data-derived and outside P. A committed report carries only
`protocol_hash` and `source_commit`, which identify the configuration and the code. Anyone holding
the same snapshots reproduces the tables from the local record.

## 5. Python → app JSON contract

**Envelope** (`envelope.schema.json`). Every file is an object with exactly these keys:

| Field | Type | Notes |
|---|---|---|
| schema_version | string `^1\.(0\|[1-9][0-9]*)\.(0\|[1-9][0-9]*)$(?!\n)` | Semver. ADR D1 says what bumps it. Every schema edit bumps at least PATCH (amended M0). |
| generated_at | string `YYYY-MM-DDTHH:MM:SS(.f)Z` | UTC |
| source_commit | string, 40 lowercase hex | |
| source_dirty | boolean | |
| data_mode | `"synthetic"` \| `"real"` | The app's badge reads this, never an env var |
| dataset | string id | Equal to the file's dataset name |
| payload | object | File-specific (below) |

**Why columnar.** The large payloads (ribbon, race, equity) are written as one array per column,
not one object per row. That way:
- keys are not repeated, which keeps the files compact;
- a column maps straight onto a `Float32Array` or a three.js buffer attribute;
- a shared `calendar` or `dates` axis plus integer indices avoids repeating date strings.

The small aggregate payload (the error surface) uses nested arrays (a dense cube) indexed by named
axes.

**Where the files live.**
- Synthetic: `app/public/data/synthetic/{dataset}.json` (🧪).
- Real: `ROOT/app-data/real/{dataset}.json` (❌). The Vite dev server serves it at `/data/real/`.
- Every file must be ≤ 2 MiB (2,097,152 bytes). `manifest.schema.json` enforces this on `files[].bytes`.
- Contract tests use one set of schemas on both sides:
  - Python: `Draft202012Validator` plus a `referencing.Registry` holding all six schemas.
  - TypeScript: `new Ajv2020({strict: true, allowUnionTypes: true})`. This side is not wired yet; it
    arrives in M1 (amended M0).
- Every `pattern` ends in `$(?!\n)`, because Python's `$` also matches before a final newline and
  ECMA-262's does not. The two validators then agree (ADR D5; amended M0).
- **Tolerant reader** (ADR D1; amended M0):
  - The producer validates strictly against its own version's schemas.
  - The app does not validate against JSON Schema at runtime. It checks MAJOR and the fields it uses,
    and ignores unknown fields.

**Rounding at export.**

| Field | Decimals |
|---|---|
| Index levels | 2 |
| Forecasts, actuals, `pi80` | 3 |
| Ratios | 4 |
| RMSE, MAE, bias, MZ coefficients, Sharpe-type statistics, p-values | 4 |
| MAPE | 5 |
| QLIKE | 6 |
| Equity, drawdown | 6 |

### 5.1 `manifest.json`: read first by the app

| Payload field | Type | Notes |
|---|---|---|
| run_id | string, 12 hex | |
| notice | string | Synthetic: "Synthetic data (seed N), not market data". Real: Cboe/Yahoo attribution and "personal, non-commercial use; not redistributed". |
| protocol | object | `protocol_hash` (64 hex), `train_start`, `dev_start`, `holdout_start`, `data_end` (dates), `horizons` (int[]), `fold_len`, `min_cell_n`, `holdout_looks` (int), `vix_regime_edges` (number[2]) |
| files | array of objects | `dataset` (id), `path` (`{dataset}.json`), `sha256` (64 hex), `bytes` (int), `license_class` (`series` or `aggregate`). A file appears only once its milestone has produced it, and the app greys out views whose file is missing. |
| models | array of objects | `model_id`, `label`, `family`, `targets` (id[]): the display registry |
| targets | array of objects | `target_id`, `label`, `units` (`vol_points`), `horizons` (int[]) |

### 5.2 `term_structure.json`: the 3D VIX term-structure ribbon (M1; `license_class: series`)

The source table is `iv_index_daily` (the term-structure group). Amended M1: the real export reads the interim table
and the frozen `configs/protocol.yaml`, not `regime_label`. The flag is computed in Python by
`ivf.features.build.is_backwardation`, the same rule `regime_label` is built with, so the two cannot disagree.

| Payload field | Type | Notes |
|---|---|---|
| dates | string[n] (`YYYY-MM-DD`) | Ascending NYSE sessions, from the first session on which all five tenors have a close to `data_end` (amended M1). M0 estimated n from the calendar alone; the start now depends on the data, so the real n is measured locally, not published (D2). |
| tenors | object[k] | `symbol`, `tenor_days` (int), `label` (e.g. `9D`), sorted ascending by `tenor_days`: 9D, 30D, 3M, 6M, 1Y |
| levels | (number\|null)[k][n] | `levels[j][i]` is the close of tenor j on `dates[i]`, in vol points. Null on a session where that tenor has no close (a gap inside the span stays null). |
| vix_vix3m | (number\|null)[n] | The ratio, null before VIX3M |
| backwardation | (boolean\|null)[n] | `vix_vix3m > 1`, strictly: equal is contango. This is the colouring flag, computed in Python from the unrounded ratio. |

Invariants: every inner array has length n; `tenors` is sorted; `dates` is strictly increasing.

### 5.3 `model_race.json`: forecast vs actual, by horizon (M2; `license_class: series`)

The source tables are `forecast`, `target_value` and `mz_regression`.

| Payload field | Type | Notes |
|---|---|---|
| calendar | string[m] | The 𝒯 slice covering every origin and target in the file |
| holdout_start_idx | int | Index in `calendar` of `holdout_start` |
| panels | object[] | One per (target_id, horizon): 3 for `vix_level` and 1 for `rv21` |
| panels[].target_id | string id | |
| panels[].horizon | int | |
| panels[].origin_idx0 | int | Index in `calendar` of the first origin. Row i has origin `calendar[origin_idx0+i]` and target `calendar[origin_idx0+i+horizon]`. |
| panels[].n | int | The number of origins. About 835 per panel (252 dev context + 583 holdout sessions, measured to 2026-10-26). |
| panels[].actual | (number\|null)[n] | `y` at the target date, in vol points |
| panels[].series | object[] | `model_id`, `yhat` (number\|null)[n]; optional `pi80_lo` and `pi80_hi` (number\|null)[n] |
| panels[].mz | object[] | `model_id`, `period` (`dev`, `holdout`), `n`, `alpha`, `beta`, `wald_p` (number\|null) |

Invariant: `origin_idx0 + n − 1 + horizon ≤ m − 1`.

### 5.4 `error_surface.json`: the 3D error surface, model × horizon × regime (M5; `license_class: aggregate`)

The source tables are `metric` and `dm_test`. Its real version holds regime cells, so it stays
local, and no real-data screenshot of it may be committed (ADR D2; amended M0, review B).

| Payload field | Type | Notes |
|---|---|---|
| periods | (`dev`\|`holdout`)[] | Axis 1 |
| regimes | object[] | Axis 4: `regime_id`, `scheme` (`all`, `vix_level`, `term_structure`), `label` (e.g. `VIX < 15`). Regimes from different schemes overlap; they are slices, not a partition. |
| surfaces | object[] | One per target_id |
| surfaces[].target_id | string id | |
| surfaces[].benchmark_model_id | string id | `rw` |
| surfaces[].models | string id[] | Axis 2 |
| surfaces[].horizons | int[] | Axis 3 |
| surfaces[].values | object | `bias`, `rmse`, `mae`, `mape`, `qlike`. Each is a cube `[period][model][horizon][regime]` of number\|null, where null means suppressed (n < `min_cell_n`, a local reliability rule; amended M0, review B) or not applicable. |
| surfaces[].n | int cube | The same shape. The count behind each cell. |
| surfaces[].dm_p_vs_benchmark | object | `sq_error` and `qlike` cubes of number\|null. Null for the benchmark itself and for suppressed cells. |

### 5.5 `equity_curves.json`: equity with regime ribbons (M6; `license_class: series`)

The source tables are `backtest_daily`, `strategy_summary`, `regime_label` and `tradable_segment`.

| Payload field | Type | Notes |
|---|---|---|
| dates | string[n] | Sessions inside usable tradable segments |
| holdout_start_idx | int\|null | |
| cost_bps | number | |
| strategies | object[] | `strategy_id`, `label`, `kind` (`forecast`, `benchmark`), `model_id` (id\|null), `horizon` (int\|null), `equity` number[n] (> 0, starts at 1), `drawdown` number[n] (≤ 0) |
| regime_segments | object[] | Run-length encoded: `scheme`, `regime_id`, `start_idx`, `end_idx` (inclusive) |
| summary | object[] | `strategy_id`, `period`, `n_days`, `ann_return`, `ann_vol`, `sharpe_ann`, `psr0`, `dsr`, `n_trials`, `max_drawdown`, `turnover_ann` (number\|null) |
| caveats | string[] | e.g. the SVXY leverage cut, the VXX series switch, the cost model |

### Size estimates (to be measured in M1-M6)

The term_structure and model_race session counts are measured on the NYSE calendar. The equity
count and all byte sizes are estimates.

| File | Values (estimate) | Estimated size |
|---|---|---|
| term_structure | ~3,977 × 7 | ~0.2 MB |
| model_race | ~835 × 4 panels × ~10 series | ~0.2-0.3 MB |
| error_surface | < 10⁴ | < 0.1 MB |
| equity_curves | ~2,200 × ~6 strategies × 2 | ~0.2 MB |

All of these are well under the 2 MB budget.
