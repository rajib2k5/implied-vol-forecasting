# Learning log

The index of this project as a course: one milestone at a time, with the concept notes behind it and the questions you answer along the way.

**How a milestone works.**

1. The session builds one milestone, then writes a walkthrough in `docs/milestones/`.
2. You answer the **predict-first** questions *before* reading the results they ask about.
3. You read the walkthrough, then answer the **explain-it-back** questions in your own words.
4. The session grades your answers and you record them under "Your answers & grades" below.
5. The session stops with a one-paragraph recap until you say "continue" (or steer).

Questions are listed here by title only. The full text is in each milestone's walkthrough, and the explain-it-back questions have no answers there by design.

## Milestones

| # | Milestone | Status | Walkthrough |
|---|---|---|---|
| 0 | Scaffold and design | **Current** | [`docs/milestones/00-scaffold-and-design.md`](docs/milestones/00-scaffold-and-design.md) |
| 1 | Data and EDA: volatility clustering, persistence, AR(1) half-life, mean reversion, IV-RV spread, vol-of-vol | Planned | Written with the milestone |
| 2 | Baselines: random walk, historical mean, ARIMA on log-IV, HAR | Planned | Written with the milestone |
| 3 | GARCH done right: GARCH, GJR and EGARCH on underlying returns; ARIMA-GARCH on log-IV changes | Planned | Written with the milestone |
| 4 | LSTM (PyTorch) on the same walk-forward splits | Planned | Written with the milestone |
| 5 | Evaluation: RMSE, MAE, MAPE, QLIKE at horizons 1, 5 and 21; Diebold-Mariano, Mincer-Zarnowitz, regimes | Planned | Written with the milestone |
| 6 | Forecast-driven VXX/SVXY backtest with costs; PSR and DSR | Planned | Written with the milestone |

## Concepts

Theory notes in [`docs/concepts/`](docs/concepts/): the maths, the assumptions, and where each idea breaks.

| # | Note | In one line |
|---|---|---|
| 00 | [Background](docs/concepts/00-background.md) | The background and sources behind the project and the hypotheses in the [PR/FAQ](docs/prfaq.md). |
| 01 | [Forecast targets and annualisation](docs/concepts/01-forecast-targets-and-annualisation.md) | What is forecast, in what units, and why a GARCH number and a VIX number are not directly comparable. |
| 02 | [Walk-forward evaluation and look-ahead](docs/concepts/02-walk-forward-evaluation-and-look-ahead.md) | How to score a forecaster so the score means what it would have meant in real time, and the ways a backtest quietly sees the future. |
| 03 | [Reproducibility and data contracts](docs/concepts/03-reproducibility-and-data-contracts.md) | What it takes for a number to be re-made by someone else, and for the app to read it without guessing. |

## Questions and answers by milestone

### Milestone 0: scaffold and design

Walkthrough: [`docs/milestones/00-scaffold-and-design.md`](docs/milestones/00-scaffold-and-design.md).

**Predict first** ([full text](docs/milestones/00-scaffold-and-design.md#1-predict-before-you-look); answers are in [section 6](docs/milestones/00-scaffold-and-design.md#6-answers-to-the-predict-first-questions)):

- P1. A CUDA build reports `is_available() == True`: does it necessarily run an LSTM?
- P2. GPU against CPU epoch time for a 64-unit LSTM on about 5,000 daily samples.
- P3. Torch's default 20 threads against an explicit 8 threads.
- P4. What PyPI's `torch` gives you on Linux aarch64: CPU or CUDA, and how big.
- P5. How many calendar days do 21 trading sessions span?
- P6. Does a `data/` line in `.gitignore` ignore `app/src/data/`?
- P7. What do the random walk's one-day errors sum to, and why does that matter for a public repo?
- P8. Which constant forecast minimises expected MAPE for a lognormal VIX-like variable: the mean, the median, or something lower?
- P9. Can you recover individual VIX closes from a published average one-day random-walk error over n days, and does a larger n protect them?

**What the adversarial reviews found** ([section 7](docs/milestones/00-scaffold-and-design.md#7-what-the-adversarial-reviews-found-and-what-changed), ledgers in [`docs/validation/`](docs/validation/)): five reviews (A code and numerics, B maths and documents, C the literature brief, D the fixes, E the final attack), as a table of claim, verdict, evidence, fix and lesson. The best interview material of the milestone.

**Explain it back** ([full text](docs/milestones/00-scaffold-and-design.md#8-explain-it-back)):

- E1. What happens, line by line, when `export_app_data.py --mode real --out-dir app/public/data` is run, including the device-and-inode guard over every repo folder and the write path?
- E2. Why can't the app compute the metrics itself?
- E3. What exactly would leak if the scaler were fitted on the full sample?
- E4. Why is GARCH on the VIX level a category error?
- E5. Why does `--check` ignore `run_id` but not `fold_len`, and why does `serialise` allow only string keys, sorted and compact?
- E6. Why CPU torch on a machine with a GPU?
- E7. Which losses rank mean forecasts correctly, and why can a lower MAE or MAPE come from a worse one?
- E8. Why did a 20-date minimum per published cell fail, and why does D2 now publish only a closed set of unitless numbers that must stay unchanged under rescaling even given public S&P 500 data, with the "no single day" promise deleted?

**Decisions waiting on you:** [section 9](docs/milestones/00-scaffold-and-design.md#9-decisions-waiting-on-you) (holdout start, CPU or CUDA torch, the PR/FAQ's open questions, the real-data publication set and its open items O7 to O10).

#### Your answers & grades: milestone 0

| Question | Your answer | Grade | Feedback |
|---|---|---|---|
| P1 | | | |
| P2 | | | |
| P3 | | | |
| P4 | | | |
| P5 | | | |
| P6 | | | |
| P7 | | | |
| P8 | | | |
| P9 | | | |
| E1 | | | |
| E2 | | | |
| E3 | | | |
| E4 | | | |
| E5 | | | |
| E6 | | | |
| E7 | | | |
| E8 | | | |

### Milestone 1: data and EDA

Predict-first and explain-it-back lists arrive with the milestone walkthrough.

#### Your answers & grades: milestone 1

| Question | Your answer | Grade | Feedback |
|---|---|---|---|
| | | | |

### Milestone 2: baselines

Predict-first and explain-it-back lists arrive with the milestone walkthrough.

#### Your answers & grades: milestone 2

| Question | Your answer | Grade | Feedback |
|---|---|---|---|
| | | | |

### Milestone 3: GARCH done right

Predict-first and explain-it-back lists arrive with the milestone walkthrough.

#### Your answers & grades: milestone 3

| Question | Your answer | Grade | Feedback |
|---|---|---|---|
| | | | |

### Milestone 4: LSTM

Predict-first and explain-it-back lists arrive with the milestone walkthrough.

#### Your answers & grades: milestone 4

| Question | Your answer | Grade | Feedback |
|---|---|---|---|
| | | | |

### Milestone 5: evaluation

Predict-first and explain-it-back lists arrive with the milestone walkthrough.

#### Your answers & grades: milestone 5

| Question | Your answer | Grade | Feedback |
|---|---|---|---|
| | | | |

### Milestone 6: trading backtest

Predict-first and explain-it-back lists arrive with the milestone walkthrough.

#### Your answers & grades: milestone 6

| Question | Your answer | Grade | Feedback |
|---|---|---|---|
| | | | |
