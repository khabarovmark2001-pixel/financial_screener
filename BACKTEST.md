# Historical fair value — backtest input

`valuation_history.py` → `data/valuation_history.parquet`

**61,508 rows** — one per (ticker, filing date) — covering **2,617 tickers** from
**2019-09-09 to 2026-09-04**, 17.6 MB, built in ~44 seconds.

```bash
python3 valuation_history.py run
python3 valuation_history.py run --min-history 5   # stricter cohort
python3 valuation_history.py status
```

Each row is the fair value that **could actually have been computed on that
day**, from information published by then. That is the only property that makes
it a backtest signal rather than a hindsight-fitted curve.

---

## 1. Same model, rewound

The methodology is not re-implemented. `dcf_forecast.project` and the engine's
`classify`, `model_ddm`, `model_rim`, `model_passthrough` and `blend` are all
vectorised over rows and indifferent to whether those rows are 2,258 companies
today or 61,508 company-dates across seven years. This module builds the inputs
and calls those same functions, so if the live engine changes this changes with
it — there is no second copy of the model free to drift.

Only the relative model needed a variant: peer medians must come from the
cross-section **as it stood that quarter**, not from today's.

Agreement with the live engine on each ticker's latest row: **median ratio
1.004**, 88.8% within 25%, **94.9% identical profile**. The residual difference
is deliberate — history uses expanding sigma and contemporaneous sector anchors
where the live engine uses the full sample and today's.

---

## 2. What was made point-in-time

| Input | Live engine | Here |
|---|---|---|
| Drivers | EWM over trailing 5y from today | EWM evaluated **at each row** — uses rows ≤ t only |
| Sector margin anchors | today's cross-section | **that quarter's** cross-section |
| Peer multiples | today's cross-section | **that quarter's** cross-section |
| Cash-flow sigma | full sample | **expanding** — volatility observable on that date |
| Cost of capital | latest | already per-filing in `metrics.parquet` |
| Market price | last close | **close on the filing date** |
| Years of history | full-sample span | **elapsed since first filing** |

### The lookahead bug this surfaced

`metrics.parquet` stamps the **full-sample** history span on every row — Apple
carries 9.76 years on its 2016 filing as much as on its 2026 one. That is
correct for the live engine, which only ever reads the latest row. Used here it
would have let a 2019 valuation claim a decade of history, pass the maturity
test, earn a reliability bonus none of it had, and — worst — pass the gate
deciding which rows enter the file at all.

Recomputed as elapsed time since each ticker's first filing. The effect was not
cosmetic: the 3-year gate went from dropping 10,873 rows to **31,928**, and the
usable history correctly starts in **2019** rather than appearing to start in
2017.

---

## 3. Forward returns

`forward_return_1m / 3m / 6m / 12m` — total return from the filing date, on
split- and dividend-adjusted closes, in trading sessions.

These are the **only** future-looking columns in the file and carry a
`forward_` prefix precisely so nothing else can consume one by accident: 44
input columns, 4 forward columns, prefix-isolated.

---

## 4. Does the signal work

Cross-sectional decile test, ranking `fair_value_upside` **within each quarter**
so it measures stock selection rather than market timing. 51,042 observations
across 25 quarters.

| Decile | n | median upside | median fwd 12m | median fwd 3m |
|---|---|---|---|---|
| 0 (most overvalued) | 5,115 | −62% | **+6.1%** | +2.0% |
| 1 | 5,103 | −40% | +7.4% | +1.7% |
| 2 | 5,101 | −23% | +9.4% | +1.9% |
| 3 | 5,099 | −8% | +9.9% | +1.9% |
| 4 | 5,110 | +8% | +11.5% | +2.5% |
| 5 | 5,097 | +26% | +11.5% | +3.2% |
| 6 | 5,098 | +49% | +12.6% | +3.2% |
| 7 | 5,102 | +83% | +14.3% | +3.4% |
| 8 | 5,102 | +147% | +10.8% | +2.6% |
| 9 (most undervalued) | 5,115 | +301% | **+14.0%** | +3.8% |

**Long-short spread (D9 − D0): +7.92% over 12 months**, median.

Rank correlation with forward returns, by horizon:

| Horizon | Spearman |
|---|---|
| 1 month | +0.038 |
| 3 months | +0.043 |
| 6 months | +0.053 |
| 12 months | **+0.093** |

The correlation **rising with horizon** is the signature you want from a
valuation signal: mispricing corrects slowly, and a factor that only worked at
one month would more likely be momentum or a data artefact. A Spearman of 0.09
at twelve months is respectable for a single cross-sectional factor. Had it come
back at 0.5, the right response would have been to hunt for the leak, not to
celebrate.

Filtering to the high-confidence half barely moves it (+8.06% vs +7.92%), so
confidence is not currently adding selection power on top of the ranking.

---

## 5. What this is not

**Not a tradeable backtest.** No transaction costs, no bid-ask, no borrow costs
on the short leg, no position sizing, no rebalancing calendar. Observations
overlap heavily — a 12-month forward return from consecutive quarterly filings
shares most of its path — so the effective sample is far smaller than 51,042 and
naive t-statistics on it would be badly overstated.

**One regime.** 2019–2026 is a single mostly-rising market with one sharp
drawdown. A value signal's behaviour in a prolonged bear market is not tested
here and cannot be.

**Deciles 8 and 9 are extreme.** A median +301% "upside" in decile 9 mostly
identifies companies the model does not understand rather than bargains, which
is why decile 8 breaks the monotonic pattern. Deciles 4–7 are the economically
meaningful range.

**Survivorship.** The universe is companies listed *today* that pass the current
screen. Names delisted before 2026 are absent, which biases returns upward at
every decile — note the whole table is positive. The **spread** between deciles
is far more trustworthy than any single decile's level.
