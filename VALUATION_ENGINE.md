# Multi-model valuation engine

`valuation_engine.py` — one weighted Fair Value per share for every ticker,
from five models weighted by how much each can be trusted for that company.

```bash
python3 valuation_engine.py run          # ~1 s for 2,258 tickers
python3 valuation_engine.py explain JPM  # full working for one company
python3 valuation_engine.py status
```

Output: `data/valuation_engine.parquet`, 2,258 rows × 60 columns, 0.7 MB. The
key columns are also merged into `analytics.parquet` so the UI can reach them
without a second load.

---

## 1. Why not one model

A bank has no capex and no working capital, so FCFF describes nothing about it.
A loss-making biotech has no earnings to discount. A utility's value is mostly
the dividend. Forcing one DCF across 2,258 companies produces a number for all
of them and a *valuation* for some of them.

Measured on this universe — a single model against the blend:

| | median upside | \|upside\| > 200% | std |
|---|---|---|---|
| FCFF alone | +31.9% | 33.4% | 45.8 |
| FCFE alone | +29.5% | 28.2% | 49.9 |
| EPS alone | −10.4% | 21.5% | 53.0 |
| **Engine fair value** | **+2.4%** | **8.1%** | **1.15** |

A valuation of the whole market should straddle zero and should not routinely
claim a stock is worth three times its price. The single models did both.

---

## 2. Architecture

```
dcf_summary.parquet ─┐
metrics.parquet ─────┼─► load_inputs ─► classify ─► 5 models ─► blend ─► parquet
fundamentals (ART) ──┤                   (profile)   (value +
analytics.parquet ───┘                               reliability)
```

Deliberately a **join over existing artefacts, not a recomputation**. The FCFF
and FCFE values, the cost of capital and the multiples already exist and are
already point-in-time. Recomputing them here would create a second source of
truth free to drift from the first. Only the two genuinely new models — DDM and
residual income — and the peer-multiple layer are calculated in this file.

Runtime is ~1 second because every model is a matrix operation over all tickers
at once, not a loop.

---

## 3. Classification

Seven profiles, tested from most structural to least. Sector wins first: a bank
is a bank whatever its growth looked like this year, and applying an
operating-company model to it is a category error rather than a calibration
problem.

| Profile | Count | Test |
|---|---|---|
| `financial` | 405 | sector = Financial Services |
| `capital_intensive` | 405 | capex/revenue > 10% or net debt/EBITDA > 3 or D/E > 1.5 |
| `standard` | 404 | profitable, nothing else applies |
| `mature_dividend` | 378 | ≥8y history, material dividend, growth < 8% |
| `unprofitable` | 369 | net income ≤ 0 or EBIT ≤ 0 |
| `high_growth` | 150 | profitable, revenue growth > 20% |
| `reit` | 147 | sector = Real Estate |

**A material dividend, not just any dividend.** Apple pays consistently on a
~15% payout and a 0.5% yield; discounting that stream values it at $16 against
a $320 price. Paying *something* is not a dividend policy, so the test requires
a payout ratio ≥ 25% **or** a yield ≥ 1.5%. Apple classifies `standard`, where
FCFE and FCFF lead.

Features computed alongside: profitability, revenue growth, FCF stability
(from the DCF's own per-company sigma), leverage, dividend consistency and
coverage, ROE level and volatility, capital intensity, and years of filing
history.

---

## 4. The models

### `fcfe` / `fcff` — reused from `dcf_forecast`

Not recomputed. Reliability is built from what the DCF already measured:
per-company cash-flow sigma, how much of enterprise value sits in the terminal
block, length of filing history, and **corroboration from the EPS model** — an
independent read on the same equity claim. Where FCFE and EPS agree closely,
FCFE is more trustworthy; that is a better use of the EPS model than averaging
it in as a sixth opinion.

### `ddm` — two-stage dividend discount

Growth starts at the **lower** of observed dividend growth and the sustainable
rate (ROE × retention) — a payer growing dividends faster than it earns them is
funding the gap from somewhere that runs out — then fades to terminal over ten
years. Discounted at cost of equity with the same `r − g ≥ 1pp` floor the DCF
uses, because Gordon explodes as the spread closes.

Reliability: payment consistency (45%), payout coverability (30%), and how much
of the answer is the terminal block (25%).

### `rim` — residual income / excess return

```
V = B₀ + Σ (ROE_t − r) × B_{t-1} / (1+r)^t + terminal
```

The right model for banks and insurers: book equity is marked near economic
value, earnings are the return on it, and there is no capex or working capital
for FCFF to work with. Book value compounds by retained earnings (clean
surplus).

Two guards against the classic failure mode. ROE **fades toward the cost of
equity**, keeping only 40% of the starting spread by year ten — excess returns
attract competition. And the terminal block is a **capped multiple of final-year
residual income**, not a growing perpetuity: assuming a firm out-earns its cost
of capital forever is how this model manufactures fantasies.

Reliability: ROE stability (45%), book value as a share of the answer (30%),
positive equity (25%).

### `relative` — peer-median multiples

P/E, EV/EBITDA, P/S and P/B, each applied to the company's own driver, with
peers taken at **industry** level and falling back to sector below five
comparables. Only positive multiples enter a median — a negative P/E is a
loss-maker, and including it in a median that will be multiplied by someone's
earnings turns a profitable business into a negative fair value.

The four implied prices are blended by their own reliability, and that blend is
exported as **`fair_value_multiples`** — the same weighting logic applied to
multiples alone, directly comparable to the DCF models rather than sitting on a
separate scale.

Reliability: agreement between the multiples (55%) and how many were usable
(45%). If P/E and EV/EBITDA disagree threefold, neither is saying much.

---

## 5. Weighting

```
weight = base_weight(profile) × reliability × applicability_gate
FairValue = Σ (weight / Σweight) × model_value
```

**Base weights by profile** (before reliability):

| Profile | FCFE | FCFF | DDM | RIM | Relative |
|---|---|---|---|---|---|
| financial | 10% | **0%** | 20% | **50%** | 20% |
| reit | 15% | 10% | 30% | 20% | 25% |
| mature_dividend | 30% | 20% | **30%** | 5% | 15% |
| capital_intensive | 25% | **45%** | 5% | 5% | 20% |
| high_growth | 30% | 20% | 0% | 5% | **45%** |
| unprofitable | 10% | 10% | 0% | 5% | **75%** |
| standard | **40%** | 30% | 5% | 5% | 20% |

**Applicability gates** are hard structural statements, applied before any
weighting, and a high reliability score cannot recover them:

- FCFF is **zero** for financials — their debt is inventory
- FCFE is halved for financials
- DDM is zero without a dividend
- RIM is zero without positive book equity
- FCFE and FCFF are cut to 35% for loss-makers

**Sanity band.** A model value below 0.1× or above 10× the market price is
treated as a modelling failure, not an opportunity, and dropped. This is what
removes Apple's $16 DDM and NVIDIA's $3.70 DDM without needing a special case.

**Reliability floor** of 0.25: below it the model is not usable for that company
and its weight redistributes across the survivors.

### Realised weights

| Profile | FCFE | FCFF | DDM | RIM | Relative |
|---|---|---|---|---|---|
| financial | 4% | **0%** | 18% | **53%** | 26% |
| reit | 8% | 5% | 29% | 24% | 34% |
| mature_dividend | 23% | 16% | **35%** | 6% | 21% |
| capital_intensive | 18% | 29% | 4% | 8% | 40% |
| high_growth | 16% | 10% | 0% | 6% | **67%** |
| unprofitable | 2% | 1% | 0% | 2% | **93%** |
| standard | 32% | 25% | 2% | 7% | 34% |

---

## 6. Confidence

```
confidence = 0.45 × weighted reliability
           + 0.35 × agreement between surviving models
           + 0.20 × model count (capped at 3)
```

Median 58%. Agreement is the max/min ratio across the models that survived —
two values within 20% of each other say something; two that disagree threefold
do not.

---

## 7. Worked examples

**JPM** — `financial`, fair value $226.82 vs $358.64, confidence 59%

```
fcfe       629.72    2.2%   rel 36%   used
fcff       776.68    0.0%   rel 48%   NOT USED - bank
ddm        112.64   23.1%   rel 94%   used
rim        256.58   52.3%   rel 85%   used
relative   235.80   22.4%   rel 91%   used
```

FCFF is refused outright; residual income leads at 52%. FCFE survives but at
2% weight — its $630 is an outlier the low reliability score has already
demoted.

**KO** — `mature_dividend`, fair value $57.51 vs $88.07, confidence 86%

All five models used, values spanning $41.87 to $65.27, DDM leading at 37%.
The tight agreement is why confidence is high.

**AAPL** — `standard`, fair value $160.08 vs $319.97, confidence 57%

FCFE 47.5% and FCFF 36.7% lead; DDM dropped by the sanity band at $16.32.

---

## 8. Limits

**7 tickers get no fair value at all** — AEVA, CRWD, DBVT, GSAT, LPL, NET,
PLSE. Every model either produced a negative value, was gated off, or fell
outside the sanity band. The engine declines rather than inventing a number,
which is the correct behaviour for a company whose economics no model describes.

**The blend inherits its inputs' assumptions.** The 4.5% equity risk premium,
the 21% tax rate and the sector margin anchors all still sit underneath the
FCFE and FCFF values.

**Relative valuation is relative.** If a whole industry is mispriced, the peer
median is mispriced with it, and `unprofitable` names take 93% of their fair
value from exactly that source.

**Agreement is not accuracy.** Five models built on the same fundamentals can
agree and still be wrong together; confidence measures internal consistency,
not truth.


---

## 9. Where it appears in the UI

| Table | Columns added |
|---|---|
| Largest companies | Fair value, FV upside, Confidence, FV multiples |
| Cheapest / most expensive by multiple | FV multiples, Fair value, FV upside, Confidence |
| Valuation screen | Profile, Fair value, FV upside, Confidence |
| Peer tables (screen and DCF) | Fair value, FV upside, Confidence |

`fair_value_multiples` sits deliberately next to the multiple being ranked on —
that table asks what peers would pay, and this is the answer on the same
reliability weighting, with no cash-flow forecast in it.

The performance tables (movers, recently reported, sector and industry
performance) do not carry it: they answer what a price *did*, not what a
business is worth, and mixing the two in one row invites reading a valuation as
a prediction of the next move.
