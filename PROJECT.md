# Portfolio Screener — project overview

A self-hosted US equity screener: ingest, valuation, discounted cash flow, and a
local web UI. Everything runs on one machine against one paid data feed.

- **Universe:** NYSE + NASDAQ common stock, ~2,800 tickers after screening
- **Data:** Sharadar Core US Equities Bundle, 10-year tier ($49/mo) + FRED (free)
- **History:** 2016-09-06 → present, 5.39M daily bars, 99,311 filing rows
- **Stack:** Python 3.13, pandas, DuckDB + Parquet, Streamlit + Plotly
- **Code:** ~4,300 lines across 7 modules

---

## 1. How to run it

```bash
cd "/Users/markkhabarov/Desktop/financial market/Portfolio screener"
.venv/bin/streamlit run app.py          # opens http://localhost:8501
```

The scripts re-exec themselves under `.venv` automatically, so `python3 <file>`
works from any shell without activating anything.

### Full rebuild, in order

| Step | Command | Time |
|---|---|---|
| 1. Screen the market | `python3 sharadar_loader.py universe` | 14 s |
| 2. Pull history | `python3 sharadar_loader.py history` | 5 min |
| 3. Market data | `python3 valuation_metrics.py market` | 10 s |
| 4. Metrics + cash flows | `python3 valuation_metrics.py metrics` | 5 s |
| 5. DCF | `python3 dcf_forecast.py run` | 5 s |
| 5b. Valuation engine | `python3 valuation_engine.py run` | 1 s |
| 5c. Historical fair value | `python3 valuation_history.py run` | 44 s |
| 6. Analytics | `python3 analytics.py run` | 5 s |

### Routine update

`./update_data.sh` runs all six steps in order and restarts the app. That
restart matters: `ui_data.py` wraps every loader in `@st.cache_data(ttl=3600)`
and the Streamlit process outlives the update, so fresh parquet on disk is not
enough on its own — without it the page serves the frames it already had in
memory for up to an hour.

```bash
./update_data.sh                  # full daily update, then restart the app
./update_data.sh --quick          # skip the universe re-screen
./update_data.sh --no-restart     # leave the running app alone
./update_data.sh --stage metrics  # run one stage and stop
```

`sharadar_loader.py refresh` on its own only rebuilds `stocks.parquet` and
`fundamentals.parquet`. Everything downstream — metrics, cash flows, the DCF,
and the screener table that stamps the `as_of` date in the page header — keeps
whatever it was last built with, which is why running it alone leaves the page
looking stale. See `DATA_REFRESH.md`.

---

## 2. Architecture

```
Sharadar API ──┐
               ├─► cache/sharadar/*.csv.gz ──► data/*.parquet ──► DuckDB views ──► Streamlit
FRED CSV ──────┘   (immutable, hash-keyed)    (typed, sorted)     (zero-copy)
```

Three principles hold the design together:

**The cache is immutable and hash-keyed.** Every API response is stored gzipped
under a SHA-1 of its request parameters. Re-running an ingest costs zero API
calls; an interrupted pull resumes for free. The cache is also the audit trail —
what the vendor actually returned, before any transformation. That is how the
pagination data-loss bug (§6) was caught.

**Compaction is a pure local rebuild.** Turning the cache into Parquet never
touches the network, so dtypes, sort order and compression can be changed and
regenerated at will.

**Queries read Parquet directly.** DuckDB creates views over the files — no
import step, no second copy on disk. A single-ticker query over 5.4M rows
returns in 7 ms.

### Modules

| File | Lines | Role |
|---|---|---|
| `sharadar_loader.py` | 855 | Screening, ingest, cache, compaction, DuckDB access |
| `valuation_metrics.py` | 678 | TTM metrics, CAPM, WACC, multiples, margins |
| `dcf_forecast.py` | 930 | Driver fitting, 15-year projection, EPS model, Monte Carlo |
| `valuation_engine.py` | 780 | Multi-model fair value: classification, 5 models, weighting |
| `valuation_history.py` | 470 | Point-in-time fair value at every filing date, for backtesting |
| `return_drivers.ipynb` | 22 cells | Factor study: what explains forward returns |
| `screen_attractive.py` | 260 | Evidence-backed screen using only what predicted |
| `build_return_drivers_nb.py` | 430 | Regenerates that notebook from source |
| `analytics.py` | 250 | Returns, volatility, drawdown, post-report drift, sectors |
| `ui_data.py` | 461 | Cached loaders, English formatting, peers, earnings calendar |
| `app.py` | 865 | Main screener page |
| `statement_checks.py` | 310 | Identity and coverage checks on the filed statements |
| `pages/2_Company_Deep_Dive.py` | 501 | Per-company page |
| `pages/3_Financial_Statements.py` | 662 | Filed statements, quarterly / TTM / annual |

### Data files

| File | Rows × cols | Size | Grain |
|---|---|---|---|
| `stocks.parquet` | 6,158,270 × 10 | 121 MB | ticker × day |
| `fundamentals.parquet` | 225,551 × 112 | 56 MB | ticker × filing × dimension (ARQ, ART, ARY) |
| `metrics.parquet` | 99,311 × 72 | 30 MB | ticker × filing |
| `cash_flows.parquet` | 88,486 × 24 | 8 MB | ticker × filing (≥6y cohort) |
| `dcf_forecast.parquet` | 33,675 × 28 | 7.7 MB | ticker × forecast year |
| `dcf_summary.parquet` | 2,245 × 70 | 1.0 MB | ticker |
| `dcf_drivers.parquet` | 2,245 × 37 | 0.4 MB | ticker |
| `analytics.parquet` | 2,974 × 106 | 1.6 MB | ticker |
| `valuation_engine.parquet` | 2,258 × 60 | 0.7 MB | ticker |
| `valuation_history.parquet` | 61,508 × 48 | 17.6 MB | ticker × filing date |
| `group_returns.parquet` | 372,072 × 7 | 5.8 MB | sector/industry × day |
| `market_data.parquet` | 16,874 × 8 | 0.4 MB | day |

---

## 3. Methods

### 3.1 Screening

Six API calls screen the whole market before any history is pulled — this is
what keeps ingest cheap.

```
exchange ∈ {NYSE, NASDAQ}      isdelisted = 'N'      category ~ 'Common Stock'
currency = USD                 ADRs included         secondary classes excluded
last close (unadjusted)        > $2
market cap                     > $500m
average daily volume, 5 sessions > 100,000 shares
```

Unadjusted close is used for the price floor deliberately: split-adjusted close
would wrongly exclude companies that traded above $2 at the time.

### 3.2 Point-in-time fundamentals

The screen, the metrics, the DCF and the analytics all use Sharadar's **`ART`**
dimension — as-reported trailing twelve months, indexed to the **SEC filing
date**, not the period end. `MR*`
dimensions fold in restatements the market had not yet seen; screening on them
is lookahead bias.

`ARQ` (discrete quarter) and `ARY` (fiscal year) are also ingested, but only
the statements page reads them — they are there to *present* what a company
filed, not to feed anything that ranks or values. Every consumer filters
`dimension` explicitly, so which dimensions exist in the parquet is not
something a downstream query can get wrong by accident.

Verified: `as_of_snapshot('2024-06-28')` returns Apple's 10-Q filed 2024-05-03
covering the March quarter — not the June quarter, which had not been filed.

### 3.3 Cost of capital

```
cost_of_equity = rf + β_adjusted × ERP             ERP = 4.5% (assumption)
β_adjusted     = 0.67 × β_raw + 0.33               Blume adjustment
cost_of_debt   = interest TTM / average debt       average, not closing
               = rf + synthetic spread             when interest is unusable
WACC           = wE·rE + wD·rD·(1 − t)
```

Three corrections were made after Apple and Berkshire came out visibly
mispriced:

| Change | From | To | Why |
|---|---|---|---|
| Equity risk premium | 5.0% | **4.5%** | 5.0% sat above the current implied US ERP and pushed every WACC up |
| Beta | raw regression | **Blume-adjusted** | raw betas are noisy and revert toward 1; this is what Bloomberg reports |
| Cost of debt fallback | flat rf + 2% | **synthetic spread by interest coverage** | a flat 2% priced Apple's AA+ debt at 6.75%, roughly a junk yield |

Apple's cost of debt fell from 6.75% to 5.40% and its WACC from 9.99% to
9.35%.

**Beta** is a CAPM regression of excess stock return on excess market return —
weekly returns, 2-year rolling window, S&P 500 benchmark. Weekly rather than
annual because FRED's SP500 series only reaches back ten years by licensing, so
an annual regression would have at most 10 points. Median β = 1.03, median
R² = 0.20.

**Simple returns, not log.** Only simple returns are linear in portfolio
weights, and beta exists to measure contribution to portfolio risk. Measured
both ways: median |Δβ| = 0.015, correlation 0.98. `alpha_annual` is annualised
by multiplying by 52, which is correct for a regression intercept.

**Cost of debt uses average debt**, since interest accrues across the year while
debt is a point-in-time stock. 933 tickers carry debt but report no interest
expense (Apple among them); those fall back to `rf + synthetic spread`, priced
off interest coverage and flagged in `cost_of_debt_source`.

### 3.4 DCF

```
FCFF = EBIT × (1 − t) + D&A − capex − ΔWC
FCFE = FCFF − interest × (1 − t) + net borrowing
```

Five drivers projected independently over 15 years, each fading linearly from
its fitted base to its own terminal value:

| Driver | Terminal anchor | Bounds |
|---|---|---|
| revenue growth | terminal `g` | −10% … 70% |
| EBIT margin | sector median (profitable firms), shrunk | −30% … 85% |
| D&A growth | terminal `g` | −15% … 50% |
| capex / revenue | sector median, shrunk | **0.2%** … 45% |
| ΔWC / revenue | sector median, shrunk | −15% … 25% |

Bounds are enforced on the base, the terminal and every projected year, so
capex is never negative and D&A is never zero.

**Driver fitting is exponentially weighted**, halflife 730 days, over a
trailing 5 years. Weight is `0.5 ** (age_days / halflife)` computed from the
actual date gap, so recent filings count for more and irregular filing cadence
is handled correctly. Evaluating an EWM at its final point is simple
exponential smoothing — exactly **ARIMA(0,1,1)**. A fitted per-ticker ARIMA was
rejected deliberately: filing histories here are 6–10 annual observations, far
too short to identify AR and MA orders, and a mis-identified ARIMA extrapolates
trends that are not there.

**Terminal margins** are the sector median plus **65%** of the firm's own
deviation from it, then clipped to ±1.5 sector IQRs. The anchor is now the
median over **profitable firms only**. Both were changed together: the old
all-firm median (Technology 8.5%) with a 50% shrink pulled Apple's terminal
EBIT margin from 31.7% to 20.3% and made every durable franchise screen as
overvalued. On profitable medians (Technology 14.2%) with a 65% shrink it lands
at 25.6%.

### 3.3a Group return series

`analytics.build_group_returns` produces a daily cumulative return series for
every sector and every industry with at least three members, under three
weightings:

| Weighting | Answers |
|---|---|
| median | what a typical company in the group did |
| average | the same, but one runaway name moves it |
| market-cap | what the group did as a block of capital |

Daily returns are clipped to [−60%, +100%] first — a single session moving more
than that is a data artefact, and one would distort a whole sector's cumulative
line.

**Cap weights are today's market caps held fixed across history.** That is not
an investable index: a real one reweights as caps move, which needs
point-in-time shares outstanding. Fixed weights tilt the series toward whatever
has already grown, so read it as *"how the current heavyweights performed"*.
The median line carries no such tilt, which is why it is the default.

### 3.4a The EPS model

A third valuation runs alongside FCFF and FCFE, discounting earnings directly:

```
net income_t = (EBIT_t − interest_t) × (1 − tax)
EPS_t        = net income_t / shares          shares held constant
price        = Σ EPS_t/(1+rE)^t + TV(EPS₁₅)/(rE−g)/(1+rE)^15
```

It gets the same treatment as the other two — implied share price, p15/p50/p85
band, and P(above market). Share count is held flat, so a company that keeps
retiring stock will grow real EPS faster than this shows.

**Terminal growth** is 2.5%, capped so `WACC − g ≥ 1pp` for every ticker
(binding for 13). Tax fixed at **21%**, the US federal statutory rate.

**Discounting** is `(1+WACC)^year` for FCFF and `(1+cost of equity)^year` for
FCFE — FCFE is an equity claim, so WACC would overstate its value.
`pv_fcfe_at_wacc` is emitted alongside for reference.

### 3.5 Uncertainty around the implied price

The point estimate is one number; the interval comes from each firm's own
realised cash-flow volatility.

1. **σ** = standard deviation of year-on-year **log** changes in realised
   FCFF/FCFE, using only sign-consistent pairs. Clipped to [0.05, 1.00].
   Universe median 0.73; Apple 0.14, Microsoft 0.11.
2. **1,000 simulated paths** per ticker. The projected cash flow is multiplied
   by a mean-reverting **AR(1)** shock in log space (φ = 0.70), with a
   martingale correction so the shock has mean 1 and adds no drift.
3. Each path is discounted and terminal-valued exactly as the base case, giving
   a distribution of implied share prices.
4. Prices are clipped to **[0, 3 × current price]** — a share price cannot be
   negative, and beyond a few multiples the model is extrapolating.
5. Reported at the **p15 / p50 / p85** quantiles, plus
   `prob_above_market`, the share of paths landing above today's price.

**Why AR(1) rather than a random walk:** under a random walk, variance grows
linearly with the horizon, so a firm with σ = 0.6 has a year-15 standard
deviation of 0.6·√15 = 2.3 in logs — a 0.09× to 11× range. That is not forecast
uncertainty, it is a model with no anchor. A bad year does not permanently
reset a company's cash-flow level. With φ = 0.7 the shock variance converges to
σ²/(1−φ²), so long-horizon uncertainty is ~1.4× the one-year figure rather
than ~3.9×.

### 3.6 Currency

233 tickers report in IDR, KRW, COP and others while market cap and price are
always USD. `fxusd` is **local units per USD**, so conversion **divides** —
verified against Sharadar's own `revenueusd`/`ebitdausd`/`debtusd` columns on
92–99.9% of non-USD rows. All currency fields use the same rate so ratios stay
consistent; share counts and ratios are untouched.

---

## 4. The UI

Streamlit — chosen because it is pure Python, runs locally, and renders in the
browser with no JavaScript.

### Page 1 — Screener (`app.py`)

1. **Largest companies** — market cap, returns (1d/1w/1m/3m/1y), volatility, max
   drawdown, implied FCFF/FCFE prices, p15–p85 band, P(above price).
2. **Price history** — any ticker, 8 timeframes, optional rebased overlays of
   S&P 500 / Nasdaq 100 / gold / Brent / VIX, volume, log scale.
3. **Sector performance** — market-cap weighted, median or average, across all
   six periods, with a bar chart and top gainers/fallers filterable by
   **sector, industry and market cap**.
4. **Recently reported** — who filed most recently and how the stock moved
   after, at +1D, +5D, +20D and since. Filterable by sector, industry, market
   cap and how far back to look.
5. **Sector and industry multiples** — cheapest and most expensive by a chosen
   multiple, grouped by sector *or* industry, each row carrying the **group
   median** and the company's distance from it, with an **extreme-value trim**
   so a handful of 4,000× names cannot occupy the whole "expensive" list. The
   history chart takes **its own multiple selector**, plots any number of group
   medians, overlays individual companies, and can **pull in the nearest peers**
   of the first overlaid name automatically. The companion price chart rebases
   the same companies to 0% and accepts **S&P 500, Nasdaq 100, gold and Brent**
   as benchmarks.

6. **Sector and industry performance** — the same table shape as the company
   view, one row per group: 1D/1W/1M/3M/6M/1Y returns, volatility, max drawdown
   and cumulative return since 2016. Built from daily group return series, so
   volatility and drawdown are the group's own rather than an average of its
   members'. Three weightings (median, average, market-cap) and a chart that
   shows either **cumulative return** — rebased to the visible window, with
   index benchmarks — or the **median multiple** over time.
7. **Valuation screen** — composite percentile rank across P/E, P/S, EV/EBITDA,
   EV/EBIT, P/FCF and quality metrics, computed **within sector**, with an
   expandable per-company panel that includes nearest competitors.
8. **DCF constructor** — **ten** sliders that **open at the fitted values**, so
   the first thing on screen is the base case. Slider keys carry the ticker, so
   switching company resets to that company's fit rather than carrying the last
   one's edits. Series toggle covers Revenue, FCFF, FCFE, EPS, EBIT margin,
   **Capex/revenue, change in working capital, D&A and WACC** — each drawn
   against its own reported history, because a driver path is only judgeable
   against where the company has actually been. Below it, how the nearest
   competitors are valued, including P/S.

   **Terminal revenue growth is now a slider.** The `WACC − g ≥ 1pp` rule is
   applied *after* the override, so the interface cannot dial growth into the
   discount rate and blow the terminal value up — the constraint belongs to the
   model, not the UI.

**Chart controls:** price or **cumulative % change**, eight preset timeframes
plus a **custom date range**, index overlays (which force cumulative mode,
since comparing a $300 stock to a 7,700 index only means anything rebased),
volume, and log scale.

**Formatting:** every number is pre-formatted to an English string before it
reaches Streamlit. `format="compact"` renders through the *browser's* locale
and was emitting Russian abbreviations ("трлн"); money now always reads
`$1.23B` / `$45.6M`, and explicit column widths stop long values overflowing
their cells.

### Page 2 — Company Deep Dive

Seven tabs: **Historical multiples** (point-in-time series plus where each
multiple sits in its own history), **Financials & DCF** (reported vs forecast
across six panels — revenue, FCFF, FCFE, EPS, EBIT margin and net income — plus
the valuation bridge and every assumption), **Competitors** (peer table and a
bar chart on any chosen metric), **Reporting calendar**, **Corporate actions**
(dividends, splits, trailing yield), **Ownership** (13F summary, insider Form
4s), and **Risk & quality** (beta, alpha, R², drawdown, leverage, performance
vs S&P 500, index membership).

### Page 3 — Financial Statements

`pages/3_Financial_Statements.py`. The company's filed statements, on a basis
the user picks: **Quarterly** (`ARQ`, the discrete quarter), **TTM** (`ART`,
the vendor's trailing-twelve-month roll-up) or **Annual** (`ARY`, the fiscal
year). Four tabs — income statement, balance sheet, cash flow statement, and a
valuation section — with a period-count slider, a thousands/millions/billions
unit switch, and a CSV download per statement. The company selector shares its
session key with the deep dive, so a company picked on one page is still
selected on the other.

**Nothing on this page is re-added from its components.** Every subtotal —
gross profit, total operating expenses, operating income, pre-tax income, total
current assets, total liabilities, net cash from operating / investing /
financing activities, EBITDA, EBIT, enterprise value — is a field Sharadar
publishes, printed as filed. That matters beyond saving arithmetic: a subtotal
summed from the visible detail would silently disagree with the company's own
statement wherever the vendor does not break out every constituent line, and
the error would look like a rounding artefact rather than a missing line item.
The ten exceptions are marked with a dagger in the table and spelled out under
it: net debt, the equity residual, effective tax rate, capex intensity, cash
conversion, EV/Sales, EV/FCF, net debt/EBITDA, interest coverage, and
liabilities-plus-equity.

**Annual comes from the vendor, not from four quarters added up.** `ARY` was
added to `DIMENSIONS` in the loader for this (26,472 rows, 2,847 tickers, a
median of ten years each). Summing quarters would be wrong for any company on a
52/53-week fiscal calendar, for anything restated mid-year, and for stub
periods — and meaningless for the balance sheet, whose lines are stocks rather
than flows. All existing consumers filter `dimension` explicitly, so the extra
dimension changed nothing downstream.

**Three vendor conventions the page had to get right,** each verified against
the whole universe rather than assumed:

* **The balance sheet does not close on liabilities plus equity.** Sharadar's
  `equity` is the *parent's* share and it publishes no separate
  non-controlling-interest balance, so assets exceed liabilities plus equity by
  the NCI. 1,075 of 2,883 companies show a non-zero gap and 908 of those report
  NCI income; XOM's is $6.7bn, Apple's is exactly zero. Rather than print a
  "balance check" that fails for a third of the market, the statement carries
  the difference as an explicit **non-controlling interests and residual** line
  and reconciles to total assets.
* **`netinc` is already the parent's net income.** `netinccmn = netinc −
  prefdivis` holds for 2,836 of 2,847 companies; also subtracting `netincnci`
  holds for only 1,902. So net income to non-controlling interests is shown as
  disclosure in the memo block, not as a step in the waterfall where it would
  imply a subtraction that does not happen.
* **`tangibles` is tangible *assets*, not tangible book value** — it equals
  `assets − intangibles` for 2,846 of 2,847 companies, never `equity −
  intangibles`. `tbvps` is therefore labelled "tangible assets per share";
  called book value it would read as an error, since it is routinely larger
  than `bvps`.

**The cash flow statement is mostly reconstructed, and says so.** Sharadar
publishes three section subtotals plus capex, D&A and share-based compensation
— no working-capital lines at all. So the *change in receivables / inventory /
payables / deferred revenue* lines are differenced off consecutive balance
sheets and signed as cash effects, and each *other … activities* line is
defined as the vendor's subtotal less the lines printed above it. Two
consequences worth being explicit about:

* **Every section foots to the filed subtotal exactly**, because the residual
  is defined that way. Verified on XOM: the itemised operating lines sum to the
  published `ncfo` to within four cents on $52bn, which is float32 noise.
* **The residual is not small.** Median 15–23% of the largest section subtotal
  across a 300-ticker sample. A balance sheet moves for reasons that never
  reach operating cash flow — working capital arriving with an acquisition,
  currency translation, reclassification between captions — and deferred taxes,
  impairments, provisions and disposal gains are not published at all. Showing
  it as one honest plug line is the alternative to spreading it silently across
  the lines above.

The period differencing is lag-aware: `ARQ` and `ARY` difference against the
previous row, `ART` against four rows back, since consecutive TTM rows are one
quarter apart but each covers twelve months. A movement is only computed when
the two period ends are the expected distance apart (±1.2 months), so a skipped
filing or a changed fiscal year-end leaves the line blank instead of booking a
multi-period swing into one column.

The income statement needed the same treatment for the same reason: SG&A plus
R&D equals total operating expenses for only ~25% of filings, so *other
operating expenses* carries the rest.

**Presentation.** Negatives print in parentheses, subtotals are bold with a
rule and totals carry a double rule, memo lines are italic, and figures are
tabular-aligned. Empty lines are hidden by default and restorable with a
checkbox: the vendor fills an inapplicable field with `0.0` rather than null,
so without that, deposits would occupy a row of zeros on every software
company and inventory on every bank. Ratios against a zero or negative base
render as "—" rather than as a number — a loss-making quarter otherwise prints
a negative EV/EBIT that reads like a cheap multiple.

**The valuation tab is priced at each filing date, not today.** `marketcap`,
`ev`, `price` and every multiple below them are the vendor's point-in-time
values as at the filing date of their own column, which is what makes the
series comparable across time — each multiple pairs a period's earnings with
the price actually quoted when they were published. This is deliberately the
opposite choice from the DCF (see §13), and the page says so, because there the
question is "what is it worth now" and here it is "what was it worth then".

### Peer selection

Peers are chosen by **sector + industry first, then closest market cap on a log
scale** — a $10bn and a $100bn company are not comparable even though both sit
far from a $1tn one on a linear axis. If the industry is too thin, the sector
backfills, and the `Match` column always says which you are looking at so a
sector-only peer is never mistaken for a true comparable.

### Earnings calendar

Unusual Whales publishes confirmed dates, but its `/earnings` page is
client-rendered with no data in the HTML and its API returns **401 without a
paid token**. The app supports it behind `UNUSUAL_WHALES_TOKEN`; without one
every row falls back to a cadence estimate. The **Source** column states which
you are looking at on every row.

---

## 5. Assumptions and limitations

Read this section before acting on any number the app produces.

| Assumption | Value | Impact |
|---|---|---|
| Equity risk premium | 4.5% | The most load-bearing guess. Not observable; flows into every cost of equity and every WACC. |
| Tax rate | 21% | US federal statutory. Set 25% to approximate federal + state. |
| Terminal growth | 2.5% | Capped at WACC − 1pp. |
| Shock persistence | φ = 0.70 | Sets how fast forecast uncertainty stops widening. |
| Sector shrink | 65% | How much of a firm's margin edge survives 15 years. |
| Beta adjustment | Blume 0.67 | Pulls raw regression betas toward 1. |
| Sector anchor population | profitable firms only | Excludes loss-makers from terminal margin medians. |

**The model still disagrees with the market on large, expensive franchises.**
After the WACC and anchor fixes, Apple's implied FCFF price is $133 against a
market price of $309. That gap is not a bug: Apple's exponentially weighted
revenue growth fits at 3.3%, and a DCF fading that to 2.5% terminal growth
cannot reproduce a 35× earnings multiple. The market is pricing growth the
model does not forecast. Treat a large negative upside on a quality name as
*"the market expects more than the trailing numbers support"*, not as a sell
signal.

**Berkshire is the opposite case and should be ignored.** It screens at +158%
upside because it is classified Financial Services, whose profitable-firm
median EBIT margin is 33.3% — an insurance-like anchor applied to a
conglomerate. It carries the `dcf_caution_sector` flag for exactly this reason.

**FCFF DCF is the wrong model for 552 tickers** — Financial Services and Real
Estate, flagged `dcf_caution_sector`. For banks, debt is raw material rather
than financing, and capex and working capital carry no operating meaning.

**Other gaps:** 194 tickers have negative year-15 FCFF and therefore a negative
terminal value; 575 share counts are derived as `marketcap / price` rather than
reported (flagged in `shares_source`); the reporting calendar is **estimated**
from each firm's own filing cadence, not announced, because the Bundle carries
no forward earnings calendar; analyst consensus is a separate Sharadar product
not in this subscription, so no market-implied estimates are shown; the gold
series is **GLD**, which tracks spot less a 0.40%/yr expense ratio.

**Licensing:** the Sharadar personal tier is non-display, internal use only.
Redistribution or showing this to others moves you to business pricing.

---

## 6. Bugs found and fixed during development

Each of these was silent — none raised an error, and none would have been
caught by a spot-check.

**`skip` pagination dropped 565 price bars.** The API does not guarantee stable
row ordering between pages, so rows shifted across page boundaries were served
twice to one page and skipped by the next. Exactly 565 duplicates and 565
missing bars across 535 of 2,565 tickers, clustered on specific dates so a
per-ticker check would never have found them. Fixed by windowing every request
to fit in a single page; `fetch_windowed` raises rather than truncates if a
window ever hits the limit.

**ADRs corrupted every cross-currency ratio.** Telkom Indonesia showed a P/S of
0.0001 and the file's maximum FCFF was 44.8 *trillion*. Fixed by dividing all
reporting-currency fields by `fxusd`, verified against Sharadar's own USD
columns.

**`capex` is signed.** It is negative for outflows, so FCFF and FCFE *add* it.
77 tickers — REITs and shipping — legitimately report positive capex during net
asset sales, and `abs()` would have inverted the sign for exactly those.

**Filing cadence broke every year-on-year comparison for 158 tickers.**
`pct_change(4)` and `shift(4)` assume four filings a year; foreign private
issuers file a 20-F annually with empty interim rows. For Infosys, four rows
back was four years back and landed on an empty row. Both are now date-based
`merge_asof` lookbacks.

**`groupby.last()` fabricated 4,137 values.** Pandas skips NaN independently
per column, so the "latest" row was assembled from several different filings —
one company showed revenue of 0 from its newest filing beside an R&D/revenue
ratio of 745× carried over from an older period. 522 of 2,947 tickers affected.
Fixed with `drop_duplicates(keep="last")`, which keeps whole rows.

**An alpha "fix" that was itself wrong.** Compounding the regression intercept
instead of multiplying it inflated every estimate, taking the maximum from 21.0
to 47,273,731. Reverted; Jensen's inequality applies to projecting wealth from
a return series, not to a linear model's intercept.

---

## 7. Validation

Checked on every row, not a sample.

**Ingest** — 0 duplicated `(ticker, date)` pairs, 0 calendar gaps across 2,565
tickers, 0 null bars.

**DCF** — all 33,675 rows, zero NaN:

```
FCFF = NOPAT + D&A − capex − ΔWC          ✓
FCFE = FCFF − interest(1−t) + net borrow  ✓
NOPAT = EBIT × (1 − 21%)                  ✓
discount factor = (1 + WACC)^year         ✓
TV = FCFF₁₅(1+g)/(WACC−g)                 ✓
EV = Σ PV + PV(TV); equity = EV − net debt ✓
revenue growth in year 15 == terminal g   ✓
capex ≥ 0, D&A > 0, revenue > 0           ✓
WACC − g ≥ 1% for every ticker            ✓ (min exactly 0.0100)
p15 ≤ p50 ≤ p85; prices ≥ 0; p85 ≤ 3× market ✓
```

**UI** — both pages run through Streamlit's `AppTest` harness with no
exceptions: every sidebar filter, all three sector weightings, all nine DCF
sliders, ticker and timeframe switches, index overlays, and five deep-dive
companies spanning a mega-cap, a mid-cap, a small-cap, a financial, and a
ticker with no DCF.

---

## 8. Cost

| | Monthly |
|---|---|
| Sharadar Bundle, 10-year tier | $49 (~€45) |
| FRED | free |
| **Total** | **~€45**, inside the €50 budget |

The 10-year tier was chosen over 5-year ($29) because a screener tuned only on
2021–2026 has never seen a bear market, and over full history ($69) because
that breaks the budget. Steady-state usage is a few hundred API calls a week
against a 100,000/day limit.


---

## 9. Changes in the latest revision

**Model**
- Revenue growth ceiling raised to **70%**; margin and ratio bounds widened.
- Cost of capital corrected: ERP 5.0% → **4.5%**, **Blume-adjusted beta**, and
  a **synthetic credit spread by interest coverage** replacing the flat rf+2%
  fallback (Apple's cost of debt 6.75% → 5.40%, WACC 9.99% → 9.35%).
- Terminal margin anchors now use **profitable firms only**, with shrink raised
  from 50% to **65%** (Apple's terminal EBIT margin 20.3% → 25.6%).
- **EPS model** added alongside FCFF and FCFE, with the same implied price,
  p15/p50/p85 band and P(above market).
- **575 missing share counts** derived as `marketcap / price`, flagged in
  `shares_source`.
- Driver fitting is **exponentially weighted** (halflife 730 days, computed
  from real date gaps) — simple exponential smoothing, i.e. ARIMA(0,1,1).
- Price distributions come from a **mean-reverting AR(1)** shock rather than a
  random walk, clipped to [0, 3× current price].

**UI**
- Cumulative % change mode and a custom date range on the price chart.
- All numbers formatted to English strings; column widths fixed so nothing
  overflows.
- Sector *and* industry filters on movers, reporters and multiples; industry
  shown throughout.
- New: recently-reported table with post-report drift; cheapest/most expensive
  by any chosen multiple; median-multiple history by sector or industry with
  company overlays; a companion rebased price chart; peer tables on both the
  DCF and the multiples views.
- DCF constructor opens at fitted values and resets cleanly on ticker change.

**Bug found and fixed**
The synthetic-spread lookup iterated the table as written, so the widest spread
overwrote every earlier match and *every* company was assigned the 15% junk
bucket. Now sorted ascending by coverage threshold so the highest bucket a
company clears is the one that sticks.


---

## 10. Latest additions

**DCF constructor**
- Four new driver charts — **capex/revenue, change in working capital, D&A and
  WACC** — each against reported history. The working-capital series is
  reconstructed from the fundamentals with the same date-based year-ago lookback
  the model uses, so the history line is comparable to the forecast drawn on it.
- **Terminal revenue growth is a slider**, clamped by `WACC − g ≥ 1pp` after the
  override rather than before.
- Competitor table gained **P/S**.

**Sector and industry multiples**
- Group **median multiple** and each company's distance from it, on both the
  cheapest and most expensive tables.
- **Extreme-value trim** (0–20th percentile) before ranking.
- The history chart has **its own multiple selector**, independent of the table.
- **Nearest peers** can be added to the chart in one click.
- **Index benchmarks** on the companion price chart.

**New: sector and industry performance**
- Table in the same shape as the company view — period returns, volatility,
  drawdown, cumulative since 2016 — for all 11 sectors and 137 industries.
- Chart toggles between cumulative return (rebased to the visible window, with
  benchmarks) and median multiple over time.
- Backed by a new `group_returns.parquet`, 372,072 daily rows.


---

## 11. Final revision — global filters and DOM handles

### Sidebar filters are global

The sidebar now carries **Sector, Industry and Tickers**, and all three apply to
every section on the page rather than only the tables nearest them.

Two scopes are derived once and reused everywhere, which is what makes the
behaviour consistent:

| Frame | Carries | Used by |
|---|---|---|
| `scoped` | sidebar sector / industry / tickers | group-level sections (movers, reporters, sector performance) |
| `filtered` | `scoped` + the market-cap floor | company tables |

The ticker options themselves cascade from the sector and industry above them,
so the list narrows as you drill down instead of always offering all 2,800
names. Selecting tickers also narrows the sector and industry performance tables
to the groups those companies belong to — the group *return series* are still
computed over full membership, since filtering a display is not the same as
re-deriving a group from a subset.

### Tickers are added to charts, not just filtered

Sidebar tickers are **unioned into** every comparison chart's overlay list
rather than used as the widget's default. This matters: Streamlit honours
`default=` only on a widget's *first* render, so a changed sidebar selection
would silently stop appearing after any interaction. The union always holds, and
each chart's own picker still adds on top of it — verified by selecting
AAPL/MSFT/NVDA in the sidebar and then adding GOOGL locally, which yields all
four traces.

The sector/industry performance chart gained a company overlay on the same
basis, so a stock can be compared directly against its sector's cumulative line.

### Discount-rate chart

The WACC series became **Discount rates** and now draws four things: reported
WACC, reported cost of equity, and flat base/what-if segments for both across
the forecast window.

The overlap you saw came from `add_hline`, whose annotations pin to the plot
edge — two of them landed on top of each other and on the historical series.
They are now ordinary traces, so they get legend entries and hover values and
cannot collide.

### Chart timeline

The **Timeline (years)** control moved out of the section header and sits
directly above the cumulative-return chart with the other chart controls, since
it affects only the chart and not the table.

### DOM handles for inspect element

Every table, chart and section now carries a stable `key`, which Streamlit
renders as a `st-key-<name>` class on the element's wrapper div:

- **11 section anchors** on the main page, 7 on the deep dive —
  `st-key-section_largest_companies`, `st-key-section_dcf_constructor`, …
- **11 tables** and **6 charts** on the main page —
  `st-key-table_valuation_screen`, `st-key-chart_price_history`, …
- **10 tables** and **5 charts** on the deep dive —
  `st-key-table_peers`, `st-key-chart_financials_grid`, …

Names describe the content, not the position, so they stay valid if sections are
reordered.


---

## 12. Sector performance is static to the sidebar

Both group-level comparison views now ignore the sidebar filters entirely:

- **Sector performance** (table + bar chart + biggest movers)
- **Sector and industry performance** (table + cumulative-return chart)

A cross-sector comparison only means something against the whole market.
Filtering to one sector left a single row with nothing to compare it against,
which made the section useless exactly when someone had drilled into a sector.

Each keeps its own controls, so nothing was lost: the sector performance section
gained its **own minimum-market-cap selector** to replace the sidebar's, and
still has weighting and period; the movers table keeps its sector, industry and
market-cap pickers; the group performance chart keeps level, weighting, display
mode, benchmarks, company overlays and timeline.

Everything else still follows the sidebar — largest companies, recently
reported, multiples, the valuation screen and the DCF constructor. Sidebar
tickers also continue to be added to the comparison charts.

Verified with the sidebar set to three tickers: the company tables drop to those
three, the multiples chart plots them, and both performance tables stay at the
full 11 sectors / 137 industries.


---

## 13. Market price bug — filing date vs last close

`dcf_summary.market_price` was Sharadar's `price` field from the **filing row**,
i.e. the price **on the day the company filed**, not the latest close. ASML last
filed on 2026-07-15 at $1,815.27 while the stock closed at $1,714.88 on
2026-09-04 — a 5.9% gap, and the DCF section was the only place on the site
showing the stale figure.

It was not confined to the label. The same value fed:

- `upside_fcff`, `upside_fcfe`, `upside_eps`
- `prob_above_market_*` — the share of simulated paths above "today's" price
- the simulation's `3 × current price` clip

**Across the cohort, 1,203 of 2,245 tickers differed by more than 5% and 182 by
more than 20%**, so every upside and every probability was measured against a
price that in many cases was months old.

**Fix.** `market_price` is now the latest traded close, joined from the prices
table, and carries `market_price_date`. The filing-date price is kept as
`price_at_filing` because it still has one legitimate job: the 575 derived share
counts are `marketcap / price`, and that market cap is also from the filing
date — mixing in today's close would invent a share count that never existed.

Verified: DCF market price now equals `analytics.last_close` for all 2,245
tickers, maximum absolute difference 0.0000.

The DCF section also now states both dates explicitly, because they are
genuinely different and the distinction matters:

> Valuation basis: filing of **2026-07-15** (price then $1,815.27) · Market
> price: last close of **2026-09-04**. Upside and P(above market) are measured
> against the last close, not the filing-date price.

---

## 14. Silent staleness in the refresh path

The page was showing prices as of 4 September however often the loader was
re-run. Five independent defects in the ingest path, all silent — every run
exited successfully.

**Empty responses were cached, which pinned the data permanently.** `fetch()`
wrote every response to the hash-keyed disk cache, including ones with no rows.
Because the key is a hash of the request and a same-day re-run makes the
identical request, the second run was served the empty frame from disk and
never reached the API — it printed `cached already: 339`, `to fetch: 0`, and
exited clean. Start a top-up before the vendor publishes the session and that
"nothing new" is frozen forever. There were 339 poisoned entries on disk, one
full run's worth (113 price batches plus 113 × 2 fundamental dimensions),
written at 20:55 local — 14:55 New York, with the session still open. Fixed:
`fetch()` never caches an empty frame, and `sharadar_loader.py purge-empty`
clears what the old behaviour left behind.

**The top-up window started one day too late.** `refresh()` fetched from
`max(stored date) + 1`. But `download_history()` drives prices *and*
fundamentals off one `start`, and `/data/fundamentals` filters `from`/`to` on
the **filing** date, not `calendardate` — `from=2026-07-15` returns AAPL's
`calendardate` 2026-06-30 row, because it was filed 2026-07-31, while
`from=2026-08-01` returns nothing. So a filing landing on the last stored
session after that day's run fell into a gap no future window ever covered, and
a vendor restatement of an already-stored day was never re-read. Fixed:
`REFRESH_OVERLAP_DAYS = 7`, ending at the last stored session rather than
starting after it; `compact()` already drops the duplicate rows. The first run
recovered 4,691 price rows and 2 filings the old window had stepped over.

**`HISTORY_START` moved every day, so the cache could never hit.** It was
`today − 10 years`, and `download_history()` measures its 366-day windows
forward from it, so every boundary shifted daily and a `history` re-run
silently re-downloaded all ten years. Both generations were still on disk — 118
batches anchored at 2016-09-06, 103 anchored at Jan 1, ~500 MB of duplicate
download. Fixed: anchored to 1 January of `year − 10`, so boundaries hold still
for a year and `history` is idempotent.

**New universe members arrived with no history.** The screen is re-run on each
update and membership drifts, but `refresh` fetched only the short top-up
window, so a newly admitted name got a handful of bars — not enough for a
trailing metric or the DCF, and it reached the page as a row of blanks. Fixed:
`refresh()` diffs the universe against the tickers in `stocks.parquet` and
backfills newcomers from `HISTORY_START` first.

**Compaction kept both versions of a revised row.** Surfaced by the overlap
fix, and the reason that fix was needed. `_optimise()` deduplicated with a
full-row `drop_duplicates()`, which only removes byte-identical rows — so once
`refresh` re-read days it already had, a vendor correction and the row it
corrected differed in a field or two and both survived: 9,386 duplicated
`(ticker, date)` across 2,408 tickers, from revisions Sharadar made between 4
and 7 September. Fatal rather than untidy, because `valuation_metrics` pivots
prices to date x ticker and a duplicated key makes the reshape raise. Fixed:
`compact()` takes an explicit natural key and folds each key to the row with
the latest `lastupdated` — `(ticker, date)` for prices, and `(ticker,
dimension, calendardate, date)` for fundamentals, where the filing date belongs
in the key because one `calendardate` legitimately carries several filings when
a company amends (6,905 of them), and those are distinct point-in-time
observations, not duplicates.

A further one, on the UI side: `ui_data.institutional_holdings` passed no date
parameter to `fetch`, so unlike the other live deep-dive loaders its cache key
never rotated and a ticker showed the 13F holdings it was first looked up with
indefinitely. Fixed with `use_cache=False`; the `st.cache_data` wrapper still
caps it at one request per ticker per hour.

**Not a bug:** prices are end-of-day. During the trading session the newest
close available is the previous session's, and around 8 September 2026 that was
Friday the 4th — the 5th and 6th were the weekend, the 7th was Labor Day. The
check for whether the vendor or this repo is at fault is in `DATA_REFRESH.md`.

---

## 15. How reliable are the statements?

`statement_checks.py` answers this rather than leaving it to be assumed, and
writes `RELIABILITY.md`. It separates two questions that are easy to conflate:

**Identity** — relationships that must hold in any set of accounts. A deviation
is the vendor disagreeing with itself.

**Coverage** — how much of a subtotal the lines printed above it account for.
A deviation is *not* an error; it is the part of the statement the vendor does
not break out, which the statements page shows as an explicit *other …*
residual so every section still foots.

Deviations are measured relative to a denominator chosen per statement —
revenue for the income statement, total assets for the balance sheet, and the
largest of the three section subtotals for the cash flow statement — because an
absolute dollar error is meaningless across companies four orders of magnitude
apart in size. The pass bar is 0.1%, not exact equality: figures are stored as
float32 (see `_optimise`), which is about seven significant digits, so on a
$400bn balance sheet exact equality is unattainable even when the arithmetic is
perfect.

### Result, on a 300-ticker sample across all three bases

**33 of 39 identity checks hold to 0.1% on at least 99% of filings, and every
one of the six failures is in the cash flow statement.** Nothing fails in the
income statement or the balance sheet:

| Statement | Result |
|---|---|
| Income | revenue − cost of revenue = gross profit, gross profit − opex = operating income, EBITDA − D&A = EBIT, pre-tax − tax = net income, net income − preferred dividends = net income to common: **100.0%** within 0.1%, on all three bases |
| Balance | current + non-current = total, for assets, liabilities and debt; current assets − current liabilities = working capital; assets − intangibles = tangible assets: **100.0%** within 0.1% |
| Cash flow | operating + investing + financing + FX = net change in cash: **91.2% / 89.4% / 86.7%** (ARQ / ART / ARY), p95 deviation 2.7–4.8% |
| Cash flow | operating cash flow − \|capex\| = free cash flow: **97.5–98.5%** |

So the income statement and balance sheet can be read as filed. The cash flow
statement's section subtotals are sound, but roughly **one filing in ten does
not foot** from the three sections to the reported net change in cash. That is
the vendor's arithmetic, not something this repository introduces — which is
why the page prints it as an explicit **unreconciled** line instead of
distributing it across the sections where it would be invisible.

### Coverage is much lower, by construction

| Subtotal | Median residual | Within 0.1% |
|---|---|---|
| Operating cash flow, from net income + D&A + SBC + working capital | 15–23% | ~0.6% |
| Investing cash flow, from capex + acquisitions + investments | 0–0.1% | 50–59% |
| Financing cash flow, from debt + equity + dividends | 1.6–2.7% | 16–28% |
| Total operating expenses, from SG&A + R&D | 2.3–2.6% | 23–28% |
| Total assets, from liabilities + parent equity | ~0% | 63–65% |

The operating line is the weak one and unavoidably so: the vendor publishes no
working-capital lines, so those are differenced off the balance sheet, and it
publishes nothing at all for deferred taxes, impairments, provisions or gains
on disposal. The last row is the NCI residual from §4 seen from the other
side — the 35% that misses is minority interests, not error.

```bash
python3 statement_checks.py                  # 300-ticker sample, all bases
python3 statement_checks.py --sample 0       # every ticker
python3 statement_checks.py --write-report   # refresh RELIABILITY.md
```

The checker imports the derivation functions from the statements page rather
than reimplementing them, so the numbers it reports are the numbers the page
draws — a second copy would be free to drift away from the thing it is
supposed to be checking.



---

## 14. Multi-model valuation engine

`valuation_engine.py` replaces "one DCF for everyone" with five models weighted
per company: FCFE and FCFF (reused from the DCF), a two-stage dividend discount
model, residual income on book equity, and peer-median multiples. Each returns a
value **and** a reliability score; base weights come from a seven-way company
classification, are multiplied by reliability, gated on economic applicability,
and renormalised over the survivors.

Calibration against the single-model DCF, on the same 2,258 tickers:

| | median upside | \|upside\| > 200% |
|---|---|---|
| FCFF alone | +31.9% | 33.4% |
| **Engine** | **+2.4%** | **8.1%** |

`fair_value_multiples` applies the same reliability weighting to the multiples
alone, as its own column.

Full write-up in **`VALUATION_ENGINE.md`**. `app.py` is unchanged; the engine's
columns are merged into `analytics.parquet` so the UI can pick them up when you
choose to surface them.


---

## 15. Historical fair value for backtesting

`valuation_history.py` reruns the engine rewound to every filing date — 61,508
(ticker, date) rows, 2019-09 to 2026-09 — by feeding one row per company-date
through the *same* `dcf.project`, `classify` and `blend` functions the live
engine uses. Point-in-time throughout: drivers from an EWM evaluated at each
row, sector and peer medians from that quarter's cross-section, expanding sigma,
close on the filing date.

Forward returns at 1/3/6/12 months are included, prefix-isolated so they cannot
leak into an input.

**Validation.** Latest-row agreement with the live engine: median ratio 1.004,
94.9% identical profile. Cross-sectional decile spread D9−D0 of **+7.92%** over
12 months, with rank correlation rising from 0.038 at one month to **0.093** at
twelve — the slow-correction pattern a valuation signal should show.

**A lookahead bug was caught building this**: `years_of_history` in
`metrics.parquet` is the full-sample span stamped on every row, which would have
let 2019 valuations claim a decade of history. Fixed to elapsed-since-first-
filing; the history gate went from dropping 10,873 rows to 31,928.

Full write-up, including what the backtest is *not*, in **`BACKTEST.md`**.


---

## 16. What explains forward returns

`return_drivers.ipynb` evaluates `valuation_history.parquet` as a factor study:
local (firm-level) determinants against global (sector, industry, size) ones,
with forward return as the target. Everything is measured cross-sectionally
within quarter, and every t-statistic is Newey–West adjusted for the overlap
that quarterly sampling of a 12-month return creates.

**Headline findings, 12-month horizon:**

| | |
|---|---|
| Predictive (\|t\| > 2) | sales yield 0.131, peer-multiple upside 0.089, blended fair value 0.084, book yield 0.082, RIM upside 0.074, FCFF 0.066, FCFE 0.053 |
| Not distinguishable from noise | ROE, ROIC, EBIT margin, net margin, confidence, reliability, models used, model agreement |
| Inverse | size −0.079 (smaller outperformed) |

Sector alone explains 0.5% of return variance and industry 2.5% — but
`quarter × industry` explains **19.8%**. The local signal survives sector
neutralisation (+0.073, t = 2.75), so it is not a disguised sector bet. It works
in Utilities (0.143), Financials (0.114) and Energy (0.106) and is **worthless in
Technology (0.018, t = 0.50)**.

A walk-forward gradient-boosted tree reaches an out-of-sample IC of +0.102
(t = 2.63, positive in 94% of 16 quarters) — no better than the best single
linear feature, so there is no large interaction being missed.

Regenerate with `python3 build_return_drivers_nb.py` then execute the notebook.


---

## 17. From research to a screen

`STRATEGY.md` turns §16's findings into rules, and `screen_attractive.py`
implements them. The screen uses **only** the seven features that cleared
|t| > 2 — every quality ratio and every engine self-assessment metric is
deliberately excluded because neither predicted anything.

Five rules came out of the evidence: rank inside the **industry** not the sector
(this alone removed McKesson and Cardinal Health, which screened top-five on a
price-to-sales that is structurally normal for drug distribution); treat upside
above +300% as a **warning** (returns fall in deciles 8–9); do not use it in
**Technology** (IC 0.018, t 0.50); skip negative book equity; and accept that
the size filter needed to make names tradeable **halves the signal**
(IC 0.081 → 0.049, spread +6.7% → +3.1%).

The seven-feature composite is also **not stronger than price-to-sales alone**
(IC 0.076 vs 0.082) — only more consistent quarter to quarter.
