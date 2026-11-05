# Sharadar ingest — implementation notes

Everything in [`sharadar_loader.py`](sharadar_loader.py), written to satisfy one
constraint: **never download the whole market.**

- **Subscription:** Sharadar Bundle, 10-year tier ($49/mo). History starts
  `2016-09-06`. Confirmed empirically — AAPL returns 2,514 daily bars, not the
  ~7,000 a full-history tier would give.
- **Python:** `.venv/` next to the script (python.org 3.13; pandas 3.0.5,
  requests, pyarrow, duckdb, jupyter).

  This machine has four Pythons on `PATH`. The default `python3` is Homebrew
  3.14, which is PEP 668 "externally managed" — pip refuses to install into it
  — and there is no bare `python` at all. So the script **re-execs itself under
  `.venv/bin/python`** when started by the wrong interpreter. `python3
  sharadar_loader.py …` works from any shell with no activation. Set
  `SHARADAR_NO_REEXEC=1` to disable.

---

## 1. Architecture

Three phases, each independently re-runnable.

| Phase | Command | Cost | Output |
|---|---|---|---|
| 1 · Screen | `universe` | ~14 s, 6 requests | `universe.csv`, `data/universe.parquet` |
| 2 · Fetch | `history` | one-off, resumable | `cache/sharadar/**/*.csv.gz` |
| 3 · Compact | `compact` | seconds, no API | `data/*.parquet` |

The separation matters: **phase 3 never touches the API.** Change a dtype, a
sort order or a compression setting and rebuild locally at zero cost.

### Why two storage layers

```
cache/sharadar/{dataset}/{sha1-of-request}.csv.gz   ← verbatim API responses
data/{dataset}.parquet                              ← typed, sorted, queryable
```

The cache key is a SHA-1 of the request parameters (the credential excluded).
Consequences:

- An interrupted pull resumes for free — already-cached requests are skipped.
- Re-running `history` costs **zero** API calls if nothing changed.
- The cache is append-only and immutable, so it is the audit trail: what the
  API actually returned, before any of my transformations.
- Corrupt Parquet? Delete `data/` and run `compact`. Nothing is re-downloaded.

---

## 2. Phase 1 — the screen

Screens every US listing from **6 small requests**, before any history is
pulled. This is the whole reason the ingest stays cheap.

| Step | Table | Returns |
|---|---|---|
| Securities master | `tickers?table=stocks` | ~20k rows → 4,323 live NYSE/NASDAQ common stocks |
| Recent bars | `stocks?from=…&to=…` | last 21 calendar days, all tickers |
| Market cap | `daily?from=…&to=…` | one date, all tickers |

Filters applied, in order:

```
exchange ∈ {NYSE, NASDAQ}     isdelisted = 'N'      category ~ 'Common Stock'
currency = USD                excl. ADR             excl. Secondary Class
lastpricedate within 21 days
closeunadj      > $2
marketcap       > $500m
volume (5 sessions, summed) > 100,000
traded all 5 sessions
```

### Live result — 2026-09-04, 13.8 s

```
listed NYSE/NASDAQ common stock : 4,323
with price and market cap       : 4,303
  close > $2                      3,858 pass
  mktcap > $500m                  2,577 pass
  week volume > 100,000           3,963 pass
  traded all week                 4,302 pass

universe: 2,565 tickers
```

Market cap is the binding filter — it removes 40% of listings on its own. The
$2 price floor and the volume floor overlap with it heavily.

### Three judgement calls

- **`closeunadj`, not `close`, for the $2 floor.** `closeunadj` is the price
  actually printed. Split-adjusted close would wrongly exclude companies that
  traded above $2 at the time but look sub-$2 after adjustment.
- **`VOLUME_AGG = "sum"`** over 5 sessions — the literal reading of "volume
  traded for the last week". Set it to `"mean"` for average daily volume, which
  makes the threshold 5× stricter.
- **Phase 1 is deliberately uncached** (`use_cache=False`). The screen must
  reflect today's market, not a stale snapshot.

---

## 3. Phase 2 — history for survivors only

### Batching

The `ticker` parameter accepts a comma-separated list. Measured ceiling:

| Tickers per request | Result |
|---|---|
| 25 | 200 OK |
| 30 | 200 OK |
| 40 | **400 Bad Request** |

Set to **25**, leaving headroom for long symbols like `BRK.B`. Requests run
4-wide through a thread pool.

### `skip` pagination is unstable — this cost real data

The first full run pulled 5,388,169 price rows. Auditing them against the
session calendar found:

- **565 duplicated `(ticker, date)` pairs**
- **565 missing bars** — 535 of 2,565 tickers had holes

Exactly equal, which identifies the cause: with `limit`/`skip` pagination the
server does not guarantee a stable row order between pages. Rows shift across a
page boundary, so each shifted row is served twice to one page and skipped by
the next. Duplicates are cosmetic — `drop_duplicates` catches them. **The
dropped rows are silent data loss**, and they clustered on specific dates
(`2023-03-17`, `2025-01-08`) across hundreds of unrelated tickers, so a
per-ticker spot-check would never have found them.

**Fix — `fetch_windowed()`: size every request to fit in one page, so `skip` is
never used.**

| Request | Rows | Pages |
|---|---|---|
| 25 tickers × 1 year of prices | ~6,250 | 1 |
| 25 tickers × 10 years of fundamentals, one dimension | ~1,000 | 1 |
| All tickers × 1 session (phase 1) | ~6,000 | 1 |

`fetch_windowed` raises rather than truncates if a window ever returns
`PAGE_LIMIT` rows, so this cannot regress silently.

**Verified after the fix:** 5,388,169 raw rows, 5,388,169 distinct
`(ticker, date)` pairs — 0 duplicates — and **0 of 2,565 tickers with a gap**
against the session calendar. The 565 lost bars are back.

Where paging is genuinely unavoidable — the ~20k-row `tickers` master — the
request passes `sort=permaticker.asc`. A **unique** sort key gives a total
ordering, which is what makes `skip` safe. Sorting on `ticker` would not have
worked: it is not unique in that table.

### Point-in-time fundamentals

Stored as **`ARQ`** and **`ART`** — as-reported. `MR*` dimensions fold in
restatements the market had not seen yet; screening on them is lookahead bias.

The join key is **`date`** (the SEC filing date), *not* `calendardate` (the
period the numbers describe). The Sharadar API column is `date` — the legacy
Nasdaq Data Link name `datekey` does not exist here.

Verified: `as_of_snapshot('2024-06-30')` returns AAPL's 10-Q **filed 2024-05-03**
covering the March quarter — correctly *not* the June quarter, which had not
been filed on that date.

### Measured cost

| Run | Requests | Fetch | Wall clock |
|---|---|---|---|
| `skip` pagination (buggy) | 309 | 308 s | 5m 08s |
| Windowed (correct) | 1,133 | 282 s | 4m 53s |

Windowing needs 3.7× the requests but is **not slower** — the same bytes move,
in more and smaller responses, and 4-wide concurrency absorbs the extra round
trips. Correctness here was free.

Steady state is far cheaper: `refresh` fetches only sessions newer than
`MAX(date)`, so a weekly top-up is ~103 requests.

---

## 4. Phase 3 — columnar storage

`compact()` reads the cache, optimises dtypes, sorts, and writes Parquet:

- `float64 → float32` — halves every numeric column. Prices do not need 15
  significant digits.
- Low-cardinality strings → dictionary-encoded (`ticker` costs ~2 bytes/row
  instead of ~6).
- Date columns parsed to real timestamps.
- **Sorted by `ticker, date`**, so Parquet row-group statistics let a query
  filtering on one name skip whole 250k-row blocks without decompressing them.
- **ZSTD level 6**, ~30% better than snappy at negligible read cost.

Parquet lands at roughly the same size as the gzipped CSV cache — 157 MB of
Parquet against 171 MB of cache. Compression is a wash. The win is that DuckDB
reads *one column* off disk without decompressing the file, which gzipped CSV
cannot do.

### Measured — 5.39M price rows, 2,565 tickers

| Operation | Time |
|---|---|
| One ticker, full 10y history (2,514 rows) | 0.007 s |
| Full scan, 2 columns, grouped | 0.011 s |
| One rebalance date (2,373 rows) | 0.126 s |
| `as_of_snapshot()` — 2,373 × 119 cols | 0.304 s |
| Two columns from Parquet | **0.24 s** |
| Same two columns from the gzip cache | **2.12 s** (9× slower) |

A 522-week backtest at ~0.3 s per rebalance is ~2.5 minutes of data access.

### Querying

`connect()` returns a DuckDB connection with `prices`, `fundamentals` and
`universe` as views over the Parquet files. There is no import step and no
second copy on disk.

```python
import sharadar_loader as sl

con = sl.connect()
con.execute("SELECT ticker, date, close FROM prices WHERE ticker='AAPL'").df()

snap  = sl.as_of_snapshot('2024-06-30')          # the rebalance input
dates = sl.rebalance_dates('2016-09-01', '2026-09-05')  # weekly session ends
```

---

## 5. Commands

A subcommand is required — running the file bare prints usage and exits.

```bash
python3 sharadar_loader.py universe          # phase 1
python3 sharadar_loader.py history           # phases 2 + 3
python3 sharadar_loader.py history --limit 50
python3 sharadar_loader.py compact           # rebuild parquet, no API calls
python3 sharadar_loader.py refresh           # weekly top-up
python3 sharadar_loader.py status            # what is on disk
```

For the notebook, select the `.venv` kernel — the re-exec trick only covers the
command line.

`refresh` reads `MAX(date)` from the stored prices and fetches only sessions
newer than that — the steady-state weekly cost is a handful of requests.

---

## 6. API gotchas found by testing

| Issue | Detail |
|---|---|
| **Legacy table codes fail** | Docs list `SEP` / `SF1`; the live API needs `stocks` / `fundamentals`. `table=SEP` silently returns an empty body, not an error. |
| **`marketcap` units differ by table** | `daily` quotes it in **millions**; `fundamentals` in **raw dollars**. The screen reads `daily`, so `MIN_MARKETCAP_MUSD = 500` means $500m. |
| **Duplicate rows in `tickers`** | A ticker appears once per data table it belongs to. `table=stocks` both restricts to equities and deduplicates. |
| **Default date range is 1 year** | Omitting `from` silently truncates history. Always pass it. |
| **`table` is a query parameter** | So the fetch helper's first argument is named `dataset`, not `table`. |
| **400 = too many tickers** | Not a malformed request. The handler says so explicitly. |

---

## 7. Security note

The API key leaked into a `requests` exception message during testing —
`raise_for_status()` embeds the full URL, query string included.

Fixed: `_redact()` strips `api_key=…` from every error path, and the 400/403/5xx
branches no longer echo the URL.

**The key was exposed in that traceback and should be rotated** at
<https://sharadar.com/account>. `.env` is the only place it should live.


---

## 8. Current state on disk

```
universe.csv                2,565 tickers
cache/sharadar/             171 MB   1,442 gzipped responses
data/stocks.parquet         107 MB   5,388,169 rows  2016-09-06 -> 2026-09-04
data/fundamentals.parquet    46 MB     178,024 rows  ARQ + ART, 2,565 tickers
data/universe.parquet       0.2 MB
```

Integrity: 0 tickers without prices, 0 without fundamentals, 0 null bars,
0 duplicated `(ticker, date)` pairs, 0 calendar gaps.

Point-in-time spot check, `as_of_snapshot('2024-06-28')`:

| Ticker | Filing used | Period | Correct? |
|---|---|---|---|
| MSFT | 2024-04-25 | 2024-03-31 | yes — June quarter not yet filed |
| NVDA | 2024-05-29 | 2024-03-31 | yes |
| AAPL | 2024-05-03 | 2024-03-31 | yes |
| AVGO | 2024-06-13 | 2024-06-30 | yes — fiscal quarter ends early, filed before the date |

---

# valuation_metrics.py — TTM cash flows, cost of capital, multiples

Reads the Parquet from `sharadar_loader.py`, adds FRED market data, writes two
files.

```bash
python3 valuation_metrics.py market    # FRED series only
python3 valuation_metrics.py metrics   # rebuild from local data, ~5 s
python3 valuation_metrics.py all
```

## Outputs — three files

| File | Rows | Size | Scope |
|---|---|---|---|
| `data/market_data.parquet` | 16,874 × 8 | 0.4 MB | daily market series |
| `data/metrics.parquet` | 99,311 × 68 | 28.5 MB | **all 2,947 tickers** |
| `data/cash_flows.parquet` | 88,486 × 24 | 8.2 MB | **2,290 tickers with ≥6y history** |

`metrics` carries betas, alphas, WACC, multiples, margins, R&D/revenue and
quality ratios for every ticker. `cash_flows` carries FCFF, FCFE, their inputs,
and everything derived from them (P/FCFF, P/FCFE, EV/FCFF, FCFE yield, FCFF and
FCFE margins) — only for the ≥6-year cohort, so the file has no null-FCFF rows
to filter around. Plain FCF (`cfo + capex`) needs no history and stays in
`metrics`.

### Market series

| Column | Source | Coverage |
|---|---|---|
| `rf_10y`, `rf_3m` | FRED DGS10, DGS3MO | 1962 / 1981 → |
| `sp500` | FRED SP500 | 2016 → (licensing limit) |
| `nasdaq100` | FRED NASDAQ100 | 1986 → |
| `brent` | FRED DCOILBRENTEU | 1987 → |
| `vix` | FRED VIXCLS | 1990 → |
| `gold` | **Sharadar `funds`, GLD** | 2016 → |

**Gold is not on FRED as a daily price.** The London AM/PM fixings
(`GOLDAMGBD228NLBM`, `GOLDPMGBD228NLBM`) are discontinued; what remains is a
PPI index ending in 2021 or gold *volatility* (`GVZCLS`). Stooq's XAUUSD feed
is behind a bot check. So gold comes from Sharadar's `funds` table via GLD —
already inside the Bundle subscription, no third vendor. GLD is physically
backed, so it tracks spot less a 0.40%/yr expense ratio: a proxy, near-exact in
returns, not a spot quote.

Rates are forward-filled across holidays so any date lookup resolves. Index and
commodity levels are **not** filled — a missing close stays missing rather than
fabricating a flat return.

## TTM basis

Everything uses Sharadar's **`ART`** dimension — as-reported trailing twelve
months, time-indexed to the SEC filing date. No row uses a number the market
had not seen on that date.

## Definitions

```
tax_rate_eff = taxexp / ebt         (only where ebt > 0; clipped to [0, 50%],
                                     else 21%)
FCFF = CFO + interest × (1 − t) + capex
FCFE = CFO + capex + net borrowing
```

**`capex` is a signed cash-flow item** — negative is an outflow — so both
formulas *add* it. Verified: `fcf = ncfo + capex` holds on 94,706 of 94,727
rows, and 77 tickers (REITs, shipping) legitimately report positive capex in a
period of net asset sales. `abs()` would have inverted those.

### Cost of debt — average, not closing, debt

```
cost_of_debt = interest expense TTM / average(debt_t, debt_{t−4 filings})
```

Average wins because interest accrues across the whole trailing year while debt
is a point-in-time stock; the closing balance misprices any company that
changed leverage mid-year. Falls back to the closing balance in a ticker's
first year.

**933 of 2,947 tickers carry debt but report no interest expense** (Apple among
them). Those fall back to `risk-free + 2.00%`, and `cost_of_debt_source` marks
every row as `reported` or `fallback` — 69% reported, 31% fallback.

### Cost of equity and WACC

```
cost_of_equity = rf + beta × ERP          ERP = 5.0%  (an assumption)
WACC = wE × cost_of_equity + wD × cost_of_debt × (1 − t)
```

`EQUITY_RISK_PREMIUM` is the most load-bearing guess in the file — it is not
observable, and it flows straight into every WACC.

### Beta — weekly, not annual

CAPM regression of excess stock return on excess market return, 2-year rolling
window on **weekly** returns, benchmark S&P 500.

Weekly rather than annual because **FRED's SP500 series only goes back ten
years** (licensing), so annual frequency would give at most 10 points per
regression. Weekly gives ~104. Set `BETA_FREQ="YE"` and
`BETA_WINDOW_YEARS=10` to force the annual version.

Result: 1,121,543 estimates over 2,815 tickers, median beta **1.03**, median
R² **0.20**.

**Returns are simple (`pct_change`), not log — deliberately.** Only simple
returns are linear in portfolio weights (`r_p = Σ wᵢrᵢ`); log returns aggregate
across *time*, not across *assets*, and beta exists to measure contribution to
portfolio risk. Measured both ways on this universe: median |Δbeta| = 0.0146,
correlation 0.980, with divergence rising from 0.007 (low-vol quartile) to
0.046 (high-vol). The 47 names differing by >0.25 are all extreme-volatility
tickers whose regressions are unreliable either way — gate them on `beta_r2`,
not on the return definition.

**`alpha_annual` is annualised by multiplying by 52, not compounding.** Alpha
is a regression intercept — a per-period excess-return premium in a linear
model — so it scales linearly. Compounding it was tried and reverted: it
inflated every estimate, taking the maximum from 21.0 to 47,273,731. Jensen's
inequality applies to projecting wealth from a return series, not to a model
intercept. Returns come from `closeadj` (splits + dividends), while the FRED
indices are price-only; the resulting bias in a ratio of covariances is second
order.

## Currency — the bug enabling ADRs introduced

Turning on `INCLUDE_ADR` pulled in **233 filers reporting in IDR, KRW, COP**
and others, while `marketcap`, `ev` and `price` are always USD. Every ratio
spanning the two was wrong:

| | before | after |
|---|---|---|
| TLK (Telkom Indonesia) P/S | 0.0001 | 1.97 |
| KEP (Korea Electric) P/E | 0.0022 | 3.26 |
| max FCFF in the file | **44.8 trillion** | 136.7 bn (Apple) |

`fxusd` is **local currency units per USD**, so the conversion *divides* —
the opposite of what the name suggests. Verified rather than assumed:
`revenue / fxusd` reproduces Sharadar's own `revenueusd` on 92–99.9% of
non-USD rows, likewise ebitda, ebit, equity, debt and cash. All currency
fields use the same rate so sums and ratios stay consistent; share counts and
ratios are untouched. `fx_to_usd` records the rate applied.

## Validated

AAPL, filing 2026-07-31 — all three identities reconcile exactly:

| | |
|---|---|
| FCFF = CFO + int×(1−t) + capex | 136,682,995,712 |
| FCFE = CFO + capex + net borrowing | 119,409,000,448 |
| WACC = wE·rE + wD·rD·(1−t) | 0.0999 |

P/E 34.97, P/S 9.66, EV/EBITDA 26.94, net margin 27.6%, beta 1.065.

Remaining extremes (P/S up to 128k, P/E to ±14k) are genuine: zero-revenue
clinical-stage biotechs (SVRA, CLDX, ASMB) and thin-earnings names (CRWD,
BEP). Filter them in the screen rather than in the data.


## `latest()` — a bug worth knowing about

The convenience accessor originally used `groupby("ticker").last()`. Pandas
skips NaN **independently per column** there, so it returns a row assembled
from several different filings. Revvity Medicines came out with revenue of 0
from its 2026-08-05 filing sitting next to an R&D/revenue ratio of 745× carried
over from an older period when revenue was still positive.

Measured across the universe: **4,137 fabricated cells over 522 of 2,947
tickers** — 18% of the file. Now `sort_values("date").drop_duplicates("ticker",
keep="last")`, which keeps whole rows, so every field shares one filing date.

The Parquet files were never affected; this was only the read helper.

## R&D intensity

Sharadar reports `rnd = 0`, not null, for the ~1,800 tickers with no R&D line —
so a naive cross-sectional median of `rnd_to_revenue_ttm` is 0.000 and means
"most listed companies do no R&D", not "data missing". Filter `rnd_ttm > 0`
first. Among the 1,105 actual R&D spenders: median **12.2%** of revenue,
quartiles 4.5% / 12.2% / 24.5%.

Also added: `sgna_to_revenue_ttm`, `opex_to_revenue_ttm`,
`sbcomp_to_revenue_ttm`, `capex_to_revenue_ttm`.


---

# dcf_forecast.py — 15-year DCF

```bash
python3 dcf_forecast.py run       # ~2 s
python3 dcf_forecast.py status
```

| File | Rows | Size | Content |
|---|---|---|---|
| `data/dcf_forecast.parquet` | 33,675 × 25 | 6.2 MB | one row per (ticker, year 1–15) |
| `data/dcf_summary.parquet` | 2,245 × 46 | 0.6 MB | one row per ticker |

Runs only on the `cash_flows` cohort (≥6y history). 2,290 → 2,245 after dropping
24 tickers with no WACC or non-positive base revenue.

## Drivers

Five, each fitted independently on its own trailing 3-year median and each
fading linearly to its own terminal value by year 15.

| Driver | Terminal target | Bounds |
|---|---|---|
| revenue growth | terminal `g` | −10% … 35% |
| EBIT margin | sector median, shrunk | −20% … 60% |
| D&A growth | terminal `g` | −10% … 30% (D&A/rev 0.2%…25%) |
| capex / revenue | sector median, shrunk | **0.2%** … 30% |
| ΔWC / revenue | sector median, shrunk | −10% … 15% |

Tax fixed at **21%** (US federal statutory). Terminal growth 2.5%, capped so
`WACC − g ≥ 1pp` — binding for 13 tickers.

Terminal drivers are the sector median plus 50% of the firm's own deviation
from it, then clipped to ±1.5 sector IQRs. That is the smoothing: a firm keeps
part of its edge but cannot carry an outlier history into year 15.

## Verified

All 33,675 rows, zero NaN:

```
FCFF = NOPAT + D&A − capex − ΔWC          True
FCFE = FCFF − interest(1−t) + net borrow  True
discount factor = (1+WACC)^year           True
TV = FCFF₁₅(1+g)/(WACC−g)                 True
EV = Σ PV + PV(TV);  equity = EV − net debt   True
revenue growth in year 15 == terminal g   True
capex ≥ 0, D&A > 0, revenue > 0           True
WACC − g ≥ 1% for every ticker            True (min exactly 0.0100)
```

## Known weaknesses

**Sector medians are dragged down by loss-makers.** 43% of Healthcare and 22%
of Technology names have negative EBIT margins, so the all-firm median is far
below what a viable firm earns:

| Sector | All firms | Profitable only | % loss-making |
|---|---|---|---|
| Healthcare | 3.2% | 13.7% | 43% |
| Technology | 8.5% | 13.5% | 22% |
| Communication Services | 10.5% | 14.6% | 20% |

Anchoring on that pulls Apple's terminal EBIT margin from 31.7% to 20.1% and
NVIDIA's from 60% to 34%, which is why most mega-caps screen 20–76% overvalued.
Switching the anchor to the median of *profitable* firms is a one-line change
in `sector_terminals`.

**FCFE is discounted at cost of equity, not WACC.** FCFE is an equity claim, so
the cost of equity is the correct rate; discounting it at WACC overstates equity
value. `pv_fcfe_at_wacc` is emitted alongside for reference.

**570 tickers have no diluted share count**, so no implied price. Enterprise and
equity values are still computed.

**552 tickers are Financial Services or Real Estate**, flagged
`dcf_caution_sector`: for banks debt is raw material rather than financing, and
capex and working capital carry no operating meaning. An FCFF DCF is
structurally the wrong model there.

**194 tickers have negative year-15 FCFF**, so a negative terminal value.

## Filing-cadence bug found here (and fixed in valuation_metrics too)

`pct_change(4)` and `shift(4)` assume four filings a year. True for US domestic
filers, false for the **158 tickers that file annually** — foreign private
issuers on 20-F, whose interim rows carry no financials at all. For Infosys,
four rows back is four years back and lands on an empty row, so every driver
came out NaN and voided the whole 15-year path.

Both are now date-based lookbacks (`merge_asof` on `date − 1 year`, 200-day
tolerance), which is correct for any filing cadence. 6,515 empty ART rows are
dropped before anything is measured. The same fix was applied to
`avg_total_debt` in `valuation_metrics.py`, where it feeds cost of debt and
therefore WACC.
