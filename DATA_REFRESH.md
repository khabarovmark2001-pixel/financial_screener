# Keeping the page current

`./update_data.sh` — one command, runs every stage in dependency order and
restarts the app.

```bash
./update_data.sh                  # full daily update, then restart the app
./update_data.sh --quick          # skip the universe re-screen
./update_data.sh --no-restart     # leave the running app alone
./update_data.sh --stage metrics  # run one stage and stop
```

It ends by printing the newest date now stored in each parquet, so you can see
at a glance whether the page will actually show something new.

---

## Why one command, and why this order

Six artefacts feed the page and each is derived from the one above it:

```
universe.csv ─► stocks.parquet ─► metrics.parquet ─► dcf_summary.parquet ─► analytics.parquet
                fundamentals      cash_flows         dcf_forecast           group_returns
                                       ▲
                              market_data.parquet (FRED + gold)
```

`sharadar_loader.py refresh` only ever rebuilt the first two. Everything
downstream — the valuation history, the cash-flow panel, the DCF, and the
screener table itself — kept whatever it was last built with. The header on
page 1 prints `analytics["as_of"]`, so a stale `analytics.parquet` makes the
whole page *read* as stale even when the prices underneath it are current.

The last stage is a restart, and it is not optional. Every loader in
`ui_data.py` is wrapped in `@st.cache_data(ttl=3600)` and the Streamlit process
outlives the update, so fresh parquet on disk is not enough on its own — for up
to an hour the page keeps serving the frames it already had in memory.

---

## Bugs this exposed, and what was changed

### Empty API responses were cached, which froze the data permanently

`fetch()` wrote every response to `cache/sharadar/{dataset}/{hash}.csv.gz`,
including responses with no rows. The cache key is a hash of the request, and a
same-day re-run makes the *identical* request — so the second run was served
the empty frame from disk and never reached the API. The loader reported
`cached already: 339`, `to fetch: 0`, exited successfully, and the parquet
never moved.

This is the whole failure mode in one line: **start a top-up before the vendor
has published the session, and that "nothing new" is pinned forever.** There
were 339 such poisoned entries on disk (113 price batches + 113 × 2 fundamental
dimensions) — exactly one full run's worth, written at 20:55 on 8 September,
which is 14:55 New York, while the session was still open.

*Fix.* `fetch()` no longer caches an empty frame. A miss now costs one wasted
request on re-run instead of freezing the answer. `sharadar_loader.py
purge-empty` deletes entries written by the old behaviour, and `refresh` calls
it before it does anything else.

### The top-up window started one day too late

`refresh()` fetched from `max(stored price date) + 1`. Two things fall through
that gap:

* **Filings.** `download_history()` drives the price and fundamentals requests
  off a single `start`, but `/data/fundamentals` filters `from`/`to` on the
  **filing date**, not on `calendardate` — verified directly: `from=2026-07-15`
  returns AAPL's `calendardate` 2026-06-30 row because it was *filed*
  2026-07-31, while `from=2026-08-01` returns nothing. So a filing that lands
  on the last stored session *after* that day's run is never covered by any
  future window. It is lost, silently and permanently.
* **Restatements.** Sharadar revises rows after publishing them. A
  strictly-forward window never re-reads a day it already has, so a correction
  is never seen.

*Fix.* `refresh()` now re-fetches a `REFRESH_OVERLAP_DAYS = 7` window ending at
the last stored session rather than starting after it. Re-fetched rows are
identical in the normal case and `compact()` drops the duplicates, so the
overlap costs bandwidth and nothing else. The first run with it recovered 4,691
price rows and 2 filings that the old window had already stepped over.

### `HISTORY_START` moved every day, so the cache never hit

It was `today − 10 years`. `download_history()` cuts the range into 366-day
windows measured forward from that date, so a start that moves daily moves
every window boundary with it, and no entry written yesterday could be hit
today. A `history` re-run silently re-downloaded all ten years. Both
generations were still on disk — 118 batches on windows anchored at 2016-09-06,
another 103 anchored at Jan 1, ~500 MB of duplicated download.

*Fix.* `HISTORY_START` is anchored to 1 January of `year − 10`. Boundaries now
hold still for a year at a time, so re-running `history` is cheap and
idempotent. Depth is unchanged at ten years or more.

### New universe members had no history

The screen is re-run on every update and membership drifts with price and
market cap, but `refresh` fetched only the short top-up window — for a name
admitted today, that is a handful of bars. Not enough for any trailing metric
or for the DCF, so it reached the page as a row of blanks.

*Fix.* `refresh()` compares the universe against the tickers already in
`stocks.parquet` and backfills newcomers from `HISTORY_START` before the
top-up.

### Compaction kept both versions of a revised row

Surfaced by the overlap fix, and it is the reason the overlap was needed.
`_optimise()` deduplicated with a full-row `drop_duplicates()`, which only ever
removes byte-identical rows. Sharadar revises bars after publishing them —
between 4 and 7 September it moved A's 2026-09-03 high from 154.300003 to
154.294998, and AA's 3 September volume from 2,554,000 to 2,553,000 — so once
`refresh` re-read days it already had, the correction and the row it corrected
differed in a field or two and both survived. 9,386 duplicated `(ticker, date)`
rows across 2,408 tickers.

Downstream that is fatal rather than untidy: `valuation_metrics` pivots prices
to date × ticker, and a duplicated key makes the reshape raise
`Index contains duplicate entries, cannot reshape` — the metrics stage died
outright.

*Fix.* `compact()` now takes an explicit natural key and folds each key down to
the vendor's most recent version, ordering by `lastupdated`. The key is
`(ticker, date)` for prices, and `(ticker, dimension, calendardate, date)` for
fundamentals — the filing date belongs in the key there, because one
`calendardate` legitimately carries several filings when a company amends, and
those are distinct point-in-time observations rather than duplicates (6,905 of
them). Rows without a `lastupdated` fall back to first-wins rather than to an
error.

### 13F holdings were frozen from first lookup

`ui_data.institutional_holdings` called `fetch("holdings_ticker", ticker=...)`
with no date parameter. The other live loaders on the deep-dive page pass
`to=today`, so their cache key rotates daily and they refresh themselves; this
one's key never changed, so a ticker kept showing the holdings it was first
looked up with, indefinitely.

*Fix.* `use_cache=False`. The `@st.cache_data(ttl=3600)` wrapper still holds
the result, so this costs at most one request per ticker per hour.

---

## When the data is current and the page still says 4 September

Sometimes nothing is broken. Prices are end-of-day: the vendor publishes a
session after the US close, so during the trading day the newest close
available is the *previous* session's. Around 8 September 2026 the last one was
Friday 4 September — the 5th and 6th were the weekend and the 7th was Labor
Day.

To check the vendor rather than this repo:

```bash
.venv/bin/python -c "
import sharadar_loader as sl
d = sl.fetch('stocks', use_cache=False, ticker='AAPL',
             **{'from':'2026-08-25','to':'2026-12-31'})
print(d[['date','close','lastupdated']].sort_values('date').tail())"
```

If that shows nothing newer than `data/stocks.parquet` already holds, the
pipeline is current and there is nothing to fix.

## Manual sector / industry edits survive the refresh

The `sector` and `industry` columns in `data/analytics.parquet` come from the
Sharadar securities master, and the `analytics` stage rewrites that file from
the vendor on every run. Corrections made by hand therefore cannot live in the
parquet.

They live in `classification_overrides.json` at the project root — a file no
stage of `update_data.sh` touches:

```json
{ "AAPL": { "sector": "Technology", "industry": "Consumer Hardware" } }
```

`ui_data.load_analytics()` re-applies it over the freshly loaded parquet, so
every table, filter and peer group on every page reads the corrected label,
and a refresh or an app restart leaves the edits in place. Edit them in the
app: **sidebar → Edit sector / industry** (pick a company, choose an existing
label or type a new one; *Reset to vendor* drops the edit). Only fields that
differ from the vendor are stored, so a label you leave alone keeps following
the vendor.

The parquet keeps the vendor's values, exposed as `vendor_sector` /
`vendor_industry`, and `classification_overridden` marks the edited rows.

One limit: figures precomputed against the vendor's grouping — the sector
margin anchors in `dcf_summary.parquet`, for one — are not recomputed by an
override. Re-run the DCF and analytics stages if you need those to follow a
re-labelled company.
