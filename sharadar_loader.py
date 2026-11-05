#!/usr/bin/env python3
"""
Filtered Sharadar ingest for the portfolio screener.

Never downloads the whole market. Three layers:

  Phase 1  build_universe()     ~14s   screens every US listing from 6 small calls
  Phase 2  download_history()   ~min   pulls history ONLY for survivors, into a cache
  Phase 3  compact()            ~s     turns the cache into query-ready Parquet

Storage is deliberately two-layer:

  cache/sharadar/{dataset}/{hash}.csv.gz   verbatim API responses, immutable
  data/{dataset}.parquet                   typed, sorted, columnar, queryable

The cache is keyed by a hash of the request parameters, so a re-run never
re-downloads anything it already has - an interrupted pull resumes for free and
costs nothing. Compaction is a pure local rebuild: delete the Parquet and
regenerate it without touching the API.

Screen (all thresholds overridable on the command line):
      currently traded on NYSE or NASDAQ
      last close                       > $2
      market cap                       > $500m
      average daily volume, last week  > 100,000 shares

Usage
      python3 sharadar_loader.py universe          # phase 1 -> universe.csv
      python3 sharadar_loader.py history           # phase 2 + 3 -> data/*.parquet
      python3 sharadar_loader.py history --limit 50
      python3 sharadar_loader.py compact           # rebuild Parquet from cache
      python3 sharadar_loader.py refresh           # daily/weekly top-up
      python3 sharadar_loader.py purge-empty       # drop no-row cache entries
      python3 sharadar_loader.py status            # what is on disk

Fields are documented at https://sharadar.com/docs
"""

from __future__ import annotations

# --------------------------------------------------------------------------
# Interpreter bootstrap.
#
# This machine has four Pythons on PATH. The default `python3` is Homebrew
# 3.14, which is PEP 668 "externally managed" - pip refuses to install into it,
# so it has no pandas. The dependencies live in .venv next to this file.
#
# Rather than require `source .venv/bin/activate` in every new shell, re-exec
# under the venv interpreter when we were started by the wrong one. Runs once,
# before any third-party import, and is a no-op inside the venv.
# --------------------------------------------------------------------------
import os
import sys
from pathlib import Path as _Path


def _reexec_in_venv() -> None:
    venv_python = _Path(__file__).resolve().parent / ".venv" / "bin" / "python"
    if not venv_python.exists() or os.environ.get("SHARADAR_NO_REEXEC"):
        return
    if _Path(sys.executable).resolve() == venv_python.resolve():
        return
    try:
        import pandas  # noqa: F401
    except ImportError:
        os.execv(str(venv_python), [str(venv_python), *sys.argv])


_reexec_in_venv()

import argparse
import gzip
import hashlib
import io
import json
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import requests

from datetime import datetime
from dateutil.relativedelta import relativedelta

# --------------------------------------------------------------------------
# configuration
# --------------------------------------------------------------------------

BASE_URL = "https://api.sharadar.com/v1.0"
HERE = Path(__file__).resolve().parent

CACHE_DIR = HERE / "cache" / "sharadar"
DATA_DIR = HERE / "data"
UNIVERSE_CSV = HERE / "universe.csv"

# Screen thresholds
MIN_CLOSE = 2.0            # USD, unadjusted - the price actually printed
MIN_MARKETCAP_MUSD = 500.0 # USD millions. Careful: the `daily` table quotes
                           # marketcap in millions, but the `fundamentals`
                           # table quotes it in raw dollars. The screen reads
                           # `daily`, so 500 here means $500m.
MIN_AVG_WEEK_VOLUME = 100_000  # shares, averaged per session over the trailing
                           # week. With VOLUME_AGG="mean" this is ~5x stricter
                           # than the same number applied to the weekly total.
VOLUME_WINDOW_SESSIONS = 5 # "last week" = 5 trading sessions
VOLUME_AGG = "mean"        # "mean" for per-day average, or "sum" over the week

EXCHANGES = ("NYSE", "NASDAQ")
CATEGORY_PATTERN = r"Common Stock"
INCLUDE_ADR = True
INCLUDE_SECONDARY_CLASS = False

# History depth. Your Sharadar tier caps this regardless (5y / 10y / full);
# asking for more than you pay for simply returns less.
# Anchored to January 1 rather than to "ten years ago today" on purpose. The
# request cache is keyed by a hash of the request, and download_history() cuts
# the range into fixed-length windows measured FORWARD FROM THIS DATE. A start
# that moves every day moves every window boundary with it, so no cache entry
# written yesterday can ever be hit today and a re-run silently re-downloads
# the entire ten years. (Both generations of that are still on disk here: one
# set of windows anchored at 2016-09-06, another at Jan 1.) Snapping to Jan 1
# keeps the boundaries - and therefore the cache - stable for a whole year,
# while still buying at least ten years of depth.
HISTORY_START = date(datetime.today().year - 10, 1, 1).isoformat()

# AR* are as-reported / point-in-time and are the only honest choice for
# backtesting. MR* fold in restatements the market had not seen yet.
#
#   ARQ  discrete quarter        ART  trailing twelve months        ARY  fiscal year
#
# ARY is the vendor's own annual roll-up. The statements page presents it
# as-is rather than summing four ARQ quarters, because a summed year is wrong
# wherever a company has a 52/53-week fiscal calendar, restates mid-year, or
# reports a stub period - and because the balance-sheet lines are stocks, not
# flows, so summing them is meaningless in the first place.
DIMENSIONS = ("ARQ", "ART", "ARY")

# The API rejects long ticker lists with a 400. Empirically the ceiling sits
# between 30 and 40; 25 leaves headroom for long symbols like "BRK.B".
TICKER_BATCH = 25
PRICE_WINDOW_DAYS = 366    # 25 tickers x 1 year ~ 6,250 rows < PAGE_LIMIT
MAX_WORKERS = 4            # parallel batches; the API tolerates this comfortably
PAGE_LIMIT = 10_000        # API default and maximum per request
REQUEST_PAUSE = 0.10
MAX_RETRIES = 4

# How far back a `refresh` re-fetches before the last stored session. Covers
# late-arriving filings and vendor restatements - see refresh().
REFRESH_OVERLAP_DAYS = 7


def get_api_key() -> str:
    """SHARADAR_API_KEY, else KEY from .env, else the free DJIA-only test key."""
    for var in ("SHARADAR_API_KEY", "KEY"):
        if os.environ.get(var):
            return os.environ[var].strip()

    env_file = HERE / ".env"
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            name, _, value = line.partition("=")
            if name.strip() in ("SHARADAR_API_KEY", "KEY"):
                return value.strip().strip("'\"")

    print(
        "! No API key found - falling back to 'test-api-key', which only "
        "returns the 30 Dow Jones companies.\n"
        "  Set SHARADAR_API_KEY in .env.",
        file=sys.stderr,
    )
    return "test-api-key"


API_KEY = get_api_key()
SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "portfolio-screener/1.0"})


def _redact(text: str) -> str:
    """Strip the API key from anything that might be printed or raised."""
    text = re.sub(r"(api_key|apiKey)=[^&\s]+", r"\1=REDACTED", text)
    return text.replace(API_KEY, "REDACTED") if API_KEY else text


# --------------------------------------------------------------------------
# transport + cache
# --------------------------------------------------------------------------

def _cache_path(dataset: str, params: dict) -> Path:
    """Cache key = hash of the request, minus the credential."""
    material = json.dumps({k: v for k, v in sorted(params.items())}, sort_keys=True)
    digest = hashlib.sha1(material.encode()).hexdigest()[:16]
    return CACHE_DIR / dataset / f"{digest}.csv.gz"


def _request(dataset: str, params: dict) -> pd.DataFrame:
    """One page. Returns an empty frame when the API has nothing to say."""
    url = f"{BASE_URL}/data/{dataset}"
    query = {**params, "api_key": API_KEY, "format": "csv"}

    for attempt in range(MAX_RETRIES):
        try:
            response = SESSION.get(url, params=query, timeout=180)
        except requests.RequestException as exc:
            if attempt == MAX_RETRIES - 1:
                raise SystemExit(f"/{dataset} unreachable: {_redact(str(exc))}")
            time.sleep(2 ** attempt)
            continue

        if response.status_code in (429, 500, 502, 503, 504):
            wait = 2 ** attempt
            print(f"  {response.status_code} on /{dataset}, retry in {wait}s", file=sys.stderr)
            time.sleep(wait)
            continue

        if response.status_code == 400:
            raise SystemExit(
                f"400 from /{dataset}. Usually too many tickers in one request - "
                f"lower TICKER_BATCH (currently {TICKER_BATCH})."
            )
        if response.status_code == 403:
            raise SystemExit(
                f"403 from /{dataset}. Either the key is wrong, or your plan does "
                f"not cover this table or this far back."
            )
        if not response.ok:
            raise SystemExit(f"{response.status_code} from /{dataset}: "
                             f"{_redact(response.text[:300])}")

        text = response.text.strip()
        if not text:
            return pd.DataFrame()
        try:
            return pd.read_csv(io.StringIO(text))
        except pd.errors.EmptyDataError:
            return pd.DataFrame()

    raise SystemExit(f"/{dataset} kept failing after {MAX_RETRIES} attempts.")


def fetch(dataset: str, *, use_cache: bool = True, **params) -> pd.DataFrame:
    """Paged, cached fetch.

    `dataset` is the Sharadar table being read. It is deliberately not called
    `table`, because `table` is itself a query parameter on /data/tickers.

    WARNING: if this has to page (>10,000 rows), pass a `sort` on a UNIQUE
    column. Without a total ordering the server may return rows in a different
    order between pages, which silently both duplicates and DROPS rows. Prefer
    fetch_windowed(), which sizes each request to fit in a single page.
    """
    path = _cache_path(dataset, params)
    if use_cache and path.exists():
        with gzip.open(path, "rt") as handle:
            try:
                return pd.read_csv(handle)
            except pd.errors.EmptyDataError:
                return pd.DataFrame()

    pages, skip = [], 0
    while True:
        page = _request(dataset, {**params, "limit": PAGE_LIMIT, "skip": skip})
        if page.empty:
            break
        pages.append(page)
        if len(page) < PAGE_LIMIT:
            break
        skip += PAGE_LIMIT
        time.sleep(REQUEST_PAUSE)

    frame = pd.concat(pages, ignore_index=True) if pages else pd.DataFrame()

    # An empty response is never cached. "No rows" is not a fact about the
    # data, it is a fact about *when you asked*: a top-up run started before
    # the vendor has published the session returns nothing, and caching that
    # pins the answer to this cache key forever. Because the key is a hash of
    # the request - and a same-day re-run makes the identical request - the
    # second, later run would be served the empty frame from disk and never
    # reach the API. That is the silent-staleness bug: the loader reports
    # "cached already" and the parquet never moves past the old last session.
    # Leaving the miss uncached costs one wasted request per re-run and makes
    # re-running always correct.
    if use_cache and not frame.empty:
        path.parent.mkdir(parents=True, exist_ok=True)
        with gzip.open(path, "wt", compresslevel=6) as handle:
            frame.to_csv(handle, index=False)
    return frame


def fetch_windowed(dataset: str, start: str, end: str, *, window_days: int = 366,
                   use_cache: bool = True, **params) -> pd.DataFrame:
    """Fetch a date range in windows small enough that no request ever pages.

    This exists because `skip` pagination is not stable on this API. Measured:
    a 10-year pull of 2,565 tickers came back with 565 duplicated rows and
    exactly 565 missing ones - the same rows shifted across a page boundary,
    so each was served twice to one page and skipped by the next.

    A one-year window of 25 tickers is ~6,250 rows, comfortably inside the
    10,000-row page limit, so `skip` is never used and nothing can shift.
    """
    frames = []
    cursor = pd.Timestamp(start)
    stop = pd.Timestamp(end)
    while cursor <= stop:
        window_end = min(cursor + pd.Timedelta(days=window_days - 1), stop)
        frame = fetch(dataset, use_cache=use_cache,
                      **{**params, "from": cursor.date().isoformat(),
                         "to": window_end.date().isoformat()})
        if not frame.empty:
            if len(frame) >= PAGE_LIMIT:
                raise SystemExit(
                    f"/{dataset} window {cursor.date()}..{window_end.date()} hit the "
                    f"{PAGE_LIMIT}-row page limit. Lower window_days or TICKER_BATCH."
                )
            frames.append(frame)
        cursor = window_end + pd.Timedelta(days=1)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _window_jobs(dataset: str, start: str, end: str, window_days: int,
                 **params) -> list[tuple[str, dict]]:
    """The individual single-page requests fetch_windowed would make."""
    jobs = []
    cursor, stop = pd.Timestamp(start), pd.Timestamp(end)
    while cursor <= stop:
        window_end = min(cursor + pd.Timedelta(days=window_days - 1), stop)
        jobs.append((dataset, {**params, "from": cursor.date().isoformat(),
                               "to": window_end.date().isoformat()}))
        cursor = window_end + pd.Timedelta(days=1)
    return jobs


# --------------------------------------------------------------------------
# phase 1 - the screen
# --------------------------------------------------------------------------

def load_listed_tickers() -> pd.DataFrame:
    """The securities master, cut down to live NYSE/NASDAQ common stock."""
    # A ticker appears in the master once per data table it belongs to, so
    # table="stocks" both restricts us to equities and deduplicates. Note the
    # modern table names - the legacy codes (SEP, SF1) return nothing.
    # ~20k rows, so this pages. permaticker is unique, which is what makes
    # skip-pagination safe here (see the warning on fetch).
    master = fetch("tickers", use_cache=False,
                   **{"table": "stocks", "sort": "permaticker.asc"})
    if master.empty:
        raise SystemExit("tickers table came back empty - check the API key.")

    keep = (
        master["exchange"].isin(EXCHANGES)
        & master["isdelisted"].eq("N")
        & master["category"].str.contains(CATEGORY_PATTERN, na=False)
    )
    if "currency" in master.columns:
        keep &= master["currency"].fillna("USD").eq("USD")
    if not INCLUDE_ADR:
        keep &= ~master["category"].str.contains("ADR", na=False)
    if not INCLUDE_SECONDARY_CLASS:
        keep &= ~master["category"].str.contains("Secondary", na=False)

    listed = master.loc[keep].copy()

    # A ticker can be flagged live but have stopped printing prices.
    if "lastpricedate" in listed.columns:
        last_price = pd.to_datetime(listed["lastpricedate"], errors="coerce")
        listed = listed.loc[last_price >= pd.Timestamp.today() - pd.Timedelta(days=21)]

    columns = [
        c for c in ("ticker", "permaticker", "name", "exchange", "category",
                    "sector", "industry", "siccode", "lastpricedate")
        if c in listed.columns
    ]
    return listed[columns].reset_index(drop=True)


def load_recent_bars(calendar_days: int = 21) -> pd.DataFrame:
    """Every ticker's last few weeks of bars, in one paged call."""
    end = date.today()
    start = end - timedelta(days=calendar_days)
    # ~6,000 tickers x 15 sessions would page, so take one session per
    # request: ~6,000 rows each, always a single page.
    bars = fetch_windowed(
        "stocks", start.isoformat(), end.isoformat(), window_days=1, use_cache=False,
        **{"fields": "ticker,date,close,closeunadj,volume"},
    )
    if bars.empty:
        raise SystemExit("stocks returned no recent bars - is the plan active?")
    bars["date"] = pd.to_datetime(bars["date"])
    return bars


def summarise_recent(bars: pd.DataFrame) -> pd.DataFrame:
    """Latest close and trailing-week volume, per ticker."""
    sessions = sorted(bars["date"].unique())
    window = sessions[-VOLUME_WINDOW_SESSIONS:]
    latest = sessions[-1]

    recent = bars[bars["date"].isin(window)]
    volume = (
        recent.groupby("ticker")["volume"].agg(VOLUME_AGG)
        .rename(f"volume_{VOLUME_AGG}_{VOLUME_WINDOW_SESSIONS}d")
    )
    sessions_traded = recent.groupby("ticker")["date"].nunique().rename("sessions_in_window")

    last_bar = (
        bars[bars["date"] == latest].set_index("ticker")[["close", "closeunadj"]]
        .rename(columns={"close": "close_split_adj", "closeunadj": "close_raw"})
    )

    summary = last_bar.join([volume, sessions_traded])
    summary["as_of"] = pd.Timestamp(latest).date().isoformat()
    return summary.reset_index()


def load_marketcap(as_of: str) -> pd.DataFrame:
    """Market cap for every ticker on the given date, in USD millions."""
    fields = "ticker,date,marketcap,pe,pb,ev"
    daily = fetch("daily", use_cache=False, **{"from": as_of, "to": as_of, "fields": fields})
    if daily.empty:
        # The daily table can lag the price table by a session.
        fallback = (pd.Timestamp(as_of) - pd.Timedelta(days=5)).date().isoformat()
        daily = fetch("daily", use_cache=False,
                      **{"from": fallback, "to": as_of, "fields": fields})
        if daily.empty:
            raise SystemExit(f"daily table returned nothing around {as_of}.")
        daily = daily.sort_values("date").groupby("ticker", as_index=False).last()
    return daily.rename(columns={"date": "marketcap_date"})


def build_universe(verbose: bool = True) -> pd.DataFrame:
    """Phase 1. Returns the screened universe and writes universe.csv."""
    listed = load_listed_tickers()
    if verbose:
        print(f"  listed NYSE/NASDAQ common stock : {len(listed):>6,}")

    bars = load_recent_bars()
    summary = summarise_recent(bars)
    as_of = summary["as_of"].iloc[0]
    if verbose:
        print(f"  price/volume as of              : {as_of}")

    caps = load_marketcap(as_of)
    universe = listed.merge(summary, on="ticker", how="inner").merge(caps, on="ticker", how="left")

    volume_column = f"volume_{VOLUME_AGG}_{VOLUME_WINDOW_SESSIONS}d"
    volume_label = "avg daily volume" if VOLUME_AGG == "mean" else "week volume"
    tests = {
        f"close > ${MIN_CLOSE:g}": universe["close_raw"] > MIN_CLOSE,
        f"mktcap > ${MIN_MARKETCAP_MUSD:g}m": universe["marketcap"] > MIN_MARKETCAP_MUSD,
        f"{volume_label} > {MIN_AVG_WEEK_VOLUME:,.0f}":
            universe[volume_column] > MIN_AVG_WEEK_VOLUME,
        "traded all week": universe["sessions_in_window"] >= VOLUME_WINDOW_SESSIONS,
    }

    if verbose:
        print(f"  with price and market cap       : {len(universe):>6,}")
        for label, mask in tests.items():
            print(f"    {label:<28} {int(mask.sum()):>6,} pass")

    passed = pd.Series(True, index=universe.index)
    for mask in tests.values():
        passed &= mask.fillna(False)

    screened = universe.loc[passed].sort_values("marketcap", ascending=False).reset_index(drop=True)
    screened.to_csv(UNIVERSE_CSV, index=False)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    screened.to_parquet(DATA_DIR / "universe.parquet", index=False)

    if verbose:
        print(f"\n  universe: {len(screened):,} tickers -> {UNIVERSE_CSV.name}")
    return screened


# --------------------------------------------------------------------------
# phase 2 - history, only for what survived
# --------------------------------------------------------------------------

def _batches(items: list[str], size: int) -> list[list[str]]:
    return [items[i:i + size] for i in range(0, len(items), size)]


def download_history(
    tickers: list[str] | None = None,
    start: str = HISTORY_START,
    workers: int = MAX_WORKERS,
) -> None:
    """Phase 2. Fills the cache. Already-cached batches cost nothing."""
    if tickers is None:
        if not UNIVERSE_CSV.exists():
            raise SystemExit("No universe.csv yet. Run: sharadar_loader.py universe")
        tickers = pd.read_csv(UNIVERSE_CSV)["ticker"].tolist()

    end = date.today().isoformat()
    batches = _batches(tickers, TICKER_BATCH)

    jobs: list[tuple[str, dict]] = []
    for batch in batches:
        joined = ",".join(batch)
        # Prices: one request per ticker-batch per year (~6,250 rows), so no
        # request ever pages. See fetch_windowed for why that matters.
        jobs += _window_jobs("stocks", start, end, PRICE_WINDOW_DAYS, ticker=joined)
        # Fundamentals: 25 tickers x ~40 quarters = ~1,000 rows, single page.
        for dimension in DIMENSIONS:
            jobs.append(("fundamentals",
                         {"ticker": joined, "dimension": dimension, "from": start, "to": end}))

    pending = [j for j in jobs if not _cache_path(j[0], j[1]).exists()]
    print(f"{len(tickers):,} tickers -> {len(batches)} batches, {len(jobs)} requests")
    print(f"  cached already : {len(jobs) - len(pending)}")
    print(f"  to fetch       : {len(pending)}")
    if not pending:
        return

    started = time.time()
    done = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(fetch, dataset, **params): (dataset, params)
                   for dataset, params in pending}
        for future in as_completed(futures):
            frame = future.result()
            done += 1
            if done % 25 == 0 or done == len(pending):
                elapsed = time.time() - started
                rate = done / elapsed
                remaining = (len(pending) - done) / rate if rate else 0
                print(f"  {done:>4}/{len(pending)}  {elapsed:>6.0f}s elapsed, "
                      f"~{remaining:>4.0f}s left")

    print(f"  fetched in {time.time() - started:.0f}s")


# --------------------------------------------------------------------------
# phase 3 - compaction into columnar storage
# --------------------------------------------------------------------------

# Downcasting is where the size win lives: float64 -> float32 halves every
# numeric column, and ticker as a dictionary column costs ~2 bytes per row
# instead of ~6. Prices do not need 15 significant digits.
def _dedupe(frame: pd.DataFrame, key: list[str]) -> pd.DataFrame:
    """Collapse each natural key to the vendor's most recent version of it.

    A full-row drop_duplicates() is not enough once refresh() re-reads days it
    already has. Sharadar revises published bars - between 4 and 7 September
    it moved A's 2026-09-03 high from 154.300003 to 154.294998 and AA's volume
    from 2,554,000 to 2,553,000 - so the re-read row and the stored row differ
    in a field or two and both survive. Downstream that is fatal rather than
    merely untidy: valuation_metrics pivots prices to date x ticker, and a
    duplicated (ticker, date) makes the reshape raise outright.

    Ordering by `lastupdated` and keeping the last one keeps the correction
    and drops what it corrected. Rows with no `lastupdated` keep their
    existing order, so this degrades to first-wins rather than to an error.
    """
    key = [c for c in key if c in frame.columns]
    if not key:
        return frame
    if "lastupdated" in frame.columns:
        frame = frame.sort_values("lastupdated", kind="stable", na_position="first")
    return frame.drop_duplicates(subset=key, keep="last")


def _optimise(frame: pd.DataFrame, sort_by: list[str], key: list[str]) -> pd.DataFrame:
    frame = _dedupe(frame.drop_duplicates(), key)
    for column in frame.columns:
        if column in ("date", "calendardate", "reportperiod", "lastupdated", "datekey"):
            frame[column] = pd.to_datetime(frame[column], errors="coerce")
        elif frame[column].dtype == "float64":
            frame[column] = frame[column].astype("float32")
        elif frame[column].dtype == "object":
            if frame[column].nunique(dropna=True) < len(frame) / 4:
                frame[column] = frame[column].astype("category")
    keys = [c for c in sort_by if c in frame.columns]
    return frame.sort_values(keys).reset_index(drop=True) if keys else frame


def compact(dataset: str, sort_by: list[str], key: list[str],
            verbose: bool = True) -> Path:
    """Rebuild data/{dataset}.parquet from the cache. Never touches the API."""
    source = CACHE_DIR / dataset
    files = sorted(source.glob("*.csv.gz"))
    if not files:
        raise SystemExit(f"Nothing cached for '{dataset}'. Run the history command first.")

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    target = DATA_DIR / f"{dataset}.parquet"

    chunks, rows = [], 0
    for path in files:
        with gzip.open(path, "rt") as handle:
            try:
                chunk = pd.read_csv(handle)
            except pd.errors.EmptyDataError:
                continue
        if not chunk.empty:
            chunks.append(chunk)
            rows += len(chunk)

    combined = pd.concat(chunks, ignore_index=True)
    frame = _optimise(combined, sort_by, key)
    dropped = len(combined) - len(frame)

    # ZSTD beats snappy by roughly 30% here at negligible read cost, and
    # sorting by ticker first lets row-group statistics skip whole blocks
    # when a query filters on one name.
    pq.write_table(
        pa.Table.from_pandas(frame, preserve_index=False),
        target, compression="zstd", compression_level=6, row_group_size=250_000,
    )

    if verbose:
        raw = sum(p.stat().st_size for p in files) / 1e6
        out = target.stat().st_size / 1e6
        print(f"  {dataset:<13} {len(frame):>9,} rows  "
              f"cache {raw:>6.1f} MB -> parquet {out:>6.1f} MB  "
              f"({raw / out:.1f}x)  {dropped:,} duplicate or superseded rows folded")
    return target


def compact_all() -> None:
    # The key is what makes a row unique, which is not the same as the sort.
    # For fundamentals the filing date belongs in the key: one calendardate
    # legitimately has several filings when a company amends, and those are
    # distinct point-in-time observations, not duplicates.
    compact("stocks", ["ticker", "date"], ["ticker", "date"])
    compact("fundamentals", ["ticker", "dimension", "date"],
            ["ticker", "dimension", "calendardate", "date"])


# --------------------------------------------------------------------------
# querying
# --------------------------------------------------------------------------

def connect():
    """DuckDB connection with `prices`, `fundamentals` and `universe` views.

    DuckDB reads the Parquet files directly - there is no import step and no
    second copy of the data on disk.
    """
    import duckdb

    connection = duckdb.connect()
    for view, filename in (("prices", "stocks.parquet"),
                           ("fundamentals", "fundamentals.parquet"),
                           ("universe", "universe.parquet")):
        path = DATA_DIR / filename
        if path.exists():
            connection.execute(
                f"CREATE VIEW {view} AS SELECT * FROM read_parquet('{path}')")
    return connection


def as_of_snapshot(as_of: str, dimension: str = "ART") -> pd.DataFrame:
    """Everything knowable on `as_of`, one row per ticker - the rebalance input.

    Joins each ticker's most recent close on or before `as_of` to the most
    recent fundamentals *filed* on or before `as_of`. Filtering on the filing
    date rather than the period end is what keeps this free of lookahead: Q2
    numbers are not usable until the 10-Q lands, often 40+ days later.
    """
    connection = connect()
    query = """
        WITH px AS (
            SELECT *, ROW_NUMBER() OVER (PARTITION BY ticker ORDER BY date DESC) AS rn
            FROM prices WHERE date <= ?
        ),
        fx AS (
            SELECT * EXCLUDE (lastupdated),
                   ROW_NUMBER() OVER (PARTITION BY ticker ORDER BY date DESC) AS rn
            FROM fundamentals WHERE date <= ? AND dimension = ?
        )
        SELECT px.* EXCLUDE (rn, lastupdated),
               fx.* EXCLUDE (rn, ticker, date),
               fx.date AS filed_on
        FROM px LEFT JOIN fx ON fx.ticker = px.ticker AND fx.rn = 1
        WHERE px.rn = 1
        ORDER BY px.ticker
    """
    frame = connection.execute(query, [as_of, as_of, dimension]).df()
    connection.close()
    return frame


def rebalance_dates(start: str, end: str, weekday: str = "FRI") -> list[str]:
    """Trading sessions to rebalance on - the last session of each week."""
    connection = connect()
    frame = connection.execute("""
        SELECT MAX(date) AS d FROM prices
        WHERE date BETWEEN ? AND ?
        GROUP BY date_trunc('week', date) ORDER BY d
    """, [start, end]).df()
    connection.close()
    return [d.date().isoformat() for d in frame["d"]]


# --------------------------------------------------------------------------

def purge_empty_cache(verbose: bool = True) -> int:
    """Delete cached responses that hold no rows.

    Pre-dates the "never cache an empty frame" rule in fetch(), and is still
    worth running: a cache poisoned by an earlier version of this file will
    otherwise keep answering top-up requests with nothing forever.
    """
    removed = 0
    for path in CACHE_DIR.glob("*/*.csv.gz"):
        try:
            with gzip.open(path, "rt") as handle:
                frame = pd.read_csv(handle)
            empty = frame.empty
        except (pd.errors.EmptyDataError, EOFError, OSError):
            empty = True          # truncated or zero-length: no use to anyone
        if empty:
            path.unlink()
            removed += 1
    if verbose:
        print(f"  purged {removed} empty cache entries")
    return removed


def refresh(start: str | None = None, overlap_days: int = REFRESH_OVERLAP_DAYS) -> None:
    """Top-up: re-fetch the tail of the stored history and recompact.

    The window deliberately OVERLAPS what is already stored instead of
    starting the day after it, for two reasons:

      * Filing dates, not price dates. download_history() drives both tables
        off one `start`, but /data/fundamentals filters `from`/`to` on the
        FILING date. Starting at last-price + 1 permanently loses any filing
        that landed on the last stored session after the previous run - the
        window never covers that day again.
      * Restatements. Sharadar revises published rows; a strictly
        forward-only window never sees a correction to a day it already has.

    Re-fetched rows are identical to the stored ones in the normal case, and
    compact() drops the duplicates, so the overlap costs bandwidth and
    nothing else.
    """
    connection = connect()
    last = connection.execute("SELECT MAX(date) FROM prices").fetchone()[0]
    connection.close()
    if last is None:
        raise SystemExit("Nothing stored yet - run the history command first.")

    purge_empty_cache()

    start = start or (pd.Timestamp(last)
                      - pd.Timedelta(days=overlap_days)).date().isoformat()
    tickers = pd.read_csv(UNIVERSE_CSV)["ticker"].tolist()
    print(f"stored through {pd.Timestamp(last).date()}; "
          f"fetching from {start} ({overlap_days}d overlap)")

    # Names the screen has only just admitted have no stored history at all,
    # and the short top-up window would give them a few days of prices - not
    # enough for any trailing metric or for the DCF, so they would show up on
    # the page as blanks. Backfill those from HISTORY_START before the top-up.
    connection = connect()
    known = set(connection.execute("SELECT DISTINCT ticker FROM prices").df()["ticker"])
    connection.close()
    newcomers = [t for t in tickers if t not in known]
    if newcomers:
        print(f"  {len(newcomers)} new to the universe - backfilling full history")
        download_history(newcomers, start=HISTORY_START)

    download_history(tickers, start=start)
    compact_all()

    connection = connect()
    now = connection.execute("SELECT MAX(date) FROM prices").fetchone()[0]
    connection.close()
    print(f"  prices now stored through {pd.Timestamp(now).date()}")


def status() -> None:
    """What is on disk, and how big."""
    print(f"universe : ", end="")
    if UNIVERSE_CSV.exists():
        print(f"{len(pd.read_csv(UNIVERSE_CSV)):,} tickers")
    else:
        print("not built")

    for dataset in ("tickers", "stocks", "daily", "fundamentals"):
        folder = CACHE_DIR / dataset
        files = list(folder.glob("*.csv.gz")) if folder.exists() else []
        size = sum(p.stat().st_size for p in files) / 1e6
        print(f"cache/{dataset:<13} {len(files):>5} files  {size:>8.1f} MB")

    if DATA_DIR.exists():
        for path in sorted(DATA_DIR.glob("*.parquet")):
            meta = pq.ParquetFile(path).metadata
            print(f"data/{path.name:<20} {meta.num_rows:>10,} rows  "
                  f"{path.stat().st_size / 1e6:>8.1f} MB")


def main() -> None:
    global MIN_CLOSE, MIN_MARKETCAP_MUSD, MIN_AVG_WEEK_VOLUME

    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command",
                        choices=["universe", "history", "compact", "refresh",
                                 "purge-empty", "status"])
    parser.add_argument("--limit", type=int, help="only process the first N tickers")
    parser.add_argument("--start", default=HISTORY_START, help="history start date")
    parser.add_argument("--workers", type=int, default=MAX_WORKERS)
    parser.add_argument("--min-close", type=float, default=MIN_CLOSE)
    parser.add_argument("--min-marketcap", type=float, default=MIN_MARKETCAP_MUSD,
                        help="USD millions")
    parser.add_argument("--min-volume", type=float, default=MIN_AVG_WEEK_VOLUME,
                        help=f"shares, {VOLUME_AGG} over "
                             f"{VOLUME_WINDOW_SESSIONS} sessions")
    args = parser.parse_args()

    MIN_CLOSE = args.min_close
    MIN_MARKETCAP_MUSD = args.min_marketcap
    MIN_AVG_WEEK_VOLUME = args.min_volume

    if args.command == "universe":
        build_universe()

    elif args.command == "history":
        if not UNIVERSE_CSV.exists():
            print("No universe.csv - running the screen first.\n")
            build_universe()
        tickers = pd.read_csv(UNIVERSE_CSV)["ticker"].tolist()
        if args.limit:
            tickers = tickers[:args.limit]
            print(f"(limited to the {len(tickers)} largest names)")
        download_history(tickers, start=args.start, workers=args.workers)
        print()
        compact_all()

    elif args.command == "compact":
        compact_all()

    elif args.command == "refresh":
        refresh()

    elif args.command == "purge-empty":
        purge_empty_cache()

    elif args.command == "status":
        status()


if __name__ == "__main__":
    main()
