#!/usr/bin/env python3
"""
Cached data access for the Streamlit app.

Every loader is wrapped in st.cache_data so a page rerun - which Streamlit does
on every widget interaction - re-reads nothing. Parquet is read once per
session; live Sharadar calls are additionally cached on disk by
sharadar_loader.fetch, so they cost one request ever.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
import streamlit as st

import analytics
import classification_overrides
import currency
import current_multiples as cm
import dcf_forecast as dcf
import valuation_metrics as vm
from sharadar_loader import DATA_DIR, connect, fetch

CACHE_TTL = 3600


# --------------------------------------------------------------------------
# local parquet
# --------------------------------------------------------------------------

#: Columns `analytics.parquet` does not carry but the repriced multiples need.
#: Pulled from the metrics table rather than added to the analytics build, so
#: this works against a parquet written before any of this existed.
_METRICS_EXTRAS = ("equity_book", "ebit_ttm", "fcff_ttm", "fcfe_ttm",
                   "net_debt", "total_debt", "cashneq", "cfo_ttm", "capex_ttm",
                   "shareswadil", "fx_to_usd")


@st.cache_data(ttl=CACHE_TTL)
def _latest_metrics() -> pd.DataFrame:
    """The most recent filed metrics row per ticker, whole rows only.

    Whole rows because GroupBy.last() would assemble one from several filings -
    see valuation_metrics.latest, which this mirrors.
    """
    frame = vm.latest()
    keep = ["ticker", "date", "price"] + [c for c in _METRICS_EXTRAS
                                          if c in frame.columns]
    return frame[keep].rename(columns={"date": "metrics_date",
                                       "price": "price_at_filing"})


@st.cache_data(ttl=CACHE_TTL)
def _closes_around_filing() -> pd.DataFrame:
    """Latest close and the close on each ticker's last filing date.

    Both from the split-adjusted `close` column, which is what makes the ratio
    between them a valid market-cap scale factor across a split. See
    current_multiples for why `closeadj` would be the wrong column here.
    """
    prices = (DATA_DIR / "stocks.parquet").as_posix()
    metrics = (DATA_DIR / "metrics.parquet").as_posix()
    connection = connect()
    frame = connection.execute(f"""
        WITH last_filing AS (
            SELECT ticker, max(date) AS filing_date
            FROM read_parquet('{metrics}') GROUP BY ticker
        ),
        latest_bar AS (
            SELECT ticker, date AS close_now_date, close AS close_now
            FROM read_parquet('{prices}')
            QUALIFY row_number() OVER (PARTITION BY ticker ORDER BY date DESC) = 1
        ),
        filed_bar AS (
            SELECT f.ticker, p.close AS close_at_filing
            FROM last_filing f
            JOIN read_parquet('{prices}') p
              ON p.ticker = f.ticker AND p.date <= f.filing_date
            QUALIFY row_number() OVER (
                PARTITION BY f.ticker ORDER BY p.date DESC) = 1
        )
        SELECT l.ticker, l.close_now, l.close_now_date, b.close_at_filing
        FROM latest_bar l LEFT JOIN filed_bar b USING (ticker)
    """).df()
    connection.close()
    return frame


@st.cache_data(ttl=CACHE_TTL)
def _load_analytics_raw() -> pd.DataFrame:
    """Analytics with every multiple rebuilt against the latest close.

    The stored multiples pair each filing's earnings with the market cap on
    that filing date. That is right for a point-in-time history and wrong for
    a screen, which has to rank on a price someone can trade - so the screen
    columns are repriced here and the filing-date values are kept alongside
    them under a `_at_filing` suffix.
    """
    frame = analytics.load()
    extras = _latest_metrics()
    frame = frame.merge(
        extras[[c for c in extras.columns if c not in frame.columns
                or c == "ticker"]], on="ticker", how="left")

    closes = _closes_around_filing()
    frame = frame.merge(closes, on="ticker", how="left")
    scale = cm.price_scale(frame["close_now"], frame["close_at_filing"])
    return cm.reprice(frame, scale)


def load_analytics() -> pd.DataFrame:
    """Analytics as stored, with manual sector / industry edits applied.

    The parquet read stays cached; the overrides are re-applied on every call
    so saving an edit takes effect on the next rerun without having to throw
    away the (expensive) parquet read. A data refresh rewrites the parquet
    with the vendor's labels again - re-applying here is what makes the edits
    survive it.
    """
    return classification_overrides.apply(_load_analytics_raw())


@st.cache_data(ttl=CACHE_TTL)
def load_summary() -> pd.DataFrame:
    return pd.read_parquet(DATA_DIR / "dcf_summary.parquet")


@st.cache_data(ttl=CACHE_TTL)
def load_drivers() -> pd.DataFrame:
    return dcf.load_drivers()


@st.cache_data(ttl=CACHE_TTL)
def load_market_data() -> pd.DataFrame:
    return vm.load_market_data().reset_index()


@st.cache_data(ttl=CACHE_TTL)
def load_prices(ticker: str) -> pd.DataFrame:
    """One ticker's daily bars, oldest first.

    All three close columns, because they answer different questions and the
    portfolio page needs all three at once. `close` is split-adjusted and is
    what a holding is worth; `closeadj` adds dividends back and is what a
    return or a volatility is measured on; `closeunadj` is the quote as it was
    printed, and its ratio to `close` is the cumulative split factor - which is
    what restates a trade entered in pre-split shares.
    """
    connection = connect()
    frame = connection.execute(
        "SELECT date, open, high, low, close, closeadj, closeunadj, volume "
        "FROM prices WHERE ticker = ? ORDER BY date", [ticker]).df()
    connection.close()
    return frame


@st.cache_data(ttl=CACHE_TTL)
def load_metrics_history(ticker: str) -> pd.DataFrame:
    connection = connect()
    path = (DATA_DIR / "metrics.parquet").as_posix()
    frame = connection.execute(
        f"SELECT * FROM read_parquet('{path}') WHERE ticker = ? ORDER BY date",
        [ticker]).df()
    connection.close()
    return frame


@st.cache_data(ttl=CACHE_TTL)
def load_cashflow_history(ticker: str) -> pd.DataFrame:
    connection = connect()
    path = (DATA_DIR / "cash_flows.parquet").as_posix()
    frame = connection.execute(
        f"SELECT * FROM read_parquet('{path}') WHERE ticker = ? ORDER BY date",
        [ticker]).df()
    connection.close()
    return frame


@st.cache_data(ttl=CACHE_TTL)
def load_statements(ticker: str, dimension: str,
                    basis: str = currency.USD) -> pd.DataFrame:
    """Every filed statement for one ticker on one dimension, oldest first.

    ARQ is the discrete quarter, ART the trailing twelve months, ARY the
    fiscal year - all three as filed, straight from the vendor. Nothing here
    sums or re-derives a period: the aggregates the statements page prints as
    subtotals (gross profit, operating income, current assets, net cash from
    operations, EBITDA, EV) are columns Sharadar already publishes.

    `basis` picks the currency. On the default, every reporting-currency line
    is restated in US dollars at the filing's own rate before it reaches the
    page, so the derived lines the page computes on top - working-capital
    movements, section residuals, net debt - are built from dollars and agree
    with the statement they sit in. Restating afterwards would leave those
    derived lines in whatever currency they were differenced from.

    One `calendardate` can carry several filings when a company amends, so
    the last filing for each period wins - a statement should show what the
    company currently says the period was, not its first draft. The
    superseded filings stay in the parquet; they are what makes the rest of
    the app point-in-time.
    """
    connection = connect()
    path = (DATA_DIR / "fundamentals.parquet").as_posix()
    frame = connection.execute(
        f"SELECT * FROM read_parquet('{path}') "
        f"WHERE ticker = ? AND dimension = ? ORDER BY calendardate, date",
        [ticker, dimension]).df()
    connection.close()
    if frame.empty:
        return frame
    frame = (frame.drop_duplicates(subset=["calendardate"], keep="last")
                  .reset_index(drop=True))
    return currency.convert(frame, basis)


@st.cache_data(ttl=CACHE_TTL)
def load_forecast(ticker: str) -> pd.DataFrame:
    connection = connect()
    path = (DATA_DIR / "dcf_forecast.parquet").as_posix()
    frame = connection.execute(
        f"SELECT * FROM read_parquet('{path}') WHERE ticker = ? ORDER BY year",
        [ticker]).df()
    connection.close()
    return frame


# --------------------------------------------------------------------------
# live Sharadar tables
# --------------------------------------------------------------------------

@st.cache_data(ttl=CACHE_TTL)
def corporate_actions(ticker: str, years: int = 5) -> pd.DataFrame:
    start = (pd.Timestamp.today() - pd.DateOffset(years=years)).date().isoformat()
    frame = fetch("actions", ticker=ticker,
                  **{"from": start, "to": pd.Timestamp.today().date().isoformat()})
    return frame.sort_values("date", ascending=False) if not frame.empty else frame


@st.cache_data(ttl=CACHE_TTL)
def institutional_holdings(ticker: str) -> pd.DataFrame:
    # use_cache=False on purpose. The other live loaders on this page carry a
    # `to=today` in their request, so their disk-cache key rotates daily and
    # they refresh themselves. This request has no date parameter at all, so
    # its key never changes - with the disk cache on, the 13F holdings a
    # ticker was first looked up with would be served forever. The
    # st.cache_data wrapper above still holds the answer for CACHE_TTL, so
    # skipping the disk layer costs at most one request per ticker per hour.
    frame = fetch("holdings_ticker", use_cache=False, ticker=ticker)
    if frame.empty:
        return frame
    if "calendardate" in frame.columns:
        frame = frame.sort_values("calendardate", ascending=False)
    return frame


@st.cache_data(ttl=CACHE_TTL)
def insider_transactions(ticker: str, years: int = 2) -> pd.DataFrame:
    start = (pd.Timestamp.today() - pd.DateOffset(years=years)).date().isoformat()
    frame = fetch("insiders", ticker=ticker,
                  **{"from": start, "to": pd.Timestamp.today().date().isoformat()})
    date_column = next((c for c in ("filingdate", "transactiondate", "date")
                        if c in frame.columns), None)
    if not frame.empty and date_column:
        frame = frame.sort_values(date_column, ascending=False)
    return frame


@st.cache_data(ttl=CACHE_TTL)
def sp500_membership() -> pd.DataFrame:
    return fetch("sp500", **{"from": "2016-01-01",
                             "to": pd.Timestamp.today().date().isoformat()})


@st.cache_data(ttl=CACHE_TTL)
def upcoming_reports(tickers: tuple[str, ...]) -> pd.DataFrame:
    """Estimated next filing date per ticker, from each firm's own cadence.

    Sharadar's Bundle carries no forward earnings calendar, so this is derived,
    not reported: the median gap between a ticker's own recent filings, added
    to its latest one. It is accurate to a few days for regular filers and is
    labelled `estimated` so it is never mistaken for a confirmed date.

    Analyst consensus estimates are a separate Sharadar product not included in
    this subscription, so the "market implied" column cannot be populated - the
    DCF forecast is shown in its place.
    """
    connection = connect()
    path = (DATA_DIR / "metrics.parquet").as_posix()
    frame = connection.execute(f"""
        SELECT ticker, date, revenue_ttm, netinc_ttm, eps_placeholder FROM (
            SELECT ticker, date, revenue_ttm, netinc_ttm, NULL AS eps_placeholder,
                   ROW_NUMBER() OVER (PARTITION BY ticker ORDER BY date DESC) rn
            FROM read_parquet('{path}')
        ) WHERE rn <= 9
    """).df()
    connection.close()

    frame = frame.sort_values(["ticker", "date"])
    gaps = frame.groupby("ticker")["date"].diff().dt.days
    cadence = gaps.groupby(frame["ticker"]).median()
    last = frame.groupby("ticker")["date"].max()

    reports = pd.DataFrame({
        "ticker": last.index,
        "last_filing": last.values,
        "median_gap_days": cadence.reindex(last.index).values,
    })
    reports["next_report_estimated"] = (
        pd.to_datetime(reports["last_filing"])
        + pd.to_timedelta(reports["median_gap_days"].fillna(91), unit="D"))
    reports["days_away"] = (reports["next_report_estimated"]
                            - pd.Timestamp.today().normalize()).dt.days
    if tickers:
        reports = reports[reports["ticker"].isin(tickers)]
    return reports.sort_values("days_away")


# --------------------------------------------------------------------------
# currency basis
# --------------------------------------------------------------------------
# One session-wide choice, rendered into every page's sidebar under the same
# key so switching it on one page carries to the others. Streamlit re-runs a
# page's sidebar from scratch on navigation, so the widget has to be created
# per page; the shared key is what makes it one setting rather than three.

CURRENCY_KEY = "currency_basis"

#: Analytics columns that come from the filed statements, and so are the only
#: ones an "as reported" view restates. Market capitalisation, enterprise
#: value, the share price and every DCF output stay in US dollars in both
#: modes - a US listing is quoted in dollars, and there is no sense in which
#: they were ever "reported" in anything else.
_ANALYTICS_MONEY = ("revenue_ttm", "netinc_ttm", "ebitda_ttm", "ebit_ttm",
                    "fcf_ttm", "fcff_ttm", "fcfe_ttm", "equity_book",
                    "net_debt", "total_debt", "cashneq", "cfo_ttm",
                    "capex_ttm")


def currency_basis() -> str:
    """The currency the session is currently displaying."""
    return st.session_state.get(CURRENCY_KEY, currency.USD)


def currency_selector(container=None) -> str:
    """The sidebar control. Returns the chosen basis."""
    target = st if container is None else container
    return target.radio(
        "Currency", list(currency.BASES), key=CURRENCY_KEY,
        help="**US dollars** restates every figure a company filed in another "
             "currency at the exchange rate on its own report date, using "
             "Sharadar's `fxusd`. It is the only basis on which the tables "
             "here are internally consistent, because market cap, enterprise "
             "value and the share price are always in dollars.\n\n"
             "**As reported** prints the filed statements in the currency the "
             "company filed them in. Prices, market cap and every multiple "
             "stay in dollars either way - a multiple is a dollar price over "
             "a dollar earnings figure in both modes, so only the absolute "
             "amounts change.")


def analytics_in_basis(frame: pd.DataFrame, basis: str) -> pd.DataFrame:
    """Put the statement-derived money columns back into filing currency.

    `analytics.parquet` is built in dollars throughout, so US dollars is the
    no-op direction and "as reported" is the one that does work: multiply back
    by the same per-filing rate the metrics build divided by.
    """
    if basis == currency.USD or "fx_to_usd" not in frame.columns:
        return frame
    out = frame.copy()
    rate = pd.to_numeric(out["fx_to_usd"], errors="coerce").fillna(1.0)
    rate = rate.where(rate > 0, 1.0)
    for column in _ANALYTICS_MONEY:
        if column in out.columns:
            out[column] = pd.to_numeric(out[column], errors="coerce") * rate
    return out


# --------------------------------------------------------------------------
# formatting helpers
# --------------------------------------------------------------------------

def human_money(value: float) -> str:
    """Compact currency, because a screener table of raw dollars is unreadable."""
    if value is None or not np.isfinite(value):
        return "-"
    for cutoff, suffix in ((1e12, "T"), (1e9, "B"), (1e6, "M"), (1e3, "K")):
        if abs(value) >= cutoff:
            return f"${value / cutoff:,.2f}{suffix}"
    return f"${value:,.2f}"


def timeframe_start(label: str, latest: pd.Timestamp) -> pd.Timestamp:
    offsets = {"1W": pd.DateOffset(weeks=1),
        "1M": pd.DateOffset(months=1), "3M": pd.DateOffset(months=3),
        "6M": pd.DateOffset(months=6), "YTD": None,
        "1Y": pd.DateOffset(years=1), "3Y": pd.DateOffset(years=3),
        "5Y": pd.DateOffset(years=5), "Max": None,
    }
    if label == "YTD":
        return pd.Timestamp(year=latest.year, month=1, day=1)
    if label == "Max":
        return pd.Timestamp("1900-01-01")
    return latest - offsets[label]


PERIOD_LABELS = {"1d": "1 day", "1w": "1 week", "1m": "1 month",
                 "3m": "3 months", "6m": "6 months", "1y": "1 year"}

MARKETCAP_TIERS = {
    "All": 0.0,
    "Micro+ ($300M)": 300e6,
    "Small+ ($2B)": 2e9,
    "Mid+ ($10B)": 10e9,
    "Large+ ($50B)": 50e9,
    "Mega ($200B)": 200e9,
}


# --------------------------------------------------------------------------
# formatting — always English, never locale-dependent
# --------------------------------------------------------------------------

def money(value: float, unit: str = "auto", symbol: str = "$") -> str:
    """Format a currency amount in millions or billions, in English.

    Streamlit's built-in `format="compact"` renders through the browser's
    locale, which produced Russian abbreviations ("трлн") on this machine.
    Formatting to a string here keeps every table in English regardless of
    browser settings, and keeps column widths predictable so numbers stop
    overflowing their cells.

    `symbol` is empty for a figure shown in its filing currency, which for a
    mixed table is not one currency and so cannot carry one sign.
    """
    if value is None or not np.isfinite(value):
        return "—"
    if unit == "B" or (unit == "auto" and abs(value) >= 1e9):
        return f"{symbol}{value / 1e9:,.2f}B"
    if unit == "M" or (unit == "auto" and abs(value) >= 1e6):
        return f"{symbol}{value / 1e6:,.1f}M"
    if abs(value) >= 1e3:
        return f"{symbol}{value / 1e3:,.1f}K"
    return f"{symbol}{value:,.2f}"


def money_column(series: pd.Series, unit: str = "auto",
                 symbol: str = "$") -> pd.Series:
    return series.map(lambda v: money(v, unit, symbol))


def money_symbol(basis: str) -> str:
    """The sign a statement-derived figure carries on a given basis."""
    return "$" if basis == currency.USD else ""


def percent(value: float, decimals: int = 1) -> str:
    if value is None or not np.isfinite(value):
        return "—"
    return f"{value:.{decimals}%}"


def number(value: float, decimals: int = 2) -> str:
    if value is None or not np.isfinite(value):
        return "—"
    return f"{value:,.{decimals}f}"


# --------------------------------------------------------------------------
# peers
# --------------------------------------------------------------------------

def competitors(frame: pd.DataFrame, ticker: str, count: int = 6) -> pd.DataFrame:
    """Nearest listed peers: same sector, same industry, closest market cap.

    Industry first, because a payments network and a regional bank share the
    Financial Services sector while sharing almost no economics. If the
    industry is too thin to fill the list, the sector backfills it — flagged in
    `peer_match` so a sector-only peer is never mistaken for a true comparable.
    """
    if ticker not in set(frame["ticker"]):
        return pd.DataFrame()
    subject = frame[frame["ticker"] == ticker].iloc[0]
    if pd.isna(subject.get("sector")):
        return pd.DataFrame()

    pool = frame[(frame["ticker"] != ticker) & frame["marketcap"].notna()]
    same_industry = pool[(pool["sector"] == subject["sector"])
                         & (pool["industry"] == subject["industry"])].copy()
    same_industry["peer_match"] = "sector + industry"
    same_sector = pool[(pool["sector"] == subject["sector"])
                       & ~pool["ticker"].isin(same_industry["ticker"])].copy()
    same_sector["peer_match"] = "sector only"

    peers = pd.concat([same_industry, same_sector], ignore_index=True)
    if peers.empty:
        return peers
    # Rank by log distance in market cap: a $10B and a $100B company are not
    # comparable even though both sit far from a $1T one on a linear scale.
    reference = max(float(subject["marketcap"]), 1.0)
    peers["cap_distance"] = (np.log(peers["marketcap"].clip(lower=1.0))
                             - np.log(reference)).abs()
    # Industry matches always sort ahead of sector-only ones.
    peers["_order"] = np.where(peers["peer_match"] == "sector + industry", 0, 1)
    return (peers.sort_values(["_order", "cap_distance"])
            .head(count).drop(columns=["_order"]).reset_index(drop=True))


# --------------------------------------------------------------------------
# earnings calendar
# --------------------------------------------------------------------------

UNUSUAL_WHALES_TOKEN = os.environ.get("UNUSUAL_WHALES_TOKEN", "").strip()


@st.cache_data(ttl=CACHE_TTL)
def unusual_whales_earnings() -> pd.DataFrame:
    """Confirmed earnings dates from Unusual Whales, when a token is present.

    Their /earnings page is a client-rendered app with no data in the HTML, and
    the public API answers 401 without a bearer token, so this cannot work on
    the free tier. Set UNUSUAL_WHALES_TOKEN in the environment to enable it;
    without one the calendar falls back to filing-cadence estimates and every
    row is labelled with which source it came from.
    """
    if not UNUSUAL_WHALES_TOKEN:
        return pd.DataFrame()
    try:
        import requests
        response = requests.get(
            "https://api.unusualwhales.com/api/earnings/afterhours",
            headers={"Authorization": f"Bearer {UNUSUAL_WHALES_TOKEN}",
                     "Accept": "application/json"},
            timeout=20)
        if response.status_code != 200:
            return pd.DataFrame()
        payload = response.json()
        records = payload.get("data", payload) if isinstance(payload, dict) else payload
        frame = pd.DataFrame(records)
        if frame.empty:
            return frame
        ticker_column = next((c for c in ("ticker", "symbol") if c in frame.columns), None)
        date_column = next((c for c in ("report_date", "date", "earnings_date")
                            if c in frame.columns), None)
        if not ticker_column or not date_column:
            return pd.DataFrame()
        return pd.DataFrame({
            "ticker": frame[ticker_column],
            "next_report": pd.to_datetime(frame[date_column], errors="coerce"),
        }).dropna()
    except Exception:
        # A calendar is a convenience; it must never take the page down.
        return pd.DataFrame()


@st.cache_data(ttl=CACHE_TTL)
def earnings_calendar(tickers: tuple[str, ...]) -> pd.DataFrame:
    """Next report date per ticker, preferring a confirmed source over estimate."""
    estimates = upcoming_reports(tickers).rename(
        columns={"next_report_estimated": "next_report"})
    estimates["source"] = "Estimated (filing cadence)"

    confirmed = unusual_whales_earnings()
    if not confirmed.empty:
        confirmed = confirmed[confirmed["ticker"].isin(tickers)].copy()
        confirmed["source"] = "Unusual Whales (confirmed)"
        estimates = estimates[~estimates["ticker"].isin(confirmed["ticker"])]
        estimates = pd.concat([confirmed, estimates], ignore_index=True)

    estimates["days_away"] = (pd.to_datetime(estimates["next_report"])
                              - pd.Timestamp.today().normalize()).dt.days
    return estimates.sort_values("days_away")


@st.cache_data(ttl=CACHE_TTL)
def load_group_returns() -> pd.DataFrame:
    return analytics.load_group_returns()


@st.cache_data(ttl=CACHE_TTL)
def load_driver_history(ticker: str) -> pd.DataFrame:
    """Reported history of the DCF drivers, on the same definitions the model uses.

    Capex/revenue and D&A come straight from the metrics file, but the change
    in working capital does not exist there - it is a year-on-year difference
    of a balance-sheet stock, so it has to be rebuilt from the fundamentals
    with the same date-based lookback the model uses. Reconstructing it here
    rather than approximating keeps the history line comparable to the forecast
    line drawn against it.
    """
    connection = connect()
    frame = connection.execute("""
        SELECT date, fxusd, revenue, workingcapital, depamor, capex
        FROM fundamentals
        WHERE ticker = ? AND dimension = 'ART'
        ORDER BY date
    """, [ticker]).df()
    connection.close()
    if frame.empty:
        return frame

    rate = pd.to_numeric(frame["fxusd"], errors="coerce")
    rate = rate.where(rate > 0, 1.0)
    for column in ("revenue", "workingcapital", "depamor", "capex"):
        frame[column] = frame[column] / rate
    frame = frame.dropna(subset=["revenue"])
    frame = frame[frame["revenue"] > 0].copy()
    frame["date"] = pd.to_datetime(frame["date"])

    prior = frame[["date", "workingcapital"]].copy()
    prior["date"] = prior["date"] + pd.DateOffset(years=1)
    prior = prior.rename(columns={"workingcapital": "wc_prior"})
    merged = pd.merge_asof(frame.sort_values("date"), prior.sort_values("date"),
                           on="date", direction="backward",
                           tolerance=pd.Timedelta(days=200))

    merged["capex_to_revenue"] = merged["capex"].abs() / merged["revenue"]
    merged["da"] = merged["depamor"].abs()
    merged["da_to_revenue"] = merged["da"] / merged["revenue"]
    merged["delta_working_capital"] = merged["workingcapital"] - merged["wc_prior"]
    merged["dwc_to_revenue"] = merged["delta_working_capital"] / merged["revenue"]
    return merged[["date", "revenue", "capex_to_revenue", "da", "da_to_revenue",
                   "delta_working_capital", "dwc_to_revenue"]]


@st.cache_data(ttl=CACHE_TTL)
def load_wacc_history(ticker: str) -> pd.DataFrame:
    connection = connect()
    path = (DATA_DIR / "metrics.parquet").as_posix()
    frame = connection.execute(
        f"SELECT date, wacc, cost_of_equity, cost_of_debt, beta "
        f"FROM read_parquet('{path}') WHERE ticker = ? ORDER BY date",
        [ticker]).df()
    connection.close()
    return frame
