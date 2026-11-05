#!/usr/bin/env python3
"""
TTM valuation metrics, cash flows and cost of capital.

Reads the Parquet written by sharadar_loader.py, adds market data from FRED,
and produces two files:

  data/market_data.parquet   risk-free rates, S&P 500, Nasdaq 100, Brent,
                             gold, VIX - daily

  data/metrics.parquet       ALL tickers, one row per (ticker, filing date):
      scale        revenue, net income, EBIT, EBITDA, FCF, R&D  (all TTM)
      cost of cap  cost of equity (CAPM), cost of debt, WACC, beta, alpha
      multiples    P/E, P/S, P/B, P/EBITDA, EV/EBITDA, EV/EBIT, EV/Sales,
                   P/FCF, FCF yield, earnings yield
      margins      gross, EBITDA, EBIT, net, FCF
      intensity    R&D/revenue, SG&A/revenue, opex/revenue, SBC/revenue,
                   capex/revenue
      quality      ROE, ROA, ROIC, D/E, net debt/EBITDA, interest coverage

  data/cash_flows.parquet    ONLY tickers with >= MIN_HISTORY_YEARS of filing
                             history:
      FCFF, FCFE and their inputs, plus P/FCFF, P/FCFE, EV/FCFF, FCFE yield
      and the FCFF/FCFE margins

TTM comes from Sharadar's `ART` dimension - as-reported trailing twelve months,
time-indexed to the SEC filing date. Nothing here uses a number the market had
not yet seen on that date.

Usage
      python3 valuation_metrics.py market    # refresh FRED series only
      python3 valuation_metrics.py metrics   # rebuild metrics from local data
      python3 valuation_metrics.py all       # both
      python3 valuation_metrics.py status
"""

from __future__ import annotations

# Same interpreter bootstrap as sharadar_loader - see the note there.
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
import io
import time

import numpy as np
import pandas as pd
import requests

from sharadar_loader import DATA_DIR, connect

# --------------------------------------------------------------------------
# configuration
# --------------------------------------------------------------------------

# FRED serves CSV without an API key from this endpoint.
FRED_CSV = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={series}"
FRED_SERIES = {
    "rf_10y": "DGS10",      # 10-Year Treasury constant maturity, % annual
    "rf_3m": "DGS3MO",      # 3-Month Treasury, % annual
    "sp500": "SP500",       # S&P 500 index level (FRED holds only ~10 years)
    "nasdaq100": "NASDAQ100",
    "brent": "DCOILBRENTEU",  # Brent crude, USD/barrel
    "vix": "VIXCLS",          # CBOE volatility index
}

# Gold is NOT on FRED as a daily price. The London AM/PM fixing series
# (GOLDAMGBD228NLBM / GOLDPMGBD228NLBM) have been discontinued, and what
# remains is either a PPI index that stops in 2021 or gold *volatility*
# (GVZCLS). Stooq's XAUUSD feed is behind a bot check.
#
# So gold comes from Sharadar's `funds` table instead - already covered by the
# Bundle subscription, same API and cache, no third vendor. GLD is a physically
# backed ETF, so it tracks spot gold less a 0.40%/yr expense ratio: a proxy,
# accurate in direction and near-exact in returns, not a spot quote.
GOLD_PROXY_TICKER = "GLD"
GOLD_HISTORY_START = "2010-01-01"

MARKET_PARQUET = DATA_DIR / "market_data.parquet"
METRICS_PARQUET = DATA_DIR / "metrics.parquet"
CASH_FLOWS_PARQUET = DATA_DIR / "cash_flows.parquet"

# FCFF/FCFE require this much filing history.
MIN_HISTORY_YEARS = 6.0

# --- assumptions, not data. Change them here and rerun `metrics`. ----------
# The equity risk premium cannot be observed, only estimated. 4.5% is close to
# Damodaran's current implied US ERP; the earlier 5.0% sat above it and pushed
# every cost of equity - and so every WACC and every DCF - too high. It remains
# the single most load-bearing guess in the project.
EQUITY_RISK_PREMIUM = 0.045

# Raw regression betas are noisy and revert toward 1 out of sample. The Blume
# adjustment (0.67*raw + 0.33*1) is the standard correction and is what
# Bloomberg reports as "adjusted beta". It pulls in the extremes - a 5.4 raw
# beta becomes 3.9 - without touching a stock that is already near the market.
BETA_BLUME_WEIGHT = 0.67

# Synthetic credit spreads over the risk-free rate by interest coverage, after
# Damodaran's large-cap table. Used when reported interest expense cannot give
# a usable cost of debt - which is 933 tickers, Apple among them. A flat
# rf + 2% priced Apple's debt at 6.75%, roughly a junk-bond yield for a AA+
# credit; on its coverage it earns the top bucket instead.
# Each entry is (minimum coverage, spread).
SYNTHETIC_SPREADS = (
    (8.50, 0.0065), (6.50, 0.0080), (5.50, 0.0100), (4.25, 0.0130),
    (3.00, 0.0180), (2.50, 0.0220), (2.25, 0.0300), (2.00, 0.0375),
    (1.75, 0.0475), (1.50, 0.0650), (1.25, 0.0800), (0.80, 0.1000),
    (0.65, 0.1150), (0.20, 0.1300), (-np.inf, 0.1500),
)
# Used when effective tax rate is unusable (pre-tax loss, missing tax expense).
DEFAULT_TAX_RATE = 0.21
# Effective rates outside this band are refits of odd one-offs, not economics.
TAX_RATE_BOUNDS = (0.0, 0.50)
# 671 tickers carry debt but report no interest expense. For those, cost of
# debt falls back to the risk-free rate plus this spread.
CREDIT_SPREAD_FALLBACK = 0.02
# Sanity band on the derived cost of debt; outside it we use the fallback.
COST_OF_DEBT_BOUNDS = (0.005, 0.25)

# --- beta -----------------------------------------------------------------
# Beta is estimated on weekly total returns over a trailing window, then
# annualised. Weekly rather than annual: FRED's S&P 500 series only goes back
# ten years, so an annual-frequency regression would have at most 10 points -
# far too few for a usable standard error. Weekly gives ~104 points per window
# and is the industry default (Bloomberg uses weekly/2y). Set BETA_FREQ="YE"
# and BETA_WINDOW_YEARS=10 if you want the annual version regardless.
BETA_FREQ = "W-FRI"
BETA_WINDOW_YEARS = 2
BETA_BENCHMARK = "sp500"     # or "nasdaq100"
BETA_MIN_OBS = 52            # skip windows thinner than this
PERIODS_PER_YEAR = {"W-FRI": 52, "ME": 12, "YE": 1}


# --------------------------------------------------------------------------
# market data - FRED
# --------------------------------------------------------------------------

# One attempt with a short timeout. FRED sometimes answers 200 with its "Site
# Maintenance" HTML page; when that happens the stored series is extended
# instead (see _extend_last_observation), and the next full run rewrites it.
FRED_TIMEOUT = 30


def _fetch_fred_csv(session: requests.Session, series: str) -> str:
    """Return the CSV text for one FRED series, or raise if FRED is unavailable.

    A maintenance page passes raise_for_status(), so the body is checked too.
    Without that check pandas fails on the HTML with "Expected 1 fields in
    line 5", which says nothing about the real cause.
    """
    response = session.get(FRED_CSV.format(series=series), timeout=FRED_TIMEOUT)
    response.raise_for_status()
    if response.text.lstrip().startswith("<"):
        raise ValueError("got HTML instead of CSV (FRED maintenance page?)")
    return response.text


def _extend_last_observation(name: str, reason: Exception) -> pd.Series:
    """Stored series for `name`, extended to today by repeating its last value.

    Used only when FRED cannot be reached. The result is a placeholder: the next
    successful `market` run fetches the full history again and overwrites it.
    """
    if not MARKET_PARQUET.exists():
        raise RuntimeError(
            f"FRED {name} unreachable ({reason}) and no stored market data to fall back on"
        ) from reason
    stored = pd.read_parquet(MARKET_PARQUET, columns=["date", name]).dropna(subset=[name])
    dates = pd.to_datetime(stored["date"])
    last_date = dates.max()
    last_value = stored.loc[dates == last_date, name].iloc[-1]
    gap = pd.bdate_range(last_date + pd.Timedelta(days=1), pd.Timestamp.today().normalize())
    print(f"  ! FRED {name} unreachable ({reason}); repeating {last_value} "
          f"for {len(gap)} business days after {last_date.date()}", file=sys.stderr)
    return pd.concat([pd.Series(stored[name].values, index=dates),
                      pd.Series(last_value, index=gap)])


def fetch_market_data(verbose: bool = True) -> pd.DataFrame:
    """Download the FRED series and write data/market_data.parquet.

    Rates are stored as decimals (0.0479), not percent, so they can be used in
    arithmetic without a unit mistake. Index levels are stored as published.
    """
    session = requests.Session()
    session.headers.update({"User-Agent": "portfolio-screener/1.0"})
    columns = {}

    for name, series in FRED_SERIES.items():
        try:
            frame = pd.read_csv(io.StringIO(_fetch_fred_csv(session, series)))
        except (requests.RequestException, ValueError) as exc:
            # Already in stored units (rates as decimals), so no conversion below.
            columns[name] = _extend_last_observation(name, exc)
            continue
        date_column = frame.columns[0]
        frame[date_column] = pd.to_datetime(frame[date_column])
        # FRED writes "." for non-trading days.
        values = pd.to_numeric(frame[series], errors="coerce")
        if name.startswith("rf_"):
            values = values / 100.0
        columns[name] = pd.Series(values.values, index=frame[date_column])
        if verbose:
            valid = values.dropna()
            print(f"  {series:<10} {len(valid):>6,} obs  "
                  f"{frame[date_column].min().date()} -> {frame[date_column].max().date()}")
        time.sleep(0.2)

    # Gold, from Sharadar rather than FRED - see GOLD_PROXY_TICKER above.
    from sharadar_loader import fetch as sharadar_fetch

    gold = sharadar_fetch(
        "funds", use_cache=False, ticker=GOLD_PROXY_TICKER,
        **{"from": GOLD_HISTORY_START, "to": pd.Timestamp.today().date().isoformat(),
           "fields": "ticker,date,closeadj"},
    )
    if gold.empty:
        print(f"  ! {GOLD_PROXY_TICKER} returned nothing - gold column will be missing",
              file=sys.stderr)
    else:
        gold["date"] = pd.to_datetime(gold["date"])
        columns["gold"] = gold.set_index("date")["closeadj"].sort_index()
        if verbose:
            print(f"  {GOLD_PROXY_TICKER + ' (gold)':<10} {len(gold):>6,} obs  "
                  f"{gold['date'].min().date()} -> {gold['date'].max().date()}")

    market = pd.DataFrame(columns).sort_index()
    market.index.name = "date"
    # Rates are published with gaps (holidays); carry the last quote forward so
    # a lookup on any calendar date resolves. Index and commodity levels are
    # NOT filled - a missing close should stay missing so returns are not
    # fabricated from a flat carried-forward price.
    for name in FRED_SERIES:
        if name.startswith("rf_"):
            market[name] = market[name].ffill()

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    market.reset_index().to_parquet(MARKET_PARQUET, index=False)
    if verbose:
        print(f"\n  -> {MARKET_PARQUET.name}  {len(market):,} rows  "
              f"{MARKET_PARQUET.stat().st_size / 1e6:.1f} MB")
    return market


def load_market_data() -> pd.DataFrame:
    if not MARKET_PARQUET.exists():
        raise SystemExit("No market data yet. Run: valuation_metrics.py market")
    market = pd.read_parquet(MARKET_PARQUET)
    return market.set_index("date").sort_index()


# --------------------------------------------------------------------------
# beta and alpha
# --------------------------------------------------------------------------

def _periodic_returns(frequency: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Wide per-ticker returns and the matching market/rf frame."""
    connection = connect()
    prices = connection.execute(
        "SELECT ticker, date, closeadj FROM prices ORDER BY ticker, date").df()
    connection.close()

    # closeadj is adjusted for splits, dividends and spinoffs, so pct_change on
    # it is a total return. The FRED indices are price-only, which biases beta
    # slightly; the effect on a ratio of covariances is second order.
    wide = prices.pivot(index="date", columns="ticker", values="closeadj")
    wide = wide.resample(frequency).last()
    stock_returns = wide.pct_change()

    market = load_market_data()
    benchmark = (market[[BETA_BENCHMARK, "rf_10y"]]
                 .resample(frequency).last())
    benchmark["market_return"] = benchmark[BETA_BENCHMARK].pct_change()

    periods = PERIODS_PER_YEAR.get(frequency, 52)
    benchmark["rf_period"] = benchmark["rf_10y"] / periods

    aligned = benchmark.reindex(stock_returns.index)
    return stock_returns, aligned


def compute_betas(frequency: str = BETA_FREQ,
                  window_years: int = BETA_WINDOW_YEARS,
                  verbose: bool = True) -> pd.DataFrame:
    """Rolling beta, alpha and R^2 per ticker per period.

    Ordinary CAPM regression of excess stock return on excess market return:

        (r_i - r_f) = alpha + beta * (r_m - r_f) + e

    beta  = cov(r_i - r_f, r_m - r_f) / var(r_m - r_f)
    alpha = mean(r_i - r_f) - beta * mean(r_m - r_f),  annualised

    Computed on the whole wide frame at once, so all ~2,900 tickers regress in
    a handful of vectorised passes rather than one loop per name.
    """
    stock_returns, benchmark = _periodic_returns(frequency)
    periods = PERIODS_PER_YEAR.get(frequency, 52)
    window = max(int(window_years * periods), BETA_MIN_OBS)

    excess_stock = stock_returns.sub(benchmark["rf_period"], axis=0)
    excess_market = benchmark["market_return"] - benchmark["rf_period"]

    rolling = excess_stock.rolling(window, min_periods=BETA_MIN_OBS)
    market_rolling = excess_market.rolling(window, min_periods=BETA_MIN_OBS)
    covariance = rolling.cov(excess_market)
    variance = market_rolling.var()

    beta = covariance.div(variance, axis=0)

    # Jensen's alpha, per period, from the same simple-return regression.
    alpha_period = rolling.mean().sub(beta.mul(market_rolling.mean(), axis=0))

    # Annualised by MULTIPLYING by `periods`. This is the standard convention
    # and it is the right one: alpha is a regression intercept - a per-period
    # excess-return premium in a linear model - not a growth rate, so it scales
    # linearly and does not compound.
    #
    # Compounding it was tried and is wrong. Jensen's inequality (the mean of
    # simple returns exceeding realised compound growth) applies to projecting
    # wealth from a return series; it does not make a linear model's intercept
    # need compounding. Applying (1+a)**52 inflated rather than deflated the
    # estimates - every ticker moved up, and the maximum went from 21.0 to
    # 47,273,731 because a per-period intercept of 0.405 compounds to 4.7e7.
    alpha = alpha_period * periods

    correlation = rolling.corr(excess_market)
    observations = excess_stock.rolling(window, min_periods=1).count()

    def melt(frame: pd.DataFrame, name: str) -> pd.DataFrame:
        return (frame.stack(future_stack=True).rename(name)
                .reset_index().rename(columns={"level_0": "date"}))

    betas = melt(beta, "beta")
    betas["alpha_annual"] = melt(alpha, "alpha_annual")["alpha_annual"]
    betas["beta_r2"] = melt(correlation, "r")["r"] ** 2
    betas["beta_obs"] = melt(observations, "n")["n"]
    betas = betas.dropna(subset=["beta"]).reset_index(drop=True)

    if verbose:
        print(f"  beta: {frequency} returns, {window}-period window, "
              f"benchmark {BETA_BENCHMARK}")
        print(f"        {len(betas):,} estimates over "
              f"{betas['ticker'].nunique():,} tickers, "
              f"median beta {betas['beta'].median():.2f}, "
              f"median R^2 {betas['beta_r2'].median():.2f}")
    return betas


# --------------------------------------------------------------------------
# TTM fundamentals and derived metrics
# --------------------------------------------------------------------------

def _safe_divide(numerator, denominator):
    """Divide, returning NaN where the denominator is zero or missing.

    Negative denominators are preserved rather than nulled: a negative P/E is
    information (the company lost money), and silently dropping it would hide
    exactly the names a screen should be excluding on purpose.
    """
    denominator = pd.to_numeric(denominator, errors="coerce")
    numerator = pd.to_numeric(numerator, errors="coerce")
    return numerator / denominator.replace(0, np.nan)


# Every field below is denominated in the filer's *reporting* currency, which
# for 233 of our tickers is not USD. See _to_usd for why that matters.
CURRENCY_FIELDS = (
    "revenue", "netinc", "netinccmn", "gp", "opinc", "ebit", "ebitda", "ebt",
    "taxexp", "intexp", "depamor", "ncfo", "capex", "fcf", "ncfdebt", "ncfdiv",
    "debt", "debtc", "debtnc", "cashneq", "equity", "assets", "invcap",
    "rnd", "sgna", "opex", "sbcomp",
)


def _to_usd(frame: pd.DataFrame) -> pd.DataFrame:
    """Restate reporting-currency figures in USD.

    Enabling ADRs pulled in 233 filers that report in IDR, KRW, COP and so on,
    while marketcap, ev and price are always USD. Mixing the two silently
    wrecks every ratio that spans them: Telkom Indonesia came out at a P/S of
    0.0001 and an FCFF of 44 *trillion* before this.

    fxusd is local currency units per USD, so the conversion DIVIDES. That is
    the opposite of what the name suggests, so it is verified rather than
    assumed: revenue/fxusd reproduces Sharadar's own revenueusd column, and
    likewise for ebitda, ebit, equity, debt and cash, on 92-99.9% of non-USD
    rows (misses are near-zero denominators and definitional differences on
    netinc-to-common).

    Everything is converted with the same rate, so sums and ratios stay
    internally consistent. Share counts and ratios are left alone.
    """
    rate = pd.to_numeric(frame["fxusd"], errors="coerce")
    # fxusd is 1.0 for USD filers, and must never be 0 or NaN.
    rate = rate.where(rate > 0, 1.0)
    for column in CURRENCY_FIELDS:
        if column in frame.columns:
            frame[column] = frame[column] / rate
    frame["fx_to_usd"] = rate
    return frame


def _load_ttm() -> pd.DataFrame:
    """Every ART (trailing-twelve-month, as-reported) filing row we hold."""
    connection = connect()
    frame = connection.execute("""
        SELECT ticker, date, calendardate, reportperiod, fxusd,
               revenue, netinc, netinccmn, gp, opinc, ebit, ebitda, ebt,
               taxexp, intexp, depamor, rnd, sgna, opex, sbcomp,
               ncfo, capex, fcf, ncfdebt, ncfdiv,
               debt, debtc, debtnc, cashneq, equity, assets, invcap,
               marketcap, ev, price, shareswadil, sharesbas,
               roe, roa, roic, currentratio, de,
               grossmargin, ebitdamargin, netmargin
        FROM fundamentals
        WHERE dimension = 'ART'
        ORDER BY ticker, date
    """).df()
    connection.close()
    return _to_usd(frame)


def compute_ttm_metrics(verbose: bool = True) -> pd.DataFrame:
    """The main table: one row per ticker per filing date."""
    frame = _load_ttm()
    if frame.empty:
        raise SystemExit("No ART fundamentals found. Run sharadar_loader.py history.")

    frame["date"] = pd.to_datetime(frame["date"])
    grouped = frame.groupby("ticker", sort=False)

    # --- history length gate --------------------------------------------
    span = grouped["date"].agg(["min", "max"])
    span["years_of_history"] = (span["max"] - span["min"]).dt.days / 365.25
    frame = frame.merge(
        span[["years_of_history"]].reset_index(), on="ticker", how="left")
    frame["has_6y_history"] = frame["years_of_history"] >= MIN_HISTORY_YEARS

    # --- effective tax rate ---------------------------------------------
    # taxexp / ebt is only meaningful with positive pre-tax income; a loss-maker
    # produces a tax *benefit* and a nonsensical rate.
    effective_tax = _safe_divide(frame["taxexp"], frame["ebt"].where(frame["ebt"] > 0))
    low, high = TAX_RATE_BOUNDS
    frame["tax_rate_eff"] = effective_tax.where(
        effective_tax.between(low, high), DEFAULT_TAX_RATE)

    # --- headline TTM figures -------------------------------------------
    frame = frame.rename(columns={
        "revenue": "revenue_ttm", "netinc": "netinc_ttm", "ebitda": "ebitda_ttm",
        "ebit": "ebit_ttm", "gp": "gross_profit_ttm", "ncfo": "cfo_ttm",
        "capex": "capex_ttm", "fcf": "fcf_ttm", "intexp": "interest_expense_ttm",
        "ncfdebt": "net_borrowing_ttm", "depamor": "d_and_a_ttm",
        "rnd": "rnd_ttm", "sgna": "sgna_ttm", "opex": "opex_ttm",
        "sbcomp": "sbcomp_ttm",
    })

    # --- FCFF / FCFE -----------------------------------------------------
    # capex is a SIGNED cash-flow item in Sharadar (negative = outflow), which
    # is why both formulas ADD it. Verified against Sharadar's own fcf field:
    # fcf = ncfo + capex holds for 99.98% of rows, and 77 tickers - mostly
    # REITs and shipping - legitimately report positive capex in a period of
    # net asset sales. Taking abs() would invert the sign for those.
    after_tax_interest = (frame["interest_expense_ttm"].fillna(0.0)
                          * (1.0 - frame["tax_rate_eff"]))

    frame["fcff_ttm"] = frame["cfo_ttm"] + after_tax_interest + frame["capex_ttm"]
    frame["fcfe_ttm"] = (frame["cfo_ttm"] + frame["capex_ttm"]
                         + frame["net_borrowing_ttm"].fillna(0.0))

    # Only for names with enough history to be worth modelling.
    frame.loc[~frame["has_6y_history"], ["fcff_ttm", "fcfe_ttm"]] = np.nan

    # --- balance sheet ---------------------------------------------------
    frame["total_debt"] = frame["debt"]
    frame["net_debt"] = frame["total_debt"] - frame["cashneq"]
    frame["equity_book"] = frame["equity"]

    # --- cost of debt ----------------------------------------------------
    # Interest expense accrues across the whole trailing year, so the matching
    # denominator is average debt over that year, not the closing balance.
    #
    # The year-ago balance is found by DATE, not by row position. shift(4)
    # assumes four filings a year, which is true for US domestic filers but not
    # for the 158 tickers here that file annually - foreign private issuers on
    # 20-F, whose interim rows carry no financials. For those, four rows back
    # is four years back, or an empty row.
    prior = frame[["ticker", "date", "total_debt"]].copy()
    prior["date"] = prior["date"] + pd.DateOffset(years=1)
    prior = prior.rename(columns={"total_debt": "debt_prior"})
    # merge_asof needs the left side sorted by `date` and returns it that way,
    # while `frame` is ordered by (ticker, date). Carry the original index
    # through so the result is realigned rather than assigned positionally.
    left = frame[["ticker", "date"]].reset_index().sort_values("date")
    debt_year_ago = (
        pd.merge_asof(left, prior.sort_values("date"),
                      on="date", by="ticker", direction="backward",
                      tolerance=pd.Timedelta(days=200))
        .set_index("index")["debt_prior"].reindex(frame.index)
    )
    frame["avg_total_debt"] = frame[["total_debt"]].assign(
        prior=debt_year_ago).mean(axis=1)
    # Fall back to the closing balance for the first year of each ticker.
    frame["avg_total_debt"] = frame["avg_total_debt"].fillna(frame["total_debt"])

    derived_cost_of_debt = _safe_divide(
        frame["interest_expense_ttm"],
        frame["avg_total_debt"].where(frame["avg_total_debt"] > 0))
    low, high = COST_OF_DEBT_BOUNDS
    usable = derived_cost_of_debt.between(low, high)
    frame["cost_of_debt"] = derived_cost_of_debt.where(usable)
    frame["cost_of_debt_source"] = np.where(usable, "reported", "fallback")

    if verbose:
        print(f"  cost of debt: {int(usable.sum()):,} of {len(frame):,} rows from "
              f"reported interest expense, {int((~usable).sum()):,} on fallback")

    # --- market data as of each filing date ------------------------------
    market = load_market_data()
    risk_free = market["rf_10y"].dropna()
    frame = frame.sort_values("date")
    frame["risk_free_rate"] = pd.merge_asof(
        frame[["date"]], risk_free.rename("rf").reset_index(),
        on="date", direction="backward")["rf"].values

    # Names whose reported interest expense gives no usable rate fall back to a
    # synthetic rating: risk-free plus the spread their interest coverage earns.
    # Coverage is EBIT / interest expense, so a company with no interest expense
    # has no coverage ratio either - those get the top bucket, which is right,
    # since carrying no measurable interest cost is itself a sign of quality.
    coverage = _safe_divide(frame["ebit_ttm"], frame["interest_expense_ttm"])
    spread = pd.Series(SYNTHETIC_SPREADS[-1][1], index=frame.index)
    # Ascending by coverage threshold, so the highest bucket a company clears is
    # the one that sticks. Iterating the table as written would let the widest
    # spread overwrite every other match, and every firm would look like junk.
    for minimum, value in sorted(SYNTHETIC_SPREADS, key=lambda item: item[0]):
        spread = spread.mask(coverage >= minimum, value)
    # No interest expense at all and positive EBIT: treat as the best bucket.
    no_interest = frame["interest_expense_ttm"].fillna(0).eq(0) & (frame["ebit_ttm"] > 0)
    spread = spread.mask(no_interest, SYNTHETIC_SPREADS[0][1])
    frame["synthetic_spread"] = spread
    frame["cost_of_debt"] = frame["cost_of_debt"].fillna(
        frame["risk_free_rate"] + spread)

    # --- beta, alpha, cost of equity, WACC -------------------------------
    betas = compute_betas(verbose=verbose)
    frame = pd.merge_asof(
        frame.sort_values("date"),
        betas.sort_values("date")[["date", "ticker", "beta", "alpha_annual",
                                   "beta_r2", "beta_obs"]],
        on="date", by="ticker", direction="backward",
    )

    # Blume-adjusted beta feeds the cost of equity; the raw regression beta is
    # kept alongside it so the adjustment is visible rather than baked in.
    frame["beta_raw"] = frame["beta"]
    frame["beta"] = (BETA_BLUME_WEIGHT * frame["beta_raw"]
                     + (1.0 - BETA_BLUME_WEIGHT) * 1.0)
    frame["cost_of_equity"] = (frame["risk_free_rate"]
                               + frame["beta"] * EQUITY_RISK_PREMIUM)

    equity_value = frame["marketcap"]
    debt_value = frame["total_debt"].clip(lower=0)
    capital = equity_value + debt_value
    frame["weight_equity"] = _safe_divide(equity_value, capital)
    frame["weight_debt"] = _safe_divide(debt_value, capital)
    frame["wacc"] = (frame["weight_equity"] * frame["cost_of_equity"]
                     + frame["weight_debt"] * frame["cost_of_debt"]
                     * (1.0 - frame["tax_rate_eff"]))

    # --- multiples (at the filing-date market cap) -----------------------
    frame["pe_ttm"] = _safe_divide(frame["marketcap"], frame["netinc_ttm"])
    frame["ps_ttm"] = _safe_divide(frame["marketcap"], frame["revenue_ttm"])
    frame["pb"] = _safe_divide(frame["marketcap"], frame["equity_book"])
    frame["p_ebitda_ttm"] = _safe_divide(frame["marketcap"], frame["ebitda_ttm"])
    frame["ev_ebitda_ttm"] = _safe_divide(frame["ev"], frame["ebitda_ttm"])
    frame["ev_ebit_ttm"] = _safe_divide(frame["ev"], frame["ebit_ttm"])
    frame["ev_sales_ttm"] = _safe_divide(frame["ev"], frame["revenue_ttm"])
    frame["p_fcf_ttm"] = _safe_divide(frame["marketcap"], frame["fcf_ttm"])
    frame["p_fcff_ttm"] = _safe_divide(frame["marketcap"], frame["fcff_ttm"])
    frame["p_fcfe_ttm"] = _safe_divide(frame["marketcap"], frame["fcfe_ttm"])
    frame["ev_fcff_ttm"] = _safe_divide(frame["ev"], frame["fcff_ttm"])
    frame["fcf_yield"] = _safe_divide(frame["fcf_ttm"], frame["marketcap"])
    frame["fcfe_yield"] = _safe_divide(frame["fcfe_ttm"], frame["marketcap"])
    frame["earnings_yield"] = _safe_divide(frame["netinc_ttm"], frame["marketcap"])

    # --- margins ---------------------------------------------------------
    revenue = frame["revenue_ttm"]
    frame["gross_margin_ttm"] = _safe_divide(frame["gross_profit_ttm"], revenue)
    frame["ebitda_margin_ttm"] = _safe_divide(frame["ebitda_ttm"], revenue)
    frame["ebit_margin_ttm"] = _safe_divide(frame["ebit_ttm"], revenue)
    frame["net_margin_ttm"] = _safe_divide(frame["netinc_ttm"], revenue)
    frame["fcf_margin_ttm"] = _safe_divide(frame["fcf_ttm"], revenue)
    frame["fcff_margin_ttm"] = _safe_divide(frame["fcff_ttm"], revenue)
    frame["fcfe_margin_ttm"] = _safe_divide(frame["fcfe_ttm"], revenue)

    # --- spend intensity --------------------------------------------------
    # Sharadar reports rnd = 0 (not null) for the ~1,800 tickers with no R&D
    # line, so a cross-sectional median of R&D/revenue is 0.000 and means
    # "most listed companies do no R&D", not "the data is missing". Filter on
    # rnd_ttm > 0 before averaging across a sector.
    frame["rnd_to_revenue_ttm"] = _safe_divide(frame["rnd_ttm"], revenue)
    frame["sgna_to_revenue_ttm"] = _safe_divide(frame["sgna_ttm"], revenue)
    frame["opex_to_revenue_ttm"] = _safe_divide(frame["opex_ttm"], revenue)
    frame["sbcomp_to_revenue_ttm"] = _safe_divide(frame["sbcomp_ttm"], revenue)
    frame["capex_to_revenue_ttm"] = _safe_divide(frame["capex_ttm"].abs(), revenue)

    # --- quality and leverage --------------------------------------------
    frame["net_debt_to_ebitda"] = _safe_divide(frame["net_debt"], frame["ebitda_ttm"])
    frame["debt_to_equity"] = _safe_divide(frame["total_debt"], frame["equity_book"])
    frame["interest_coverage"] = _safe_divide(
        frame["ebit_ttm"], frame["interest_expense_ttm"])
    frame["roe_ttm"] = frame["roe"]
    frame["roa_ttm"] = frame["roa"]
    frame["roic_ttm"] = frame["roic"]

    return frame.sort_values(["ticker", "date"]).reset_index(drop=True)


# Everything derived from FCFF/FCFE lives in the cash-flow dataset, because it
# only exists for the >=6y cohort. Plain FCF (cfo + capex) needs no history and
# stays in metrics.
CASH_FLOW_COLUMNS = [
    "ticker", "date", "calendardate", "reportperiod",
    "years_of_history", "fx_to_usd",
    # inputs, so the figures can be re-derived without the other file
    "revenue_ttm", "cfo_ttm", "capex_ttm", "fcf_ttm",
    "interest_expense_ttm", "net_borrowing_ttm", "tax_rate_eff",
    "total_debt", "marketcap", "ev",
    # the cash flows themselves
    "fcff_ttm", "fcfe_ttm",
    # and what is built on them
    "fcff_margin_ttm", "fcfe_margin_ttm",
    "p_fcff_ttm", "p_fcfe_ttm", "ev_fcff_ttm", "fcfe_yield",
]

METRICS_COLUMNS = [
    "ticker", "date", "calendardate", "reportperiod",
    "years_of_history", "has_6y_history", "fx_to_usd",
    # scale
    "revenue_ttm", "netinc_ttm", "gross_profit_ttm", "ebit_ttm", "ebitda_ttm",
    "d_and_a_ttm", "cfo_ttm", "capex_ttm", "fcf_ttm",
    "rnd_ttm", "sgna_ttm", "opex_ttm", "sbcomp_ttm",
    # balance sheet
    "interest_expense_ttm", "net_borrowing_ttm",
    "total_debt", "avg_total_debt", "net_debt", "cashneq", "equity_book",
    "marketcap", "ev", "price", "shareswadil",
    # cost of capital
    "risk_free_rate", "beta", "beta_raw", "alpha_annual", "beta_r2", "beta_obs",
    "cost_of_equity", "cost_of_debt", "cost_of_debt_source", "synthetic_spread",
    "weight_equity", "weight_debt", "wacc", "tax_rate_eff",
    # multiples
    "pe_ttm", "ps_ttm", "pb", "p_ebitda_ttm", "ev_ebitda_ttm", "ev_ebit_ttm",
    "ev_sales_ttm", "p_fcf_ttm", "fcf_yield", "earnings_yield",
    # margins
    "gross_margin_ttm", "ebitda_margin_ttm", "ebit_margin_ttm",
    "net_margin_ttm", "fcf_margin_ttm",
    # spend intensity
    "rnd_to_revenue_ttm", "sgna_to_revenue_ttm", "opex_to_revenue_ttm",
    "sbcomp_to_revenue_ttm", "capex_to_revenue_ttm",
    # quality
    "roe_ttm", "roa_ttm", "roic_ttm", "debt_to_equity",
    "net_debt_to_ebitda", "interest_coverage", "currentratio",
]


def split_outputs(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Metrics for every ticker; cash flows only for the >=6y cohort."""
    metrics = frame[[c for c in METRICS_COLUMNS if c in frame.columns]].copy()
    cash_flows = (frame.loc[frame["has_6y_history"],
                            [c for c in CASH_FLOW_COLUMNS if c in frame.columns]]
                  .copy().reset_index(drop=True))
    return metrics, cash_flows


# --------------------------------------------------------------------------

def build(verbose: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    frame = compute_ttm_metrics(verbose=verbose)
    metrics, cash_flows = split_outputs(frame)

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    for table, path in ((metrics, METRICS_PARQUET), (cash_flows, CASH_FLOWS_PARQUET)):
        table.to_parquet(path, index=False, compression="zstd", compression_level=6)

    if verbose:
        print()
        for table, path, note in (
            (metrics, METRICS_PARQUET, "all tickers"),
            (cash_flows, CASH_FLOWS_PARQUET,
             f">={MIN_HISTORY_YEARS:g}y history only"),
        ):
            print(f"  -> {path.name:<20} {len(table):>7,} rows x "
                  f"{len(table.columns):>2} cols  "
                  f"{path.stat().st_size / 1e6:>5.1f} MB  "
                  f"{table['ticker'].nunique():>5,} tickers  ({note})")
    return metrics, cash_flows


def latest(as_of: str | None = None, cash_flows: bool = False) -> pd.DataFrame:
    """Most recent filed row per ticker on or before `as_of`.

    drop_duplicates, NOT groupby(...).last(): GroupBy.last() skips NaN
    independently per column, so it returns a row assembled from several
    different filings. Revvity Medicines came out with revenue of 0 from its
    August 2026 filing next to an R&D/revenue ratio of 745x carried over from
    an older period when revenue was still positive. drop_duplicates keeps
    whole rows, so every field on a row shares one filing date.
    """
    frame = pd.read_parquet(CASH_FLOWS_PARQUET if cash_flows else METRICS_PARQUET)
    if as_of is not None:
        frame = frame[frame["date"] <= pd.Timestamp(as_of)]
    return (frame.sort_values("date")
            .drop_duplicates("ticker", keep="last")
            .reset_index(drop=True))


def status() -> None:
    for path in (MARKET_PARQUET, METRICS_PARQUET, CASH_FLOWS_PARQUET):
        if path.exists():
            frame = pd.read_parquet(path)
            tickers = (f"{frame['ticker'].nunique():,} tickers"
                       if "ticker" in frame.columns else "")
            print(f"{path.name:<22} {len(frame):>9,} rows x {len(frame.columns):>3} cols "
                  f"{path.stat().st_size / 1e6:>7.1f} MB  {tickers}")
        else:
            print(f"{path.name:<22} not built")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["market", "metrics", "all", "status"])
    args = parser.parse_args()

    if args.command in ("market", "all"):
        fetch_market_data()
    if args.command in ("metrics", "all"):
        build()
    if args.command == "status":
        status()


if __name__ == "__main__":
    main()
