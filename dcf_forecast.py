#!/usr/bin/env python3
"""
15-year DCF: forecast FCFF and FCFE, discount at WACC, value the terminal.

Runs only for the tickers in data/cash_flows.parquet - the >=6y-history cohort,
which is the only one with enough observations to fit drivers on.

Five drivers are projected independently, each from its own trimmed historical
base and each fading to its own terminal value over the horizon:

    revenue growth      -> terminal growth g, held >= MIN_SPREAD below WACC
    EBIT margin         -> sector median, shrunk toward the firm's own level
    D&A growth          -> terminal growth (D&A cannot outgrow revenue forever)
    capex / revenue     -> sector median, shrunk toward the firm's own level
    delta WC / revenue  -> sector median, shrunk toward the firm's own level

Every driver is clipped to a bound that keeps it economically meaningful:
capex is never negative, D&A is never zero, margins stay inside a plausible
band. The tax rate is fixed at the US federal statutory rate.

    FCFF = EBIT x (1 - tax) + D&A - capex - delta WC
    FCFE = FCFF - interest x (1 - tax) + net borrowing

Outputs:

    data/dcf_forecast.parquet   one row per (ticker, forecast year) - every
                                driver and every intermediate value
    data/dcf_summary.parquet    one row per ticker - WACC, terminal growth,
                                PV of each stream, terminal value, enterprise
                                and equity value, implied price vs market

Usage
      python3 dcf_forecast.py run
      python3 dcf_forecast.py status
"""

from __future__ import annotations

# Same interpreter bootstrap as the other modules - see sharadar_loader.
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

import numpy as np
import pandas as pd

import valuation_metrics as vm
from sharadar_loader import DATA_DIR, connect, fetch

FORECAST_PARQUET = DATA_DIR / "dcf_forecast.parquet"
SUMMARY_PARQUET = DATA_DIR / "dcf_summary.parquet"
DRIVERS_PARQUET = DATA_DIR / "dcf_drivers.parquet"

# --------------------------------------------------------------------------
# assumptions
# --------------------------------------------------------------------------

HORIZON_YEARS = 15

# US federal statutory corporate rate. Held fixed rather than using each firm's
# effective rate: effective rates swing on one-off credits and settlements, and
# projecting one of those 15 years forward is worse than assuming the statutory
# rate the firm will actually face. Set 0.25 to approximate federal + state.
TAX_RATE = 0.21

# Terminal growth. Capped so it can never approach WACC - a DCF's terminal
# value explodes as the spread goes to zero, and a firm cannot outgrow the
# economy forever. MIN_SPREAD is the floor on (WACC - g) demanded of every
# ticker; TERMINAL_GROWTH_TARGET is what we use when WACC leaves room.
TERMINAL_GROWTH_TARGET = 0.025      # ~long-run nominal GDP
TERMINAL_GROWTH_FLOOR = 0.000
MIN_SPREAD = 0.010                  # WACC - g >= 1pp, as required

# How much of the firm's own history survives into its terminal driver, with
# the rest coming from the sector median. 0.65 keeps roughly two thirds of the
# firm's deviation from its sector: competition erodes an advantage over 15
# years but does not erase it, and 0.50 proved too harsh - it pulled Apple's
# terminal EBIT margin from 31.7% to 20.3% and made every durable franchise
# screen as overvalued.
SECTOR_SHRINK = 0.65

# Sector anchors are medians over PROFITABLE firms only. Including loss-makers
# drags them far below what a viable business earns - Healthcare's all-firm
# median EBIT margin is 3.2% against 13.7% for profitable names, because 43% of
# listed healthcare companies lose money. Anchoring a going concern's terminal
# margin on a median that half the sample cannot sustain is not conservatism,
# it is a bias.
SECTOR_MEDIAN_PROFITABLE_ONLY = True
# After shrinking, the terminal driver is clipped to the sector median plus or
# minus this many interquartile ranges. This is the smoothing step - it stops a
# single firm's outlier history from producing an outlier terminal assumption.
SECTOR_BAND_IQR = 1.50

# Trailing window used to fit each driver's starting level, in YEARS rather
# than filings: long enough to average out a bad quarter, short enough to still
# describe the business as it is now. Measured in years because filing
# frequency is not constant across the universe - see _prior_year.
BASE_WINDOW_YEARS = 5
# How far either side of the one-year mark a prior observation may sit and
# still count as the year-ago comparison.
YOY_TOLERANCE_DAYS = 200

# --- driver estimation: exponentially weighted, recency-favouring -----------
# Each driver's starting level is an exponentially weighted mean over the
# window, with weight 0.5 ** (age / halflife). Two years of halflife means a
# filing from two years ago counts half as much as the latest one.
#
# This is simple exponential smoothing, which is exactly ARIMA(0,1,1): the
# optimal one-step forecast of a random walk with noise. It suits these series
# better than a fitted ARIMA per ticker would - filing histories here are 6-10
# annual observations, far too short to identify AR and MA orders, and a
# mis-identified ARIMA extrapolates trends that are not there. Exponential
# smoothing weights the recent past more heavily without inventing structure.
#
# Because the weight is computed from the actual date gap, irregular filing
# cadence (annual 20-F filers vs quarterly 10-Q filers) is handled correctly.
DRIVER_HALFLIFE_DAYS = 730.0

# --- uncertainty ------------------------------------------------------------
# The forecast is a point estimate; the spread around it comes from how much
# each firm's own historical FCFF/FCFE actually moved year to year. That
# dispersion is applied to the projected path as a multiplicative random walk,
# and the resulting distribution of implied share prices is reported at these
# quantiles.
SIMULATIONS = 1000
QUANTILE_LOW = 0.15
QUANTILE_HIGH = 0.85
# Cash-flow volatility is clipped: below the floor the interval is uselessly
# tight, above the cap a single loss-making year makes the distribution
# meaningless. The universe median of raw sigma is 0.73 (AAPL 0.14, MSFT 0.11,
# and a long tail of small caps above 1.5), so a 0.60 cap would clip more than
# half the sample and throw away real dispersion.
CASHFLOW_SIGMA_BOUNDS = (0.05, 1.00)

# Shocks are persistent but mean-reverting - an AR(1) in log space rather than
# a random walk. This matters: under a pure random walk, variance grows
# linearly with the horizon, so a firm with sigma 0.6 has a year-15 standard
# deviation of 0.6*sqrt(15) = 2.3 in logs, an 0.09x-11x range. That is not
# forecast uncertainty, it is a model with no anchor. A bad year does not
# permanently reset a company's cash-flow level; it decays back toward the
# projected path. With phi = 0.7 the shock's variance converges to
# sigma^2/(1-phi^2), so long-horizon uncertainty is ~1.4x the one-year figure
# instead of ~3.9x.
SHOCK_PERSISTENCE = 0.70
# Simulated prices are clipped to a share price that can exist: never negative,
# and never more than this multiple of the current market price. Beyond that
# the model is extrapolating, not valuing.
PRICE_FLOOR = 0.0
PRICE_CAP_MULTIPLE = 3.0
SIMULATION_SEED = 20260906

# Bounds. Each is a hard clip applied to both the base and the terminal value,
# and again to every projected year.
BOUNDS = {
    # Revenue growth: the ceiling is 70%, high enough to let a genuine
    # hypergrowth company start there. It still fades to terminal growth by
    # year 15, so the cap governs the first years, not the whole path.
    "revenue_growth": (-0.10, 0.70),
    # Margin ceilings are set where the most profitable real businesses sit -
    # software, exchanges, royalty and pipeline companies all clear 60%.
    "ebit_margin": (-0.30, 0.85),
    # Capex is a cash outflow expressed as a positive fraction of revenue.
    # Never negative - a company cannot un-spend capital - and never zero,
    # because every going concern replaces assets.
    "capex_to_revenue": (0.002, 0.45),
    # D&A likewise is never zero for a firm with any fixed or intangible base.
    "da_to_revenue": (0.002, 0.40),
    "da_growth": (-0.15, 0.50),
    # Working capital can be released (negative) or absorbed (positive).
    "dwc_to_revenue": (-0.15, 0.25),
    # Terminal growth cannot exceed long-run nominal GDP by much without the
    # firm eventually becoming the entire economy. The WACC - g >= 1pp rule
    # still binds on top of this.
    "terminal_growth": (0.000, 0.050),
}

# Sectors where an FCFF DCF is structurally the wrong model: for banks and
# insurers debt is raw material rather than financing, and capex and working
# capital carry no operating meaning. Flagged, not excluded.
DCF_CAUTION_SECTORS = ("Financial Services", "Real Estate")


# --------------------------------------------------------------------------
# inputs
# --------------------------------------------------------------------------

def _sector_map() -> pd.Series:
    """ticker -> sector, from the Sharadar securities master."""
    master = fetch("tickers", use_cache=False,
                   **{"table": "stocks", "sort": "permaticker.asc",
                      "fields": "ticker,sector,industry"})
    return (master.dropna(subset=["ticker"])
            .drop_duplicates("ticker")
            .set_index("ticker")["sector"])


def _history() -> pd.DataFrame:
    """ART history of everything the drivers are fitted on."""
    connection = connect()
    frame = connection.execute("""
        SELECT ticker, date, fxusd,
               revenue, ebit, depamor, capex, workingcapital, debt
        FROM fundamentals
        WHERE dimension = 'ART'
        ORDER BY ticker, date
    """).df()
    connection.close()

    # Same currency restatement as valuation_metrics: fxusd is local units per
    # USD, so converting to USD divides.
    rate = pd.to_numeric(frame["fxusd"], errors="coerce")
    rate = rate.where(rate > 0, 1.0)
    for column in ("revenue", "ebit", "depamor", "capex", "workingcapital", "debt"):
        frame[column] = frame[column] / rate
    frame["date"] = pd.to_datetime(frame["date"])
    return frame


def _clip(series: pd.Series, key: str) -> pd.Series:
    low, high = BOUNDS[key]
    return series.clip(low, high)


def _prior_year(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Attach each column's value from roughly one year earlier, by DATE.

    Not by row position. A positional shift(4) assumes four filings a year,
    which holds for US domestic filers (three 10-Qs and a 10-K) but not for the
    158 tickers here that file annually - foreign private issuers on 20-F,
    whose interim 6-K rows carry no financials at all. For Infosys, four rows
    back is four years back, and the row it lands on is empty, so every driver
    came out NaN and the whole 15-year path with it.

    Shifting the prior frame's dates forward by a year and asof-matching
    backward finds the latest observation at or before t-1y, for any filing
    cadence, with a tolerance so a stale year is not matched to a fresh one.
    """
    prior = frame[["ticker", "date", *columns]].copy()
    prior["date"] = prior["date"] + pd.DateOffset(years=1)
    prior = prior.rename(columns={c: f"{c}_prior" for c in columns})
    return pd.merge_asof(
        frame.sort_values("date"), prior.sort_values("date"),
        on="date", by="ticker", direction="backward",
        tolerance=pd.Timedelta(days=YOY_TOLERANCE_DAYS),
    )


def _exponentially_weighted(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Per-ticker exponentially weighted mean of each column.

    Weight is 0.5 ** (age_in_days / DRIVER_HALFLIFE_DAYS), measured from each
    ticker's own latest filing, so the most recent observation dominates and
    influence decays smoothly into the past. Evaluating an EWM at its final
    point is simple exponential smoothing - ARIMA(0,1,1) - which is why the age
    is taken in days rather than rows: a 20-F filer's annual observations decay
    at the same rate per calendar year as a 10-Q filer's quarterly ones.

    Weights are renormalised per column over that column's non-null rows, so a
    driver missing from some filings is still weighted correctly rather than
    being diluted toward zero.
    """
    frame = frame.copy()
    age_days = (frame.groupby("ticker")["date"].transform("max")
                - frame["date"]).dt.days.astype(float)
    frame["_weight"] = 0.5 ** (age_days / DRIVER_HALFLIFE_DAYS)

    out = {}
    for column in columns:
        values = pd.to_numeric(frame[column], errors="coerce")
        valid = values.notna()
        weight = frame["_weight"].where(valid, 0.0)
        weighted_sum = (values.fillna(0.0) * weight).groupby(frame["ticker"]).sum()
        weight_sum = weight.groupby(frame["ticker"]).sum()
        out[column] = weighted_sum / weight_sum.replace(0.0, np.nan)
    return pd.DataFrame(out)


def base_drivers(verbose: bool = True) -> pd.DataFrame:
    """One row per ticker: the starting level of every driver.

    Each is an exponentially weighted mean over the trailing window rather than
    the latest value, so a single odd quarter cannot set the whole 15-year path
    but recent quarters still count for more than old ones.
    """
    history = _history()

    # Interim filings from annual filers are row placeholders with no
    # financials in them. Drop them before anything is measured, so a gap in
    # the schedule cannot be mistaken for a gap in the business.
    before = len(history)
    history = history.dropna(subset=["revenue"])
    history = history[history["revenue"] > 0].copy()
    if verbose:
        print(f"  dropped {before - len(history):,} ART rows carrying no financials")

    history = _prior_year(history, ["revenue", "depamor", "workingcapital"])
    history["revenue_yoy"] = (
        history["revenue"] / history["revenue_prior"].where(history["revenue_prior"] > 0)
    ) - 1.0
    history["da_yoy"] = (
        history["depamor"].abs()
        / history["depamor_prior"].abs().where(history["depamor_prior"].abs() > 0)
    ) - 1.0
    history["delta_wc"] = history["workingcapital"] - history["workingcapital_prior"]

    revenue = history["revenue"].where(history["revenue"] > 0)
    history["ebit_margin"] = history["ebit"] / revenue
    history["da_to_revenue"] = history["depamor"].abs() / revenue
    # capex is signed (negative = outflow); the driver is its magnitude.
    history["capex_to_revenue"] = history["capex"].abs() / revenue
    history["dwc_to_revenue"] = history["delta_wc"] / revenue
    history["debt_to_revenue"] = history["debt"] / revenue

    # Window by date, not by row count: BASE_WINDOW_YEARS of history means the
    # same thing to a quarterly and an annual filer.
    last_filing = history.groupby("ticker")["date"].transform("max")
    recent = history[history["date"]
                     >= last_filing - pd.DateOffset(years=BASE_WINDOW_YEARS)].copy()

    driver_columns = ["revenue_yoy", "ebit_margin", "da_yoy", "da_to_revenue",
                      "capex_to_revenue", "dwc_to_revenue", "debt_to_revenue"]
    drivers = _exponentially_weighted(recent, driver_columns).rename(columns={
        "revenue_yoy": "revenue_growth", "da_yoy": "da_growth"})

    # Working capital is null for ~600 tickers, overwhelmingly financials that
    # do not report it meaningfully. Treat as no working-capital drag rather
    # than dropping the ticker, and record that it was assumed.
    drivers["dwc_assumed_zero"] = drivers["dwc_to_revenue"].isna()
    drivers["dwc_to_revenue"] = drivers["dwc_to_revenue"].fillna(0.0)

    # A ticker with only one populated filing in the window has no year-ago
    # comparison and so no growth estimate. Fall back to the terminal growth
    # rate - a firm we cannot measure is assumed to grow with the economy -
    # and flag it rather than letting NaN silently void the whole projection.
    drivers["revenue_growth_assumed"] = drivers["revenue_growth"].isna()
    drivers["revenue_growth"] = drivers["revenue_growth"].fillna(TERMINAL_GROWTH_TARGET)
    drivers["ebit_margin"] = drivers["ebit_margin"].fillna(0.0)

    # D&A is the one driver with no sensible zero: a firm reporting none still
    # consumes capital. Fall back to the bound, not to nothing.
    drivers["da_to_revenue"] = drivers["da_to_revenue"].fillna(BOUNDS["da_to_revenue"][0])
    drivers["da_growth"] = drivers["da_growth"].fillna(TERMINAL_GROWTH_TARGET)
    drivers["capex_to_revenue"] = drivers["capex_to_revenue"].fillna(
        BOUNDS["capex_to_revenue"][0])
    drivers["debt_to_revenue"] = drivers["debt_to_revenue"].fillna(0.0).clip(lower=0.0)

    for key in ("revenue_growth", "ebit_margin", "da_growth",
                "da_to_revenue", "capex_to_revenue", "dwc_to_revenue"):
        drivers[key] = _clip(drivers[key], key)

    if verbose:
        print(f"  fitted base drivers for {len(drivers):,} tickers "
              f"over {BASE_WINDOW_YEARS} trailing years")
    return drivers.reset_index()


def sector_terminals(drivers: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """Terminal driver levels: sector median, shrunk toward the firm, smoothed.

    A firm out-earning its sector keeps part of that edge, a firm below it
    recovers part of the gap, and the result is then clipped into a band around
    the sector median so no single outlier history escapes into the terminal
    year.
    """
    smoothed = ("ebit_margin", "capex_to_revenue", "dwc_to_revenue")
    # The anchor population excludes loss-makers; the firms being anchored do
    # not. A loss-making firm still gets a terminal margin, it is just pulled
    # toward what a profitable peer earns rather than toward the sector's own
    # losses.
    population = (drivers[drivers["ebit_margin"] > 0]
                  if SECTOR_MEDIAN_PROFITABLE_ONLY else drivers)
    statistics = (population.groupby("sector")[list(smoothed)]
                  .agg(["median", lambda s: s.quantile(0.75) - s.quantile(0.25)]))
    statistics.columns = [f"{field}_{'median' if kind == 'median' else 'iqr'}"
                          for field, kind in statistics.columns]
    drivers = drivers.merge(statistics.reset_index(), on="sector", how="left")

    for field in smoothed:
        median = drivers[f"{field}_median"]
        iqr = drivers[f"{field}_iqr"].fillna(0.0)
        # Shrink the firm's deviation from its sector toward the sector.
        blended = median + SECTOR_SHRINK * (drivers[field] - median)
        # Then smooth: nothing may sit further than SECTOR_BAND_IQR from the
        # sector median.
        banded = blended.clip(median - SECTOR_BAND_IQR * iqr,
                              median + SECTOR_BAND_IQR * iqr)
        # A ticker with no sector falls back to its own history.
        drivers[f"terminal_{field}"] = _clip(banded.fillna(drivers[field]), field)

    if verbose:
        summary = statistics[["ebit_margin_median", "capex_to_revenue_median"]]
        print("\n  sector medians used as terminal anchors:")
        for sector, row in summary.sort_values("ebit_margin_median",
                                               ascending=False).iterrows():
            print(f"    {sector:<24} EBIT margin {row.iloc[0]:>7.1%}   "
                  f"capex/rev {row.iloc[1]:>6.1%}")
    return drivers


# --------------------------------------------------------------------------
# projection
# --------------------------------------------------------------------------

def cashflow_sigma(verbose: bool = True) -> pd.DataFrame:
    """Per-ticker dispersion of historical FCFF and FCFE, as a log-scale sigma.

    Measured as the standard deviation of year-on-year log changes in the
    realised cash flow. Log changes rather than levels so the number is
    scale-free and can be applied multiplicatively to any forecast; only
    sign-consistent pairs are used, because the log ratio of a cash flow that
    crossed zero is undefined and would otherwise silently drop the most
    volatile firms from the sample.
    """
    history = pd.read_parquet(vm.CASH_FLOWS_PARQUET,
                              columns=["ticker", "date", "fcff_ttm", "fcfe_ttm"])
    history = history.sort_values(["ticker", "date"])
    history = _prior_year(history, ["fcff_ttm", "fcfe_ttm"])

    # Net income comes from the metrics file, since it is not in cash_flows.
    earnings = pd.read_parquet(vm.METRICS_PARQUET,
                               columns=["ticker", "date", "netinc_ttm"])
    earnings = _prior_year(earnings.sort_values(["ticker", "date"]), ["netinc_ttm"])

    out = {}
    for stream, frame in (("fcff", history), ("fcfe", history), ("eps", earnings)):
        column = "netinc_ttm" if stream == "eps" else f"{stream}_ttm"
        current = frame[column]
        prior = frame[f"{column}_prior"]
        both_positive = (current > 0) & (prior > 0)
        ratio = np.log(current.where(both_positive) / prior.where(both_positive))
        out[f"sigma_{stream}"] = ratio.groupby(frame["ticker"]).std()

    sigma = pd.DataFrame(out)
    sigma.index.name = "ticker"
    # A ticker with too few sign-consistent pairs gets the universe median
    # rather than a fabricated zero.
    for column in sigma.columns:
        sigma[column] = sigma[column].fillna(sigma[column].median())
        sigma[column] = sigma[column].clip(*CASHFLOW_SIGMA_BOUNDS)
    if verbose:
        print(f"  cash-flow sigma: median FCFF {sigma['sigma_fcff'].median():.3f}, "
              f"FCFE {sigma['sigma_fcfe'].median():.3f}, "
              f"EPS {sigma['sigma_eps'].median():.3f}")
    return sigma.reset_index()


def _simulate_prices(cash_flow: np.ndarray, discount: np.ndarray, sigma: np.ndarray,
                     terminal_growth: np.ndarray, discount_rate: np.ndarray,
                     net_debt: np.ndarray | None, shares: np.ndarray,
                     market_price: np.ndarray,
                     rng: np.random.Generator) -> tuple[np.ndarray, ...]:
    """Distribution of implied share price under cash-flow uncertainty.

    The projected path is multiplied by a driftless multiplicative random walk
    calibrated to the firm's own historical cash-flow volatility, so
    uncertainty compounds with the forecast horizon - year 15 is far less
    certain than year 1, which is how forecast error actually behaves. The
    terminal value is shocked by the year-15 factor, since it is a perpetuity
    on that year's cash flow.

    Returns the low quantile, median, high quantile, and the share of paths
    landing above today's market price.
    """
    n_tickers, years = cash_flow.shape
    low = np.full(n_tickers, np.nan)
    median = np.full(n_tickers, np.nan)
    high = np.full(n_tickers, np.nan)
    probability = np.full(n_tickers, np.nan)

    # Shared standard normals: one (SIMULATIONS, years) draw scaled per ticker.
    # Common random numbers also make tickers comparable to each other.
    steps = rng.standard_normal((SIMULATIONS, years))

    # AR(1) in log space: x_t = phi*x_{t-1} + e_t, in units of sigma. Unlike a
    # random walk this converges, so year 15 stays a forecast rather than a
    # lottery. `variance` is the analytic variance of x_t in the same units,
    # used below for the martingale correction.
    phi = SHOCK_PERSISTENCE
    walk = np.empty_like(steps)
    previous = np.zeros(SIMULATIONS)
    variance = np.empty(years)
    running = 0.0
    for t in range(years):
        previous = phi * previous + steps[:, t]
        walk[:, t] = previous
        running = phi ** 2 * running + 1.0
        variance[t] = running
    horizon = variance[None, :]

    spread = np.maximum(discount_rate - terminal_growth, MIN_SPREAD)
    price_cap = np.where(market_price > 0, market_price * PRICE_CAP_MULTIPLE, np.nan)

    chunk = 200
    for start in range(0, n_tickers, chunk):
        stop = min(start + chunk, n_tickers)
        s = sigma[start:stop, None, None]
        # Martingale correction so the shock has mean 1 and does not add drift.
        shock = np.exp(s * walk[None, :, :] - 0.5 * s ** 2 * horizon[None, :, :])

        shocked = cash_flow[start:stop, None, :] * shock
        present_value = (shocked / discount[start:stop, None, :]).sum(axis=2)

        terminal = (shocked[:, :, -1] * (1.0 + terminal_growth[start:stop, None])
                    / spread[start:stop, None])
        present_value = present_value + terminal / discount[start:stop, None, -1]

        if net_debt is not None:
            present_value = present_value - net_debt[start:stop, None]

        share_count = shares[start:stop, None]
        with np.errstate(divide="ignore", invalid="ignore"):
            prices = np.where(share_count > 0, present_value / share_count, np.nan)

        # A share price cannot be negative, and beyond a few multiples of the
        # current price the model is extrapolating rather than valuing.
        prices = np.clip(prices, PRICE_FLOOR, price_cap[start:stop, None])

        valid = np.isfinite(prices)
        enough = valid.sum(axis=1) > SIMULATIONS // 10
        with np.errstate(invalid="ignore"):
            quantiles = np.nanquantile(
                np.where(valid, prices, np.nan),
                [QUANTILE_LOW, 0.5, QUANTILE_HIGH], axis=1)
        low[start:stop] = np.where(enough, quantiles[0], np.nan)
        median[start:stop] = np.where(enough, quantiles[1], np.nan)
        high[start:stop] = np.where(enough, quantiles[2], np.nan)

        above = (prices > market_price[start:stop, None]) & valid
        probability[start:stop] = np.where(
            enough, above.sum(axis=1) / np.maximum(valid.sum(axis=1), 1), np.nan)

    return low, median, high, probability


def _fade(start: np.ndarray, end: np.ndarray, years: int) -> np.ndarray:
    """Linear path from `start` in year 1 to `end` in year `years`."""
    weights = np.linspace(0.0, 1.0, years)[None, :]
    return start[:, None] * (1.0 - weights) + end[:, None] * weights


def project(drivers: pd.DataFrame, verbose: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build the year-by-year forecast and the per-ticker summary."""
    years = HORIZON_YEARS
    n = len(drivers)
    year_index = np.arange(1, years + 1)

    wacc = drivers["wacc"].to_numpy()
    cost_of_equity = drivers["cost_of_equity"].to_numpy()
    cost_of_debt = drivers["cost_of_debt"].to_numpy()

    # Terminal growth: the target unless the caller overrides it, but never
    # within MIN_SPREAD of WACC. The cap is applied after the override, so a
    # user cannot dial growth up into the discount rate and blow the terminal
    # value to infinity - the constraint is the model's, not the interface's.
    requested = np.full_like(wacc, TERMINAL_GROWTH_TARGET)
    if "terminal_growth_override" in drivers.columns:
        override = pd.to_numeric(drivers["terminal_growth_override"],
                                 errors="coerce").to_numpy(dtype=float)
        requested = np.where(np.isfinite(override), override, requested)
    terminal_growth = np.minimum(requested, wacc - MIN_SPREAD)
    terminal_growth = np.clip(terminal_growth, TERMINAL_GROWTH_FLOOR, None)
    spread_binding = terminal_growth < requested - 1e-12

    # --- revenue: growth fades from its base to terminal growth by year 15 --
    growth_path = _fade(drivers["revenue_growth"].to_numpy(), terminal_growth, years)
    growth_path = np.clip(growth_path, *BOUNDS["revenue_growth"])
    base_revenue = drivers["base_revenue"].to_numpy()[:, None]
    revenue = base_revenue * np.cumprod(1.0 + growth_path, axis=1)

    # --- margins and ratios fade to their sector-anchored terminals ---------
    ebit_margin = np.clip(_fade(drivers["ebit_margin"].to_numpy(),
                                drivers["terminal_ebit_margin"].to_numpy(), years),
                          *BOUNDS["ebit_margin"])
    capex_ratio = np.clip(_fade(drivers["capex_to_revenue"].to_numpy(),
                                drivers["terminal_capex_to_revenue"].to_numpy(), years),
                          *BOUNDS["capex_to_revenue"])
    dwc_ratio = np.clip(_fade(drivers["dwc_to_revenue"].to_numpy(),
                              drivers["terminal_dwc_to_revenue"].to_numpy(), years),
                        *BOUNDS["dwc_to_revenue"])

    # --- D&A is driven by its own growth rate, then sanity-bounded ----------
    da_growth = np.clip(_fade(drivers["da_growth"].to_numpy(), terminal_growth, years),
                        *BOUNDS["da_growth"])
    base_da = (drivers["da_to_revenue"].to_numpy() * drivers["base_revenue"].to_numpy())
    da = base_da[:, None] * np.cumprod(1.0 + da_growth, axis=1)
    # Compounding a growth rate for 15 years can drift D&A away from any
    # plausible share of revenue, so the ratio is clipped and D&A rebuilt.
    da_ratio = np.clip(np.divide(da, revenue, out=np.full_like(da, np.nan),
                                 where=revenue > 0), *BOUNDS["da_to_revenue"])
    da = da_ratio * revenue

    ebit = ebit_margin * revenue
    nopat = ebit * (1.0 - TAX_RATE)
    capex = capex_ratio * revenue
    delta_wc = dwc_ratio * revenue

    fcff = nopat + da - capex - delta_wc

    # --- FCFE: hold leverage constant against revenue -----------------------
    # Debt grows with the business, interest accrues on the opening balance,
    # and the change in debt is the net borrowing that accrues to equity.
    debt = drivers["debt_to_revenue"].to_numpy()[:, None] * revenue
    opening_debt = np.concatenate(
        [drivers["base_debt"].to_numpy()[:, None], debt[:, :-1]], axis=1)
    interest = cost_of_debt[:, None] * opening_debt
    net_borrowing = debt - opening_debt
    fcfe = fcff - interest * (1.0 - TAX_RATE) + net_borrowing

    # --- earnings stream, for the EPS model ---------------------------------
    # Net income after interest and tax, projected on the same drivers. Share
    # count is held constant: forecasting buybacks would need a payout policy
    # the model does not have, so a firm that keeps retiring stock will see its
    # real EPS grow faster than this.
    net_income = (ebit - interest) * (1.0 - TAX_RATE)
    share_count = drivers["shares"].to_numpy(dtype=float)
    eps = net_income / np.where(share_count > 0, share_count, np.nan)[:, None]

    # --- discounting --------------------------------------------------------
    discount_wacc = (1.0 + wacc)[:, None] ** year_index[None, :]
    discount_coe = (1.0 + cost_of_equity)[:, None] ** year_index[None, :]
    pv_fcff = fcff / discount_wacc
    # FCFE is an equity claim, so cost of equity is the correct rate. The WACC
    # discounting is emitted too because it was asked for, but it understates
    # the required return and so overstates equity value - prefer _at_coe.
    pv_fcfe_at_coe = fcfe / discount_coe
    pv_fcfe_at_wacc = fcfe / discount_wacc
    # Earnings accrue to equity, so the EPS stream discounts at cost of equity.
    pv_eps = eps / discount_coe

    # --- terminal value -----------------------------------------------------
    spread_wacc = np.maximum(wacc - terminal_growth, MIN_SPREAD)
    terminal_value_fcff = fcff[:, -1] * (1.0 + terminal_growth) / spread_wacc
    pv_terminal_fcff = terminal_value_fcff / discount_wacc[:, -1]

    spread_coe = np.maximum(cost_of_equity - terminal_growth, MIN_SPREAD)
    terminal_value_fcfe = fcfe[:, -1] * (1.0 + terminal_growth) / spread_coe
    pv_terminal_fcfe = terminal_value_fcfe / discount_coe[:, -1]

    terminal_value_eps = eps[:, -1] * (1.0 + terminal_growth) / spread_coe
    pv_terminal_eps = terminal_value_eps / discount_coe[:, -1]

    # --- long-format forecast ----------------------------------------------
    base_year = pd.to_datetime(drivers["base_date"]).dt.year.to_numpy()
    blocks = {
        "revenue_growth": growth_path, "revenue": revenue,
        "ebit_margin": ebit_margin, "ebit": ebit, "nopat": nopat,
        "da_growth": da_growth, "da_to_revenue": da_ratio, "da": da,
        "capex_to_revenue": capex_ratio, "capex": capex,
        "dwc_to_revenue": dwc_ratio, "delta_working_capital": delta_wc,
        "fcff": fcff,
        "debt": debt, "interest": interest, "net_borrowing": net_borrowing,
        "fcfe": fcfe,
        "discount_factor_wacc": discount_wacc,
        "discount_factor_coe": discount_coe,
        "pv_fcff": pv_fcff,
        "pv_fcfe_at_coe": pv_fcfe_at_coe,
        "pv_fcfe_at_wacc": pv_fcfe_at_wacc,
        "net_income": net_income, "eps": eps, "pv_eps": pv_eps,
    }
    forecast = pd.DataFrame({
        "ticker": np.repeat(drivers["ticker"].to_numpy(), years),
        "year": np.tile(year_index, n),
        "calendar_year": np.repeat(base_year, years) + np.tile(year_index, n),
        **{name: values.reshape(-1) for name, values in blocks.items()},
    })

    # --- per-ticker summary -------------------------------------------------
    sum_pv_fcff = pv_fcff.sum(axis=1)
    sum_pv_fcfe = pv_fcfe_at_coe.sum(axis=1)
    sum_pv_eps = pv_eps.sum(axis=1)
    # The EPS model values a share directly, so its present value IS the price.
    implied_price_eps = sum_pv_eps + pv_terminal_eps
    enterprise_value = sum_pv_fcff + pv_terminal_fcff
    net_debt = drivers["net_debt"].to_numpy()
    equity_value = enterprise_value - net_debt
    equity_value_fcfe = sum_pv_fcfe + pv_terminal_fcfe
    shares = drivers["shares"].to_numpy()
    market_price = drivers["market_price"].to_numpy(dtype=float)

    with np.errstate(divide="ignore", invalid="ignore"):
        implied_price = np.where(shares > 0, equity_value / shares, np.nan)
        implied_price_fcfe = np.where(shares > 0, equity_value_fcfe / shares, np.nan)
        upside = np.where(market_price > 0, implied_price / market_price - 1.0, np.nan)
        upside_fcfe = np.where(market_price > 0,
                               implied_price_fcfe / market_price - 1.0, np.nan)
        upside_eps = np.where(market_price > 0,
                              implied_price_eps / market_price - 1.0, np.nan)
        terminal_share = np.where(enterprise_value != 0,
                                  pv_terminal_fcff / enterprise_value, np.nan)

    # --- price distribution from historical cash-flow dispersion ------------
    rng = np.random.default_rng(SIMULATION_SEED)
    sigma_fcff = drivers["sigma_fcff"].to_numpy()
    sigma_fcfe = drivers["sigma_fcfe"].to_numpy()

    fcff_low, fcff_mid, fcff_high, fcff_probability = _simulate_prices(
        fcff, discount_wacc, sigma_fcff, terminal_growth, wacc,
        net_debt, shares, market_price, rng)
    # FCFE is already an equity claim, so no net-debt bridge is applied.
    fcfe_low, fcfe_mid, fcfe_high, fcfe_probability = _simulate_prices(
        fcfe, discount_coe, sigma_fcfe, terminal_growth, cost_of_equity,
        None, shares, market_price, rng)
    # EPS is already per share, so there is nothing to divide by and no net
    # debt to bridge - pass a unit share count.
    sigma_eps = drivers["sigma_eps"].to_numpy()
    unit_shares = np.ones_like(shares, dtype=float)
    eps_low, eps_mid, eps_high, eps_probability = _simulate_prices(
        eps, discount_coe, sigma_eps, terminal_growth, cost_of_equity,
        None, unit_shares, market_price, rng)

    summary = pd.DataFrame({
        "ticker": drivers["ticker"].to_numpy(),
        "sector": drivers["sector"].to_numpy(),
        "base_date": drivers["base_date"].to_numpy(),
        "dcf_caution_sector": drivers["sector"].isin(DCF_CAUTION_SECTORS).to_numpy(),
        "dwc_assumed_zero": drivers["dwc_assumed_zero"].to_numpy(),
        "revenue_growth_assumed": drivers["revenue_growth_assumed"].to_numpy(),
        # rates
        "wacc": wacc, "cost_of_equity": cost_of_equity, "cost_of_debt": cost_of_debt,
        "tax_rate": TAX_RATE,
        "terminal_growth": terminal_growth,
        "wacc_minus_growth": wacc - terminal_growth,
        "spread_binding": spread_binding,
        # base drivers
        "base_revenue": drivers["base_revenue"].to_numpy(),
        "base_revenue_growth": drivers["revenue_growth"].to_numpy(),
        "base_ebit_margin": drivers["ebit_margin"].to_numpy(),
        "base_da_growth": drivers["da_growth"].to_numpy(),
        "base_capex_to_revenue": drivers["capex_to_revenue"].to_numpy(),
        "base_dwc_to_revenue": drivers["dwc_to_revenue"].to_numpy(),
        "base_debt_to_revenue": drivers["debt_to_revenue"].to_numpy(),
        # terminal drivers
        "terminal_ebit_margin": drivers["terminal_ebit_margin"].to_numpy(),
        "sector_median_ebit_margin": drivers["ebit_margin_median"].to_numpy(),
        "terminal_capex_to_revenue": drivers["terminal_capex_to_revenue"].to_numpy(),
        "sector_median_capex_to_revenue": drivers["capex_to_revenue_median"].to_numpy(),
        "terminal_dwc_to_revenue": drivers["terminal_dwc_to_revenue"].to_numpy(),
        # year-15 state
        "revenue_year15": revenue[:, -1],
        "fcff_year15": fcff[:, -1],
        "fcfe_year15": fcfe[:, -1],
        # valuation
        "sum_pv_fcff": sum_pv_fcff,
        "terminal_value_fcff": terminal_value_fcff,
        "pv_terminal_fcff": pv_terminal_fcff,
        "terminal_share_of_ev": terminal_share,
        "enterprise_value_dcf": enterprise_value,
        "net_debt": net_debt,
        "equity_value_dcf": equity_value,
        "sum_pv_fcfe": sum_pv_fcfe,
        "terminal_value_fcfe": terminal_value_fcfe,
        "pv_terminal_fcfe": pv_terminal_fcfe,
        "equity_value_fcfe": equity_value_fcfe,
        "eps_year1": eps[:, 0],
        "eps_year15": eps[:, -1],
        "sum_pv_eps": sum_pv_eps,
        "terminal_value_eps": terminal_value_eps,
        "pv_terminal_eps": pv_terminal_eps,
        # market comparison
        "shares": shares,
        "shares_source": drivers["shares_source"].to_numpy(),
        "market_price": market_price,
        "market_price_date": drivers["market_price_date"].to_numpy(),
        "price_at_filing": drivers["price_at_filing"].to_numpy(),
        "marketcap": drivers["marketcap"].to_numpy(),
        "enterprise_value_market": drivers["ev_market"].to_numpy(),
        "implied_share_price_fcff": implied_price,
        "implied_share_price_fcfe": implied_price_fcfe,
        "upside_fcff": upside,
        "upside_fcfe": upside_fcfe,
        # distribution of the implied price under cash-flow uncertainty
        "sigma_fcff": sigma_fcff,
        "sigma_fcfe": sigma_fcfe,
        "sigma_eps": sigma_eps,
        "implied_share_price_eps": implied_price_eps,
        "upside_eps": upside_eps,
        "price_eps_p15": eps_low,
        "price_eps_p50": eps_mid,
        "price_eps_p85": eps_high,
        "prob_above_market_eps": eps_probability,
        "price_fcff_p15": fcff_low,
        "price_fcff_p50": fcff_mid,
        "price_fcff_p85": fcff_high,
        "prob_above_market_fcff": fcff_probability,
        "price_fcfe_p15": fcfe_low,
        "price_fcfe_p50": fcfe_mid,
        "price_fcfe_p85": fcfe_high,
        "prob_above_market_fcfe": fcfe_probability,
    })

    if verbose:
        print(f"\n  projected {n:,} tickers x {years} years = {len(forecast):,} rows")
    return forecast, summary


# --------------------------------------------------------------------------

def build(verbose: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    cash_flows = vm.latest(cash_flows=True)
    metrics = vm.latest().set_index("ticker")

    drivers = base_drivers(verbose=verbose)
    drivers = drivers[drivers["ticker"].isin(cash_flows["ticker"])].copy()

    anchor = cash_flows.set_index("ticker")
    drivers["base_date"] = drivers["ticker"].map(anchor["date"])
    # The latest traded close, which is what "is this cheap today?" has to
    # compare against. Sharadar's `price` on a filing row is the price ON THE
    # FILING DATE, which can be months stale - ASML last filed 2026-07-15 at
    # 1,815 while the stock last closed at 1,715, a 5.9% gap. Across the
    # cohort 1,203 of 2,245 tickers differ by more than 5% and 182 by more
    # than 20%, so the filing price was silently mispricing every upside,
    # every P(above market) and the simulation cap - not just the label.
    connection = connect()
    latest_close = connection.execute(
        "SELECT ticker, close AS market_price, date AS market_price_date "
        "FROM (SELECT ticker, close, date, ROW_NUMBER() OVER "
        "(PARTITION BY ticker ORDER BY date DESC) rn FROM prices) "
        "WHERE rn = 1").df().set_index("ticker")
    connection.close()
    drivers["market_price"] = drivers["ticker"].map(latest_close["market_price"])
    drivers["market_price_date"] = drivers["ticker"].map(
        latest_close["market_price_date"])
    drivers["base_revenue"] = drivers["ticker"].map(anchor["revenue_ttm"])
    drivers["base_debt"] = drivers["ticker"].map(anchor["total_debt"]).fillna(0.0)
    for source, target in (("wacc", "wacc"), ("cost_of_equity", "cost_of_equity"),
                           ("cost_of_debt", "cost_of_debt"), ("net_debt", "net_debt"),
                           ("shareswadil", "shares"),
                           ("price", "price_at_filing"),
                           ("marketcap", "marketcap"), ("ev", "ev_market")):
        drivers[target] = drivers["ticker"].map(metrics[source])

    drivers["sector"] = drivers["ticker"].map(_sector_map())
    # Null means "use TERMINAL_GROWTH_TARGET"; the UI writes a number here.
    drivers["terminal_growth_override"] = np.nan
    drivers = drivers.merge(cashflow_sigma(verbose=verbose), on="ticker", how="left")
    for column in ("sigma_fcff", "sigma_fcfe"):
        drivers[column] = drivers[column].fillna(drivers[column].median())

    # Sharadar leaves shareswadil empty for ~570 tickers. Market cap divided by
    # price recovers the share count exactly, since marketcap is built from the
    # same two quantities. Recorded so a consumer can tell which is which.
    reported_shares = drivers["shares"]
    # Share count is derived from the FILING-date price, because the market
    # cap it divides into is also from the filing date. Mixing in today's
    # close would invent a share count that never existed.
    with np.errstate(divide="ignore", invalid="ignore"):
        derived_shares = np.where(
            drivers["price_at_filing"] > 0,
            drivers["marketcap"] / drivers["price_at_filing"], np.nan)
    drivers["shares_source"] = np.where(reported_shares.notna() & (reported_shares > 0),
                                        "reported", "derived_from_marketcap")
    drivers["shares"] = reported_shares.where(
        reported_shares.notna() & (reported_shares > 0), pd.Series(derived_shares,
                                                                  index=drivers.index))
    if verbose:
        derived = int((drivers["shares_source"] == "derived_from_marketcap").sum())
        print(f"  share count: {len(drivers) - derived:,} reported, "
              f"{derived:,} derived from marketcap / price")

    # A ticker with no recent bar falls back to the filing price rather than
    # dropping out of the model entirely.
    drivers["market_price"] = drivers["market_price"].fillna(
        drivers["price_at_filing"])
    required = ["wacc", "cost_of_equity", "cost_of_debt", "base_revenue"]
    before = len(drivers)
    drivers = drivers.dropna(subset=required)
    drivers = drivers[drivers["base_revenue"] > 0]
    if verbose and len(drivers) < before:
        print(f"  dropped {before - len(drivers):,} tickers missing a rate "
              f"or with non-positive base revenue")

    drivers = sector_terminals(drivers, verbose=verbose)
    forecast, summary = project(drivers, verbose=verbose)

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    for table, path in ((forecast, FORECAST_PARQUET), (summary, SUMMARY_PARQUET)):
        table.to_parquet(path, index=False, compression="zstd", compression_level=6)
    # Persisted so the UI's what-if constructor can revalue a single ticker
    # without re-reading 100k filing rows to refit its drivers.
    drivers.to_parquet(DRIVERS_PARQUET, index=False,
                       compression="zstd", compression_level=6)

    if verbose:
        print()
        for table, path in ((forecast, FORECAST_PARQUET), (summary, SUMMARY_PARQUET)):
            print(f"  -> {path.name:<22} {len(table):>7,} rows x "
                  f"{len(table.columns):>2} cols  "
                  f"{path.stat().st_size / 1e6:>5.1f} MB")
    return forecast, summary


# Drivers the UI's what-if constructor exposes, with a human label. The slider
# range for each comes from BOUNDS, so the UI cannot produce an input the model
# would reject.
EDITABLE_DRIVERS = {
    "revenue_growth": "Revenue growth, year 1",
    "terminal_growth_override": "Revenue growth, terminal",
    "ebit_margin": "EBIT margin, year 1",
    "terminal_ebit_margin": "EBIT margin, terminal",
    "capex_to_revenue": "Capex / revenue, year 1",
    "terminal_capex_to_revenue": "Capex / revenue, terminal",
    "dwc_to_revenue": "Change in working capital / revenue",
    "da_growth": "D&A growth",
    "wacc": "WACC",
    "cost_of_equity": "Cost of equity",
}

# Which BOUNDS entry governs each editable driver.
DRIVER_BOUND_KEY = {
    "revenue_growth": "revenue_growth",
    "terminal_growth_override": "terminal_growth",
    "ebit_margin": "ebit_margin",
    "terminal_ebit_margin": "ebit_margin",
    "capex_to_revenue": "capex_to_revenue",
    "terminal_capex_to_revenue": "capex_to_revenue",
    "dwc_to_revenue": "dwc_to_revenue",
    "da_growth": "da_growth",
}


def load_drivers() -> pd.DataFrame:
    if not DRIVERS_PARQUET.exists():
        raise SystemExit("No drivers yet. Run: python3 dcf_forecast.py run")
    return pd.read_parquet(DRIVERS_PARQUET)


def revalue(ticker: str, overrides: dict | None = None,
            drivers: pd.DataFrame | None = None) -> tuple[pd.DataFrame, pd.Series]:
    """Re-run one ticker's DCF with driver overrides. Powers the UI constructor.

    Returns the same (forecast, summary) shape as `project`, for one ticker, so
    the UI can chart a what-if against the stored base case without any
    separate code path - what you see in the constructor is the same model that
    produced the stored numbers.
    """
    if drivers is None:
        drivers = load_drivers()
    row = drivers[drivers["ticker"] == ticker]
    if row.empty:
        raise KeyError(f"{ticker} has no fitted drivers")
    row = row.copy().reset_index(drop=True)

    for field, value in (overrides or {}).items():
        if field not in row.columns:
            raise KeyError(f"{field} is not a driver")
        bound = DRIVER_BOUND_KEY.get(field)
        if bound is not None:
            low, high = BOUNDS[bound]
            value = float(np.clip(value, low, high))
        row.loc[0, field] = value

    forecast, summary = project(row, verbose=False)
    return forecast, summary.iloc[0]


def status() -> None:
    for path in (FORECAST_PARQUET, SUMMARY_PARQUET, DRIVERS_PARQUET):
        if path.exists():
            frame = pd.read_parquet(path)
            print(f"{path.name:<24} {len(frame):>8,} rows x {len(frame.columns):>3} cols "
                  f"{path.stat().st_size / 1e6:>6.1f} MB  "
                  f"{frame['ticker'].nunique():,} tickers")
        else:
            print(f"{path.name:<24} not built")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["run", "status"])
    args = parser.parse_args()
    if args.command == "run":
        build()
    else:
        status()


if __name__ == "__main__":
    main()
