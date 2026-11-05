#!/usr/bin/env python3
"""
Point-in-time fair value for every ticker at every filing date - backtest input.

Runs the same engine as `valuation_engine.py`, but rewound: for each of the
~72,000 (ticker, filing date) pairs it computes the fair value that could
actually have been calculated on that day, using only information published by
then. That is what makes it usable as a backtest signal rather than a
hindsight-fitted curve.

    data/valuation_history.parquet   one row per (ticker, filing date)

The methodology is not re-implemented. `dcf_forecast.project` and the engine's
`classify`, `model_ddm`, `model_rim` and `blend` are all vectorised over rows
and indifferent to whether those rows are 2,258 companies today or 72,000
company-dates across ten years, so this module builds the inputs and calls the
same functions. If the live engine changes, this changes with it - there is no
second copy of the model to drift.

Only the relative model needed a variant: peer medians here must be taken from
the cross-section *as it stood that quarter*, not from today's.

Avoiding lookahead is the whole job:
    drivers          exponentially weighted mean over filings up to that date
    sector anchors   contemporaneous cross-section, that quarter only
    peer multiples   contemporaneous cross-section, that quarter only
    cost of capital  already point-in-time in metrics.parquet
    market price     the close on the filing date
    forward returns  deliberately future, and named so nothing else can use them

Usage
      python3 valuation_history.py run
      python3 valuation_history.py run --min-history 3
      python3 valuation_history.py status
"""

from __future__ import annotations

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

import dcf_forecast as dcf
import valuation_engine as ve
from sharadar_loader import DATA_DIR, connect

HISTORY_PARQUET = DATA_DIR / "valuation_history.parquet"

# The live engine values only the >=6y cohort. Held to six years here the
# backtest could not start until 2022, because the price and filing history
# itself only begins in September 2016. Three years of prior filings is enough
# to fit a trailing driver, and `years_of_history` is on every row so a
# stricter cut can be applied at analysis time without recomputing.
MIN_HISTORY_YEARS = 3.0

# Forward windows in trading sessions, for whatever the backtest wants to
# regress against. These are the only future-looking columns in the file.
FORWARD_WINDOWS = {"1m": 21, "3m": 63, "6m": 126, "12m": 252}


# --------------------------------------------------------------------------
# point-in-time inputs
# --------------------------------------------------------------------------

def _filing_panel() -> pd.DataFrame:
    """Every filing row with the metrics and statement fields the models need.

    metrics.parquet is already point-in-time - each row is what was knowable on
    that filing date, including the cost of capital, which was estimated from a
    beta window ending then. The statement fields the engine needs but metrics
    does not carry (dividends, book value per share, payout) come from the
    fundamentals table on the same key.
    """
    connection = connect()
    metrics = connection.execute("""
        SELECT ticker, date, calendardate, revenue_ttm, netinc_ttm, ebit_ttm,
               ebitda_ttm, fcf_ttm, cfo_ttm, capex_ttm, d_and_a_ttm,
               interest_expense_ttm, total_debt, net_debt, cashneq, equity_book,
               marketcap, ev, price, shareswadil, wacc, cost_of_equity,
               cost_of_debt, pe_ttm, ps_ttm, pb, ev_ebitda_ttm, ev_ebit_ttm,
               ebit_margin_ttm, net_margin_ttm, roe_ttm, roic_ttm,
               debt_to_equity, net_debt_to_ebitda, interest_coverage,
               capex_to_revenue_ttm, years_of_history
        FROM read_parquet(?)
        ORDER BY ticker, date
    """, [(DATA_DIR / "metrics.parquet").as_posix()]).df()

    statements = connection.execute("""
        SELECT ticker, date, fxusd, dps, bvps, payoutratio, workingcapital,
               equity, depamor, capex AS capex_signed, revenue, ebit, debt
        FROM fundamentals
        WHERE dimension = 'ART'
        ORDER BY ticker, date
    """).df()
    connection.close()

    rate = pd.to_numeric(statements["fxusd"], errors="coerce")
    rate = rate.where(rate > 0, 1.0)
    for column in ("dps", "bvps", "workingcapital", "equity", "depamor",
                   "capex_signed", "revenue", "ebit", "debt"):
        statements[column] = statements[column] / rate

    metrics["date"] = pd.to_datetime(metrics["date"])
    statements["date"] = pd.to_datetime(statements["date"])
    panel = metrics.merge(statements.drop(columns=["fxusd"]),
                          on=["ticker", "date"], how="left")
    panel = panel.sort_values(["ticker", "date"]).reset_index(drop=True)

    # metrics.parquet stamps the FULL-SAMPLE span on every row - Apple carries
    # 9.76 years on its 2016 filing as much as on its 2026 one. That is correct
    # for the live engine, which only ever reads the latest row, and is
    # lookahead here: it would let a 2019 valuation claim a decade of history,
    # pass the maturity test, and earn a reliability bonus none of it had. It
    # also gates which rows enter the file at all. Recomputed as elapsed time
    # since the ticker's first filing.
    first_filing = panel.groupby("ticker")["date"].transform("min")
    panel["years_of_history_fullsample"] = panel["years_of_history"]
    panel["years_of_history"] = (panel["date"] - first_filing).dt.days / 365.25
    return panel


def _prior_year_by_date(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Each column's value roughly a year earlier, matched on date not position.

    Same rule as the live pipeline: filing cadence is not four a year for the
    foreign private issuers that file annually, so a positional shift would
    reach back four years for them.
    """
    prior = frame[["ticker", "date", *columns]].copy()
    prior["date"] = prior["date"] + pd.DateOffset(years=1)
    prior = prior.rename(columns={c: f"{c}_prior" for c in columns})
    return pd.merge_asof(
        frame.sort_values("date"), prior.sort_values("date"),
        on="date", by="ticker", direction="backward",
        tolerance=pd.Timedelta(days=dcf.YOY_TOLERANCE_DAYS))


def _expanding_ewm(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Exponentially weighted mean of each column at every row, per ticker.

    This is the historical analogue of the engine's driver fit. Evaluating an
    EWM at row t uses rows up to t and nothing after, so every value in the
    output is one that existed on its own date - the property the whole file
    depends on. Decay is by elapsed days rather than row count so an annual
    filer and a quarterly one lose influence at the same calendar rate.
    """
    out = {}
    halflife = pd.Timedelta(days=dcf.DRIVER_HALFLIFE_DAYS)
    for column in columns:
        values = pd.to_numeric(frame[column], errors="coerce")
        # ewm(times=...) will not accept NaT or NaN values alongside times, so
        # each column is smoothed on its own valid subset and reindexed back.
        parts = []
        for ticker, block in frame.assign(_v=values).groupby("ticker", sort=False):
            valid = block.dropna(subset=["_v"])
            if valid.empty:
                parts.append(pd.Series(np.nan, index=block.index))
                continue
            smoothed = valid["_v"].ewm(halflife=halflife,
                                       times=valid["date"]).mean()
            parts.append(smoothed.reindex(block.index).ffill())
        out[column] = pd.concat(parts).reindex(frame.index)
    return pd.DataFrame(out, index=frame.index)


def build_drivers(panel: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """One driver row per (ticker, filing date), fitted on data up to that date."""
    frame = panel[panel["revenue"] > 0].copy()
    frame = _prior_year_by_date(
        frame, ["revenue", "depamor", "workingcapital", "fcf_ttm", "netinc_ttm"])

    revenue = frame["revenue"]
    frame["revenue_yoy"] = (revenue / frame["revenue_prior"]
                            .where(frame["revenue_prior"] > 0)) - 1.0
    frame["da_yoy"] = (frame["depamor"].abs()
                       / frame["depamor_prior"].abs()
                       .where(frame["depamor_prior"].abs() > 0)) - 1.0
    frame["ebit_margin"] = frame["ebit"] / revenue
    frame["da_to_revenue"] = frame["depamor"].abs() / revenue
    frame["capex_to_revenue"] = frame["capex_signed"].abs() / revenue
    frame["dwc_to_revenue"] = (frame["workingcapital"]
                               - frame["workingcapital_prior"]) / revenue
    frame["debt_to_revenue"] = frame["debt"] / revenue

    driver_columns = ["revenue_yoy", "ebit_margin", "da_yoy", "da_to_revenue",
                      "capex_to_revenue", "dwc_to_revenue", "debt_to_revenue"]
    smoothed = _expanding_ewm(frame, driver_columns).rename(
        columns={"revenue_yoy": "revenue_growth", "da_yoy": "da_growth"})
    frame = frame.drop(columns=driver_columns).join(smoothed)

    frame["dwc_assumed_zero"] = frame["dwc_to_revenue"].isna()
    frame["dwc_to_revenue"] = frame["dwc_to_revenue"].fillna(0.0)
    frame["revenue_growth_assumed"] = frame["revenue_growth"].isna()
    frame["revenue_growth"] = frame["revenue_growth"].fillna(
        dcf.TERMINAL_GROWTH_TARGET)
    frame["ebit_margin"] = frame["ebit_margin"].fillna(0.0)
    frame["da_growth"] = frame["da_growth"].fillna(dcf.TERMINAL_GROWTH_TARGET)
    frame["da_to_revenue"] = frame["da_to_revenue"].fillna(
        dcf.BOUNDS["da_to_revenue"][0])
    frame["capex_to_revenue"] = frame["capex_to_revenue"].fillna(
        dcf.BOUNDS["capex_to_revenue"][0])
    frame["debt_to_revenue"] = frame["debt_to_revenue"].fillna(0.0).clip(lower=0.0)

    for key in ("revenue_growth", "ebit_margin", "da_growth", "da_to_revenue",
                "capex_to_revenue", "dwc_to_revenue"):
        frame[key] = dcf._clip(frame[key], key)

    if verbose:
        print(f"  fitted drivers on {len(frame):,} filing rows")
    return frame


def add_sector_anchors(frame: pd.DataFrame, sectors: pd.Series,
                       verbose: bool = True) -> pd.DataFrame:
    """Terminal driver levels from the cross-section as it stood that quarter.

    The live engine anchors on today's sector medians. Using those here would
    hand a 2019 valuation the margin structure of 2026, which is exactly the
    lookahead this file exists to avoid. Medians are therefore recomputed for
    every quarter, over profitable firms only, as the live engine does.
    """
    frame = frame.copy()
    frame["sector"] = frame["ticker"].map(sectors)
    frame["quarter"] = frame["date"].dt.to_period("Q")

    smoothed = ("ebit_margin", "capex_to_revenue", "dwc_to_revenue")
    population = (frame[frame["ebit_margin"] > 0]
                  if dcf.SECTOR_MEDIAN_PROFITABLE_ONLY else frame)
    grouped = population.groupby(["quarter", "sector"])[list(smoothed)]
    statistics = grouped.agg(["median",
                              lambda s: s.quantile(0.75) - s.quantile(0.25)])
    statistics.columns = [f"{field}_{'median' if kind == 'median' else 'iqr'}"
                          for field, kind in statistics.columns]
    frame = frame.merge(statistics.reset_index(), on=["quarter", "sector"],
                        how="left")

    for field in smoothed:
        median = frame[f"{field}_median"]
        iqr = frame[f"{field}_iqr"].fillna(0.0)
        blended = median + dcf.SECTOR_SHRINK * (frame[field] - median)
        banded = blended.clip(median - dcf.SECTOR_BAND_IQR * iqr,
                              median + dcf.SECTOR_BAND_IQR * iqr)
        frame[f"terminal_{field}"] = dcf._clip(banded.fillna(frame[field]), field)

    if verbose:
        quarters = frame["quarter"].nunique()
        print(f"  sector anchors recomputed for {quarters} quarters")
    return frame


def add_sigma(frame: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """Cash-flow dispersion using only each row's own past.

    An expanding standard deviation of year-on-year log changes: at every row
    it is the volatility observable on that date. The live engine uses the full
    sample, which is correct there and would be lookahead here.
    """
    frame = frame.copy()
    for name, column in (("fcff", "fcf_ttm"), ("fcfe", "fcf_ttm"),
                         ("eps", "netinc_ttm")):
        current = frame[column]
        prior = frame[f"{column}_prior"]
        both_positive = (current > 0) & (prior > 0)
        ratio = np.log(current.where(both_positive) / prior.where(both_positive))
        expanding = (ratio.groupby(frame["ticker"])
                     .expanding().std().reset_index(level=0, drop=True))
        frame[f"sigma_{name}"] = expanding.reindex(frame.index)

    universe_median = {name: frame[f"sigma_{name}"].median()
                       for name in ("fcff", "fcfe", "eps")}
    for name in ("fcff", "fcfe", "eps"):
        column = f"sigma_{name}"
        frame[column] = (frame[column].fillna(universe_median[name])
                         .clip(*dcf.CASHFLOW_SIGMA_BOUNDS))
    if verbose:
        print(f"  sigma: median FCFF {frame['sigma_fcff'].median():.3f}, "
              f"EPS {frame['sigma_eps'].median():.3f}")
    return frame


def add_market_price(frame: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """The close on the filing date, plus forward returns for the backtest."""
    connection = connect()
    prices = connection.execute(
        "SELECT ticker, date, close, closeadj FROM prices ORDER BY ticker, date"
    ).df()
    connection.close()
    prices["date"] = pd.to_datetime(prices["date"])

    # The filing date is not always a session; take the first close on or after.
    matched = pd.merge_asof(
        frame[["ticker", "date"]].reset_index().sort_values("date"),
        prices.rename(columns={"date": "price_date"})
        .assign(date=lambda d: d["price_date"]).sort_values("date"),
        on="date", by="ticker", direction="forward",
        tolerance=pd.Timedelta(days=7))
    matched = matched.set_index("index").reindex(frame.index)

    frame = frame.copy()
    frame["market_price"] = matched["close"].to_numpy()
    frame["market_price_date"] = matched["price_date"].to_numpy()
    frame["price_at_filing"] = frame["market_price"].fillna(frame["price"])
    frame["closeadj_at_filing"] = matched["closeadj"].to_numpy()

    # Forward total returns from the filing date. The only columns in this file
    # that look ahead, and named so they cannot be mistaken for inputs.
    wide = prices.pivot(index="date", columns="ticker", values="closeadj")
    sessions = {date: position for position, date in enumerate(wide.index)}
    start_position = matched["price_date"].map(sessions)
    last_position = len(wide) - 1

    ticker_column = {ticker: position
                     for position, ticker in enumerate(wide.columns)}
    values = wide.to_numpy()
    column_index = frame["ticker"].map(ticker_column).to_numpy()
    base = frame["closeadj_at_filing"].to_numpy(dtype=float)
    positions = start_position.to_numpy(dtype=float)

    for label, offset in FORWARD_WINDOWS.items():
        target = positions + offset
        valid = np.isfinite(target) & np.isfinite(column_index) & (base > 0) \
            & (target <= last_position)
        result = np.full(len(frame), np.nan)
        rows = np.where(valid)[0]
        if len(rows):
            future = values[target[rows].astype(int), column_index[rows].astype(int)]
            result[rows] = future / base[rows] - 1.0
        frame[f"forward_return_{label}"] = result

    if verbose:
        covered = frame["market_price"].notna().mean()
        print(f"  market price matched on {covered:.1%} of rows")
    return frame


# --------------------------------------------------------------------------
# the relative model, quarter by quarter
# --------------------------------------------------------------------------

def model_relative_history(frame: pd.DataFrame) -> pd.DataFrame:
    """Peer-median multiples from the cross-section of that quarter.

    Identical arithmetic to `valuation_engine.model_relative`, with the peer
    group scoped to the same quarter as well as the same industry. Today's
    medians would be a decade of hindsight applied to a 2019 valuation.
    """
    shares = frame["shares"].to_numpy(dtype=float)
    shares = np.where(shares > 0, shares, np.nan)
    net_debt = frame["net_debt"].fillna(0.0).to_numpy(dtype=float)
    quarter = frame["quarter"]

    values, weights, detail = {}, {}, {}
    for multiple, (kind, driver_column) in ve.MULTIPLE_SPECS.items():
        if multiple not in frame.columns or driver_column not in frame.columns:
            continue
        positive = frame[multiple].where(frame[multiple] > 0)

        industry_key = [quarter, frame["industry"]]
        sector_key = [quarter, frame["sector"]]
        industry_median = positive.groupby(industry_key).transform("median")
        industry_count = positive.groupby(industry_key).transform("count")
        sector_median = positive.groupby(sector_key).transform("median")
        sector_count = positive.groupby(sector_key).transform("count")

        median = industry_median.where(industry_count >= ve.MIN_PEERS_INDUSTRY,
                                       sector_median)
        count = industry_count.where(industry_count >= ve.MIN_PEERS_INDUSTRY,
                                     sector_count)

        driver = frame[driver_column].to_numpy(dtype=float)
        peer_multiple = median.to_numpy(dtype=float)
        if kind == "enterprise":
            implied = (peer_multiple * driver - net_debt) / shares
        else:
            implied = peer_multiple * driver / shares
        implied = np.where((driver > 0) & (peer_multiple > 0)
                           & np.isfinite(implied) & (implied > 0), implied, np.nan)

        depth = np.clip(count.to_numpy(dtype=float) / 20.0, 0.0, 1.0)
        values[multiple] = implied
        weights[multiple] = np.where(np.isfinite(implied), 0.4 + 0.6 * depth, 0.0)
        detail[f"rel_{multiple}_value"] = implied

    if not values:
        return pd.DataFrame(index=frame.index)

    stacked = np.vstack([values[m] for m in values])
    stacked_weights = np.where(np.isfinite(stacked),
                               np.vstack([weights[m] for m in values]), 0.0)
    total = stacked_weights.sum(axis=0)
    blended = np.where(total > 0,
                       np.nansum(stacked * stacked_weights, axis=0)
                       / np.where(total > 0, total, 1.0), np.nan)
    used = np.isfinite(stacked).sum(axis=0)

    with np.errstate(invalid="ignore", divide="ignore"):
        spread = np.nanmax(stacked, axis=0) / np.nanmin(stacked, axis=0)
    agreement = np.clip(1.5 - np.nan_to_num(spread, nan=5.0) / 4.0, 0.0, 1.0)
    reliability = np.where(used > 0,
                           0.55 * agreement + 0.45 * np.clip(used / 3.0, 0.0, 1.0),
                           0.0)

    out = pd.DataFrame(detail, index=frame.index)
    out["relative_value"] = blended
    out["relative_reliability"] = np.clip(reliability, 0.0, 1.0)
    out["relative_multiples_used"] = used
    out["fair_value_multiples"] = blended
    return out


# --------------------------------------------------------------------------

def build(min_history: float = MIN_HISTORY_YEARS, verbose: bool = True) -> pd.DataFrame:
    panel = _filing_panel()
    if verbose:
        print(f"  {len(panel):,} filing rows, {panel['ticker'].nunique():,} tickers, "
              f"{panel['date'].min().date()} to {panel['date'].max().date()}")

    reference = pd.read_parquet(DATA_DIR / "analytics.parquet",
                                columns=["ticker", "sector", "industry"])
    sectors = reference.set_index("ticker")["sector"]
    industries = reference.set_index("ticker")["industry"]

    frame = build_drivers(panel, verbose=verbose)
    frame = add_sector_anchors(frame, sectors, verbose=verbose)
    frame["industry"] = frame["ticker"].map(industries)
    frame = add_sigma(frame, verbose=verbose)
    frame = add_market_price(frame, verbose=verbose)

    # Enough prior filings to have fitted a trailing driver at all.
    before = len(frame)
    frame = frame[(frame["years_of_history"] >= min_history)
                  & frame["market_price"].notna()
                  & frame["wacc"].notna() & frame["cost_of_equity"].notna()
                  & (frame["revenue"] > 0)].reset_index(drop=True)
    if verbose:
        print(f"  {len(frame):,} rows survive the {min_history:g}y history gate "
              f"(dropped {before - len(frame):,})")

    # --- shape into exactly what dcf.project expects ---------------------
    frame["base_date"] = frame["date"]
    frame["base_revenue"] = frame["revenue"]
    frame["base_debt"] = frame["debt"].fillna(0.0)
    frame["ev_market"] = frame["ev"]
    frame["terminal_growth_override"] = np.nan
    reported_shares = frame["shareswadil"]
    derived = np.where(frame["price_at_filing"] > 0,
                       frame["marketcap"] / frame["price_at_filing"], np.nan)
    frame["shares_source"] = np.where(
        reported_shares.notna() & (reported_shares > 0), "reported",
        "derived_from_marketcap")
    frame["shares"] = reported_shares.where(
        reported_shares.notna() & (reported_shares > 0),
        pd.Series(derived, index=frame.index))

    forecast, summary = dcf.project(frame, verbose=False)
    if verbose:
        print(f"  projected {len(summary):,} point-in-time DCFs")

    # --- the engine, unchanged -------------------------------------------
    engine_input = frame[[
        "ticker", "date", "quarter", "sector", "industry", "years_of_history",
        "netinc_ttm", "ebitda_ttm", "ebit_ttm", "revenue_ttm", "equity_book",
        "net_debt", "marketcap", "pe_ttm", "ps_ttm", "pb", "ev_ebitda_ttm",
        "ev_ebit_ttm", "ebit_margin_ttm", "net_margin_ttm", "roe_ttm",
        "roic_ttm", "debt_to_equity", "net_debt_to_ebitda", "interest_coverage",
        "capex_to_revenue_ttm", "cost_of_equity", "wacc", "bvps", "dps",
        "payoutratio", "shares",
    ] + [f"forward_return_{k}" for k in FORWARD_WINDOWS]].copy()

    engine_input["base_revenue_growth"] = summary["base_revenue_growth"].to_numpy()
    engine_input["market_price"] = summary["market_price"].to_numpy()
    engine_input["implied_share_price_fcfe"] = \
        summary["implied_share_price_fcfe"].to_numpy()
    engine_input["implied_share_price_fcff"] = \
        summary["implied_share_price_fcff"].to_numpy()
    engine_input["implied_share_price_eps"] = \
        summary["implied_share_price_eps"].to_numpy()
    engine_input["terminal_share_of_ev"] = summary["terminal_share_of_ev"].to_numpy()
    engine_input["sigma_fcff"] = summary["sigma_fcff"].to_numpy()
    engine_input["sigma_fcfe"] = summary["sigma_fcfe"].to_numpy()

    # Dividend record as it stood: paid-share and growth over the trailing
    # eight filings only, so a company that started paying in 2023 does not
    # look like a payer in 2019.
    engine_input["dps_ttm"] = engine_input["dps"]
    grouped = engine_input.groupby("ticker", sort=False)["dps"]
    engine_input["dps_paid_share"] = grouped.transform(
        lambda s: (s.fillna(0) > 0).rolling(8, min_periods=2).mean())
    engine_input["dps_growth"] = grouped.transform(
        lambda s: s.where(s > 0).pct_change(4))
    engine_input["payout_ratio"] = engine_input["payoutratio"]
    engine_input["roe_volatility"] = (
        engine_input.groupby("ticker", sort=False)["roe_ttm"]
        .transform(lambda s: s.expanding().std()))

    classified = ve.classify(engine_input)
    for part in (ve.model_ddm(classified), ve.model_rim(classified),
                 model_relative_history(classified)):
        classified = classified.join(part)
    classified["fcfe_value"] = classified["implied_share_price_fcfe"]
    classified["fcff_value"] = classified["implied_share_price_fcff"]

    passthrough = ve.model_passthrough(classified)
    classified["fcfe_reliability"] = passthrough["fcfe_reliability"]
    classified["fcff_reliability"] = passthrough["fcff_reliability"]
    for model in ve.MODELS:
        if f"{model}_value" not in classified.columns:
            classified[f"{model}_value"] = np.nan
        if f"{model}_reliability" not in classified.columns:
            classified[f"{model}_reliability"] = 0.0

    result = classified.join(ve.blend(classified, verbose=verbose))

    keep = (["ticker", "date", "quarter", "sector", "industry", "profile",
             "years_of_history", "market_price", "shares", "marketcap",
             "cost_of_equity", "wacc", "fair_value", "fair_value_upside",
             "fair_value_multiples", "confidence", "models_used",
             "weighted_reliability", "model_agreement",
             "pe_ttm", "ps_ttm", "pb", "ev_ebitda_ttm", "roe_ttm", "roic_ttm",
             "ebit_margin_ttm", "net_margin_ttm", "revenue_ttm", "netinc_ttm"]
            + [f"{m}_value" for m in ve.MODELS]
            + [f"{m}_weight" for m in ve.MODELS]
            + [f"{m}_reliability" for m in ve.MODELS]
            + [f"forward_return_{k}" for k in FORWARD_WINDOWS])

    output = result[[c for c in keep if c in result.columns]].copy()
    output["quarter"] = output["quarter"].astype(str)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    output.to_parquet(HISTORY_PARQUET, index=False,
                      compression="zstd", compression_level=6)
    if verbose:
        valued = output["fair_value"].notna().sum()
        print(f"\n  -> {HISTORY_PARQUET.name}  {len(output):,} rows x "
              f"{len(output.columns)} cols  "
              f"{HISTORY_PARQUET.stat().st_size / 1e6:.1f} MB")
        print(f"     {valued:,} rows with a fair value, "
              f"{output['ticker'].nunique():,} tickers, "
              f"{output['date'].min().date()} to {output['date'].max().date()}")
    return output


def load() -> pd.DataFrame:
    if not HISTORY_PARQUET.exists():
        raise SystemExit("No history. Run: python3 valuation_history.py run")
    return pd.read_parquet(HISTORY_PARQUET)


def status() -> None:
    if not HISTORY_PARQUET.exists():
        print("valuation_history.parquet   not built")
        return
    frame = load()
    print(f"valuation_history.parquet  {len(frame):,} rows x {len(frame.columns)} cols "
          f"{HISTORY_PARQUET.stat().st_size / 1e6:.1f} MB")
    print(f"  tickers        : {frame['ticker'].nunique():,}")
    print(f"  dates          : {frame['date'].min().date()} to "
          f"{frame['date'].max().date()}")
    print(f"  with fair value: {frame['fair_value'].notna().sum():,}")
    print(f"  forward 12m    : {frame['forward_return_12m'].notna().sum():,}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["run", "status"])
    parser.add_argument("--min-history", type=float, default=MIN_HISTORY_YEARS)
    args = parser.parse_args()
    if args.command == "run":
        build(min_history=args.min_history)
    else:
        status()


if __name__ == "__main__":
    main()
