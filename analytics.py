#!/usr/bin/env python3
"""
Per-ticker market analytics: returns, volatility, drawdown, sector labels.

Precomputed rather than derived in the UI, so the app opens instantly instead
of scanning 5.4M price rows on every interaction.

    data/analytics.parquet   one row per ticker

Usage
      python3 analytics.py run
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

import valuation_metrics as vm
from sharadar_loader import DATA_DIR, connect, fetch

ANALYTICS_PARQUET = DATA_DIR / "analytics.parquet"
GROUP_RETURNS_PARQUET = DATA_DIR / "group_returns.parquet"

# Trading sessions per period. US markets run ~252 sessions a year.
WINDOWS = {"1d": 1, "1w": 5, "1m": 21, "3m": 63, "6m": 126, "1y": 252}
VOL_WINDOW = 252
TRADING_DAYS = 252


def _reference_data() -> pd.DataFrame:
    """Names and sectors from the Sharadar securities master."""
    master = fetch("tickers", use_cache=False,
                   **{"table": "stocks", "sort": "permaticker.asc",
                      "fields": "ticker,name,exchange,sector,industry,famaindustry"})
    return master.dropna(subset=["ticker"]).drop_duplicates("ticker")


def build(verbose: bool = True) -> pd.DataFrame:
    connection = connect()
    prices = connection.execute(
        "SELECT ticker, date, closeadj, close, volume FROM prices ORDER BY date").df()
    connection.close()

    wide = prices.pivot(index="date", columns="ticker", values="closeadj").sort_index()
    latest_date = wide.index[-1]
    last = wide.iloc[-1]

    rows = {"last_close": last}
    for label, sessions in WINDOWS.items():
        if len(wide) > sessions:
            past = wide.iloc[-1 - sessions]
            rows[f"return_{label}"] = last / past.replace(0, np.nan) - 1.0
        else:
            rows[f"return_{label}"] = pd.Series(np.nan, index=wide.columns)

    daily = wide.pct_change()
    tail = daily.tail(VOL_WINDOW)
    rows["volatility_1y"] = tail.std() * np.sqrt(TRADING_DAYS)
    rows["volatility_3m"] = daily.tail(63).std() * np.sqrt(TRADING_DAYS)
    # Worst peak-to-trough over the last year - a plain standard deviation says
    # nothing about whether the downside actually arrived.
    window = wide.tail(VOL_WINDOW)
    rows["max_drawdown_1y"] = (window / window.cummax() - 1.0).min()
    rows["sessions_available"] = wide.notna().sum()

    # Dollar volume says more about tradability than share count.
    dollar = (prices.assign(dollar=prices["close"] * prices["volume"])
              .pivot(index="date", columns="ticker", values="dollar"))
    rows["avg_dollar_volume_3m"] = dollar.tail(63).mean()

    analytics = pd.DataFrame(rows)
    analytics.index.name = "ticker"
    analytics = analytics.reset_index()
    analytics["as_of"] = latest_date

    # --- valuation and fundamentals -------------------------------------
    metrics = vm.latest()
    keep = ["ticker", "marketcap", "ev", "revenue_ttm", "netinc_ttm", "ebitda_ttm",
            "fcf_ttm", "rnd_to_revenue_ttm", "pe_ttm", "ps_ttm", "pb",
            "ev_ebitda_ttm", "ev_ebit_ttm", "ev_sales_ttm", "p_fcf_ttm",
            "fcf_yield", "earnings_yield", "gross_margin_ttm", "ebitda_margin_ttm",
            "ebit_margin_ttm", "net_margin_ttm", "roe_ttm", "roa_ttm", "roic_ttm",
            "debt_to_equity", "net_debt_to_ebitda", "interest_coverage",
            "beta", "beta_raw", "alpha_annual", "beta_r2", "wacc", "cost_of_equity",
            "cost_of_debt", "risk_free_rate", "date"]
    analytics = analytics.merge(
        metrics[[c for c in keep if c in metrics.columns]]
        .rename(columns={"date": "last_filing"}), on="ticker", how="left")

    # --- DCF ------------------------------------------------------------
    summary = pd.read_parquet(DATA_DIR / "dcf_summary.parquet")
    dcf_columns = ["ticker", "implied_share_price_fcff", "implied_share_price_fcfe",
                   "implied_share_price_eps",
                   "price_fcff_p15", "price_fcff_p50", "price_fcff_p85",
                   "price_fcfe_p15", "price_fcfe_p50", "price_fcfe_p85",
                   "price_eps_p15", "price_eps_p50", "price_eps_p85",
                   "prob_above_market_fcff", "prob_above_market_fcfe",
                   "prob_above_market_eps",
                   "upside_fcff", "upside_fcfe", "upside_eps", "terminal_growth",
                   "terminal_share_of_ev", "dcf_caution_sector", "shares",
                   "shares_source", "sigma_fcff", "sigma_fcfe", "sigma_eps",
                   "eps_year1", "base_ebit_margin", "terminal_ebit_margin",
                   "sector_median_ebit_margin"]
    analytics = analytics.merge(
        summary[[c for c in dcf_columns if c in summary.columns]],
        on="ticker", how="left")

    # --- multi-model valuation engine ------------------------------------
    # Optional: analytics still builds if the engine has not been run yet, so
    # the pipeline stays valid in either order on a fresh checkout.
    engine_path = DATA_DIR / "valuation_engine.parquet"
    if engine_path.exists():
        engine = pd.read_parquet(engine_path)
        engine_columns = ["ticker", "profile", "fair_value", "fair_value_upside",
                          "fair_value_multiples", "confidence", "models_used",
                          "weighted_reliability", "model_agreement", "rationale",
                          "fcfe_weight", "fcff_weight", "ddm_weight",
                          "rim_weight", "relative_weight",
                          "ddm_value", "rim_value", "relative_value"]
        analytics = analytics.merge(
            engine[[c for c in engine_columns if c in engine.columns]],
            on="ticker", how="left")

    # --- post-report performance ----------------------------------------
    # The filing date is the closest thing this dataset has to an earnings
    # date, so "after the report" is measured from the first session on or
    # after the filing. Sessions rather than calendar days, so a Friday filing
    # is not credited with a weekend.
    filings = analytics[["ticker", "last_filing"]].dropna()
    session_index = {d: i for i, d in enumerate(wide.index)}
    positions = pd.to_datetime(filings["last_filing"]).map(
        lambda d: next((session_index[s] for s in wide.index if s >= d), None))
    drift = {"return_1d_after_report": 1, "return_5d_after_report": 5,
             "return_20d_after_report": 20}
    for label in drift:
        analytics[label] = np.nan
    analytics["return_since_report"] = np.nan

    last_position = len(wide) - 1
    for ticker, start in zip(filings["ticker"], positions):
        if start is None or ticker not in wide.columns:
            continue
        base = wide[ticker].iloc[start]
        if not np.isfinite(base) or base == 0:
            continue
        row_index = analytics.index[analytics["ticker"] == ticker]
        for label, offset in drift.items():
            end = start + offset
            if end <= last_position:
                analytics.loc[row_index, label] = wide[ticker].iloc[end] / base - 1.0
        analytics.loc[row_index, "return_since_report"] = (
            wide[ticker].iloc[last_position] / base - 1.0)
    analytics["days_since_report"] = (
        pd.Timestamp(latest_date) - pd.to_datetime(analytics["last_filing"])).dt.days

    reference = _reference_data()
    analytics = analytics.merge(
        reference[["ticker", "name", "exchange", "sector", "industry"]],
        on="ticker", how="left")

    # Only names that survived the screen are investable here.
    universe = pd.read_csv(_Path(__file__).resolve().parent / "universe.csv")
    analytics["in_universe"] = analytics["ticker"].isin(universe["ticker"])

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    analytics.to_parquet(ANALYTICS_PARQUET, index=False,
                         compression="zstd", compression_level=6)
    if verbose:
        print(f"  -> {ANALYTICS_PARQUET.name}  {len(analytics):,} tickers x "
              f"{len(analytics.columns)} cols  "
              f"{ANALYTICS_PARQUET.stat().st_size / 1e6:.1f} MB")
        print(f"     as of {pd.Timestamp(latest_date).date()}  |  "
              f"{int(analytics['in_universe'].sum()):,} in universe  |  "
              f"{int(analytics['implied_share_price_fcff'].notna().sum()):,} with a DCF")
    return analytics


def build_group_returns(analytics: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """Daily cumulative return series for every sector and industry.

    Three weightings, because they answer different questions:

      median          what a typical company in the group did
      mean            the same, but sensitive to one runaway name
      cap-weighted    what the group did as a block of capital

    Cap weights are today's market caps held fixed across history. That is not
    a real index - a true one reweights as caps move, and would need
    point-in-time shares outstanding. Fixed weights tilt the series toward
    whatever has already grown, so read it as "how the current heavyweights
    performed", not as an investable benchmark. The median line carries no such
    tilt, which is why it is the default.
    """
    connection = connect()
    prices = connection.execute(
        "SELECT ticker, date, closeadj FROM prices ORDER BY date").df()
    connection.close()

    wide = prices.pivot(index="date", columns="ticker", values="closeadj").sort_index()
    daily = wide.pct_change()
    # A single session moving more than 100% is a data artefact, not a return,
    # and one of them would distort a whole sector's cumulative line.
    daily = daily.clip(-0.60, 1.00)

    caps = analytics.set_index("ticker")["marketcap"]
    frames = []
    for group_type, column in (("sector", "sector"), ("industry", "industry")):
        labels = analytics.dropna(subset=[column]).set_index("ticker")[column]
        for group, members in labels.groupby(labels):
            tickers = [t for t in members.index if t in daily.columns]
            if len(tickers) < 3:      # too thin to describe a group
                continue
            block = daily[tickers]
            weights = caps.reindex(tickers).fillna(0.0)
            total = weights.sum()
            weighted = ((block * weights).sum(axis=1) / total
                        if total > 0 else block.mean(axis=1))
            frames.append(pd.DataFrame({
                "group_type": group_type,
                "group": group,
                "date": block.index,
                "cum_median": (1 + block.median(axis=1).fillna(0)).cumprod() - 1,
                "cum_mean": (1 + block.mean(axis=1).fillna(0)).cumprod() - 1,
                "cum_capweighted": (1 + weighted.fillna(0)).cumprod() - 1,
                "members": len(tickers),
            }))

    result = pd.concat(frames, ignore_index=True)
    result.to_parquet(GROUP_RETURNS_PARQUET, index=False,
                      compression="zstd", compression_level=6)
    if verbose:
        groups = result.groupby("group_type")["group"].nunique().to_dict()
        print(f"  -> {GROUP_RETURNS_PARQUET.name}  {len(result):,} rows  "
              f"{GROUP_RETURNS_PARQUET.stat().st_size / 1e6:.1f} MB  {groups}")
    return result


def load_group_returns() -> pd.DataFrame:
    if not GROUP_RETURNS_PARQUET.exists():
        raise SystemExit("No group returns yet. Run: python3 analytics.py run")
    return pd.read_parquet(GROUP_RETURNS_PARQUET)


def group_period_returns(group_returns: pd.DataFrame, group_type: str,
                         weighting: str = "cum_median") -> pd.DataFrame:
    """Period returns, volatility and drawdown per group, from the daily series.

    Same shape as the company table, so a sector reads like a stock.
    """
    frame = group_returns[group_returns["group_type"] == group_type]
    wide = frame.pivot(index="date", columns="group", values=weighting).sort_index()
    level = 1.0 + wide                      # cumulative return -> index level

    rows = {"members": frame.groupby("group")["members"].last()}
    for label, sessions in WINDOWS.items():
        rows[f"return_{label}"] = (level.iloc[-1] / level.iloc[-1 - sessions] - 1.0
                                   if len(level) > sessions else np.nan)
    daily = level.pct_change()
    rows["volatility_1y"] = daily.tail(VOL_WINDOW).std() * np.sqrt(TRADING_DAYS)
    recent = level.tail(VOL_WINDOW)
    rows["max_drawdown_1y"] = (recent / recent.cummax() - 1.0).min()
    rows["cumulative_total"] = level.iloc[-1] / level.iloc[0] - 1.0

    out = pd.DataFrame(rows)
    out.index.name = group_type
    return out.reset_index()


def load() -> pd.DataFrame:
    if not ANALYTICS_PARQUET.exists():
        raise SystemExit("No analytics yet. Run: python3 analytics.py run")
    return pd.read_parquet(ANALYTICS_PARQUET)


def sector_industry_performance(analytics: pd.DataFrame, weighting: str = "marketcap",
                       min_marketcap: float = 0.0, sector_industry: str = "sector") -> pd.DataFrame:
    """Sector returns, weighted by market cap or as a plain median/average.

    Market-cap weighting answers "what did the sector do"; the median answers
    "what did a typical company in it do". They diverge sharply when a sector
    is dominated by a few names, which is exactly when the question matters.
    """
    frame = analytics[analytics["marketcap"] >= min_marketcap].dropna(subset=[sector_industry])
    columns = [f"return_{label}" for label in WINDOWS]

    if weighting == "marketcap":
        def aggregate(group: pd.DataFrame) -> pd.Series:
            weights = group["marketcap"].fillna(0.0)
            out = {}
            for column in columns:
                values = group[column]
                mask = values.notna() & (weights > 0)
                total = weights[mask].sum()
                out[column] = ((values[mask] * weights[mask]).sum() / total
                               if total > 0 else np.nan)
            return pd.Series(out)
        result = frame.groupby(sector_industry)[columns + ["marketcap"]].apply(aggregate)
    elif weighting == "median":
        result = frame.groupby(sector_industry)[columns].median()
    else:
        result = frame.groupby(sector_industry)[columns].mean()

    result["companies"] = frame.groupby(sector_industry).size()
    result["total_marketcap"] = frame.groupby(sector_industry)["marketcap"].sum()
    return result.reset_index()


def movers(analytics: pd.DataFrame, period: str = "1m", sector: str | None = None,
           min_marketcap: float = 0.0, count: int = 10) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Best and worst performers over a period, filtered by size and sector."""
    column = f"return_{period}"
    frame = analytics[analytics["marketcap"] >= min_marketcap].dropna(subset=[column])
    if sector and sector != "All":
        frame = frame[frame["sector"] == sector]
    show = ["ticker", "name", "sector", "marketcap", column, "last_close"]
    return (frame.nlargest(count, column)[show], frame.nsmallest(count, column)[show])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["run"])
    parser.parse_args()
    frame = build()
    build_group_returns(frame)


if __name__ == "__main__":
    main()
