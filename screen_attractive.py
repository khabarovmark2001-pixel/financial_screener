#!/usr/bin/env python3
"""
Evidence-backed attractiveness screen.

Built only from the variables that survived the factor study in
`return_drivers.ipynb`. Every feature here cleared |t| > 2 against 12-month
forward returns on 61,508 point-in-time observations, Newey-West adjusted for
overlapping windows. Everything that did not clear that bar is deliberately
absent, including things a screener would normally lean on:

    dropped, no predictive power     ROE, ROIC, EBIT margin, net margin
    dropped, no predictive power     confidence, weighted_reliability,
                                     models_used, model_agreement
    dropped, collinear               earnings yield (positive alone, flips sign
                                     next to sales and book yield)

    data/attractive.parquet          scored universe, one row per ticker

Usage
      python3 screen_attractive.py run
      python3 screen_attractive.py run --tier "Small+ ($2B)" --top 40
      python3 screen_attractive.py status
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

import analytics
import valuation_engine as ve
from sharadar_loader import DATA_DIR

ATTRACTIVE_PARQUET = DATA_DIR / "attractive.parquet"

# Mean 12-month information coefficient measured in return_drivers.ipynb.
# Recorded for reference and for the report - NOT used as weights. IC-weighting
# a seven-factor composite on 25 quarters fits the noise in those estimates;
# equal weighting across features that all cleared significance is the more
# robust choice and is what SCORE_FEATURES below implements.
MEASURED_IC = {
    "sales_yield": 0.131,
    "upside_multiples": 0.089,
    "upside_blended": 0.084,
    "book_yield": 0.082,
    "upside_rim": 0.074,
    "upside_fcff": 0.066,
    "upside_fcfe": 0.053,
}
SCORE_FEATURES = list(MEASURED_IC)

# Where the signal was measured to work, by sector. Technology's IC was 0.018
# with t = 0.50 - indistinguishable from zero - so a high score on a technology
# stock is not evidence of anything and is labelled as such rather than shown
# beside a Utilities score as if the two meant the same.
SECTOR_IC = {
    "Utilities": 0.143, "Financial Services": 0.114, "Energy": 0.106,
    "Consumer Cyclical": 0.074, "Healthcare": 0.070, "Consumer Defensive": 0.069,
    "Basic Materials": 0.062, "Communication Services": 0.056,
    "Industrials": 0.048, "Real Estate": 0.042, "Technology": 0.018,
}
SECTOR_UNRELIABLE = {"Technology", "Real Estate", "Industrials"}

# The study found the signal concentrated in micro caps (IC 0.115) - which is
# also where spreads are widest and capacity smallest. Liquidity floor rather
# than a size floor: it is tradability that binds, not market cap as such.
MIN_DOLLAR_VOLUME = 5e6


def _safe_ratio(numerator, denominator):
    denominator = pd.to_numeric(denominator, errors="coerce").replace(0, np.nan)
    return pd.to_numeric(numerator, errors="coerce") / denominator


def build_features(frame: pd.DataFrame) -> pd.DataFrame:
    """The seven surviving features, on the current cross-section."""
    frame = frame.copy()
    price = frame["market_price"]

    frame["sales_yield"] = _safe_ratio(1.0, frame["ps_ttm"])
    frame["book_yield"] = _safe_ratio(1.0, frame["pb"])
    frame["upside_blended"] = frame["fair_value_upside"]
    frame["upside_multiples"] = _safe_ratio(frame["fair_value_multiples"], price) - 1
    for model in ("fcfe", "fcff", "rim"):
        frame[f"upside_{model}"] = _safe_ratio(frame[f"{model}_value"], price) - 1
    return frame


# An industry needs at least this many names before it can define its own
# median; below it the comparison falls back to the sector.
MIN_INDUSTRY_PEERS = 8

# The decile study found returns peaking at decile 7 and *falling* in deciles 8
# and 9, where median claimed upside was +147% and +301%. Beyond this point a
# large upside stops being evidence of cheapness and starts being evidence that
# the model does not describe the company. Flagged rather than dropped, because
# a handful are genuine.
IMPLAUSIBLE_UPSIDE = 3.0


def score(frame: pd.DataFrame, sector_neutral: bool = True) -> pd.DataFrame:
    """Percentile-rank each feature, then average.

    Ranked **within industry**, falling back to sector where the industry is too
    thin. Sector alone is not a tight enough comparison: drug distributors trade
    on a structurally low price-to-sales because they book the whole drug price
    as revenue on a 1% margin, so against the rest of Healthcare they look
    permanently cheap. Ranked against other distributors they look ordinary,
    which is the truth. This is the same industry-first logic the valuation
    engine uses to pick peers.

    Equal weights, for the reason in MEASURED_IC.
    """
    frame = frame.copy()
    if sector_neutral:
        industry_counts = frame.groupby("industry")["ticker"].transform("size")
        group = frame["industry"].where(industry_counts >= MIN_INDUSTRY_PEERS,
                                        frame["sector"])
        frame["peer_group"] = group
        frame["peer_group_kind"] = np.where(
            industry_counts >= MIN_INDUSTRY_PEERS, "industry", "sector")
    else:
        group = pd.Series("all", index=frame.index)
        frame["peer_group"] = "all"
        frame["peer_group_kind"] = "market"

    ranks = pd.DataFrame(index=frame.index)
    for feature in SCORE_FEATURES:
        values = frame[feature]
        # Only positive yields are meaningful: a negative book yield means
        # negative equity, which is not cheapness.
        if feature.endswith("_yield"):
            values = values.where(values > 0)
        ranks[feature] = values.groupby(group).rank(pct=True)

    frame["attractiveness"] = ranks.mean(axis=1, skipna=True)
    frame["features_scored"] = ranks.notna().sum(axis=1)
    for feature in SCORE_FEATURES:
        frame[f"rank_{feature}"] = ranks[feature]

    # A score built from two of seven features is not the same measurement as
    # one built from all seven.
    frame.loc[frame["features_scored"] < 4, "attractiveness"] = np.nan

    frame["sector_signal_ic"] = frame["sector"].map(SECTOR_IC)
    frame["signal_reliable_here"] = ~frame["sector"].isin(SECTOR_UNRELIABLE)

    # Two conditions that make the score unreadable rather than merely uncertain.
    frame["negative_book_equity"] = frame["pb"].fillna(1.0) < 0
    frame["implausible_upside"] = (
        frame["fair_value_upside"].fillna(0) > IMPLAUSIBLE_UPSIDE)
    frame["clean"] = ~(frame["negative_book_equity"] | frame["implausible_upside"])
    return frame


def build(verbose: bool = True) -> pd.DataFrame:
    engine = ve.load()
    market = analytics.load()

    columns = ["ticker", "name", "sector", "industry", "marketcap", "last_close",
               "avg_dollar_volume_3m", "return_1y", "volatility_1y",
               "max_drawdown_1y", "ps_ttm", "pb", "pe_ttm", "ev_ebitda_ttm",
               "roic_ttm", "ebit_margin_ttm", "net_debt_to_ebitda",
               "interest_coverage", "in_universe"]
    frame = engine.merge(market[[c for c in columns if c in market.columns]],
                         on="ticker", how="inner", suffixes=("", "_mkt"))
    frame = frame[frame["in_universe"]]

    frame = score(build_features(frame))
    frame["liquid_enough"] = frame["avg_dollar_volume_3m"].fillna(0) >= MIN_DOLLAR_VOLUME

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(ATTRACTIVE_PARQUET, index=False,
                     compression="zstd", compression_level=6)
    if verbose:
        scored = frame["attractiveness"].notna().sum()
        print(f"  scored {scored:,} of {len(frame):,} tickers on "
              f"{len(SCORE_FEATURES)} evidence-backed features")
        print(f"  -> {ATTRACTIVE_PARQUET.name}  "
              f"{ATTRACTIVE_PARQUET.stat().st_size / 1e6:.1f} MB")
    return frame


def load() -> pd.DataFrame:
    if not ATTRACTIVE_PARQUET.exists():
        raise SystemExit("Not built. Run: python3 screen_attractive.py run")
    return pd.read_parquet(ATTRACTIVE_PARQUET)


def shortlist(frame: pd.DataFrame | None = None, top: int = 25,
              min_marketcap: float = 2e9, reliable_sectors_only: bool = True,
              require_liquidity: bool = True,
              exclude_flagged: bool = True) -> pd.DataFrame:
    """The names the evidence actually supports, most attractive first."""
    frame = load() if frame is None else frame
    pool = frame.dropna(subset=["attractiveness"])
    pool = pool[pool["marketcap"] >= min_marketcap]
    if require_liquidity:
        pool = pool[pool["liquid_enough"]]
    if reliable_sectors_only:
        pool = pool[pool["signal_reliable_here"]]
    if exclude_flagged:
        pool = pool[pool["clean"]]
    return pool.nlargest(top, "attractiveness")


def report(top: int = 25, min_marketcap: float = 2e9) -> None:
    frame = load()
    picks = shortlist(frame, top=top, min_marketcap=min_marketcap)

    print(f"Most attractive {len(picks)} names "
          f"(market cap >= ${min_marketcap/1e9:.0f}bn, liquid, "
          f"sectors where the signal measured reliably)\n")
    show = picks[["ticker", "name", "sector", "attractiveness", "marketcap",
                  "last_close", "fair_value", "fair_value_upside", "ps_ttm",
                  "pb", "pe_ttm", "return_1y"]].copy()
    show["marketcap"] = show["marketcap"].map(lambda v: f"${v/1e9:,.1f}bn")
    show["last_close"] = show["last_close"].map(lambda v: f"${v:,.2f}")
    show["fair_value"] = show["fair_value"].map(lambda v: f"${v:,.2f}")
    for column in ("fair_value_upside", "return_1y"):
        show[column] = show[column].map(
            lambda v: f"{v:+.0%}" if pd.notna(v) else "-")
    for column in ("attractiveness",):
        show[column] = show[column].map(lambda v: f"{v:.3f}")
    for column in ("ps_ttm", "pb", "pe_ttm"):
        show[column] = show[column].map(
            lambda v: f"{v:,.1f}" if pd.notna(v) else "-")
    show.columns = ["Ticker", "Company", "Sector", "Score", "Mkt cap", "Price",
                    "Fair value", "Upside", "P/S", "P/B", "P/E", "1y"]
    print(show.to_string(index=False))

    print("\nBy sector, where the signal measured strongest:")
    by_sector = (picks.groupby("sector")
                 .agg(names=("ticker", "size"),
                      median_score=("attractiveness", "median"),
                      measured_ic=("sector_signal_ic", "first"))
                 .sort_values("measured_ic", ascending=False))
    print(by_sector.round(3).to_string())


def status() -> None:
    if not ATTRACTIVE_PARQUET.exists():
        print("attractive.parquet   not built")
        return
    frame = load()
    print(f"attractive.parquet   {len(frame):,} tickers, "
          f"{frame['attractiveness'].notna().sum():,} scored")
    print(f"  liquid (>= ${MIN_DOLLAR_VOLUME/1e6:.0f}m/day): "
          f"{int(frame['liquid_enough'].sum()):,}")
    print(f"  in sectors where the signal is reliable: "
          f"{int(frame['signal_reliable_here'].sum()):,}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["run", "report", "status"])
    parser.add_argument("--top", type=int, default=25)
    parser.add_argument("--min-marketcap", type=float, default=2e9)
    args = parser.parse_args()
    if args.command == "run":
        build()
        print()
        report(top=args.top, min_marketcap=args.min_marketcap)
    elif args.command == "report":
        report(top=args.top, min_marketcap=args.min_marketcap)
    else:
        status()


if __name__ == "__main__":
    main()
