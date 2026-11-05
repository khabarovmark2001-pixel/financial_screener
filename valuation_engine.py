#!/usr/bin/env python3
"""
Multi-model valuation engine: one weighted Fair Value per share, per ticker.

No single DCF fits every company. A bank has no meaningful capex or working
capital, so FCFF is nonsense for it; a loss-making biotech has no earnings to
discount; a utility's value sits mostly in its dividend. This engine runs five
models, scores how much each one can be trusted for that particular company,
and blends only the ones that survive.

    FairValue = SUM(normalised_weight_i * model_value_i)

Models
    fcfe       EPS-driven FCFE DCF - equity cash flow, the default for normal
               operating companies. Reused from dcf_forecast.
    fcff       Enterprise DCF - secondary, and the primary lens on
               capital-intensive and leverage-sensitive businesses.
    ddm        Two-stage dividend discount - mature, consistent payers.
    rim        Residual income / excess return - the right model for banks and
               insurers, where book value is economically meaningful and FCFF
               is not.
    relative   Peer-median multiples (P/E, EV/EBITDA, P/S, P/B) - a cross-check
               everywhere, and the main signal for unprofitable or
               high-growth names where every DCF is extrapolation.

Each model returns a value AND a reliability score in [0, 1]. Base weights come
from the company's profile; final weights are base * reliability, with anything
economically inappropriate or too unreliable dropped and its weight
redistributed across the survivors.

Outputs
    data/valuation_engine.parquet   one row per ticker: fair value, every
                                    model's value, weight and reliability, the
                                    multiples-blend fair value, a confidence
                                    score and a written rationale.

Usage
      python3 valuation_engine.py run
      python3 valuation_engine.py explain AAPL
      python3 valuation_engine.py status
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
import valuation_metrics as vm
from sharadar_loader import DATA_DIR, connect

ENGINE_PARQUET = DATA_DIR / "valuation_engine.parquet"

MODELS = ("fcfe", "fcff", "ddm", "rim", "relative")

# --------------------------------------------------------------------------
# assumptions
# --------------------------------------------------------------------------

# DDM and RIM run a shorter explicit horizon than the 15-year cash-flow DCF.
# Both converge on a steady state faster: a dividend policy and a return on
# equity mean-revert well inside a decade, and stretching them further just
# compounds an assumption.
HORIZON_YEARS = 10

# Excess returns decay. A company earning 25% on equity does not keep earning
# it forever - competition pulls ROE toward the cost of equity. This is the
# share of the starting spread still left in the terminal year.
EXCESS_RETURN_PERSISTENCE = 0.40
# Terminal residual income is not a growing perpetuity: assuming a company
# earns above its cost of capital in perpetuity is the single most common way
# a residual income model produces a fantasy. This caps the terminal block at
# a multiple of the final year's residual income.
RIM_TERMINAL_MULTIPLE = 8.0

# Sustainable dividend growth = ROE * retention, capped well below the
# discount rate. Gordon explodes as g approaches r.
DDM_MAX_GROWTH = 0.08
MIN_SPREAD = dcf.MIN_SPREAD          # r - g >= 1pp, same rule as the DCF

# A model whose reliability falls below this is not economically usable for
# that company; it is dropped and its weight redistributed.
RELIABILITY_FLOOR = 0.25
# A model value more than this multiple away from the market price in either
# direction is treated as a modelling failure rather than an opportunity.
SANITY_BAND = (0.10, 10.0)

# Peer sets smaller than this fall back from industry to sector.
MIN_PEERS_INDUSTRY = 5
MIN_PEERS_SECTOR = 8

# Base weights by profile. Rows sum to 1 before reliability is applied; they do
# not need to after, because weights are renormalised over survivors.
BASE_WEIGHTS = {
    #                      fcfe  fcff   ddm   rim  relative
    "financial":          (0.10, 0.00, 0.20, 0.50, 0.20),
    "reit":               (0.15, 0.10, 0.30, 0.20, 0.25),
    "mature_dividend":    (0.30, 0.20, 0.30, 0.05, 0.15),
    "capital_intensive":  (0.25, 0.45, 0.05, 0.05, 0.20),
    "high_growth":        (0.30, 0.20, 0.00, 0.05, 0.45),
    "unprofitable":       (0.10, 0.10, 0.00, 0.05, 0.75),
    "standard":           (0.40, 0.30, 0.05, 0.05, 0.20),
}

PROFILE_RATIONALE = {
    "financial": "bank or insurer - debt is raw material, so FCFF is "
                 "meaningless; residual income on book equity leads",
    "reit": "property vehicle - distributions and book assets carry the value, "
            "capex is acquisition rather than maintenance",
    "mature_dividend": "mature and consistently distributing - the dividend is "
                       "the cash the owner actually receives",
    "capital_intensive": "capital-intensive or leveraged - enterprise value "
                         "before financing is the cleaner lens",
    "high_growth": "growing fast with thin or negative margins - any DCF is "
                   "extrapolation, so peers carry more of the weight",
    "unprofitable": "no earnings to discount - valued mainly against peers",
    "standard": "ordinary operating company - equity cash flow leads",
}

# Multiples used by the relative model, with how each converts to a per-share
# equity value.
MULTIPLE_SPECS = {
    "pe_ttm": ("earnings", "netinc_ttm"),
    "ev_ebitda_ttm": ("enterprise", "ebitda_ttm"),
    "ps_ttm": ("equity", "revenue_ttm"),
    "pb": ("equity", "equity_book"),
}


# --------------------------------------------------------------------------
# inputs
# --------------------------------------------------------------------------

def _dividend_history() -> pd.DataFrame:
    """Per-ticker dividend record: level, growth and how consistently it is paid.

    Consistency matters more than level for deciding whether a DDM applies. A
    company that paid in nine of the last ten filings has a policy; one that
    paid once does not, and Gordon on a single payment is arithmetic without
    meaning.
    """
    connection = connect()
    frame = connection.execute("""
        SELECT ticker, date, fxusd, dps, ncfdiv, netinc, equity, bvps, roe,
               payoutratio
        FROM fundamentals
        WHERE dimension = 'ART'
        ORDER BY ticker, date
    """).df()
    connection.close()

    rate = pd.to_numeric(frame["fxusd"], errors="coerce")
    rate = rate.where(rate > 0, 1.0)
    for column in ("dps", "ncfdiv", "netinc", "equity", "bvps"):
        frame[column] = frame[column] / rate
    frame["date"] = pd.to_datetime(frame["date"])

    recent = frame.groupby("ticker", sort=False).tail(12)
    grouped = recent.groupby("ticker")

    out = pd.DataFrame({
        "dps_ttm": grouped["dps"].last(),
        "dps_paid_share": grouped["dps"].apply(lambda s: (s.fillna(0) > 0).mean()),
        "dps_filings": grouped["dps"].size(),
        "payout_ratio": grouped["payoutratio"].median(),
        "bvps": grouped["bvps"].last(),
        "roe_median": grouped["roe"].median(),
        # Dispersion of ROE says whether a residual income forecast has
        # anything stable to stand on.
        "roe_volatility": grouped["roe"].std(),
    })

    # Dividend growth over the window, annualised, on sign-consistent endpoints.
    def growth(series: pd.Series) -> float:
        values = series.dropna()
        values = values[values > 0]
        if len(values) < 3:
            return np.nan
        years = max(len(values) / 4.0, 1.0)
        return (values.iloc[-1] / values.iloc[0]) ** (1.0 / years) - 1.0

    out["dps_growth"] = grouped["dps"].apply(growth)
    return out.reset_index()


def load_inputs(verbose: bool = True) -> pd.DataFrame:
    """Everything the engine needs, one row per ticker.

    Deliberately a join over existing artefacts rather than a recomputation:
    the DCF values, the cost of capital and the multiples already exist and are
    already point-in-time. Duplicating that work here would create a second
    source of truth that could drift from the first.
    """
    summary = pd.read_parquet(dcf.SUMMARY_PARQUET)
    metrics = vm.latest()

    metric_columns = [
        "ticker", "netinc_ttm", "ebitda_ttm", "ebit_ttm", "revenue_ttm",
        "fcf_ttm", "equity_book", "total_debt", "net_debt", "marketcap", "ev",
        "pe_ttm", "ps_ttm", "pb", "ev_ebitda_ttm", "ev_ebit_ttm",
        "ebit_margin_ttm", "net_margin_ttm", "roe_ttm", "roic_ttm",
        "debt_to_equity", "net_debt_to_ebitda", "interest_coverage",
        "capex_to_revenue_ttm", "years_of_history",
    ]
    frame = summary.merge(
        metrics[[c for c in metric_columns if c in metrics.columns]],
        on="ticker", how="left", suffixes=("", "_m"))
    frame = frame.merge(_dividend_history(), on="ticker", how="left")

    sectors = pd.read_parquet(DATA_DIR / "analytics.parquet",
                              columns=["ticker", "sector", "industry", "marketcap"])
    frame = frame.merge(sectors.rename(columns={"marketcap": "marketcap_now"}),
                        on="ticker", how="left", suffixes=("", "_a"))
    if "sector_a" in frame.columns:
        frame["sector"] = frame["sector"].fillna(frame["sector_a"])
        frame = frame.drop(columns=[c for c in frame.columns if c.endswith("_a")])

    if verbose:
        print(f"  loaded {len(frame):,} tickers")
    return frame


# --------------------------------------------------------------------------
# classification
# --------------------------------------------------------------------------

def classify(frame: pd.DataFrame) -> pd.DataFrame:
    """Assign each company a profile from what its own numbers look like.

    Order matters: the tests run from most structural to least. Sector comes
    first because a bank is a bank whatever its growth rate looks like this
    year, and applying an operating-company model to it is a category error,
    not a calibration problem.
    """
    frame = frame.copy()

    profitable = (frame["netinc_ttm"] > 0) & (frame["ebit_ttm"] > 0)
    growth_rate = frame["base_revenue_growth"]
    payer = (frame["dps_ttm"].fillna(0) > 0) & (frame["dps_paid_share"].fillna(0) >= 0.6)
    # Paying *something* is not a dividend policy. Apple pays consistently on a
    # ~15% payout and a 0.5% yield - discounting that stream values the company
    # at $16 against a $320 price, which the sanity band then throws away. A
    # company is only a dividend story if the dividend is a material share of
    # what it earns or of what the market pays for it.
    dividend_yield = (frame["dps_ttm"] / frame["market_price"]).replace(
        [np.inf, -np.inf], np.nan)
    material_dividend = payer & ((frame["payout_ratio"].fillna(0) >= 0.25)
                                 | (dividend_yield.fillna(0) >= 0.015))
    capital_heavy = (frame["capex_to_revenue_ttm"].fillna(0) > 0.10)
    levered = (frame["net_debt_to_ebitda"].fillna(0) > 3.0) | \
              (frame["debt_to_equity"].fillna(0) > 1.5)
    mature = frame["years_of_history"].fillna(0) >= 8

    profile = pd.Series("standard", index=frame.index, dtype=object)
    profile = profile.mask(~profitable, "unprofitable")
    profile = profile.mask(profitable & (growth_rate > 0.20), "high_growth")
    profile = profile.mask(profitable & (capital_heavy | levered), "capital_intensive")
    profile = profile.mask(profitable & mature & material_dividend
                           & (growth_rate < 0.08), "mature_dividend")
    profile = profile.mask(frame["sector"].eq("Real Estate"), "reit")
    profile = profile.mask(frame["sector"].eq("Financial Services"), "financial")

    frame["profile"] = profile
    frame["is_profitable"] = profitable
    frame["is_dividend_payer"] = payer
    frame["has_material_dividend"] = material_dividend
    frame["dividend_yield"] = dividend_yield
    frame["is_capital_intensive"] = capital_heavy
    frame["is_levered"] = levered
    frame["is_mature"] = mature
    return frame


# --------------------------------------------------------------------------
# models
# --------------------------------------------------------------------------

def _fade_to(start: np.ndarray, end: np.ndarray, years: int) -> np.ndarray:
    weights = np.linspace(0.0, 1.0, years)[None, :]
    return start[:, None] * (1.0 - weights) + end[:, None] * weights


def model_ddm(frame: pd.DataFrame) -> pd.DataFrame:
    """Two-stage dividend discount.

    Growth starts from the lower of the company's own dividend growth and its
    sustainable rate (ROE * retention) - a payer growing dividends faster than
    it earns them is funding the gap from somewhere that will run out - then
    fades to the terminal rate. Discounted at cost of equity, with the same
    r - g >= 1pp floor the DCF uses, because Gordon explodes as the spread
    closes.
    """
    rate = frame["cost_of_equity"].to_numpy(dtype=float)
    dps = frame["dps_ttm"].to_numpy(dtype=float)
    payout = frame["payout_ratio"].to_numpy(dtype=float)
    roe = frame["roe_ttm"].to_numpy(dtype=float)
    observed_growth = frame["dps_growth"].to_numpy(dtype=float)

    retention = np.clip(1.0 - np.nan_to_num(payout, nan=0.5), 0.0, 1.0)
    sustainable = np.nan_to_num(roe, nan=0.0) * retention
    start_growth = np.fmin(np.nan_to_num(observed_growth, nan=sustainable),
                           sustainable)
    start_growth = np.clip(start_growth, -0.05, DDM_MAX_GROWTH)

    terminal_growth = np.minimum(dcf.TERMINAL_GROWTH_TARGET, rate - MIN_SPREAD)
    terminal_growth = np.clip(terminal_growth, 0.0, None)

    path = _fade_to(start_growth, terminal_growth, HORIZON_YEARS)
    dividends = dps[:, None] * np.cumprod(1.0 + path, axis=1)
    years = np.arange(1, HORIZON_YEARS + 1)[None, :]
    discount = (1.0 + rate)[:, None] ** years

    present = (dividends / discount).sum(axis=1)
    spread = np.maximum(rate - terminal_growth, MIN_SPREAD)
    terminal = dividends[:, -1] * (1.0 + terminal_growth) / spread
    value = present + terminal / discount[:, -1]

    usable = (dps > 0) & np.isfinite(rate) & (rate > 0)
    value = np.where(usable, value, np.nan)

    # Reliability: how consistently it pays, whether the payout is coverable,
    # and how much of the answer is the terminal block.
    consistency = frame["dps_paid_share"].fillna(0.0).to_numpy(dtype=float)
    coverable = np.clip(1.5 - np.nan_to_num(payout, nan=1.0), 0.0, 1.0)
    terminal_share = np.where(value > 0, (terminal / discount[:, -1]) / value, 1.0)
    terminal_penalty = np.clip(1.3 - terminal_share, 0.0, 1.0)
    reliability = np.where(usable,
                           0.45 * consistency + 0.30 * coverable
                           + 0.25 * terminal_penalty, 0.0)

    return pd.DataFrame({
        "ddm_value": value,
        "ddm_reliability": np.clip(reliability, 0.0, 1.0),
        "ddm_start_growth": np.where(usable, start_growth, np.nan),
    }, index=frame.index)


def model_rim(frame: pd.DataFrame) -> pd.DataFrame:
    """Residual income / excess return on book equity.

        V = B0 + SUM (ROE_t - r) * B_{t-1} / (1+r)^t + terminal

    This is the right model for banks and insurers: their book equity is
    marked close to economic value, their earnings are the return on it, and
    they have no capex or working capital for an FCFF model to work with.

    ROE fades toward the cost of equity, keeping only EXCESS_RETURN_PERSISTENCE
    of the starting spread by the terminal year - excess returns attract
    competition. The terminal block is a capped multiple of final-year residual
    income rather than a growing perpetuity, because assuming a firm out-earns
    its cost of capital forever is how this model manufactures fantasies.
    """
    rate = frame["cost_of_equity"].to_numpy(dtype=float)
    book = frame["bvps"].to_numpy(dtype=float)
    roe_start = frame["roe_ttm"].to_numpy(dtype=float)
    payout = np.clip(np.nan_to_num(frame["payout_ratio"].to_numpy(dtype=float),
                                   nan=0.35), 0.0, 1.0)

    roe_terminal = rate + EXCESS_RETURN_PERSISTENCE * (roe_start - rate)
    roe_path = _fade_to(roe_start, roe_terminal, HORIZON_YEARS)
    # A forecast ROE outside this band is a broken input, not a business.
    roe_path = np.clip(roe_path, -0.50, 0.60)

    years = np.arange(1, HORIZON_YEARS + 1)[None, :]
    discount = (1.0 + rate)[:, None] ** years

    opening = np.empty_like(roe_path)
    residual = np.empty_like(roe_path)
    current = book.copy()
    for t in range(HORIZON_YEARS):
        opening[:, t] = current
        earnings = roe_path[:, t] * current
        residual[:, t] = earnings - rate * current
        # Clean surplus: book grows by retained earnings.
        current = current + earnings * (1.0 - payout)

    present = (residual / discount).sum(axis=1)
    terminal = np.clip(residual[:, -1], None, np.abs(residual[:, -1])) \
        * RIM_TERMINAL_MULTIPLE
    value = book + present + terminal / discount[:, -1]

    usable = (book > 0) & np.isfinite(rate) & (rate > 0) & np.isfinite(roe_start)
    value = np.where(usable & (value > 0), value, np.nan)

    # Reliability: stable ROE, book value that is not a rounding error against
    # market cap, and a result not dominated by the terminal block.
    roe_volatility = frame["roe_volatility"].fillna(0.20).to_numpy(dtype=float)
    stability = np.clip(1.0 - roe_volatility / 0.25, 0.0, 1.0)
    book_weight = np.where(value > 0, np.clip(book / value, 0.0, 1.0), 0.0)
    equity_positive = (frame["equity_book"].fillna(0).to_numpy() > 0).astype(float)
    reliability = np.where(usable,
                           0.45 * stability + 0.30 * book_weight
                           + 0.25 * equity_positive, 0.0)

    return pd.DataFrame({
        "rim_value": value,
        "rim_reliability": np.clip(reliability, 0.0, 1.0),
        "rim_roe_start": np.where(usable, roe_start, np.nan),
        "rim_roe_terminal": np.where(usable, roe_terminal, np.nan),
    }, index=frame.index)


def _peer_medians(frame: pd.DataFrame, column: str) -> tuple[pd.Series, pd.Series]:
    """Median multiple among peers, industry first then sector.

    Only positive values count. A negative P/E is a loss-maker, and including
    it in a median that will be multiplied by someone's earnings produces a
    negative fair value out of a positive business.
    """
    positive = frame[column].where(frame[column] > 0)
    industry = positive.groupby(frame["industry"]).transform("median")
    industry_count = positive.groupby(frame["industry"]).transform("count")
    sector = positive.groupby(frame["sector"]).transform("median")
    sector_count = positive.groupby(frame["sector"]).transform("count")

    median = industry.where(industry_count >= MIN_PEERS_INDUSTRY, sector)
    count = industry_count.where(industry_count >= MIN_PEERS_INDUSTRY, sector_count)
    return median, count


def model_relative(frame: pd.DataFrame) -> pd.DataFrame:
    """Peer-median multiples, blended by their own reliability.

    Each multiple gives its own implied price. They are then combined with the
    same reliability-weighting the top-level engine uses, which is what makes
    `fair_value_multiples` directly comparable to the DCF models rather than a
    separate scale.
    """
    shares = frame["shares"].to_numpy(dtype=float)
    shares = np.where(shares > 0, shares, np.nan)
    net_debt = frame["net_debt"].fillna(0.0).to_numpy(dtype=float)

    values, weights, detail = {}, {}, {}
    for multiple, (kind, driver_column) in MULTIPLE_SPECS.items():
        if multiple not in frame.columns or driver_column not in frame.columns:
            continue
        median, count = _peer_medians(frame, multiple)
        driver = frame[driver_column].to_numpy(dtype=float)
        peer_multiple = median.to_numpy(dtype=float)

        if kind == "enterprise":
            implied = (peer_multiple * driver - net_debt) / shares
        else:
            implied = peer_multiple * driver / shares

        # A multiple only means something applied to a positive driver.
        implied = np.where((driver > 0) & (peer_multiple > 0) & np.isfinite(implied),
                           implied, np.nan)
        implied = np.where(implied > 0, implied, np.nan)

        peer_depth = np.clip(count.to_numpy(dtype=float) / 20.0, 0.0, 1.0)
        reliability = np.where(np.isfinite(implied), 0.4 + 0.6 * peer_depth, 0.0)

        values[multiple] = implied
        weights[multiple] = reliability
        detail[f"rel_{multiple}_value"] = implied
        detail[f"rel_{multiple}_peer"] = peer_multiple

    if not values:
        return pd.DataFrame(index=frame.index)

    stacked = np.vstack([values[m] for m in values])
    stacked_weights = np.vstack([weights[m] for m in values])
    stacked_weights = np.where(np.isfinite(stacked), stacked_weights, 0.0)
    total = stacked_weights.sum(axis=0)

    blended = np.where(total > 0,
                       np.nansum(stacked * stacked_weights, axis=0)
                       / np.where(total > 0, total, 1.0), np.nan)
    used = np.isfinite(stacked).sum(axis=0)

    # Agreement between the multiples is the honest reliability signal here:
    # if P/E and EV/EBITDA disagree by 3x, neither is telling you much.
    with np.errstate(invalid="ignore", divide="ignore"):
        spread = np.nanmax(stacked, axis=0) / np.nanmin(stacked, axis=0)
    agreement = np.clip(1.5 - np.nan_to_num(spread, nan=5.0) / 4.0, 0.0, 1.0)
    depth = np.clip(used / 3.0, 0.0, 1.0)
    reliability = np.where(used > 0, 0.55 * agreement + 0.45 * depth, 0.0)

    out = pd.DataFrame(detail, index=frame.index)
    out["relative_value"] = blended
    out["relative_reliability"] = np.clip(reliability, 0.0, 1.0)
    out["relative_multiples_used"] = used
    # The multiples blend is exported on its own as well: it is the answer to
    # "what would peers pay for this", independent of any cash-flow forecast.
    out["fair_value_multiples"] = blended
    return out


def model_passthrough(frame: pd.DataFrame) -> pd.DataFrame:
    """FCFE and FCFF come from the existing 15-year DCF - not recomputed here.

    Their reliability is driven by the dispersion the DCF already measured for
    each company's own cash flows (sigma), how much of enterprise value sits in
    the terminal block, and whether the model is structurally applicable at all.
    """
    out = pd.DataFrame(index=frame.index)
    out["fcfe_value"] = frame["implied_share_price_fcfe"]
    out["fcff_value"] = frame["implied_share_price_fcff"]

    def stability(sigma_column: str) -> np.ndarray:
        sigma = frame[sigma_column].fillna(0.8).to_numpy(dtype=float)
        return np.clip(1.0 - (sigma - 0.05) / 0.95, 0.0, 1.0)

    terminal_share = frame["terminal_share_of_ev"].fillna(0.8).to_numpy(dtype=float)
    terminal_penalty = np.clip(1.25 - terminal_share, 0.0, 1.0)
    history = np.clip(frame["years_of_history"].fillna(0).to_numpy() / 10.0, 0.0, 1.0)

    # The EPS model is an independent read on the same equity claim, so how
    # closely it agrees with FCFE is a genuine confidence signal rather than a
    # sixth opinion to average in.
    eps_value = frame["implied_share_price_eps"].to_numpy(dtype=float)
    fcfe_value = out["fcfe_value"].to_numpy(dtype=float)
    with np.errstate(invalid="ignore", divide="ignore"):
        ratio = np.where((eps_value > 0) & (fcfe_value > 0),
                         np.maximum(eps_value, fcfe_value)
                         / np.minimum(eps_value, fcfe_value), np.nan)
    corroboration = np.clip(1.6 - np.nan_to_num(ratio, nan=2.0) / 2.0, 0.0, 1.0)

    out["fcfe_reliability"] = np.clip(
        0.35 * stability("sigma_fcfe") + 0.25 * terminal_penalty
        + 0.20 * history + 0.20 * corroboration, 0.0, 1.0)
    out["fcff_reliability"] = np.clip(
        0.45 * stability("sigma_fcff") + 0.35 * terminal_penalty
        + 0.20 * history, 0.0, 1.0)
    out["eps_corroboration"] = corroboration
    return out


# --------------------------------------------------------------------------
# blending
# --------------------------------------------------------------------------

def _applicability(frame: pd.DataFrame) -> pd.DataFrame:
    """Hard structural gates, applied before any weighting.

    These are not calibration - they are statements that a model does not
    describe the company at all. A weight of zero here cannot be recovered by
    a high reliability score.
    """
    gates = pd.DataFrame(1.0, index=frame.index, columns=list(MODELS))
    financial = frame["sector"].eq("Financial Services")

    # FCFF on a bank is a category error: its debt is inventory, and capex and
    # working capital have no operating meaning.
    gates.loc[financial, "fcff"] = 0.0
    # An equity DCF on a bank is defensible but weak; the engine leans on RIM.
    gates.loc[financial, "fcfe"] *= 0.5
    # No dividend, no dividend model.
    gates.loc[~frame["is_dividend_payer"].fillna(False), "ddm"] = 0.0
    # Residual income needs book equity that means something.
    gates.loc[~(frame["equity_book"].fillna(0) > 0), "rim"] = 0.0
    # A cash-flow DCF on a company with no earnings is extrapolating a loss.
    loss_making = ~frame["is_profitable"].fillna(False)
    gates.loc[loss_making, ["fcfe", "fcff"]] *= 0.35
    return gates


def blend(frame: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """Weights, fair value, confidence and rationale."""
    values = np.vstack([frame[f"{m}_value"].to_numpy(dtype=float) for m in MODELS])
    reliability = np.vstack([frame[f"{m}_reliability"].to_numpy(dtype=float)
                             for m in MODELS])

    base = np.vstack([
        np.array([BASE_WEIGHTS[p][i] for p in frame["profile"]])
        for i in range(len(MODELS))])
    gates = _applicability(frame)[list(MODELS)].to_numpy(dtype=float).T

    # A value wildly detached from the market price is a modelling failure, not
    # a signal. Dropping it here stops one broken model dragging the blend.
    market = frame["market_price"].to_numpy(dtype=float)
    with np.errstate(invalid="ignore", divide="ignore"):
        multiple_of_price = values / np.where(market > 0, market, np.nan)
    sane = (multiple_of_price >= SANITY_BAND[0]) & (multiple_of_price <= SANITY_BAND[1])

    usable = np.isfinite(values) & (values > 0) & sane \
        & (reliability >= RELIABILITY_FLOOR) & (gates > 0)

    weights = base * reliability * gates
    weights = np.where(usable, weights, 0.0)
    total = weights.sum(axis=0)
    normalised = np.where(total > 0, weights / np.where(total > 0, total, 1.0), 0.0)

    fair_value = np.where(total > 0, np.nansum(np.where(usable, values, 0.0)
                                               * normalised, axis=0), np.nan)

    models_used = usable.sum(axis=0)
    weighted_reliability = np.where(total > 0,
                                    (reliability * normalised).sum(axis=0), 0.0)

    # Agreement across the surviving models, value-weighted. Two models within
    # 20% of each other say something; two that disagree threefold do not.
    spread = np.full(len(frame), np.nan)
    for index in range(values.shape[1]):
        column = values[:, index][usable[:, index]]
        if len(column) >= 2:
            spread[index] = column.max() / max(column.min(), 1e-9)
    agreement = np.clip(1.6 - np.nan_to_num(spread, nan=2.0) / 2.5, 0.0, 1.0)
    agreement = np.where(models_used >= 2, agreement, 0.35)

    confidence = np.clip(0.45 * weighted_reliability
                         + 0.35 * agreement
                         + 0.20 * np.clip(models_used / 3.0, 0.0, 1.0), 0.0, 1.0)
    confidence = np.where(models_used > 0, confidence, np.nan)

    out = pd.DataFrame(index=frame.index)
    out["fair_value"] = fair_value
    out["models_used"] = models_used
    out["model_agreement"] = np.where(models_used >= 2, spread, np.nan)
    out["weighted_reliability"] = np.where(models_used > 0, weighted_reliability, np.nan)
    out["confidence"] = confidence
    for index, model in enumerate(MODELS):
        out[f"{model}_weight"] = normalised[index]
        out[f"{model}_used"] = usable[index]

    with np.errstate(invalid="ignore", divide="ignore"):
        out["fair_value_upside"] = np.where(market > 0, fair_value / market - 1.0,
                                            np.nan)
    out["rationale"] = _rationale(frame, normalised, usable)
    if verbose:
        print(f"  fair value for {int(np.isfinite(fair_value).sum()):,} of "
              f"{len(frame):,} tickers, median {np.nanmedian(models_used):.0f} "
              f"models each")
    return out


def _rationale(frame: pd.DataFrame, weights: np.ndarray,
               usable: np.ndarray) -> list[str]:
    """One sentence per company explaining which models carried the answer."""
    labels = {"fcfe": "FCFE DCF", "fcff": "FCFF DCF", "ddm": "dividend discount",
              "rim": "residual income", "relative": "peer multiples"}
    out = []
    for index in range(len(frame)):
        profile = frame["profile"].iloc[index]
        parts = [f"{labels[m]} {weights[position, index]:.0%}"
                 for position, m in enumerate(MODELS)
                 if usable[position, index] and weights[position, index] > 0.005]
        dropped = [labels[m] for position, m in enumerate(MODELS)
                   if not usable[position, index]]
        sentence = f"{PROFILE_RATIONALE[profile]}; weighted " + (
            ", ".join(parts) if parts else "nothing - no model survived")
        if dropped:
            sentence += f". Not used: {', '.join(dropped)}"
        out.append(sentence + ".")
    return out


# --------------------------------------------------------------------------

def build(verbose: bool = True) -> pd.DataFrame:
    frame = classify(load_inputs(verbose=verbose))
    if verbose:
        counts = frame["profile"].value_counts()
        print("  profiles: " + ", ".join(f"{k} {v:,}" for k, v in counts.items()))

    for part in (model_passthrough(frame), model_ddm(frame), model_rim(frame),
                 model_relative(frame)):
        frame = frame.join(part)

    # Any model that produced no column at all still needs one, so the blend
    # can treat "absent" and "unusable" identically instead of raising.
    for model in MODELS:
        if f"{model}_value" not in frame.columns:
            frame[f"{model}_value"] = np.nan
        if f"{model}_reliability" not in frame.columns:
            frame[f"{model}_reliability"] = 0.0

    result = frame.join(blend(frame, verbose=verbose))

    keep = ([
        "ticker", "sector", "industry", "profile", "base_date", "market_price",
        "market_price_date", "shares", "marketcap", "cost_of_equity", "wacc",
        "fair_value", "fair_value_upside", "fair_value_multiples", "confidence",
        "models_used", "model_agreement", "weighted_reliability", "rationale",
        "is_profitable", "is_dividend_payer", "has_material_dividend",
        "dividend_yield", "is_capital_intensive", "is_levered", "is_mature",
    ]
        + [f"{m}_value" for m in MODELS]
        + [f"{m}_weight" for m in MODELS]
        + [f"{m}_reliability" for m in MODELS]
        + [f"{m}_used" for m in MODELS]
        + ["ddm_start_growth", "rim_roe_start", "rim_roe_terminal",
           "relative_multiples_used", "eps_corroboration",
           "implied_share_price_eps"]
        + [c for c in result.columns if c.startswith("rel_")])

    output = result[[c for c in keep if c in result.columns]].copy()
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    output.to_parquet(ENGINE_PARQUET, index=False,
                      compression="zstd", compression_level=6)
    if verbose:
        print(f"\n  -> {ENGINE_PARQUET.name}  {len(output):,} rows x "
              f"{len(output.columns)} cols  "
              f"{ENGINE_PARQUET.stat().st_size / 1e6:.1f} MB")
    return output


def load() -> pd.DataFrame:
    if not ENGINE_PARQUET.exists():
        raise SystemExit("No engine output. Run: python3 valuation_engine.py run")
    return pd.read_parquet(ENGINE_PARQUET)


def explain(ticker: str) -> None:
    """Print the full working for one company."""
    frame = load()
    row = frame[frame["ticker"] == ticker]
    if row.empty:
        raise SystemExit(f"{ticker} is not in the engine output.")
    row = row.iloc[0]

    print(f"{ticker} - {row['sector']} / {row['industry']}")
    print(f"profile        : {row['profile']}")
    print(f"market price   : ${row['market_price']:,.2f}")
    print(f"FAIR VALUE     : ${row['fair_value']:,.2f}   "
          f"({row['fair_value_upside']:+.1%})")
    print(f"multiples only : ${row['fair_value_multiples']:,.2f}"
          if pd.notna(row["fair_value_multiples"]) else "multiples only : -")
    print(f"confidence     : {row['confidence']:.0%}  "
          f"({int(row['models_used'])} models, "
          f"reliability {row['weighted_reliability']:.0%})")
    print()
    print(f"  {'model':<18}{'value':>12}{'weight':>9}{'reliab.':>9}  used")
    for model in MODELS:
        value = row[f"{model}_value"]
        print(f"  {model:<18}"
              + (f"{value:>12,.2f}" if pd.notna(value) else f"{'-':>12}")
              + f"{row[f'{model}_weight']:>9.1%}"
              + f"{row[f'{model}_reliability']:>9.0%}"
              + f"  {'yes' if row[f'{model}_used'] else 'no'}")
    print()
    print("rationale:", row["rationale"])


def status() -> None:
    if not ENGINE_PARQUET.exists():
        print("valuation_engine.parquet   not built")
        return
    frame = load()
    print(f"valuation_engine.parquet   {len(frame):,} rows x "
          f"{len(frame.columns)} cols  "
          f"{ENGINE_PARQUET.stat().st_size / 1e6:.1f} MB")
    print(f"  fair value        : {frame['fair_value'].notna().sum():,}")
    print(f"  multiples blend   : {frame['fair_value_multiples'].notna().sum():,}")
    print(f"  median confidence : {frame['confidence'].median():.0%}")
    print("  profiles          : "
          + ", ".join(f"{k} {v:,}" for k, v in frame["profile"].value_counts().items()))


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["run", "explain", "status"])
    parser.add_argument("ticker", nargs="?")
    args = parser.parse_args()

    if args.command == "run":
        build()
    elif args.command == "explain":
        if not args.ticker:
            raise SystemExit("Usage: valuation_engine.py explain TICKER")
        explain(args.ticker.upper())
    else:
        status()


if __name__ == "__main__":
    main()
