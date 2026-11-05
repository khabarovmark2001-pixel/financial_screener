#!/usr/bin/env python3
"""
Valuation multiples at today's price rather than at the filing-date price.

`metrics.parquet` pairs each filing's trailing-twelve-month earnings with the
market capitalisation *on that filing date*, which is the right construction
for a point-in-time history: it is the multiple as it could actually have been
computed the day the numbers were published, and it is what makes the series
on the deep-dive charts honest.

It is the wrong construction for a screen. A company that reported in February
and has since fallen 40% still shows February's P/E, so the screen ranks it on
a price nobody can trade. Every multiple on the site's *current* views is
therefore rebuilt here against the latest close, holding the reported earnings
and the reported balance sheet fixed.

Only the equity leg moves. Net debt is a balance-sheet stock and is left at the
filing value - marking it to market would need a bond price this dataset does
not carry - so enterprise value moves one-for-one with market capitalisation:

    marketcap_now = marketcap_filed x (close_now / close_filed)
    ev_now        = ev_filed + (marketcap_now - marketcap_filed)

The scale factor comes from the *split-adjusted* close on both ends, never from
the raw quote. A split changes the share count and the quoted price in opposite
directions, so a raw-price ratio would rescale a market cap by the split factor
- NVDA's ten-for-one would have cut its market cap to a tenth. Sharadar's
`close` column is split-adjusted across the whole series and `closeunadj` is
not, which is what makes the ratio safe. Dividends are deliberately *not*
adjusted out: a market cap genuinely falls when a company pays one away, so
`close` is right here and `closeadj` - a total-return series - would be wrong.

The filing-date values are not discarded. Every repriced column keeps its
original under a `_at_filing` suffix, so a page can show both and the point-in-
time history stays available.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

#: Suffix under which the original filing-date value is preserved.
FILING_SUFFIX = "_at_filing"

#: Multiples rebuilt against the current price, and what each divides.
#: Read as {column: (numerator, denominator)} where the numerator names which
#: of the two repriced aggregates it uses.
EQUITY_MULTIPLES = {
    "pe_ttm": "netinc_ttm",
    "ps_ttm": "revenue_ttm",
    "pb": "equity_book",
    "p_ebitda_ttm": "ebitda_ttm",
    "p_fcf_ttm": "fcf_ttm",
}

ENTERPRISE_MULTIPLES = {
    "ev_ebitda_ttm": "ebitda_ttm",
    "ev_ebit_ttm": "ebit_ttm",
    "ev_sales_ttm": "revenue_ttm",
    "ev_fcff_ttm": "fcff_ttm",
}

#: Yields - the same ratios inverted, so they need the same treatment.
YIELDS = {
    "fcf_yield": "fcf_ttm",
    "earnings_yield": "netinc_ttm",
    "fcfe_yield": "fcfe_ttm",
}

#: Everything this module rewrites, for callers that need to know.
REPRICED_COLUMNS = (("marketcap", "ev")
                    + tuple(EQUITY_MULTIPLES)
                    + tuple(ENTERPRISE_MULTIPLES)
                    + tuple(YIELDS))


def _safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    """Division that yields NaN on a zero base, matching valuation_metrics.

    Negative denominators are preserved rather than nulled, for the same reason
    they are there: a negative P/E says the company lost money, and dropping it
    would hide exactly the names a screen should be excluding on purpose.
    """
    numerator = pd.to_numeric(numerator, errors="coerce")
    denominator = pd.to_numeric(denominator, errors="coerce")
    return numerator / denominator.replace(0, np.nan)


def price_scale(close_now: pd.Series, close_filed: pd.Series) -> pd.Series:
    """close_now / close_filed, or 1.0 where the pair cannot be trusted.

    Falling back to 1.0 leaves the filing-date multiple in place, which is the
    number the page showed before this module existed - visibly stale rather
    than wrong by an arbitrary factor. A missing filing-date close is the
    normal case for a ticker whose price history starts after its oldest
    filing, and a non-positive one is never a real quote.
    """
    now = pd.to_numeric(close_now, errors="coerce")
    filed = pd.to_numeric(close_filed, errors="coerce")
    scale = now / filed.where(filed > 0)
    return scale.where(np.isfinite(scale) & (scale > 0), 1.0)


def reprice(frame: pd.DataFrame, scale: pd.Series) -> pd.DataFrame:
    """Rebuild every price-dependent column at the current price.

    `frame` needs `marketcap` and `ev` plus whichever TTM aggregates the
    multiples it wants are built on; anything absent is skipped rather than
    raising, so this works on the trimmed column set `analytics.parquet`
    carries as well as on the full metrics table.

    `scale` must already be aligned to `frame`'s index.
    """
    out = frame.copy()
    scale = pd.to_numeric(scale, errors="coerce").reindex(out.index).fillna(1.0)
    out["price_scale"] = scale

    marketcap = pd.to_numeric(out.get("marketcap"), errors="coerce")
    if marketcap is None:
        return out

    marketcap_now = marketcap * scale
    out[f"marketcap{FILING_SUFFIX}"] = marketcap
    out["marketcap"] = marketcap_now

    # Enterprise value moves by the same *amount*, not the same ratio: the net
    # debt inside it has not moved. Scaling EV directly would silently mark the
    # debt to the equity's return, which for a levered name is a large error in
    # the wrong direction.
    if "ev" in out.columns:
        enterprise = pd.to_numeric(out["ev"], errors="coerce")
        out[f"ev{FILING_SUFFIX}"] = enterprise
        out["ev"] = enterprise + (marketcap_now - marketcap)

    enterprise_now = pd.to_numeric(out.get("ev"), errors="coerce")

    for column, denominator in EQUITY_MULTIPLES.items():
        if denominator not in out.columns:
            continue
        if column in out.columns:
            out[f"{column}{FILING_SUFFIX}"] = out[column]
        out[column] = _safe_divide(marketcap_now, out[denominator])

    for column, denominator in ENTERPRISE_MULTIPLES.items():
        if denominator not in out.columns or enterprise_now is None:
            continue
        if column in out.columns:
            out[f"{column}{FILING_SUFFIX}"] = out[column]
        out[column] = _safe_divide(enterprise_now, out[denominator])

    for column, numerator in YIELDS.items():
        if numerator not in out.columns:
            continue
        if column in out.columns:
            out[f"{column}{FILING_SUFFIX}"] = out[column]
        out[column] = _safe_divide(out[numerator], marketcap_now)

    return out


def describe(frame: pd.DataFrame) -> str:
    """One line on how far the repricing actually moved things."""
    scale = pd.to_numeric(frame.get("price_scale"), errors="coerce").dropna()
    moved = scale[~np.isclose(scale, 1.0)]
    if moved.empty:
        return "Multiples are at the filing-date price."
    return (f"Multiples are at the latest close for {len(moved):,} companies; "
            f"the price has moved a median of {(moved - 1).abs().median():.1%} "
            f"since the last filing.")
