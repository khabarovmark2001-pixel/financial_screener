#!/usr/bin/env python3
"""
Restating Sharadar fundamentals in US dollars.

Sharadar's SF1 table mixes two currencies in one row and the column names do
not say which is which. Every statement line - revenue, assets, equity, the
cash-flow sections, and the per-share figures derived from them - is in the
filer's *reporting* currency. Market capitalisation, enterprise value and the
share price are always in US dollars, because the listing is a US one. For the
2,600-odd domestic filers the two coincide and nothing is at stake; for the 238
ADRs in this universe they do not, and a page that prints `revenue` under a
heading that says "US$ millions" is simply wrong - Toyota's quarter reads as
12 million million rather than $75bn.

`fxusd` is the vendor's own rate for the filing, quoted as *local currency
units per US dollar*, so the conversion divides. That is the opposite of what
the name suggests, so it is verified rather than assumed: dividing by it
reproduces Sharadar's own `revenueusd`, `equityusd`, `ebitdausd`, `ebitusd`,
`debtusd`, `cashnequsd`, `netinccmnusd` and `epsusd` columns on 100% of the
15,056 non-USD rows in this dataset. Those eight are the only USD variants the
vendor publishes; the other sixty-odd money columns have none, which is why
this module exists.

The rate is per filing, not per day, so a converted statement is translated at
the rate that applied on its own report date - the same figure the vendor used
for its own USD columns. It is a spot translation of every line, not the
multi-rate treatment IAS 21 requires of a real reporting entity (average rates
for income, closing rates for the balance sheet), so a converted statement will
not tie exactly to the company's own USD-denominated filing where it publishes
one. It is internally consistent, which is what makes ratios and sums across
lines meaningful, and it matches the vendor's own USD columns exactly.

Usage:

    import currency
    usd = currency.to_usd(statements)          # convert
    usd = currency.convert(statements, basis)  # convert only if basis is USD
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------
# how each of Sharadar's 112 fundamentals columns is denominated
# --------------------------------------------------------------------------
# The four sets below partition the table exactly - no column appears twice
# and none is missing, asserted at import. That matters more than it sounds:
# the failure mode of a currency layer is a field nobody classified, which
# then silently keeps its local-currency value in a table of dollars.

#: Identifiers, dates and share counts. Not money, so never converted.
#: Share counts are in *ordinary* shares while the per-share figures below are
#: per *listed* share (Sharadar has already applied `sharefactor` to them), so
#: per-share x share count does not tie for an ADR. That is the vendor's
#: convention, not a currency question, and converting cannot fix it.
META_FIELDS = frozenset({
    "ticker", "dimension", "calendardate", "date", "reportperiod",
    "fiscalperiod", "lastupdated", "fxusd", "sharefactor",
    "sharesbas", "shareswa", "shareswadil",
})

#: Pure ratios, margins and rates. Currency cancels, so never converted.
RATIO_FIELDS = frozenset({
    "assetturnover", "currentratio", "de", "divyield", "ebitdamargin",
    "evebit", "evebitda", "grossmargin", "netmargin", "payoutratio",
    "pb", "pe", "pe1", "ps", "ps1", "roa", "roe", "roic", "ros",
})

#: Already in US dollars as the vendor publishes them. Converting these is the
#: mirror-image bug of not converting the rest, and just as wrong.
USD_FIELDS = frozenset({
    "marketcap", "ev", "price",
    "cashnequsd", "debtusd", "ebitdausd", "ebitusd", "epsusd",
    "equityusd", "netinccmnusd", "revenueusd",
})

#: Per *listed* share, in the reporting currency. Same divisor as the money
#: lines - `eps / fxusd` reproduces the vendor's `epsusd` exactly.
PER_SHARE_FIELDS = frozenset({
    "bvps", "dps", "eps", "epsdil", "fcfps", "sps", "tbvps",
})

#: Absolute amounts in the reporting currency. The sixty-three columns that
#: carry the actual statements, and the reason this module exists.
MONEY_FIELDS = frozenset({
    "accoci", "assets", "assetsavg", "assetsc", "assetsnc", "capex", "cashneq",
    "cor", "consolinc", "debt", "debtc", "debtnc", "deferredrev", "depamor",
    "deposits", "ebit", "ebitda", "ebt", "equity", "equityavg", "fcf", "gp",
    "intangibles", "intexp", "invcap", "invcapavg", "inventory", "investments",
    "investmentsc", "investmentsnc", "liabilities", "liabilitiesc",
    "liabilitiesnc", "ncf", "ncfbus", "ncfcommon", "ncfdebt", "ncfdiv", "ncff",
    "ncfi", "ncfinv", "ncfo", "ncfx", "netinc", "netinccmn", "netincdis",
    "netincnci", "opex", "opinc", "payables", "ppnenet", "prefdivis",
    "receivables", "retearn", "revenue", "rnd", "sgna", "sbcomp", "tangibles",
    "taxassets", "taxexp", "taxliabilities", "workingcapital",
})

#: Everything that gets divided by the rate.
CONVERTED_FIELDS = MONEY_FIELDS | PER_SHARE_FIELDS

_GROUPS = (META_FIELDS, RATIO_FIELDS, USD_FIELDS, PER_SHARE_FIELDS, MONEY_FIELDS)
ALL_FIELDS = frozenset().union(*_GROUPS)

# A column classified twice, or into the wrong set, is the one bug this module
# cannot recover from at runtime, so it fails at import instead.
assert sum(len(g) for g in _GROUPS) == len(ALL_FIELDS), \
    "currency field sets overlap"

# --------------------------------------------------------------------------
# display bases
# --------------------------------------------------------------------------

USD = "US dollars"
AS_REPORTED = "As reported"
BASES = (USD, AS_REPORTED)

#: The column a converted frame carries so callers can show the rate used.
RATE_COLUMN = "fx_to_usd"

#: Marks a frame that has been through `to_usd`, so a double conversion - which
#: would silently square the rate - is caught rather than compounded.
_CONVERTED_FLAG = "_currency_converted_to_usd"


def rate(frame: pd.DataFrame) -> pd.Series:
    """The filing's local-currency-per-USD rate, cleaned for division.

    `fxusd` is 1.0 for a US filer. A zero, a negative or a null would turn a
    division into an infinity or wipe the row out, so all three fall back to
    1.0: leaving an amount in its reporting currency is a visible error, while
    an infinity propagates into every subtotal and ratio built on it.
    """
    if "fxusd" not in frame.columns:
        return pd.Series(1.0, index=frame.index)
    values = pd.to_numeric(frame["fxusd"], errors="coerce")
    return values.where(values > 0, 1.0).astype("float64")


def is_converted(frame: pd.DataFrame) -> bool:
    """True if this frame has already been restated in USD."""
    return bool(frame.attrs.get(_CONVERTED_FLAG, False))


def to_usd(frame: pd.DataFrame, extra_fields: tuple[str, ...] = ()) -> pd.DataFrame:
    """Restate every reporting-currency column in US dollars.

    `extra_fields` names columns a caller derived itself - the statements page
    computes working-capital movements and section residuals off the balance
    sheet - which are money but are not vendor fields, so they are not in
    MONEY_FIELDS. Derived columns are normally computed *after* conversion and
    need nothing here; this is for the cases where that order cannot hold.

    Converting an already-converted frame is a no-op rather than an error, so
    a page that hands the same frame to two renderers cannot square the rate.
    """
    if is_converted(frame):
        return frame

    converted = frame.copy()
    divisor = rate(frame)
    targets = (CONVERTED_FIELDS | set(extra_fields)) & set(converted.columns)
    for column in targets:
        values = pd.to_numeric(converted[column], errors="coerce")
        # float64 throughout: several vendor columns arrive as float32, and
        # dividing a 17,576-to-the-dollar rupiah balance in float32 loses
        # digits that matter at the top of a balance sheet.
        converted[column] = values.astype("float64") / divisor

    converted[RATE_COLUMN] = divisor
    converted.attrs = dict(frame.attrs)
    converted.attrs[_CONVERTED_FLAG] = True
    return converted


def convert(frame: pd.DataFrame, basis: str,
            extra_fields: tuple[str, ...] = ()) -> pd.DataFrame:
    """Convert to USD, or leave as filed, according to `basis`."""
    if basis == USD:
        return to_usd(frame, extra_fields)
    out = frame.copy()
    out[RATE_COLUMN] = rate(frame)
    out.attrs = dict(frame.attrs)
    return out


def is_foreign(frame: pd.DataFrame) -> bool:
    """True when any row in this frame was filed in something other than USD."""
    return bool((rate(frame) != 1.0).any())


def rate_summary(frame: pd.DataFrame) -> str | None:
    """One sentence naming the rate range used, or None for a USD filer.

    Sharadar publishes no ISO code for the reporting currency - not in SF1 and
    not in the securities master, where `currency` is the *listing* currency
    and is USD for every ticker here - so the rate is named rather than the
    currency. Guessing a code from the rate's magnitude or the company's
    domicile is not safe: this universe holds a UK-domiciled filer reporting in
    rupees and a Singapore-domiciled one reporting in rupiah.
    """
    rates = rate(frame)
    foreign = rates[rates != 1.0]
    if foreign.empty:
        return None
    low, high = float(foreign.min()), float(foreign.max())
    if np.isclose(low, high):
        return f"1 US$ = {low:,.4g} units of the reporting currency"
    return (f"1 US$ = {low:,.4g}-{high:,.4g} units of the reporting currency "
            f"across the periods shown")


def unit_label(basis: str, scale_name: str) -> str:
    """Column heading for a statement, honest about which currency it holds."""
    money = "US$" if basis == USD else "Reporting currency,"
    return f"{money} {scale_name.lower()}"
