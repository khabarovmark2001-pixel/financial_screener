#!/usr/bin/env python3
"""
Financial statements: income statement, balance sheet, cash flow, and the
valuation bridge, on a quarterly / TTM / annual basis.

Every subtotal on this page - gross profit, operating income, total current
assets, net cash from operations, EBITDA, enterprise value - is a column the
vendor already publishes, printed as filed. The page adds no arithmetic of its
own except the handful of lines listed in DERIVED below, which are marked with
a dagger in the table and spelled out under it.

Two things here are not "as filed", and both are choices the reader controls:

  Currency   Sharadar files every statement line in the company's own
             reporting currency while quoting market cap, enterprise value and
             the share price in US dollars. On the default basis the statement
             lines are restated in dollars at each filing's own rate, so the
             column heading and the numbers under it agree. See currency.py.

  Price      The valuation tab defaults to the latest close rather than the
             price on each filing date, because that is the multiple you can
             act on. The point-in-time series is one radio away.
"""

from __future__ import annotations

import html

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import currency
import current_multiples as cm
import ui_data as ui

st.set_page_config(page_title="Financial Statements", page_icon="page_facing_up",
                   layout="wide")

# --------------------------------------------------------------------------
# statement layout
# --------------------------------------------------------------------------
# Each row is (label, column, style, unit). `column` is the vendor field, or
# None for a spacer, or a key in DERIVED for the few lines Sharadar does not
# publish. Styles drive nothing but presentation:
#
#   item   an ordinary line, indented under whatever subtotal it rolls into
#   sub    a subtotal the vendor computed (bold, rule above)
#   total  a statement total (bold, double rule)
#   head   a section caption inside the statement
#   memo   supplementary, printed below the statement proper
#
# Units: money (scaled), share (counts, scaled), ps (per share), x (ratio),
# pct (already a fraction in the data).

INCOME = [
    ("Revenue",                          "revenue",     "sub",   "money"),
    ("Cost of revenue",                  "cor",         "item",  "money"),
    ("Gross profit",                     "gp",          "sub",   "money"),
    ("Research and development",         "rnd",         "item",  "money"),
    ("Selling, general and administrative", "sgna",     "item",  "money"),
    ("Other operating expenses",         "_other_opex", "item",  "money"),
    ("Total operating expenses",         "opex",        "sub",   "money"),
    ("Operating income",                 "opinc",       "sub",   "money"),
    ("Interest expense",                 "intexp",      "item",  "money"),
    ("Pre-tax income",                   "ebt",         "sub",   "money"),
    ("Income tax expense",               "taxexp",      "item",  "money"),
    ("Income from discontinued operations", "netincdis", "item", "money"),
    ("Net income",                       "netinc",      "sub",   "money"),
    ("Preferred dividends",              "prefdivis",   "item",  "money"),
    ("Net income to common",             "netinccmn",   "total", "money"),
    (None, None, "spacer", None),
    ("Earnings per share, basic",        "eps",         "item",  "ps"),
    ("Earnings per share, diluted",      "epsdil",      "item",  "ps"),
    ("Dividends per share",              "dps",         "item",  "ps"),
    ("Weighted average shares",          "shareswa",    "item",  "share"),
    ("Weighted average shares, diluted", "shareswadil", "item",  "share"),
    ("Shares outstanding, basic",        "sharesbas",   "item",  "share"),
    (None, None, "spacer", None),
    ("Net income to non-controlling interests", "netincnci", "memo", "money"),
    ("EBIT",                             "ebit",        "memo",  "money"),
    ("EBITDA",                           "ebitda",      "memo",  "money"),
    ("Depreciation and amortisation",    "depamor",     "memo",  "money"),
    ("Share-based compensation",         "sbcomp",      "memo",  "money"),
    ("Gross margin",                     "grossmargin", "memo",  "pct"),
    ("EBITDA margin",                    "ebitdamargin", "memo", "pct"),
    ("Operating margin",                 "ros",         "memo",  "pct"),
    ("Net margin",                       "netmargin",   "memo",  "pct"),
    ("Effective tax rate",               "_tax_rate",   "memo",  "pct"),
]

BALANCE = [
    ("ASSETS", None, "head", None),
    ("Cash and equivalents",             "cashneq",     "item",  "money"),
    ("Short-term investments",           "investmentsc", "item", "money"),
    ("Receivables",                      "receivables", "item",  "money"),
    ("Inventory",                        "inventory",   "item",  "money"),
    ("Total current assets",             "assetsc",     "sub",   "money"),
    ("Property, plant and equipment, net", "ppnenet",   "item",  "money"),
    ("Goodwill and intangibles",         "intangibles", "item",  "money"),
    ("Long-term investments",            "investmentsnc", "item", "money"),
    ("Tax assets",                       "taxassets",   "item",  "money"),
    ("Total non-current assets",         "assetsnc",    "sub",   "money"),
    ("Total assets",                     "assets",      "total", "money"),
    (None, None, "spacer", None),
    ("LIABILITIES", None, "head", None),
    ("Payables",                         "payables",    "item",  "money"),
    ("Deferred revenue",                 "deferredrev", "item",  "money"),
    ("Current debt",                     "debtc",       "item",  "money"),
    ("Total current liabilities",        "liabilitiesc", "sub",  "money"),
    ("Non-current debt",                 "debtnc",      "item",  "money"),
    ("Tax liabilities",                  "taxliabilities", "item", "money"),
    ("Total non-current liabilities",    "liabilitiesnc", "sub",  "money"),
    ("Deposits",                         "deposits",    "item",  "money"),
    ("Total liabilities",                "liabilities", "total", "money"),
    (None, None, "spacer", None),
    ("EQUITY", None, "head", None),
    ("Retained earnings",                "retearn",     "item",  "money"),
    ("Accumulated other comprehensive income", "accoci", "item", "money"),
    ("Total equity attributable to parent", "equity",   "total", "money"),
    (None, None, "spacer", None),
    ("Liabilities and parent equity",    "_liab_equity", "sub",  "money"),
    ("Non-controlling interests and residual", "_nci_residual", "item", "money"),
    ("Total assets (reconciliation)",    "assets",      "sub",   "money"),
    (None, None, "spacer", None),
    ("Total debt",                       "debt",        "memo",  "money"),
    ("Net debt",                         "_net_debt",   "memo",  "money"),
    ("Working capital",                  "workingcapital", "memo", "money"),
    ("Tangible assets (assets less intangibles)", "tangibles", "memo", "money"),
    ("Invested capital",                 "invcap",      "memo",  "money"),
    ("Book value per share",             "bvps",        "memo",  "ps"),
    ("Tangible assets per share",        "tbvps",       "memo",  "ps"),
]

CASHFLOW = [
    ("OPERATING ACTIVITIES", None, "head", None),
    ("Net income",                       "netinc",      "item",  "money"),
    ("Depreciation and amortisation",    "depamor",     "item",  "money"),
    ("Share-based compensation",         "sbcomp",      "item",  "money"),
    ("Change in receivables",            "_wc_receivables", "item", "money"),
    ("Change in inventory",              "_wc_inventory", "item", "money"),
    ("Change in payables",               "_wc_payables", "item", "money"),
    ("Change in deferred revenue",       "_wc_deferredrev", "item", "money"),
    ("Other operating activities",       "_other_ncfo", "item",  "money"),
    ("Net cash from operating activities", "ncfo",      "sub",   "money"),
    (None, None, "spacer", None),
    ("INVESTING ACTIVITIES", None, "head", None),
    ("Capital expenditure",              "capex",       "item",  "money"),
    ("Business acquisitions and disposals", "ncfbus",   "item",  "money"),
    ("Investment purchases and sales",   "ncfinv",      "item",  "money"),
    ("Other investing activities",       "_other_ncfi", "item",  "money"),
    ("Net cash from investing activities", "ncfi",      "sub",   "money"),
    (None, None, "spacer", None),
    ("FINANCING ACTIVITIES", None, "head", None),
    ("Issuance and repayment of debt",   "ncfdebt",     "item",  "money"),
    ("Issuance and repurchase of equity", "ncfcommon",  "item",  "money"),
    ("Dividends paid",                   "ncfdiv",      "item",  "money"),
    ("Other financing activities",       "_other_ncff", "item",  "money"),
    ("Net cash from financing activities", "ncff",      "sub",   "money"),
    (None, None, "spacer", None),
    ("Effect of exchange rates",         "ncfx",        "item",  "money"),
    ("Net change in cash",               "ncf",         "total", "money"),
    ("Unreconciled (sections plus FX less net change)",
                                         "_ncf_gap",    "item",  "money"),
    (None, None, "spacer", None),
    ("Free cash flow",                   "fcf",         "memo",  "money"),
    ("Free cash flow per share",         "fcfps",       "memo",  "ps"),
    ("Change in net working capital",    "_wc_total",   "memo",  "money"),
    ("Capex as a share of revenue",      "_capex_intensity", "memo", "pct"),
    ("Cash conversion (operating cash flow / net income)",
                                         "_cash_conversion", "memo", "x"),
    ("Other operating as a share of operating cash flow",
                                         "_other_ncfo_share", "memo", "pct"),
]

VALUATION = [
    ("EARNINGS BASE", None, "head", None),
    ("Revenue",                          "revenue",     "item",  "money"),
    ("EBITDA",                           "ebitda",      "sub",   "money"),
    ("EBIT",                             "ebit",        "sub",   "money"),
    ("Net income to common",             "netinccmn",   "item",  "money"),
    ("Free cash flow",                   "fcf",         "item",  "money"),
    (None, None, "spacer", None),
    ("CAPITAL STRUCTURE", None, "head", None),
    ("Market capitalisation",            "marketcap",   "item",  "money"),
    ("Total debt",                       "debt",        "item",  "money"),
    ("Cash and equivalents",             "cashneq",     "item",  "money"),
    ("Net debt",                         "_net_debt",   "sub",   "money"),
    ("Enterprise value",                 "ev",          "total", "money"),
    ("Share price at filing",            "price",       "item",  "ps"),
    (None, None, "spacer", None),
    ("ENTERPRISE MULTIPLES", None, "head", None),
    ("EV / EBITDA",                      "evebitda",    "item",  "x"),
    ("EV / EBIT",                        "evebit",      "item",  "x"),
    ("EV / Sales",                       "_ev_sales",   "item",  "x"),
    ("EV / Free cash flow",              "_ev_fcf",     "item",  "x"),
    (None, None, "spacer", None),
    ("EQUITY MULTIPLES", None, "head", None),
    ("Price / Earnings",                 "pe",          "item",  "x"),
    ("Price / Book",                     "pb",          "item",  "x"),
    ("Price / Sales",                    "ps",          "item",  "x"),
    ("Dividend yield",                   "divyield",    "item",  "pct"),
    ("Payout ratio",                     "payoutratio", "item",  "pct"),
    (None, None, "spacer", None),
    ("RETURNS AND MARGINS", None, "head", None),
    ("Return on equity",                 "roe",         "item",  "pct"),
    ("Return on assets",                 "roa",         "item",  "pct"),
    ("Return on invested capital",       "roic",        "item",  "pct"),
    ("Gross margin",                     "grossmargin", "item",  "pct"),
    ("EBITDA margin",                    "ebitdamargin", "item", "pct"),
    ("Net margin",                       "netmargin",   "item",  "pct"),
    ("Asset turnover",                   "assetturnover", "item", "x"),
    (None, None, "spacer", None),
    ("LEVERAGE AND LIQUIDITY", None, "head", None),
    ("Net debt / EBITDA",                "_netdebt_ebitda", "item", "x"),
    ("Debt / Equity",                    "de",          "item",  "x"),
    ("Interest coverage (EBIT / interest expense)",
                                         "_interest_cover", "item", "x"),
    ("Current ratio",                    "currentratio", "item", "x"),
]

# The only lines on this page the vendor does not publish. Each is one
# arithmetic step from fields it does publish, and each is flagged with a
# dagger in the table so nothing derived can be mistaken for something filed.
DERIVED = {
    "_wc_receivables":  ("Prior period's receivables less this period's — a "
                         "balance-sheet movement, signed as a cash effect"),
    "_wc_inventory":    ("Prior period's inventory less this period's — a "
                         "balance-sheet movement, signed as a cash effect"),
    "_wc_payables":     ("This period's payables less the prior period's — a "
                         "balance-sheet movement, signed as a cash effect"),
    "_wc_deferredrev":  ("This period's deferred revenue less the prior "
                         "period's — a balance-sheet movement, signed as a "
                         "cash effect"),
    "_wc_total":        "The four working-capital movements above, added",
    "_other_ncfo":      ("Net cash from operating activities less the lines "
                         "above it. Absorbs every reconciling item the vendor "
                         "does not publish — deferred taxes, impairments, "
                         "gains on disposal, provisions — and the part of the "
                         "balance-sheet movement that never touched operating "
                         "cash flow, such as working capital acquired with a "
                         "business or restated by currency translation"),
    "_other_ncfi":      ("Net cash from investing activities less capex, "
                         "acquisitions and investment purchases"),
    "_other_ncff":      ("Net cash from financing activities less debt, "
                         "equity and dividend flows"),
    "_ncf_gap":         ("The three section subtotals plus the FX effect, "
                         "less the reported net change in cash — zero to "
                         "within 0.1% for 97% of filings"),
    "_other_ncfo_share": "Other operating activities / net cash from operations",
    "_other_opex":      ("Total operating expenses less SG&A and R&D. Needed "
                         "because those two equal opex for only 30% of "
                         "filings — the rest report expense captions the "
                         "vendor does not break out"),
    "_net_debt":        "Total debt less cash and equivalents (debt − cashneq)",
    "_liab_equity":     "Total liabilities plus parent equity (liabilities + equity)",
    "_nci_residual":    ("Total assets less liabilities and parent equity "
                         "(assets − liabilities − equity). The vendor's `equity` "
                         "is the parent's share only and it publishes no separate "
                         "non-controlling-interest balance, so this residual is "
                         "what closes the balance sheet — for 908 of the 1,075 "
                         "companies where it is non-zero, it is NCI"),
    "_tax_rate":        "Income tax expense / pre-tax income (taxexp ÷ ebt)",
    "_capex_intensity": "Capital expenditure / revenue (|capex| ÷ revenue)",
    "_cash_conversion": "Operating cash flow / net income (ncfo ÷ netinc)",
    "_ev_sales":        "Enterprise value / revenue (ev ÷ revenue)",
    "_ev_fcf":          "Enterprise value / free cash flow (ev ÷ fcf)",
    "_netdebt_ebitda":  "Net debt / EBITDA",
    "_interest_cover":  "EBIT / interest expense (ebit ÷ |intexp|)",
}

# First label each derived key is given, so the note below the statements
# names them the way the rows do without listing a key twice.
DERIVED_LABELS = {}
for _label, _column, _style, _unit in INCOME + BALANCE + CASHFLOW + VALUATION:
    if _column in DERIVED:
        DERIVED_LABELS.setdefault(_column, _label)


BASES = {
    "Quarterly": ("ARQ", "Each column is one **discrete fiscal quarter** as "
                         "filed — three months of revenue, cash flow and "
                         "earnings, and the balance sheet as it stood on the "
                         "quarter end."),
    "TTM":       ("ART", "Each column is the **trailing twelve months** ending "
                         "on that quarter, as the vendor rolls it up. Income "
                         "and cash-flow lines are four quarters summed; "
                         "balance-sheet lines are stocks, so they are the "
                         "point-in-time balance on the quarter end, not a sum."),
    "Annual":    ("ARY", "Each column is one **fiscal year** as filed. This is "
                         "the vendor's own annual statement, not four quarters "
                         "added together — which matters for 52/53-week "
                         "calendars, mid-year restatements and stub periods."),
}


# --------------------------------------------------------------------------
# derivation and formatting
# --------------------------------------------------------------------------

def _safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    """Element-wise division that yields NaN rather than inf on a zero base.

    A ratio against a zero or negative denominator is not a small number, it
    is a meaningless one - a loss-making quarter would otherwise print a
    negative EV/EBIT that reads like a cheap multiple.
    """
    denominator = denominator.where(denominator > 0)
    return numerator / denominator


# How far back the previous balance sheet sits, per basis, and how many months
# should separate the two period ends. ART rows are one quarter apart but each
# covers twelve months, so a trailing-twelve-month movement is measured four
# rows back, not one.
PRIOR_PERIOD = {"ARQ": (1, 3), "ART": (4, 12), "ARY": (1, 12)}

# A working-capital line is only meaningful if the two balance sheets it
# differences are the expected distance apart. Skipped filings and changed
# fiscal year-ends both leave gaps, and differencing across one would book a
# multi-period movement into a single column.
GAP_TOLERANCE_MONTHS = 1.2


def working_capital_moves(frame: pd.DataFrame, dimension: str) -> pd.DataFrame:
    """Balance-sheet movements, signed the way a cash flow statement prints them.

    Sharadar publishes no working-capital lines at all - the cash flow fields
    stop at the three section subtotals plus capex, D&A and SBC - so these are
    differenced off the balance sheet. Signs follow the statement, not the
    balance sheet: receivables and inventory going UP consumes cash and prints
    negative, payables and deferred revenue going up releases cash and prints
    positive.

    These are NOT the filed cash flow lines and will not equal them. A balance
    sheet moves for reasons that never touch operating cash flow - an
    acquisition arrives with a receivables book, currency translation restates
    a foreign subsidiary, items get reclassified between captions. That
    difference lands in the "other operating activities" residual below, which
    is why that line is not small. See RELIABILITY.md for the measured size.
    """
    lag, months_expected = PRIOR_PERIOD[dimension]
    ends = pd.to_datetime(frame["calendardate"])
    gap_months = (ends - ends.shift(lag)).dt.days / 30.44
    usable = (gap_months - months_expected).abs() <= GAP_TOLERANCE_MONTHS

    moves = pd.DataFrame(index=frame.index)
    for column, sign in (("receivables", -1), ("inventory", -1),
                         ("payables", 1), ("deferredrev", 1)):
        values = pd.to_numeric(frame.get(column), errors="coerce")
        if values is None:
            moves[column] = np.nan
            continue
        moves[column] = (sign * (values - values.shift(lag))).where(usable)
    return moves


def add_derived(frame: pd.DataFrame, dimension: str) -> pd.DataFrame:
    """Attach the DERIVED lines. Everything else is already in the frame."""
    frame = frame.copy()
    get = lambda name: (pd.to_numeric(frame[name], errors="coerce")
                        if name in frame.columns
                        else pd.Series(np.nan, index=frame.index))

    debt, cash, ebitda, ebit = get("debt"), get("cashneq"), get("ebitda"), get("ebit")
    revenue, ev, fcf = get("revenue"), get("ev"), get("fcf")

    # --- cash flow: working capital, then the three section residuals -----
    moves = working_capital_moves(frame, dimension)
    frame["_wc_receivables"] = moves["receivables"]
    frame["_wc_inventory"] = moves["inventory"]
    frame["_wc_payables"] = moves["payables"]
    frame["_wc_deferredrev"] = moves["deferredrev"]
    frame["_wc_total"] = moves.sum(axis=1, min_count=1)

    # Each residual is defined as the vendor's subtotal less the lines above
    # it, so every section foots to the filed number exactly. The itemised
    # lines are summed with fillna(0) - a field the company does not report is
    # absent, not unknown - but the residual is left NaN when the subtotal
    # itself is missing, so a blank period stays blank instead of printing the
    # negated sum of its own detail.
    operating_detail = (get("netinc").fillna(0) + get("depamor").fillna(0)
                        + get("sbcomp").fillna(0)
                        + frame["_wc_total"].fillna(0))
    frame["_other_ncfo"] = get("ncfo") - operating_detail

    investing_detail = (get("capex").fillna(0) + get("ncfbus").fillna(0)
                        + get("ncfinv").fillna(0))
    frame["_other_ncfi"] = get("ncfi") - investing_detail

    financing_detail = (get("ncfdebt").fillna(0) + get("ncfcommon").fillna(0)
                        + get("ncfdiv").fillna(0))
    frame["_other_ncff"] = get("ncff") - financing_detail

    frame["_ncf_gap"] = (get("ncfo").fillna(0) + get("ncfi").fillna(0)
                         + get("ncff").fillna(0) + get("ncfx").fillna(0)
                         - get("ncf"))
    frame["_other_ncfo_share"] = _safe_divide(frame["_other_ncfo"].abs(),
                                              get("ncfo").abs())

    # Same idea on the income statement: sgna + rnd equals opex for only about
    # a third of filings, so the detail needs a residual to foot as well.
    frame["_other_opex"] = get("opex") - (get("sgna").fillna(0)
                                          + get("rnd").fillna(0))

    frame["_net_debt"] = debt - cash
    frame["_liab_equity"] = get("liabilities") + get("equity")
    frame["_nci_residual"] = get("assets") - frame["_liab_equity"]
    frame["_tax_rate"] = _safe_divide(get("taxexp"), get("ebt"))
    frame["_capex_intensity"] = _safe_divide(get("capex").abs(), revenue)
    frame["_cash_conversion"] = _safe_divide(get("ncfo"), get("netinc"))
    frame["_ev_sales"] = _safe_divide(ev, revenue)
    frame["_ev_fcf"] = _safe_divide(ev, fcf)
    frame["_netdebt_ebitda"] = _safe_divide(frame["_net_debt"], ebitda)
    frame["_interest_cover"] = _safe_divide(ebit, get("intexp").abs())
    return frame


# --------------------------------------------------------------------------
# the valuation block: currency, then price
# --------------------------------------------------------------------------

AT_FILING = "At each filing date"
AT_LATEST = "At the latest close"
PRICE_BASES = (AT_LATEST, AT_FILING)

#: Multiples that divide a US-dollar market value by a statement figure.
#: `pe`, `ps`, `pb`, `evebit`, `evebitda` and `divyield` are the vendor's and
#: are already dollar-on-dollar - it computes them from its own `*usd` columns,
#: verified on this dataset. `_ev_sales` and `_ev_fcf` are this page's and are
#: not: `ev / revenue` puts a dollar numerator over a reporting-currency
#: denominator, which is what made Toyota print an EV/Sales of 0.005x.
_EQUITY_MULTIPLES = ("pe", "ps", "pb")
_ENTERPRISE_MULTIPLES = ("evebit", "evebitda", "_ev_sales", "_ev_fcf")


def fix_currency_mixing(frame: pd.DataFrame, usd: pd.DataFrame) -> pd.DataFrame:
    """Rebuild the two derived EV multiples from dollars, whatever the basis.

    A multiple is a dollar over a dollar in both display modes - the numerator
    is a market value quoted on a US exchange and there is no version of it in
    the filer's own currency. So these two are built from the dollar frame
    even when the statement above them is printing yen.
    """
    out = frame.copy()
    enterprise = pd.to_numeric(usd.get("ev"), errors="coerce")
    out["_ev_sales"] = _safe_divide(enterprise,
                                    pd.to_numeric(usd.get("revenue"),
                                                  errors="coerce"))
    out["_ev_fcf"] = _safe_divide(enterprise,
                                  pd.to_numeric(usd.get("fcf"), errors="coerce"))
    return out


def reprice_to_latest(frame: pd.DataFrame, scale: pd.Series,
                      close_now: float) -> pd.DataFrame:
    """Move the whole valuation block from each filing's price to today's.

    Equity multiples scale with the market cap directly. Enterprise multiples
    cannot: the net debt inside enterprise value has not moved, so EV changes
    by the same *amount* as the market cap rather than in the same proportion,
    and each EV multiple is rebuilt around the new EV.

    Rebuilt by inverting the stored multiple to recover its denominator rather
    than by reading that denominator off the row. The vendor computes `pe`,
    `ps` and `evebitda` on a trailing twelve months even on a quarterly row,
    so the row's own revenue is a quarter of what its `ps` divided by -
    dividing by it would quietly quadruple every quarterly multiple on the
    page. Inverting carries whichever period the vendor actually used.
    """
    out = frame.copy()
    scale = pd.to_numeric(scale, errors="coerce").reindex(out.index).fillna(1.0)

    marketcap = pd.to_numeric(out.get("marketcap"), errors="coerce")
    enterprise = pd.to_numeric(out.get("ev"), errors="coerce")
    marketcap_now = marketcap * scale
    enterprise_now = enterprise + (marketcap_now - marketcap)

    for column in _EQUITY_MULTIPLES:
        if column in out.columns:
            out[column] = pd.to_numeric(out[column], errors="coerce") * scale
    # A yield is the same ratio the other way up, so it moves inversely.
    if "divyield" in out.columns:
        out["divyield"] = (pd.to_numeric(out["divyield"], errors="coerce")
                           / scale.replace(0, np.nan))

    for column in _ENTERPRISE_MULTIPLES:
        if column not in out.columns:
            continue
        multiple = pd.to_numeric(out[column], errors="coerce")
        implied_base = enterprise / multiple.replace(0, np.nan)
        out[column] = enterprise_now / implied_base.replace(0, np.nan)

    out["marketcap"] = marketcap_now
    out["ev"] = enterprise_now
    # One price for every column now, because there is only one latest close.
    out["price"] = close_now
    return out


@st.cache_data(ttl=3600, show_spinner=False)
def latest_close_scale(ticker: str, filing_dates: tuple) -> tuple:
    """Per-column price scale factor, plus the close it moves everything to.

    Split-adjusted closes on both ends, never the raw quote: a split moves the
    share count and the printed price in opposite directions, so a raw-price
    ratio would rescale the market cap by the split factor. Sharadar's `close`
    is split-adjusted across the whole series; `closeunadj` is not. Dividends
    are deliberately left in - a market cap really does fall when a company
    pays one away - which is why this is `close` and not `closeadj`.

    A filing older than the stored price history gets a scale of 1.0 and so
    keeps its filing-date multiple: visibly stale beats wrong by a guess.
    """
    prices = ui.load_prices(ticker)
    if prices.empty:
        return [1.0] * len(filing_dates), float("nan"), None

    prices = prices[["date", "close"]].dropna().copy()
    prices["date"] = pd.to_datetime(prices["date"])
    prices = prices.sort_values("date")
    close_now = float(prices["close"].iloc[-1])
    close_date = prices["date"].iloc[-1]

    wanted = pd.DataFrame({"date": pd.to_datetime(list(filing_dates))})
    matched = pd.merge_asof(wanted.sort_values("date"), prices,
                            on="date", direction="backward")
    matched = matched.set_index(wanted.sort_values("date").index).sort_index()
    scale = cm.price_scale(pd.Series(close_now, index=matched.index),
                           matched["close"])
    return scale.tolist(), close_now, close_date


def format_value(value, unit: str, scale: float) -> str:
    """One cell. Negatives in parentheses, the way a statement prints them."""
    if value is None or not np.isfinite(value):
        return "—"
    if unit == "rate":
        # An exchange rate spans 0.73 (sterling) to 17,576 (rupiah) across this
        # universe, so it gets significant figures rather than fixed decimals.
        return f"{value:,.6g}"
    if unit == "pct":
        return f"({abs(value):.1%})" if value < 0 else f"{value:.1%}"
    if unit == "x":
        return f"({abs(value):,.2f}x)" if value < 0 else f"{value:,.2f}x"
    if unit == "ps":
        return f"({abs(value):,.2f})" if value < 0 else f"{value:,.2f}"
    scaled = value / scale
    if abs(scaled) < 0.05 and value != 0:
        return "—"
    return f"({abs(scaled):,.1f})" if scaled < 0 else f"{scaled:,.1f}"


def is_blank(series: pd.Series) -> bool:
    """True when a line has nothing to say across every period shown.

    Zero counts as nothing here, not as a reported zero: the vendor fills
    inapplicable lines with 0.0 rather than null, so `deposits` on a software
    company and `inventory` on a bank would otherwise occupy a row of zeros in
    every statement on the site.
    """
    values = pd.to_numeric(series, errors="coerce")
    return bool(values.isna().all() or (values.fillna(0) == 0).all())


def statement_html(rows, frame: pd.DataFrame, columns: list[str],
                   scale: float, unit_label: str, show_blank: bool) -> str:
    """Render one statement as a table, subtotals and all."""
    head = "".join(f"<th>{html.escape(c)}</th>" for c in columns)
    body, pending_spacer = [], False

    for label, column, style, unit in rows:
        if style == "spacer":
            pending_spacer = True
            continue
        if style == "head":
            if pending_spacer:
                body.append('<tr class="spacer"><td colspan="99"></td></tr>')
                pending_spacer = False
            body.append(f'<tr class="head"><td colspan="99">'
                        f'{html.escape(label)}</td></tr>')
            continue

        series = (frame[column] if column in frame.columns
                  else pd.Series(np.nan, index=frame.index))
        if not show_blank and style in ("item", "memo") and is_blank(series):
            continue

        if pending_spacer:
            body.append('<tr class="spacer"><td colspan="99"></td></tr>')
            pending_spacer = False

        mark = ' <span class="drv" title="derived">†</span>' if column in DERIVED else ""
        cells = "".join(
            f"<td>{format_value(v, unit, scale)}</td>"
            for v in pd.to_numeric(series, errors="coerce"))
        body.append(f'<tr class="{style}"><th>{html.escape(label)}{mark}</th>'
                    f"{cells}</tr>")

    return (f'<div class="fs-wrap"><table class="fs">'
            f"<thead><tr><th>{html.escape(unit_label)}</th>{head}</tr></thead>"
            f'<tbody>{"".join(body)}</tbody></table></div>')


def to_export(rows, frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """The same statement as raw numbers, for the CSV download."""
    records, labels = [], []
    for label, column, style, _unit in rows:
        if style in ("spacer", "head") or column is None:
            continue
        series = (pd.to_numeric(frame[column], errors="coerce")
                  if column in frame.columns
                  else pd.Series(np.nan, index=frame.index))
        labels.append(label)
        records.append(series.tolist())
    return pd.DataFrame(records, index=labels, columns=columns)


# --------------------------------------------------------------------------
# custom charts
# --------------------------------------------------------------------------
# The chart tab plots the same line items the statements above print, so its
# catalogue is built from the four statement layouts rather than maintained
# separately - anything that has a row has a chart, derived lines included.
# First occurrence wins, so a line appearing on two statements (revenue, net
# debt, the margins) is offered once, with the unit its own statement gives it.

CHART_PALETTE = ["#2F5D8A", "#B4472F", "#1B7F5E", "#8A6D2F", "#6B4C8A",
                 "#2F7F8A", "#A5541F", "#4B6F44"]

CHART_METRICS: dict[str, tuple[str, str]] = {}
for _label, _column, _style, _unit in INCOME + BALANCE + CASHFLOW + VALUATION:
    if _style in ("spacer", "head") or _column is None:
        continue
    CHART_METRICS.setdefault(_label, (_column, _unit))

ABSOLUTE, GROWTH, MARGIN = "Absolute", "Growth YoY", "% of revenue"
CHART_VIEWS = [ABSOLUTE, GROWTH, MARGIN]

# Longest first, so the slider reads left (all history) to right (recent only).
CHART_WINDOWS = {"Max": None, "20Y": 20, "15Y": 15, "10Y": 10, "7Y": 7,
                 "5Y": 5, "3Y": 3, "2Y": 2}

# How many rows back the same period one year earlier sits. ART rows are one
# quarter apart but each already covers twelve months, so its year-ago
# comparator is four rows back, not one - comparing adjacent TTM rows would
# measure a quarter of change and call it a year of growth.
YOY_LAG = {"ARQ": 4, "ART": 4, "ARY": 1}

# New charts start on different line items, so pressing Add chart twice does
# not produce two copies of the same revenue series.
CHART_DEFAULTS = ["Revenue", "Net income", "EBITDA",
                  "Research and development", "Free cash flow",
                  "Gross margin", "Capital expenditure", "Total debt"]


@st.cache_data(ttl=3600, show_spinner=False)
def chart_statements(ticker: str, dimension: str,
                     money_basis: str) -> pd.DataFrame:
    """One company's filings on one basis, with the derived lines attached.

    A chart can hold several companies at once, and on the dollar basis that
    is the only way a comparison between a domestic filer and an ADR means
    anything - two lines on one axis in two currencies is a picture of the
    exchange rate.
    """
    frame = ui.load_statements(ticker, dimension, money_basis)
    if frame.empty:
        return frame
    derived = add_derived(frame, dimension)
    if money_basis == currency.USD:
        return derived
    usd = add_derived(ui.load_statements(ticker, dimension, currency.USD),
                      dimension)
    return fix_currency_mixing(derived, usd)


#: Chart lines whose value depends on the share price, and which therefore
#: stop being current the moment the price moves off the filing-date quote.
PRICE_DEPENDENT = (set(_EQUITY_MULTIPLES) | set(_ENTERPRISE_MULTIPLES)
                   | {"marketcap", "ev", "price", "divyield"})


def current_point(ticker: str, frame: pd.DataFrame, column: str) -> tuple | None:
    """The last filed period revalued at the latest close, as (date, value).

    Computed from the statement's own last row rather than looked up from the
    analytics table on purpose. The vendor's `pe` and `ps` are trailing-twelve-
    month figures even on a quarterly row, while this page's `_ev_sales` is
    built from whatever period the row covers - so a value imported from
    elsewhere would sit on a different denominator from the line it is
    extending, and the join between them would be a step with no meaning.
    Repricing the row the line already ends on cannot drift.
    """
    if frame.empty or column not in frame.columns:
        return None
    last = frame.tail(1).reset_index(drop=True)
    scale, close_now, close_date = latest_close_scale(
        ticker, (pd.to_datetime(last["date"].iloc[0]),))
    if close_date is None:
        return None
    repriced = reprice_to_latest(last, pd.Series(scale), close_now)
    value = pd.to_numeric(repriced[column], errors="coerce").iloc[0]
    if not np.isfinite(value):
        return None
    return pd.Timestamp(close_date), float(value)


def chart_series(ticker: str, dimension: str, column: str,
                 view: str, money_basis: str) -> pd.DataFrame:
    """One plotted line: period end against the value the view asks for.

    A price-dependent line is carried one segment past its last filing to the
    latest close, so it ends where the tables and the valuation tab do rather
    than at whatever the price happened to be when the last filing landed.
    Only on the absolute view: a year-on-year growth rate needs a year-ago
    comparator the current point does not have, and a multiple expressed as a
    share of revenue is not a quantity that has a "now".

    Growth is left blank where the year-ago base is zero or negative - a swing
    out of a loss has no meaningful percentage - and where the two periods are
    not actually twelve months apart, which is what a skipped filing or a
    changed fiscal year-end leaves behind.
    """
    frame = chart_statements(ticker, dimension, money_basis)
    if frame.empty or column not in frame.columns:
        return pd.DataFrame(columns=["date", "value"])

    out = pd.DataFrame({
        "date": pd.to_datetime(frame["reportperiod"]),
        "value": pd.to_numeric(frame[column], errors="coerce"),
        "is_current": False})

    if view == ABSOLUTE and column in PRICE_DEPENDENT:
        point = current_point(ticker, frame, column)
        if point is not None:
            out = pd.concat([out, pd.DataFrame(
                [{"date": point[0], "value": point[1], "is_current": True}])],
                ignore_index=True)

    if view == GROWTH:
        lag = YOY_LAG[dimension]
        prior = out["value"].shift(lag)
        gap_months = (out["date"] - out["date"].shift(lag)).dt.days / 30.44
        usable = (gap_months - 12).abs() <= GAP_TOLERANCE_MONTHS
        out["value"] = ((out["value"] - prior)
                        / prior.where(prior > 0)).where(usable)
    elif view == MARGIN:
        out["value"] = _safe_divide(
            out["value"], pd.to_numeric(frame.get("revenue"), errors="coerce"))

    return out.dropna(subset=["value"]).sort_values("date")


def chart_axis(view: str, unit: str, scale: float,
               unit_label: str) -> tuple[str, str, float]:
    """Axis title, tick format and the divisor the values need."""
    if view == GROWTH:
        return "Year-on-year growth", ".0%", 1.0
    if view == MARGIN:
        return "Share of revenue", ".0%", 1.0
    if unit == "pct":
        return "Percent", ".1%", 1.0
    if unit == "x":
        return "Ratio", ",.2f", 1.0
    if unit == "ps":
        return "Per share", ",.2f", 1.0
    return unit_label, ",.0f", scale


def render_statement_chart(chart_id: int, options: list[str],
                           labels: dict[str, str], default_ticker: str,
                           scale: float, unit_label: str, money_basis: str,
                           removable: bool) -> None:
    """One configurable chart. Every widget key carries the chart's id, so
    charts keep their own settings and removing one leaves the rest alone."""
    with st.container(border=True):
        top = st.columns([3, 3, 2, 2])
        metric_names = list(CHART_METRICS)
        fallback = CHART_DEFAULTS[chart_id % len(CHART_DEFAULTS)]
        metric_label = top[0].selectbox(
            "Line item", metric_names, key=f"fs_chart_metric_{chart_id}",
            index=metric_names.index(fallback) if fallback in metric_names else 0,
            help="Every line the statements above print, derived lines included.")
        view = top[1].radio("View", CHART_VIEWS, horizontal=True,
                            key=f"fs_chart_view_{chart_id}")
        basis_choice = top[2].radio("Reporting", list(BASES), horizontal=True,
                                    key=f"fs_chart_basis_{chart_id}")
        window = top[3].select_slider(
            "Timeframe", list(CHART_WINDOWS), value="10Y",
            key=f"fs_chart_window_{chart_id}",
            help="How far back to plot, from all filed history down to the "
                 "last two years.")

        bottom = st.columns([9, 1], vertical_alignment="bottom")
        tickers = bottom[0].multiselect(
            "Companies", options, default=[default_ticker],
            key=f"fs_chart_tickers_{chart_id}",
            format_func=lambda t: f"{t} — {labels.get(t, '')}"[:48],
            help="Any listed company - the sector and industry of the one "
                 "selected at the top of the page do not narrow this list.")
        if removable and bottom[1].button("Remove", key=f"fs_chart_remove_{chart_id}",
                                          width="stretch"):
            st.session_state.fs_chart_ids = [
                i for i in st.session_state.fs_chart_ids if i != chart_id]
            st.rerun()

        column, unit = CHART_METRICS[metric_label]
        dimension_choice = BASES[basis_choice][0]
        years = CHART_WINDOWS[window]
        cutoff = (None if years is None
                  else pd.Timestamp.today() - pd.DateOffset(years=years))
        title, tickformat, divisor = chart_axis(view, unit, scale, unit_label)

        figure = go.Figure()
        missing, repriced_any = [], False
        for index, ticker in enumerate(tickers):
            series = chart_series(ticker, dimension_choice, column, view,
                                  money_basis)
            if cutoff is not None and not series.empty:
                series = series[series["date"] >= cutoff]
            if series.empty:
                missing.append(ticker)
                continue
            colour = CHART_PALETTE[index % len(CHART_PALETTE)]
            figure.add_trace(go.Scatter(
                x=series["date"], y=series["value"] / divisor, name=ticker,
                mode="lines+markers", marker=dict(size=4),
                line=dict(color=colour, width=2),
                hovertemplate=f"%{{y:{tickformat}}}<extra>{ticker}</extra>"))

            # The current point gets its own marker so it can never be read as
            # a filing. It is the last filed period revalued at today's close.
            current = series[series.get("is_current", False) == True]  # noqa: E712
            if not current.empty:
                repriced_any = True
                figure.add_trace(go.Scatter(
                    x=current["date"], y=current["value"] / divisor,
                    mode="markers", showlegend=False,
                    marker=dict(color=colour, size=10, symbol="diamond",
                                line=dict(width=1, color="white")),
                    hovertemplate=f"Now: %{{y:{tickformat}}}"
                                  f"<extra>{ticker}</extra>"))

        if not tickers:
            st.info("Pick at least one company to plot.")
            return

        if view in (GROWTH, MARGIN):
            figure.add_hline(y=0, line_width=1, line_color="#8894A0", opacity=0.5)
        # figure.update_layout(
        #     height=380, hovermode="x unified", margin=dict(t=30, b=0),
        #     yaxis=dict(title=title, tickformat=tickformat),
        #     legend=dict(orientation="h", y=1.12),
        #     title=dict(text=f"{metric_label} — {view.lower()}, "
        #                     f"{basis_choice.lower()}", x=0, font=dict(size=14)))
        st.plotly_chart(figure, width="stretch",
                        key=f"chart_statement_custom_{chart_id}")

        notes = []
        if view == GROWTH:
            notes.append("Growth is blank where the year-ago base is zero or "
                         "negative, and where the two periods are not twelve "
                         "months apart.")
        if view == MARGIN:
            notes.append("Blank where revenue is zero or negative. A "
                         "balance-sheet line against revenue is a ratio, not "
                         "a margin - read it as size relative to sales.")
        if view == ABSOLUTE and unit in ("money", "share"):
            notes.append(f"Values in {unit_label.lower()}, set by the Units "
                         f"control at the top of the page.")
        if repriced_any:
            notes.append("The ◆ is the last filed period revalued at the "
                         "latest close - same reported earnings and balance "
                         "sheet, market cap moved to the last trading day. "
                         "The gap between it and the marker before it is the "
                         "price move since that filing.")
        if missing:
            notes.append(f"No {basis_choice.lower()} data in this window for "
                         f"{', '.join(missing)}.")
        if notes:
            st.caption(" ".join(notes))


CSS = """
<style>
.fs-wrap { overflow-x: auto; }
table.fs { border-collapse: collapse; width: 100%; font-size: 0.86rem;
           font-variant-numeric: tabular-nums; }
table.fs th, table.fs td { padding: 0.28rem 0.7rem; white-space: nowrap; }
table.fs thead th { text-align: right; font-weight: 600; border-bottom: 2px solid;
                    position: sticky; top: 0; background: var(--background-color); }
table.fs thead th:first-child { text-align: left; }
table.fs td { text-align: right; }
table.fs tbody th { text-align: left; font-weight: 400; padding-left: 1.6rem; }
table.fs tr.sub th, table.fs tr.total th { font-weight: 700; padding-left: 0.7rem; }
table.fs tr.sub td, table.fs tr.total td { font-weight: 700; }
table.fs tr.sub th, table.fs tr.sub td { border-top: 1px solid; }
table.fs tr.total th, table.fs tr.total td { border-top: 1px solid;
                                             border-bottom: 3px double; }
table.fs tr.head td { text-align: left; font-weight: 700; letter-spacing: 0.06em;
                      font-size: 0.74rem; padding-top: 0.9rem; opacity: 0.75; }
table.fs tr.memo th { font-style: italic; opacity: 0.85; }
table.fs tr.memo td { font-style: italic; opacity: 0.85; }
table.fs tr.spacer td { padding: 0; height: 0.6rem; }
table.fs tbody tr:hover td, table.fs tbody tr:hover th { background: rgba(127,127,127,0.10); }
.fs .drv { opacity: 0.55; cursor: help; }
</style>
"""


# --------------------------------------------------------------------------
# page
# --------------------------------------------------------------------------

st.markdown(CSS, unsafe_allow_html=True)

analytics_frame = ui.load_analytics()
universe = analytics_frame[analytics_frame["in_universe"]].copy()
universe = universe.sort_values("marketcap", ascending=False)

st.title("Financial Statements")

with st.sidebar:
    st.header("Display")
    money_basis = ui.currency_selector()

options = universe["ticker"].tolist()
labels = dict(zip(universe["ticker"], universe["name"].fillna("")))

controls = st.columns([3, 2, 2, 2])
# Shares the session key with the deep dive, so picking a company on one page
# carries to the other instead of resetting to the largest name in the index.
selected = controls[0].selectbox(
    "Company", options, key="deep_dive_ticker",
    format_func=lambda t: f"{t} — {labels.get(t, '')}"[:60])
basis = controls[1].radio("Basis", list(BASES), horizontal=True,
                          key="statements_basis")
period_count = controls[2].slider("Periods shown", 4, 40, 12,
                                  key="statements_periods")
unit_choice = controls[3].radio("Units", ["Millions", "Billions", "Thousands"],
                                key="statements_units")

SCALE = {"Thousands": 1e3, "Millions": 1e6, "Billions": 1e9}[unit_choice]
UNIT_LABEL = currency.unit_label(money_basis, unit_choice)

dimension, basis_note = BASES[basis]
row = universe[universe["ticker"] == selected].iloc[0]
st.caption(f"**{row['name']}** — {row['sector']} > **{row['industry']}** — "
           f"{row['exchange']} — market cap {ui.money(row['marketcap'])}")

# Two frames, because the two questions on this page are denominated
# differently. `statements` is what the statements print and follows the
# currency control; `statements_usd` is what the multiples are built from,
# because a market value has no version in the filer's own currency. For a US
# filer the two are identical and the second costs one cached read.
statements = ui.load_statements(selected, dimension, money_basis)
if statements.empty:
    st.warning(f"No {basis.lower()} statements filed for {selected}. "
               f"Sharadar publishes the annual roll-up for most but not all "
               f"names; try another basis.")
    st.stop()
statements_usd = ui.load_statements(selected, dimension, currency.USD)

statements = add_derived(statements, dimension)
statements_usd = add_derived(statements_usd, dimension)
statements = fix_currency_mixing(statements, statements_usd)

statements = statements.tail(period_count)
# Most recent first: a statement is read left to right from the latest period.
statements = statements.iloc[::-1].reset_index(drop=True)

period_labels = [str(v) for v in statements["fiscalperiod"]]
filed = pd.to_datetime(statements["date"])
periods_end = pd.to_datetime(statements["reportperiod"])

foreign_filer = currency.is_foreign(statements)
rate_note = currency.rate_summary(statements)

st.info(f"{basis_note}\n\nColumns are labelled by fiscal period, newest first. "
        f"Latest column covers the period ended "
        f"**{periods_end.iloc[0].date()}**, filed **{filed.iloc[0].date()}**. "
        f"Where a period was amended, the most recent filing is shown.")

if foreign_filer:
    if money_basis == currency.USD:
        st.success(
            f"{selected} files in a currency other than the US dollar. Every "
            f"amount below has been restated in **US dollars at the exchange "
            f"rate on its own report date** — {rate_note} — so the column "
            f"headings and the numbers under them agree. The rate used is on "
            f"the last line of each statement. Switch the sidebar to *As "
            f"reported* to see the figures as filed.")
    else:
        st.warning(
            f"{selected} files in a currency other than the US dollar, and "
            f"the amounts below are **as filed, not in US dollars** — "
            f"{rate_note}. Market capitalisation, enterprise value, the share "
            f"price and every multiple stay in dollars, because a US listing "
            f"is quoted in dollars and there is no filed version of them. "
            f"Switch the sidebar to *US dollars* to put the whole page on one "
            f"currency.")

with st.expander("Line-item notes"):
    st.markdown(
        "Subtotals in **bold** — gross profit, operating income, total current "
        "assets, net cash from operations, EBITDA, enterprise value and the "
        "rest — are fields the vendor publishes, printed as filed. They are "
        "not re-added here, so they agree with the company's own statement "
        "even where a line the vendor does not break out is missing from the "
        "detail above them.\n\n"
        "Lines marked † are the only ones this page computes:\n\n"
        + "\n".join(f"- **{DERIVED_LABELS[key]}** — {note}"
                    for key, note in DERIVED.items())
        + "\n\nNegative amounts print in parentheses. Empty lines are hidden "
        "by default: the vendor fills an inapplicable field with zero rather "
        "than null, so deposits on a software company and inventory on a bank "
        "would otherwise show a row of zeros.\n\n"
        "Two vendor fields are easy to misread and are labelled here for what "
        "they actually contain:\n\n"
        "- **Net income** is already the parent's share. Net income to "
        "common is net income less preferred dividends — checked across the "
        "universe, that holds for 2,836 of 2,847 companies, while subtracting "
        "non-controlling interests as well holds for only 1,902. So NCI sits "
        "in the memo block as disclosure, not as a step in the waterfall.\n"
        "- **Tangible assets** is total assets less intangibles (2,846 of "
        "2,847), not equity less intangibles. `tbvps` is therefore tangible "
        "assets per share, and is labelled that way rather than as tangible "
        "book value — it is routinely larger than book value per share.")

show_blank = st.checkbox("Show lines with no reported values", value=False,
                         key="statements_show_blank")

tabs = st.tabs(["Income statement", "Balance sheet", "Cash flow statement",
                "Valuation and ratios", "Charts"])

for tab, (name, rows, key) in zip(tabs, [
        ("Income statement", INCOME, "income"),
        ("Balance sheet", BALANCE, "balance"),
        ("Cash flow statement", CASHFLOW, "cashflow"),
        ("Valuation and ratios", VALUATION, "valuation")]):
    with tab:
        st.container(key=f"section_statements_{key}")
        st.subheader(name)
        view = statements

        if key == "valuation":
            price_basis = st.radio(
                "Price basis", list(PRICE_BASES), horizontal=True,
                key="statements_price_basis",
                help="Which share price the market capitalisation, enterprise "
                     "value and every multiple below are built on.")
            if price_basis == AT_LATEST:
                scale, close_now, close_date = latest_close_scale(
                    selected, tuple(filed))
                view = reprice_to_latest(statements, pd.Series(scale), close_now)
                st.caption(
                    f"Every column is valued at the **latest close of "
                    f"${close_now:,.2f}** on "
                    f"**{pd.Timestamp(close_date).date()}**, against that "
                    f"period's reported earnings and balance sheet. So the "
                    f"left-hand column is today's multiple, and the ones "
                    f"beside it read as *what today's price would be worth "
                    f"against what the company earned back then* — the shape "
                    f"of the series is the earnings history, not the price "
                    f"history. Net debt is a balance-sheet stock and is left "
                    f"at each filing's value, so enterprise value moves by the "
                    f"same amount as market cap rather than in the same "
                    f"proportion. Switch to *{AT_FILING}* for the "
                    f"point-in-time series, where each multiple pairs a "
                    f"period's earnings with the price actually quoted when "
                    f"they were published.")
            else:
                st.caption(
                    "Market capitalisation, enterprise value, share price and "
                    "every multiple below them are the vendor's point-in-time "
                    "values **as at the filing date of each column**, not "
                    "today's price. That is what makes the series comparable "
                    "across time: each multiple pairs a period's earnings with "
                    "the price that was actually quoted when those earnings "
                    "were published.")
            if money_basis != currency.USD and foreign_filer:
                st.caption(
                    ":grey[The amounts in the earnings block follow the "
                    "sidebar and are shown as filed. Market cap, enterprise "
                    "value, the share price and every multiple are in US "
                    "dollars regardless — a US listing is quoted in dollars, "
                    "so a multiple is a dollar over a dollar in both modes.]")

        elif key == "cashflow":
            st.caption(
                "Sharadar publishes only the three section subtotals plus "
                "capex, D&A and share-based compensation — there are no filed "
                "working-capital lines. The **change in** lines are therefore "
                "differenced off the balance sheet and signed as cash effects, "
                "and each **other … activities** line is the vendor's subtotal "
                "less the lines above it, so every section foots to the filed "
                "number exactly. The operating residual is not small: across a "
                "300-ticker sample its median is 15–23% of the largest section "
                "subtotal, because a balance sheet also moves for reasons that "
                "never touch operating cash flow — working capital arriving "
                "with an acquisition, currency translation, reclassification — "
                "and because deferred taxes, impairments and disposal gains "
                "are not published at all. **Unreconciled** is the vendor's "
                "own gap between its three sections and its reported net "
                "change in cash; it is zero for about nine filings in ten. "
                "Measurements are in `RELIABILITY.md`, regenerated by "
                "`python3 statement_checks.py --write-report`.")

        elif key == "balance":
            st.caption(
                "Balance-sheet lines are stocks measured on the period end, so "
                "they read the same whichever basis is selected — only the "
                "period ends change. The statement closes on total assets "
                "rather than on liabilities plus equity because the vendor's "
                "`equity` is the parent's share alone and it publishes no "
                "separate non-controlling-interest balance: the residual line "
                "carries the NCI. It is exactly zero for a company with no "
                "minority interests, and 1,806 of 2,883 companies show zero.")

        # The rate is printed on the statement it was applied to, rather than
        # only in the banner, so a downloaded CSV carries its own conversion
        # factor and every figure in it can be taken back to what was filed.
        rows_shown = list(rows)
        if key == "valuation" and price_basis == AT_LATEST:
            # The layout labels this row for the point-in-time case, which is
            # no longer what it holds: every column now carries one price.
            rows_shown = [("Share price (latest close)", column, style, unit)
                          if column == "price" else (label, column, style, unit)
                          for label, column, style, unit in rows_shown]
        if foreign_filer:
            rows_shown += [
                (None, None, "spacer", None),
                ("Exchange rate applied — reporting currency units per US$, "
                 "on the report date", currency.RATE_COLUMN, "memo", "rate")]

        st.markdown(
            statement_html(rows_shown, view, period_labels, SCALE,
                           UNIT_LABEL, show_blank),
            unsafe_allow_html=True)

        export = to_export(rows_shown, view, period_labels)
        st.download_button(
            f"Download {name.lower()} (CSV)",
            export.to_csv().encode(),
            file_name=f"{selected}_{dimension}_{key}.csv",
            mime="text/csv", key=f"download_statements_{key}")

with tabs[4]:
    st.container(key="section_statements_charts")
    st.subheader("Line item charts")
    st.caption(
        "Plot any line the statements print, for any set of companies. Each "
        "chart carries its own line item, view, reporting basis, timeframe "
        "and company list, so two charts can compare the same companies on "
        "different measures or the same measure on different bases. Nothing "
        "here follows the company, basis or period count chosen at the top of "
        "the page — only the Units control, which sets the scale absolute "
        "money lines are drawn in. **Add chart** stacks another one below."
    )

    if "fs_chart_ids" not in st.session_state:
        st.session_state.fs_chart_ids = [0]
        st.session_state.fs_chart_next = 1

    chart_options = (analytics_frame.sort_values("marketcap", ascending=False)
                     ["ticker"].tolist())
    chart_labels = dict(zip(analytics_frame["ticker"],
                            analytics_frame["name"].fillna("")))
    default_ticker = selected if selected in chart_options else chart_options[0]

    for chart_id in list(st.session_state.fs_chart_ids):
        render_statement_chart(
            chart_id, chart_options, chart_labels, default_ticker,
            SCALE, UNIT_LABEL, money_basis,
            removable=len(st.session_state.fs_chart_ids) > 1)

    if st.button("Add chart", key="fs_chart_add", type="primary",
                 help="Adds another chart below with its own settings."):
        st.session_state.fs_chart_ids.append(st.session_state.fs_chart_next)
        st.session_state.fs_chart_next += 1
        st.rerun()
