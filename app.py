#!/usr/bin/env python3
"""
Portfolio screener - main page.

Run with:   .venv/bin/streamlit run app.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

import classification_overrides
import currency
import current_multiples as cm
import dcf_forecast as dcf
import ui_data as ui
from analytics import WINDOWS, sector_industry_performance

st.set_page_config(page_title="Portfolio Screener", page_icon="chart_with_upwards_trend",
                   layout="wide", initial_sidebar_state="expanded")

# Muted, colour-blind-safe accents. Green/red only ever encode direction.
UP, DOWN, ACCENT, MUTED = "#1B7F5E", "#B4472F", "#2F5D8A", "#8894A0"
PALETTE = ["#2F5D8A", "#B4472F", "#1B7F5E", "#8A6D2F", "#6B4C8A", "#2F7F8A"]

MULTIPLES = {
    "pe_ttm": "P/E", "ps_ttm": "P/S", "pb": "P/B",
    "ev_ebitda_ttm": "EV/EBITDA", "ev_ebit_ttm": "EV/EBIT",
    "ev_sales_ttm": "EV/Sales", "p_fcf_ttm": "P/FCF",
}

analytics_frame = ui.load_analytics()
universe = analytics_frame[analytics_frame["in_universe"]].copy()
as_of = pd.Timestamp(analytics_frame["as_of"].iloc[0]).date()

st.title("Portfolio Screener")
st.caption(
    f"US equities, NYSE + NASDAQ - prices as of **{as_of}** - "
    f"{len(universe):,} screened tickers - "
    f"{int(universe['implied_share_price_fcff'].notna().sum()):,} with a 15-year DCF"
)

# --------------------------------------------------------------------------
# sidebar
# --------------------------------------------------------------------------
with st.sidebar:
    st.header("Display")
    money_basis = ui.currency_selector()

    st.header("Filters")
    tier = st.selectbox("Minimum market cap", list(ui.MARKETCAP_TIERS), index=2,
                        key="min_marketcap")
    min_marketcap = ui.MARKETCAP_TIERS[tier]
    sectors = ["All"] + sorted(universe["sector"].dropna().unique().tolist())
    sector_choice = st.selectbox("Sector", sectors, key="sector_filter")

    industries = ["All"]
    if sector_choice != "All":
        industries += sorted(universe.loc[universe["sector"] == sector_choice,
                                          "industry"].dropna().unique().tolist())
    else:
        industries += sorted(universe["industry"].dropna().unique().tolist())
    industry_choice = st.selectbox("Industry", industries, key="industry_filter")

    # Ticker options respect the sector and industry chosen above, so the list
    # narrows as you drill down instead of always offering all 2,800 names.
    ticker_pool = universe
    if sector_choice != "All":
        ticker_pool = ticker_pool[ticker_pool["sector"] == sector_choice]
    if industry_choice != "All":
        ticker_pool = ticker_pool[ticker_pool["industry"] == industry_choice]
    focus_tickers = st.multiselect(
        "Tickers", ticker_pool.sort_values("marketcap", ascending=False)["ticker"]
        .tolist(), key="ticker_filter",
        help="Leave empty for the whole universe. Anything selected narrows "
             "every company table on the page and seeds the overlay pickers on "
             "the charts, which you can still add to locally.")

    st.divider()
    with st.expander("Edit sector / industry", expanded=False):
        st.caption(
            "Re-label a company when the vendor's classification is wrong. "
            "Edits are stored outside the data files, so a data refresh or an "
            "app restart leaves them in place, and every table, filter and "
            "peer group on every page reads the corrected label."
        )
        stored_overrides = classification_overrides.load()
        edit_pool = analytics_frame.sort_values("marketcap", ascending=False)
        edit_options = edit_pool["ticker"].tolist()
        preferred = (list(focus_tickers)[0] if focus_tickers
                     else (edit_options[0] if edit_options else None))
        edit_ticker = st.selectbox(
            "Company", edit_options, key="override_ticker",
            index=edit_options.index(preferred) if preferred in edit_options else 0)

        if edit_ticker:
            edit_row = edit_pool[edit_pool["ticker"] == edit_ticker].iloc[0]
            # The vendor columns are what the parquet held before any override,
            # so this line always shows what an edit would be replacing.
            st.caption(
                f"Vendor: **{edit_row.get('vendor_sector') or '-'}** > "
                f"**{edit_row.get('vendor_industry') or '-'}**")
            current = stored_overrides.get(edit_ticker, {})

            CUSTOM = "Type a new one..."

            def _label_picker(field, label, existing_values, suffix=""):
                """Pick an existing label or type a new one, for one field.

                The widget keys carry the ticker (and, for industry, the chosen
                sector) because Streamlit ignores `index` once a key holds
                state - without it, switching company would keep showing the
                previous company's labels.
                """
                current_value = current.get(field) or edit_row.get(field) or ""
                options = sorted({v for v in existing_values if isinstance(v, str)}
                                 | ({current_value} if current_value else set()))
                options = options + [CUSTOM]
                chosen = st.selectbox(
                    label, options, key=f"override_{field}_choice_{suffix}",
                    index=options.index(current_value) if current_value in options else 0)
                if chosen == CUSTOM:
                    chosen = st.text_input(f"New {field}", value="",
                                           key=f"override_{field}_custom_{suffix}").strip()
                return chosen

            new_sector = _label_picker(
                "sector", "Sector", analytics_frame["sector"].dropna().unique(),
                suffix=edit_ticker)
            new_industry = _label_picker(
                "industry", "Industry",
                analytics_frame.loc[analytics_frame["sector"] == new_sector, "industry"]
                .dropna().unique() if new_sector else
                analytics_frame["industry"].dropna().unique(),
                suffix=f"{edit_ticker}_{new_sector}")

            buttons = st.columns(2)
            if buttons[0].button("Save", key="override_save", width="stretch",
                                 type="primary"):
                # Only fields that actually differ from the vendor are stored.
                # A field left on its vendor value keeps following the vendor
                # through future refreshes instead of being frozen by the edit.
                to_store = {
                    field: ("" if value == (edit_row.get(f"vendor_{field}") or "")
                            else value)
                    for field, value in (("sector", new_sector),
                                         ("industry", new_industry))}
                if not any(to_store.values()) and edit_ticker not in stored_overrides:
                    st.info("Both labels already match the vendor - nothing to store.")
                else:
                    classification_overrides.set_override(edit_ticker, **to_store)
                    st.success(f"{edit_ticker} saved.")
                    st.rerun()
            if buttons[1].button("Reset to vendor", key="override_reset",
                                 width="stretch",
                                 disabled=edit_ticker not in stored_overrides):
                classification_overrides.clear_override(edit_ticker)
                st.rerun()

        if stored_overrides:
            st.caption(f"**{len(stored_overrides)} manual override(s)**")
            st.dataframe(
                pd.DataFrame([
                    {"Ticker": ticker,
                     "Sector": entry.get("sector", "-"),
                     "Industry": entry.get("industry", "-")}
                    for ticker, entry in sorted(stored_overrides.items())]),
                hide_index=True, width="stretch", key="table_overrides")
            if st.button("Clear all overrides", key="override_clear_all",
                         width="stretch"):
                classification_overrides.clear_all()
                st.rerun()

    st.divider()
    st.caption(
        "DCF output is a model, not advice. The equity risk premium (4.5%) and "
        "the 21% tax rate are assumptions. Sector margin anchors use profitable "
        "firms only. Banks, insurers and REITs are flagged - an FCFF DCF is "
        "structurally the wrong model for them."
    )

# The currency choice is applied once, here, so every section below reads one
# frame. It touches only the statement-derived money columns - revenue, net
# income, EBITDA, free cash flow and the balance-sheet stocks. Market cap,
# enterprise value, every price and every multiple stay in dollars in both
# modes: a US listing is quoted in dollars, so a multiple is a dollar over a
# dollar whichever way the toggle is set, and there is no filed version of a
# market value to switch to.
analytics_frame = ui.analytics_in_basis(analytics_frame, money_basis)
universe = ui.analytics_in_basis(universe, money_basis)

MONEY_SIGN = ui.money_symbol(money_basis)

if money_basis != currency.USD:
    foreign_count = int((universe["fx_to_usd"].fillna(1.0) != 1.0).sum())
    st.warning(
        f"**As reported.** Revenue, net income, EBITDA and free cash flow are "
        f"printed in each company's own filing currency, which for "
        f"{foreign_count} of the {len(universe):,} screened names is not the "
        f"US dollar — so those columns are not comparable across rows and "
        f"carry no currency sign. Market cap, prices and every multiple stay "
        f"in dollars. Switch the sidebar to *{currency.USD}* to put the whole "
        f"page on one currency."
    )

st.caption(
    ":grey[" + cm.describe(universe) + " Earnings and cash flow are the last "
    "trailing twelve months filed; only the price they are divided by is "
    "current. The filing-date multiples are kept alongside and are what the "
    "history charts plot.]"
)

# `scoped` carries the sidebar's sector / industry / ticker choice and nothing
# else, so group-level sections can use it without inheriting a market-cap
# floor meant for company tables. `filtered` adds that floor. Every section
# below builds on one of these two, which is what makes the sidebar global.
scoped = universe.copy()
if sector_choice != "All":
    scoped = scoped[scoped["sector"] == sector_choice]
if industry_choice != "All":
    scoped = scoped[scoped["industry"] == industry_choice]
if focus_tickers:
    scoped = scoped[scoped["ticker"].isin(focus_tickers)]

filtered = scoped[scoped["marketcap"] >= min_marketcap]

if filtered.empty:
    st.warning("No tickers match the current filters.")
    st.stop()

if focus_tickers:
    st.info(f"Filtered to {len(focus_tickers)} selected ticker(s): "
            f"**{', '.join(focus_tickers)}**. Every company table below is "
            f"limited to these, and they are added to every comparison chart "
            f"on top of whatever each chart's own picker holds. Clear the "
            f"Tickers box in the sidebar to see the whole universe again.")


def format_table(frame, money_cols=(), percent_cols=(), number_cols=(),
                 price_cols=()):
    """Pre-format numbers as English strings.

    Streamlit's `format="compact"` renders through the browser locale, which
    produced Russian abbreviations. Formatting here keeps every table in
    English and stops long numbers overflowing their cells.
    """
    out = frame.copy()
    for column in money_cols:
        if column in out:
            out[column] = out[column].map(ui.money)
    for column in percent_cols:
        if column in out:
            out[column] = out[column].map(lambda v: ui.percent(v, 1))
    for column in number_cols:
        if column in out:
            out[column] = out[column].map(lambda v: ui.number(v, 2))
    for column in price_cols:
        if column in out:
            out[column] = out[column].map(
                lambda v: f"${v:,.2f}" if pd.notna(v) and np.isfinite(v) else "-")
    return out


RETURN_COLUMNS = ["return_1d", "return_1w", "return_1m", "return_3m", "return_1y"]
RETURN_HEADERS = {"1d": "1D", "1w": "1W", "1m": "1M", "3m": "3M",
                  "6m": "6M", "1y": "1Y"}


# --------------------------------------------------------------------------
# multiple history lines, carried to today's price
# --------------------------------------------------------------------------
# Everything stored in metrics.parquet is point-in-time: each value is the
# multiple at the price quoted on that filing date. That is the right thing to
# store and the wrong thing to stop at - the last filing can be months old, so
# a line that ends there says nothing about where the stock trades now, and it
# disagrees with every table on the page, which is priced at the last close.
#
# So each line is carried one segment further: same reported earnings, market
# cap moved to the last trading date in the database. Only the numerator
# changes - the price leg of P/x, and the equity leg of EV/x, with net debt
# left at its reported value because a balance-sheet stock has not moved.

# The last session in the database, not the wall clock. On a weekend or before
# the vendor publishes a session, `today` is a date no price exists for, and
# the current point would float off the end of the axis away from the bar it
# was actually computed from.
LAST_SESSION = pd.Timestamp(as_of)


def _signed_log(values):
    """Log scale that survives negatives, which multiples routinely are."""
    values = pd.to_numeric(pd.Series(values), errors="coerce")
    return (np.log(np.abs(values)) * np.sign(values)).to_numpy()


def current_multiple(frame: pd.DataFrame, ticker: str, multiple: str) -> float:
    """One company's multiple at the latest close."""
    row = frame.loc[frame["ticker"] == ticker, multiple]
    return float(row.iloc[0]) if len(row) and pd.notna(row.iloc[0]) else np.nan


def current_group_median(frame: pd.DataFrame, group_field: str, group: str,
                         multiple: str) -> float:
    """A group's median multiple at the latest close.

    Positives only and a median rather than a mean, matching
    group_multiple_history exactly - otherwise the point carrying the line to
    today would be computed on a different population from the line itself,
    and the join between them would be a step that means nothing.
    """
    members = frame[(frame[group_field] == group) & (frame[multiple] > 0)]
    return float(members[multiple].median()) if not members.empty else np.nan


def add_multiple_series(figure, dates, values, current, name, colour,
                        width: float = 2.0, dash: str | None = None,
                        log_scale: bool = False) -> None:
    """One multiple line, with a final segment to the current price.

    The diamond marks where the filed history stops and the price adjustment
    begins, so the current point can never be mistaken for a filing. Where
    there is no current value - a company with no recent filing, or a negative
    multiple the group median excludes - the line simply ends at the last
    filing rather than being dropped.
    """
    dates = list(pd.to_datetime(pd.Series(list(dates))))
    values = list(_signed_log(values) if log_scale else
                  pd.to_numeric(pd.Series(list(values)), errors="coerce"))

    has_current = current is not None and np.isfinite(current)
    if has_current:
        plotted = float(_signed_log([current])[0] if log_scale else current)
        dates.append(LAST_SESSION)
        values.append(plotted)

    figure.add_trace(go.Scatter(
        x=dates, y=values, name=name, mode="lines",
        line=dict(color=colour, width=width, dash=dash),
        hovertemplate="%{x|%b %Y}: %{y:,.2f}<extra>" + name + "</extra>"))

    # if has_current:
    #     figure.add_trace(go.Scatter(
    #         x=[LAST_SESSION], y=[values[-1]], mode="markers", showlegend=False,
    #         marker=dict(color=colour, size=9, symbol="diamond",
    #                     line=dict(width=1, color="white")),
    #         hovertemplate="Now: %{y:,.2f}<extra>" + name + "</extra>"
    #         ))


def profile_card(row, source: pd.DataFrame, key: str,
                 show_peers: bool = True) -> None:
    """The company card: valuation, size, profitability, risk, DCF, peers.

    Every multiple here is against the latest close, not the price on the
    filing date - that is the whole point of a card you look at to decide
    something. Where the price has moved since the filing, the filing-date
    value is printed beside it in grey so the move is visible rather than
    silently folded in.
    """
    with st.container(border=True):
        st.subheader(f"{row['ticker']} — {row['name']}")
        st.caption(f"{row['sector']} > **{row['industry']}** — "
                   f"{row.get('exchange', '')} — market cap "
                   f"{ui.money(row['marketcap'])} — last close "
                   f"${row['last_close']:,.2f}")

        blocks = st.columns(5)

        blocks[0].markdown("**Valuation**")
        for label, value in [("P/E", row.pe_ttm), ("P/B", row.pb),
                             ("P/S", row.ps_ttm), ("P/FCF", row.p_fcf_ttm),
                             ("EV/EBITDA", row.ev_ebitda_ttm),
                             ("EV/EBIT", row.ev_ebit_ttm),
                             ("EV/Sales", row.ev_sales_ttm)]:
            blocks[0].text(f"{label:<11}{ui.number(value)}")

        blocks[1].markdown("**Size and yield**")
        blocks[1].text(f"{'Mkt cap':<11}{ui.money(row['marketcap'])}")
        blocks[1].text(f"{'EV':<11}{ui.money(row.get('ev'))}")
        for label, field in [("Revenue", "revenue_ttm"),
                             ("EBITDA", "ebitda_ttm"),
                             ("Net inc", "netinc_ttm"),
                             ("FCF", "fcf_ttm")]:
            blocks[1].text(
                f"{label:<11}{ui.money(row.get(field), symbol=MONEY_SIGN)}")
        blocks[1].text(f"{'FCF yld':<11}{ui.percent(row.fcf_yield)}")

        blocks[2].markdown("**Profitability**")
        for label, value in [("Gross", row.gross_margin_ttm),
                             ("EBITDA", row.ebitda_margin_ttm),
                             ("EBIT", row.ebit_margin_ttm),
                             ("Net", row.net_margin_ttm),
                             ("ROE", row.roe_ttm), ("ROIC", row.roic_ttm)]:
            blocks[2].text(f"{label:<9}{ui.percent(value)}")

        blocks[3].markdown("**Risk**")
        blocks[3].text(f"{'Beta':<12}{ui.number(row.beta)}")
        blocks[3].text(f"{'Vol 1Y':<12}{ui.percent(row.volatility_1y)}")
        blocks[3].text(f"{'Max DD':<12}{ui.percent(row.max_drawdown_1y)}")
        blocks[3].text(f"{'D/E':<12}{ui.number(row.debt_to_equity)}")
        blocks[3].text(f"{'ND/EBITDA':<12}{ui.number(row.net_debt_to_ebitda)}")
        blocks[3].text(f"{'WACC':<12}{ui.percent(row.wacc, 2)}")

        blocks[4].markdown("**Valuation models**")
        blocks[4].text(f"{'Price':<9}${row.last_close:,.2f}")
        for label, value in [("Fair val", row.get("fair_value")),
                             ("Multiples", row.get("fair_value_multiples")),
                             ("FCFF", row.implied_share_price_fcff),
                             ("FCFE", row.implied_share_price_fcfe),
                             ("EPS", row.implied_share_price_eps)]:
            blocks[4].text(f"{label:<9}"
                           + (f"${value:,.2f}" if pd.notna(value) else "—"))
        blocks[4].text(f"{'Upside':<9}{ui.percent(row.get('fair_value_upside'))}")

        scale = row.get("price_scale")
        if pd.notna(scale) and not np.isclose(float(scale), 1.0, atol=1e-4):
            filing = pd.Timestamp(row["last_filing"]).date() \
                if pd.notna(row.get("last_filing")) else "the last filing"
            st.caption(
                f":grey[Multiples are at today's close. The price is "
                f"{float(scale) - 1:+.1%} since {filing}, so on the filing-date "
                f"price these were P/E {ui.number(row.get('pe_ttm_at_filing'))}, "
                f"P/B {ui.number(row.get('pb_at_filing'))}, "
                f"P/S {ui.number(row.get('ps_ttm_at_filing'))}, "
                f"EV/EBITDA {ui.number(row.get('ev_ebitda_ttm_at_filing'))}.]")

        if row.get("rationale"):
            st.caption(f"**Valuation engine:** {row['rationale']}")

        if not show_peers:
            return
        peers = ui.competitors(source, row["ticker"], count=6)
        if peers.empty:
            return
        st.caption("**Nearest competitors** — same sector and industry, "
                   "closest market cap")
        peer_columns = ["ticker", "name", "peer_match", "marketcap", "pe_ttm",
                        "pb", "ev_ebitda_ttm", "ps_ttm", "ebit_margin_ttm",
                        "roic_ttm", "return_1y", "fair_value",
                        "fair_value_upside"]
        st.dataframe(
            format_table(peers[peer_columns], money_cols=["marketcap"],
                         percent_cols=["ebit_margin_ttm", "roic_ttm",
                                       "return_1y", "fair_value_upside"],
                         number_cols=["pe_ttm", "pb", "ev_ebitda_ttm", "ps_ttm"],
                         price_cols=["fair_value"]),
            hide_index=True, width="stretch",
            column_config={
                "ticker": st.column_config.TextColumn("Ticker", width=70),
                "name": st.column_config.TextColumn("Company", width=200),
                "peer_match": st.column_config.TextColumn("Match", width=135),
                "marketcap": st.column_config.TextColumn("Market cap", width=105),
                "pe_ttm": st.column_config.TextColumn("P/E", width=75),
                "pb": st.column_config.TextColumn("P/B", width=75),
                "ev_ebitda_ttm": st.column_config.TextColumn("EV/EBITDA", width=95),
                "ps_ttm": st.column_config.TextColumn("P/S", width=75),
                "ebit_margin_ttm": st.column_config.TextColumn("EBIT %", width=85),
                "roic_ttm": st.column_config.TextColumn("ROIC", width=80),
                "return_1y": st.column_config.TextColumn("1Y", width=80),
                "fair_value": st.column_config.TextColumn("Fair value", width=100),
                "fair_value_upside": st.column_config.TextColumn("FV upside",
                                                                 width=92),
            }, key=f"table_peers_{key}")

# ==========================================================================
# 1. largest companies
# ==========================================================================
st.container(key="section_largest_companies")
st.header("Largest companies")
st.caption(
    "Ranked by market cap. **DCF FCFF / FCFE / EPS** are the three implied "
    "share prices; **p15-p85** is the interval from simulating each firm's own "
    "historical cash-flow volatility; **P(above)** is the share of simulated "
    "paths landing above today's price."
)

count = st.slider("Rows", 10, 200, 30, step=10, key="largest_rows")
columns = (["ticker", "name", "sector", "industry", "marketcap", "last_close"]
           + RETURN_COLUMNS
           + ["volatility_1y", "max_drawdown_1y",
              "fair_value", "fair_value_upside", "confidence",
              "fair_value_multiples", "implied_share_price_fcff",
              "implied_share_price_fcfe", "implied_share_price_eps",
              "price_fcff_p15", "price_fcff_p85", "prob_above_market_fcff"])
largest = universe.nlargest(count, "marketcap")[columns]

st.dataframe(
    format_table(largest, money_cols=["marketcap"],
                 percent_cols=RETURN_COLUMNS + ["volatility_1y", "max_drawdown_1y",
                                                "prob_above_market_fcff",
                                                "fair_value_upside", "confidence"],
                 price_cols=["last_close", "fair_value", "fair_value_multiples",
                             "implied_share_price_fcff",
                             "implied_share_price_fcfe", "implied_share_price_eps",
                             "price_fcff_p15", "price_fcff_p85"]),
    hide_index=True, width="stretch", height=520,
    column_config={
        "ticker": st.column_config.TextColumn("Ticker", width=70),
        "name": st.column_config.TextColumn("Company", width=180),
        "sector": st.column_config.TextColumn("Sector", width=140),
        "industry": st.column_config.TextColumn("Industry", width=160),
        "marketcap": st.column_config.TextColumn("Market cap", width=105),
        "last_close": st.column_config.TextColumn("Price", width=90),
        **{f"return_{k}": st.column_config.TextColumn(v, width=78)
           for k, v in RETURN_HEADERS.items()},
        "volatility_1y": st.column_config.TextColumn("Vol 1Y", width=82),
        "max_drawdown_1y": st.column_config.TextColumn("Max DD", width=88),
        "fair_value": st.column_config.TextColumn(
            "Fair value", width=100,
            help="Weighted blend of up to five valuation models, chosen and "
                 "weighted per company. See VALUATION_ENGINE.md."),
        "fair_value_upside": st.column_config.TextColumn("FV upside", width=92),
        "confidence": st.column_config.TextColumn(
            "Confidence", width=95,
            help="Model reliability, agreement between models, and how many "
                 "survived."),
        "fair_value_multiples": st.column_config.TextColumn(
            "FV multiples", width=105,
            help="Peer-median P/E, EV/EBITDA, P/S and P/B blended by the same "
                 "reliability weighting - no cash-flow forecast involved."),
        "implied_share_price_fcff": st.column_config.TextColumn("DCF FCFF", width=98),
        "implied_share_price_fcfe": st.column_config.TextColumn("DCF FCFE", width=98),
        "implied_share_price_eps": st.column_config.TextColumn("DCF EPS", width=95),
        "price_fcff_p15": st.column_config.TextColumn("p15", width=88),
        "price_fcff_p85": st.column_config.TextColumn("p85", width=88),
        "prob_above_market_fcff": st.column_config.TextColumn("P(above)", width=92),
    }, key="table_largest_companies")

# ==========================================================================
# 2. price chart
# ==========================================================================
st.container(key="section_price_history")
st.header("Price history")
chart_left, chart_right = st.columns([3, 1])
ticker_options = universe.sort_values("marketcap", ascending=False)["ticker"].tolist()

with chart_right:
    chart_ticker = st.selectbox("Ticker", ticker_options, key="chart_ticker")
    display_mode = st.radio("Show as", ["Price ($)", "Cumulative % change"],
                            key="chart_mode")
    timeframe = st.radio("Timeframe",
                         ["1M", "3M", "6M", "YTD", "1Y", "3Y", "5Y", "Max", "Custom"],
                         index=4, horizontal=True, key="chart_tf")
    overlay = st.multiselect("Compare",
                             ["S&P 500", "Nasdaq 100", "Gold", "Brent", "VIX"],
                             default=[], key="chart_overlay")
    log_scale = st.checkbox("Log scale", value=False, key="chart_log")

prices = ui.load_prices(chart_ticker)
if prices.empty:
    with chart_left:
        st.warning(f"No price history for {chart_ticker}.")
else:
    prices["date"] = pd.to_datetime(prices["date"])
    first_date, last_date = prices["date"].min(), prices["date"].max()

    if timeframe == "Custom":
        with chart_right:
            custom = st.date_input(
                "Custom range",
                value=(max(first_date, last_date - pd.DateOffset(years=1)), last_date),
                min_value=first_date, max_value=last_date, key="chart_custom_range")
        if isinstance(custom, (tuple, list)) and len(custom) == 2:
            start, end = pd.Timestamp(custom[0]), pd.Timestamp(custom[1])
        else:  # mid-selection the widget returns a single date
            start, end = pd.Timestamp(custom if not isinstance(custom, (tuple, list))
                                      else custom[0]), last_date
    else:
        start, end = ui.timeframe_start(timeframe, last_date), last_date

    window = prices[(prices["date"] >= start) & (prices["date"] <= end)]
    if window.empty:
        with chart_left:
            st.warning("No sessions in the selected range.")
    else:
        # Comparing against an index only means anything on a common scale, so
        # overlays force cumulative change regardless of the display toggle.
        cumulative = display_mode == "Cumulative % change" or bool(overlay)
        primary = (window["closeadj"] / window["closeadj"].iloc[0] - 1.0
                   if cumulative else window["closeadj"])

        figure = make_subplots(specs=[[{"secondary_y": True}]])
        figure.add_trace(go.Scatter(
            x=window["date"], y=primary, name=chart_ticker,
            line=dict(color=ACCENT, width=2),
            hovertemplate="%{y:.2%}<extra></extra>" if cumulative
            else "$%{y:,.2f}<extra></extra>"), secondary_y=False)

        if overlay:
            market = ui.load_market_data()
            market["date"] = pd.to_datetime(market["date"])
            market = market[(market["date"] >= start) & (market["date"] <= end)]
            series_map = {"S&P 500": "sp500", "Nasdaq 100": "nasdaq100",
                          "Gold": "gold", "Brent": "brent", "VIX": "vix"}
            for index, label in enumerate(overlay):
                series = market[["date", series_map[label]]].dropna()
                if series.empty:
                    continue
                values = series[series_map[label]]
                figure.add_trace(go.Scatter(
                    x=series["date"], y=values / values.iloc[0] - 1.0, name=label,
                    line=dict(width=1.4, dash="dot",
                              color=PALETTE[(index + 1) % len(PALETTE)]),
                    hovertemplate="%{y:.2%}<extra></extra>"), secondary_y=False)

        figure.update_yaxes(
            title_text="Cumulative change" if cumulative else "Adjusted close",
            tickformat=".0%" if cumulative else "$,.0f", secondary_y=False)
        if cumulative:
            figure.add_hline(y=0, line_width=1, line_color=MUTED, opacity=0.5)

        figure.add_trace(go.Bar(x=window["date"], y=window["volume"], name="Volume",
                                marker_color=MUTED, opacity=0.22,
                                hovertemplate="%{y:,.0f}<extra></extra>"),
                         secondary_y=True)
        figure.update_yaxes(title_text="Volume", secondary_y=True, showgrid=False)
        if log_scale and not cumulative:
            figure.update_yaxes(type="log", secondary_y=False)
        figure.update_layout(height=460, hovermode="x unified", margin=dict(t=30, b=0),
                             legend=dict(orientation="h", y=1.08))

        row = universe[universe["ticker"] == chart_ticker].iloc[0]
        with chart_left:
            st.plotly_chart(figure, width="stretch", key="chart_price_history")
            change = window["closeadj"].iloc[-1] / window["closeadj"].iloc[0] - 1.0
            stats = st.columns(6)
            stats[0].metric("Price", f"${row.last_close:,.2f}",
                            ui.percent(row.return_1d, 2)
                            if pd.notna(row.return_1d) else None)
            stats[1].metric("Selected range", ui.percent(change, 1))
            stats[2].metric("Volatility 1Y", ui.percent(row.volatility_1y, 1))
            stats[3].metric("Beta", ui.number(row.beta, 2))
            stats[4].metric("Market cap", ui.money(row.marketcap))
            stats[5].metric("DCF (FCFF)",
                            f"${row.implied_share_price_fcff:,.2f}"
                            if pd.notna(row.implied_share_price_fcff) else "-",
                            ui.percent(row.upside_fcff, 1)
                            if pd.notna(row.upside_fcff) else None)
            st.caption(f"**{row['name']}** - {row['sector']} > **{row['industry']}**")

# ==========================================================================
# 2b. company profile
# ==========================================================================
st.container(key="section_company_profile")
st.header("Company profile")
st.caption(
    "Any company in the universe, not only the ones a table above happens to "
    "be showing — the picker ignores the sidebar filters on purpose, so a name "
    "can be looked up without clearing a screen first."
)

profile_options = (analytics_frame[analytics_frame["in_universe"]]
                   .sort_values("marketcap", ascending=False)["ticker"].tolist())
profile_labels = dict(zip(analytics_frame["ticker"],
                          analytics_frame["name"].fillna("")))
# Seeded from the sidebar and from the chart above it, so the page tends to
# stay on one company as you move down it, while the picker still overrides.
profile_default = next((t for t in (list(focus_tickers) + [chart_ticker])
                        if t in profile_options), profile_options[0])
profile_ticker = st.selectbox(
    "Company", profile_options, key="profile_ticker",
    index=profile_options.index(profile_default),
    format_func=lambda t: f"{t} — {profile_labels.get(t, '')}"[:60])

profile_card(universe[universe["ticker"] == profile_ticker].iloc[0],
             universe, key="profile")

# ==========================================================================
# 3. sector performance
# ==========================================================================
st.container(key="section_sector_industry_performance")
st.header("Sector performance")
st.caption(
    ":grey[**This section ignores the sidebar filters on purpose.**] Comparing "
    "sectors only means anything against the whole market - filtering to one "
    "sector would leave a single row and nothing to compare it with. Use the "
    "controls inside the section instead."
)
controls = st.columns([1, 1, 1])
weighting_label = controls[0].radio(
    "Weighting", ["Market-cap weighted", "Median", "Average"], key="sector_weight")
weighting = {"Market-cap weighted": "marketcap", "Median": "median",
             "Average": "mean"}[weighting_label]
period = controls[1].selectbox("Period for movers", list(WINDOWS), index=3,
                               format_func=lambda p: ui.PERIOD_LABELS[p],
                               key="mover_period")
controls[2].caption(
    "Market-cap weighting answers *what the sector did*; the median answers "
    "*what a typical company in it did*. They diverge most where a handful of "
    "names dominate - which is exactly when the difference matters."
)

# `universe`, not `scoped`: this section is deliberately market-wide.
sector_pool = universe.copy()
sector_industry = "sector"
if sector_choice != "All":
    sector_industry = "industry"
    sector_pool = sector_pool[sector_pool["sector"] == sector_choice]
filtered_pool = sector_pool[sector_pool["marketcap"] >= min_marketcap]
sectors_frame = sector_industry_performance(filtered_pool, weighting,
                                           min_marketcap, sector_industry)
all_return_columns = [f"return_{label}" for label in WINDOWS]
display = sectors_frame.sort_values(f"return_{period}", ascending=False)

st.dataframe(
    format_table(display[[sector_industry, "companies", "total_marketcap"]
                         + all_return_columns],
                 money_cols=["total_marketcap"], percent_cols=all_return_columns),
    hide_index=True, width="stretch",
    column_config={
        sector_industry: st.column_config.TextColumn("Sector" if sector_choice == "All" else "Industry", width=175),
        "companies": st.column_config.NumberColumn("Firms", format="%d", width=70),
        "total_marketcap": st.column_config.TextColumn("Total cap", width=110),
        **{f"return_{k}": st.column_config.TextColumn(v, width=80)
           for k, v in RETURN_HEADERS.items()},
    }, key="table_sector_industry_performance")

figure = go.Figure(go.Bar(
    x=display[sector_industry], y=display[f"return_{period}"],
    marker_color=[UP if v >= 0 else DOWN for v in display[f"return_{period}"]],
    hovertemplate="%{x}<br>%{y:.2%}<extra></extra>"))
figure.update_layout(height=320, yaxis_title=f"{ui.PERIOD_LABELS[period]} return",
                     yaxis_tickformat=".1%", margin=dict(t=20, b=0))
st.plotly_chart(figure, width="stretch", key="chart_sector_returns_bar")

st.container(key="section_movers")
st.subheader(f"Biggest movers - {ui.PERIOD_LABELS[period]}")

mover_pool = filtered.dropna(subset=[f"return_{period}"])
column = f"return_{period}"
mover_columns = ["ticker", "name", "sector", "industry", "marketcap",
                 column, "last_close"]
mover_config = {
    "ticker": st.column_config.TextColumn("Ticker", width=70),
    "name": st.column_config.TextColumn("Company", width=180),
    "sector": st.column_config.TextColumn("Sector", width=140),
    "industry": st.column_config.TextColumn("Industry", width=160),
    "marketcap": st.column_config.TextColumn("Market cap", width=105),
    column: st.column_config.TextColumn("Return", width=90),
    "last_close": st.column_config.TextColumn("Price", width=90),
}
left, right = st.columns(2)
with left:
    st.caption("**Top gainers**")
    st.dataframe(format_table(mover_pool.nlargest(12, column)[mover_columns],
                              money_cols=["marketcap"], percent_cols=[column],
                              price_cols=["last_close"]),
                 hide_index=True, width="stretch", column_config=mover_config, key="table_movers_gainers")
with right:
    st.caption("**Biggest fallers**")
    st.dataframe(format_table(mover_pool.nsmallest(12, column)[mover_columns],
                              money_cols=["marketcap"], percent_cols=[column],
                              price_cols=["last_close"]),
                 hide_index=True, width="stretch", column_config=mover_config, key="table_movers_losers")

# ==========================================================================
# 4. recent reporters
# ==========================================================================
st.container(key="section_recent_reporters")
st.header("Recently reported")
st.caption(
    "Companies whose latest filing landed most recently, and how the stock "
    "moved afterwards. The filing date is the closest thing this dataset has "
    "to an earnings date, so the reaction is measured from the first session "
    "on or after it - in trading sessions, not calendar days."
)

report_controls = st.columns([1])
report_window = report_controls[0].slider("Reported within (days)", 5, 120, 45,
                                          step=5, key="report_window")

reporters = scoped[(scoped["marketcap"] >= min_marketcap)
                   & (scoped["days_since_report"] <= report_window)]
if sector_choice != "All":
    reporters = reporters[reporters["sector"] == sector_choice]
if industry_choice != "All":
    reporters = reporters[reporters["industry"] == industry_choice]
reporters = reporters.sort_values("days_since_report")

if reporters.empty:
    st.info("No companies matching those filters reported in that window.")
else:
    report_columns = ["ticker", "name", "sector", "industry", "marketcap",
                      "last_filing", "days_since_report", "return_1d_after_report",
                      "return_5d_after_report", "return_20d_after_report",
                      "return_since_report", "last_close"]
    st.dataframe(
        format_table(reporters[report_columns].head(40), money_cols=["marketcap"],
                     percent_cols=["return_1d_after_report", "return_5d_after_report",
                                   "return_20d_after_report", "return_since_report"],
                     price_cols=["last_close"]),
        hide_index=True, width="stretch",
        column_config={
            "ticker": st.column_config.TextColumn("Ticker", width=70),
            "name": st.column_config.TextColumn("Company", width=180),
            "sector": st.column_config.TextColumn("Sector", width=140),
            "industry": st.column_config.TextColumn("Industry", width=155),
            "marketcap": st.column_config.TextColumn("Market cap", width=105),
            "last_filing": st.column_config.DateColumn("Reported", width=105),
            "days_since_report": st.column_config.NumberColumn("Days ago",
                                                               format="%d", width=85),
            "return_since_report": st.column_config.TextColumn("Since", width=85),
            "return_1d_after_report": st.column_config.TextColumn("+1D", width=78),
            "return_5d_after_report": st.column_config.TextColumn("+5D", width=78),
            "return_20d_after_report": st.column_config.TextColumn("+20D", width=82),
            "last_close": st.column_config.TextColumn("Price", width=90),
        }, key="table_recent_reporters")

# ==========================================================================
# 5. sector and industry multiples
# ==========================================================================
st.container(key="section_multiples")
st.header("Sector and industry multiples")

multiple_controls = st.columns([1, 1, 2])
chosen_multiple = multiple_controls[0].selectbox(
    "Multiple", list(MULTIPLES), format_func=MULTIPLES.get, key="cheap_multiple")
# group_by = multiple_controls[1].radio("Group by", ["Sector", "Industry"],
#                                       key="cheap_groupby")
multiple_controls[1].caption(
    "Every multiple here is at the **latest close** — the reported twelve "
    "months divided by today's market cap, not the one on the filing date. "
    "Only positive multiples are ranked: a negative P/E means the company "
    "lost money, which makes it the most expensive, not the cheapest."
)

group_column = "sector" if sector_industry == "sector" else "industry"
pool = filtered[filtered[chosen_multiple] > 0].dropna(
    subset=[chosen_multiple, group_column]).copy()

trim_controls = st.columns([1, 3])
trim = trim_controls[0].slider(
    "Trim extreme values (percentile)", 0, 20, 2, step=1, key="cheap_trim",
    help="Drops the top and bottom N% of the multiple before ranking. A "
         "handful of companies trading at 4,000x earnings otherwise occupy "
         "the whole 'most expensive' list and tell you nothing.")
if trim > 0 and not pool.empty:
    low = pool[chosen_multiple].quantile(trim / 100.0)
    high = pool[chosen_multiple].quantile(1.0 - trim / 100.0)
    before = len(pool)
    pool = pool[pool[chosen_multiple].between(low, high)]
    trim_controls[1].caption(
        f"Keeping {MULTIPLES[chosen_multiple]} between **{low:,.2f}** and "
        f"**{high:,.2f}** - {before - len(pool)} of {before} companies trimmed.")

# The group median is the reference that makes a single multiple mean
# something: 12x is cheap in software and expensive in a utility.
if not pool.empty:
    pool["group_median"] = pool.groupby(group_column)[chosen_multiple].transform("median")
    pool["vs_group_median"] = pool[chosen_multiple] / pool["group_median"] - 1.0

if pool.empty:
    st.info("No companies with a positive value for that multiple.")
else:
    rank_columns = ["ticker", "name", group_column, "marketcap", chosen_multiple,
                    "group_median", "vs_group_median", "last_close",
                    "fair_value_multiples", "fair_value", "fair_value_upside",
                    "confidence"]
    rank_config = {
        "ticker": st.column_config.TextColumn("Ticker", width=70),
        "name": st.column_config.TextColumn("Company", width=190),
        group_column: st.column_config.TextColumn("Sector" if sector_industry == "sector" else "Industry", width=170),
        "marketcap": st.column_config.TextColumn("Market cap", width=105),
        chosen_multiple: st.column_config.TextColumn(MULTIPLES[chosen_multiple],
                                                     width=95),
        "group_median": st.column_config.TextColumn(f"{"Sector" if sector_industry == "sector" else "Industry"} median", width=115),
        "vs_group_median": st.column_config.TextColumn("vs median", width=95),
        "last_close": st.column_config.TextColumn("Price", width=90),
        "fair_value_multiples": st.column_config.TextColumn(
            "FV multiples", width=105,
            help="What peers would pay, on the same reliability weighting."),
        "fair_value": st.column_config.TextColumn("Fair value", width=100),
        "fair_value_upside": st.column_config.TextColumn("FV upside", width=92),
        "confidence": st.column_config.TextColumn("Confidence", width=95),
    }
    cheap_left, cheap_right = st.columns(2)
    with cheap_left:
        st.caption(f"**Cheapest by {MULTIPLES[chosen_multiple]}**")
        st.dataframe(
            format_table(pool.nsmallest(15, chosen_multiple)[rank_columns],
                         money_cols=["marketcap"],
                         percent_cols=["vs_group_median", "fair_value_upside",
                                       "confidence"],
                         number_cols=[chosen_multiple, "group_median"],
                         price_cols=["last_close", "fair_value_multiples",
                                     "fair_value"]),
            hide_index=True, width="stretch", column_config=rank_config, key="table_multiples_cheapest")
    with cheap_right:
        st.caption(f"**Most expensive by {MULTIPLES[chosen_multiple]}**")
        st.dataframe(
            format_table(pool.nlargest(15, chosen_multiple)[rank_columns],
                         money_cols=["marketcap"],
                         percent_cols=["vs_group_median", "fair_value_upside",
                                       "confidence"],
                         number_cols=[chosen_multiple, "group_median"],
                         price_cols=["last_close", "fair_value_multiples",
                                     "fair_value"]),
            hide_index=True, width="stretch", column_config=rank_config, key="table_multiples_expensive")

st.container(key="section_multiples_history")
st.subheader("Historical multiples by group, with companies to compare")
st.caption(
    ":grey[Each line is the filed history — the multiple as it could have been "
    "computed on that filing date, at the price quoted then — carried one "
    "segment further to **today's price**, marked with a ◆. The last segment "
    "holds the reported earnings fixed and moves only the market cap, to the "
    "last close in the database, so where the diamond sits is exactly the "
    "number in the tables above. Read the line as the history and the diamond "
    "as where it trades now.]"
)
history_controls = st.columns([1, 2, 2, 1])
chart_multiple = history_controls[0].selectbox(
    "Multiple to chart", list(MULTIPLES), format_func=MULTIPLES.get,
    key="hist_multiple")
group_options = sorted(scoped[group_column].dropna().unique().tolist())
default_groups = ([sector_choice] if sector_industry == "sector" and sector_choice != "All"
                  else group_options[:2])
chosen_groups = history_controls[1].multiselect(
    f"{"Sector" if sector_industry == "sector" else "Industry"} medians to plot", group_options,
    default=[g for g in default_groups if g in group_options], key="hist_groups")
ticker_universe = universe.sort_values("marketcap", ascending=False)["ticker"].tolist()
compare_tickers = history_controls[2].multiselect(
    "Companies to overlay", ticker_universe,
    default=[chart_ticker] if chart_ticker in set(universe["ticker"]) else [],
    key="hist_tickers")
# Sidebar tickers are unioned in rather than used as the widget default:
# Streamlit only honours `default` on a widget's first render, so after any
# interaction a changed sidebar selection would silently stop appearing. The
# union always holds, and the local picker still adds on top of it.
compare_tickers = list(dict.fromkeys(list(focus_tickers) + list(compare_tickers)))
history_months = history_controls[3].slider(
    "Months", 3, 80, 80, key="hist_months",
    help="Length of the window both charts below cover, counted back from today.")
history_start = pd.Timestamp.today() - pd.DateOffset(months=history_months)

peer_controls = st.columns([1, 1, 3])
add_peers = peer_controls[0].checkbox(
    "Add nearest peers", value=False, key="hist_add_peers",
    help="Same sector and industry as the first overlaid company, closest "
         "market cap on a log scale.")
log_scale = peer_controls[1].checkbox("Log scale", value=False, key="chart_log_peers")
peer_source = compare_tickers[0] if compare_tickers else chart_ticker
if add_peers and peer_source:
    peer_frame = ui.competitors(universe, peer_source, count=4)
    added = [t for t in peer_frame["ticker"].tolist()
             if t not in compare_tickers] if not peer_frame.empty else []
    compare_tickers = compare_tickers + added
    if added:
        peer_controls[2].caption(
            f"Peers of **{peer_source}** added: {', '.join(added)}")


@st.cache_data(ttl=3600)
def group_multiple_history(group_field, groups, multiple, months):
    """Median multiple per quarter for each sector or industry.

    Median rather than mean, and positives only: one loss-making company with a
    P/E of -4,000 would otherwise define the whole group's line.
    """
    from sharadar_loader import DATA_DIR, connect
    if not groups:
        return pd.DataFrame()
    members = analytics_frame[analytics_frame[group_field].isin(list(groups))][
        ["ticker", group_field]]
    connection = connect()
    path = (DATA_DIR / "metrics.parquet").as_posix()
    start = (pd.Timestamp.today()
             - pd.DateOffset(months=months)).date().isoformat()
    frame = connection.execute(
        f"SELECT ticker, date, {multiple} AS value FROM read_parquet('{path}') "
        f"WHERE date >= '{start}' AND {multiple} > 0").df()
    connection.close()
    frame = frame.merge(members, on="ticker", how="inner")
    frame["quarter"] = pd.to_datetime(frame["date"]).dt.to_period("Q").dt.to_timestamp()
    return frame.groupby([group_field, "quarter"])["value"].median().reset_index()


history = group_multiple_history(group_column, tuple(chosen_groups),
                                 chart_multiple, history_months)
figure = go.Figure()
for index, group in enumerate(chosen_groups):
    series = history[history[group_column] == group] if not history.empty \
        else pd.DataFrame()
    if series.empty:
        continue
    add_multiple_series(
        figure, series["quarter"], series["value"],
        current_group_median(analytics_frame, group_column, group,
                             chart_multiple),
        name=f"{group} (median)", colour=PALETTE[index % len(PALETTE)],
        width=2.5, log_scale=log_scale)

for index, ticker in enumerate(compare_tickers):
    company = ui.load_metrics_history(ticker)
    if company.empty or chart_multiple not in company:
        continue
    series = company[["date", chart_multiple]].copy()
    series = series[pd.to_datetime(series["date"]) >= history_start]
    if series.empty:
        continue
    add_multiple_series(
        figure, series["date"], series[chart_multiple],
        current_multiple(analytics_frame, ticker, chart_multiple),
        name=ticker,
        colour=PALETTE[(index + len(chosen_groups)) % len(PALETTE)],
        width=1.6, dash=None, log_scale=log_scale)

figure.update_layout(height=420, hovermode="x unified", margin=dict(t=20, b=0),
                     yaxis_title=MULTIPLES[chart_multiple],
                     legend=dict(orientation="h", y=1.1))
st.plotly_chart(figure, width="stretch", key="chart_group_multiple_history")

price_header = st.columns([1, 1, 1, 2])
price_benchmarks = price_header[0].multiselect(
    "Benchmarks", ["S&P 500", "Nasdaq 100", "Gold", "Brent"],
    default=["S&P 500"], key="hist_benchmarks")
price_mode = price_header[1].radio(
    "Values", ["Percent", "Absolute"], horizontal=True, key="hist_price_mode",
    help="Percent rebases every line to 0% at the start of the window, which "
         "is what makes companies comparable with each other. Absolute plots "
         "the adjusted close in dollars - the actual share price, on its own "
         "scale. Benchmarks are index levels, so in absolute mode they move "
         "to a second axis on the right.")
log_scale = price_header[2].checkbox("Log scale", value=False, key="log_prices_peers")
absolute = price_mode == "Absolute"
price_header[3].caption(
    "**Price performance of the same companies** over the same window - "
    + ("prices in dollars, adjusted for splits and dividends."
       if absolute else
       "rebased to 0%.")
    + " Valuation and price are different questions, and a multiple can fall "
      "while the stock rises."
)
price_figure = go.Figure()
for index, ticker in enumerate(compare_tickers):
    series = ui.load_prices(ticker)
    if series.empty:
        continue
    series["date"] = pd.to_datetime(series["date"])
    series = series[series["date"] >= history_start]
    if series.empty:
        continue
    if absolute:
        prices_to_plot = series["closeadj"]
        hover = "$%{y:,.2f}<extra></extra>"
    elif log_scale:
        prices_to_plot = np.log(series["closeadj"] / series["closeadj"].iloc[0])
        hover = "%{y:.1%}<extra></extra>"
    else:
        prices_to_plot = series["closeadj"] / series["closeadj"].iloc[0] - 1.0
        hover = "%{y:.1%}<extra></extra>"
    price_figure.add_trace(go.Scatter(
        x=series["date"], y=prices_to_plot,
        name=ticker, line=dict(color=PALETTE[index % len(PALETTE)], width=2),
        hovertemplate=hover))

if price_benchmarks:
    benchmark_data = ui.load_market_data()
    benchmark_data["date"] = pd.to_datetime(benchmark_data["date"])
    benchmark_data = benchmark_data[benchmark_data["date"] >= history_start]
    benchmark_map = {"S&P 500": "sp500", "Nasdaq 100": "nasdaq100",
                     "Gold": "gold", "Brent": "brent"}
    for offset, label in enumerate(price_benchmarks):
        series = benchmark_data[["date", benchmark_map[label]]].dropna()
        if series.empty:
            continue
        values = series[benchmark_map[label]]
        # In absolute mode an index level (S&P ~6,000) would flatten every
        # share price against it, so the benchmarks get their own right-hand
        # axis. In percent mode everything is already on one comparable scale.
        price_figure.add_trace(go.Scatter(
            x=series["date"],
            y=values if absolute else values / values.iloc[0] - 1.0,
            name=f"{label} (right)" if absolute else label,
            yaxis="y2" if absolute else "y",
            line=dict(width=1.6, dash="dash",
                      color=PALETTE[(offset + len(compare_tickers)) % len(PALETTE)]),
            hovertemplate=("%{y:,.1f}<extra></extra>" if absolute
                           else "%{y:.1%}<extra></extra>")))

if not absolute and (compare_tickers or price_benchmarks):
    price_figure.add_hline(y=0, line_width=1, line_color=MUTED, opacity=0.5)
price_figure.update_layout(height=340, hovermode="x unified", margin=dict(t=20, b=0),
                           legend=dict(orientation="h", y=1.12))
if absolute:
    price_figure.update_layout(
        yaxis=dict(title="Adjusted close (USD)", tickprefix="$",
                   type="log" if log_scale else "linear"),
        yaxis2=dict(title="Benchmark level", overlaying="y", side="right",
                    showgrid=False, type="log" if log_scale else "linear"))
else:
    price_figure.update_layout(yaxis=dict(title="Cumulative change",
                                          tickformat=".0%"))
st.plotly_chart(price_figure, width="stretch", key="chart_group_price_history")

st.container(key="section_group_performance")
st.subheader("Sector and industry performance")
st.caption(
    "The same table as the company view, one row per group. Built from daily "
    "cumulative return series, so volatility and drawdown are the group's own, "
    "not an average of its members'. :grey[Like the sector performance section, "
    "this shows every group regardless of the sidebar filters.]"
)

performance_controls = st.columns([1, 1, 1])
perf_group_type = performance_controls[0].radio(
    "Level", ["sector", "industry"], key="perf_level",
    format_func=lambda v: v.capitalize())
perf_weighting_label = performance_controls[1].radio(
    "Weighting", ["Median", "Average", "Market-cap weighted"], key="perf_weight")
perf_weighting = {"Median": "cum_median", "Average": "cum_mean",
                  "Market-cap weighted": "cum_capweighted"}[perf_weighting_label]
perf_display = performance_controls[2].radio(
    "Chart shows", ["Cumulative return", "Median multiple"], key="perf_display")

group_returns = ui.load_group_returns()
try:
    from analytics import group_period_returns
    perf_table = group_period_returns(group_returns, perf_group_type, perf_weighting)
except Exception as error:
    st.error(f"Could not build group performance: {error}")
    perf_table = pd.DataFrame()

# Also independent of the sidebar, for the same reason as the sector
# performance section above: a comparison across groups needs every group.
if not perf_table.empty:
    perf_columns = ([perf_group_type, "members"] + all_return_columns
                    + ["volatility_1y", "max_drawdown_1y", "cumulative_total"])
    ordered = perf_table.sort_values("return_1y", ascending=False)
    st.dataframe(
        format_table(ordered[perf_columns],
                     percent_cols=all_return_columns + ["volatility_1y",
                                                        "max_drawdown_1y",
                                                        "cumulative_total"]),
        hide_index=True, width="stretch", height=420,
        column_config={
            perf_group_type: st.column_config.TextColumn(
                perf_group_type.capitalize(), width=230),
            "members": st.column_config.NumberColumn("Firms", format="%d", width=70),
            **{f"return_{k}": st.column_config.TextColumn(v, width=80)
               for k, v in RETURN_HEADERS.items()},
            "volatility_1y": st.column_config.TextColumn("Vol 1Y", width=82),
            "max_drawdown_1y": st.column_config.TextColumn("Max DD", width=88),
            "cumulative_total": st.column_config.TextColumn("Since 2016", width=100),
        }, key="table_group_performance")

    # Chart controls sit with the chart, not with the table above it.
    chart_controls = st.columns([2, 1, 1, 1])
    plot_options = ordered[perf_group_type].tolist()
    plotted = chart_controls[0].multiselect(
        f"{perf_group_type.capitalize()}s to chart", plot_options,
        default=plot_options[:5], key="perf_plot")
    perf_benchmarks = chart_controls[1].multiselect(
        "Benchmarks", ["S&P 500", "Nasdaq 100", "Gold", "Brent"],
        default=["S&P 500"] if perf_display == "Cumulative return" else [],
        key="perf_benchmarks")
    perf_companies = chart_controls[2].multiselect(
        "Companies to overlay", ticker_universe, key="perf_companies")
    # Same union as the multiples chart - see the note there.
    perf_companies = list(dict.fromkeys(list(focus_tickers)
                                        + list(perf_companies)))
    perf_months = chart_controls[3].slider(
        "Timeline (months)", 1, 60, 60, step=1, key="perf_months",
        help="Continuous from one month to five years. The chart rebases to "
             "the start of whatever window you pick.")

    perf_figure = go.Figure()
    window_start = pd.Timestamp.today() - pd.DateOffset(months=perf_months)

    if perf_display == "Cumulative return":
        for index, group in enumerate(plotted):
            series = group_returns[(group_returns["group_type"] == perf_group_type)
                                   & (group_returns["group"] == group)]
            series = series[pd.to_datetime(series["date"]) >= window_start]
            if series.empty:
                continue
            # Rebase to the start of the visible window so every line begins at
            # 0% - otherwise a 5-year chart inherits nine years of drift.
            level = 1.0 + series[perf_weighting].to_numpy()
            perf_figure.add_trace(go.Scatter(
                x=series["date"], y=level / level[0] - 1.0, name=group,
                line=dict(color=PALETTE[index % len(PALETTE)], width=2),
                hovertemplate="%{y:.1%}<extra></extra>"))
        for offset, ticker in enumerate(perf_companies):
            series = ui.load_prices(ticker)
            if series.empty:
                continue
            series["date"] = pd.to_datetime(series["date"])
            series = series[series["date"] >= window_start]
            if series.empty:
                continue
            values = series["closeadj"].to_numpy()
            perf_figure.add_trace(go.Scatter(
                x=series["date"], y=values / values[0] - 1.0, name=ticker,
                line=dict(width=1.6, dash="dot",
                          color=PALETTE[(offset + len(plotted)) % len(PALETTE)]),
                hovertemplate="%{y:.1%}<extra></extra>"))
        for offset, label in enumerate(perf_benchmarks):
            benchmark_data = ui.load_market_data()
            benchmark_data["date"] = pd.to_datetime(benchmark_data["date"])
            benchmark_map = {"S&P 500": "sp500", "Nasdaq 100": "nasdaq100",
                             "Gold": "gold", "Brent": "brent"}
            series = benchmark_data[["date", benchmark_map[label]]].dropna()
            series = series[series["date"] >= window_start]
            if series.empty:
                continue
            values = series[benchmark_map[label]].to_numpy()
            perf_figure.add_trace(go.Scatter(
                x=series["date"], y=values / values[0] - 1.0, name=label,
                line=dict(width=1.8, dash="dash",
                          color=PALETTE[(offset + len(plotted)
                                         + len(perf_companies)) % len(PALETTE)]),
                hovertemplate="%{y:.1%}<extra></extra>"))
        perf_figure.add_hline(y=0, line_width=1, line_color=MUTED, opacity=0.5)
        perf_figure.update_layout(yaxis_title="Cumulative return",
                                  yaxis_tickformat=".0%")
    else:
        multiple_history = group_multiple_history(
            perf_group_type, tuple(plotted), chart_multiple, perf_months)
        for index, group in enumerate(plotted):
            series = (multiple_history[multiple_history[perf_group_type] == group]
                      if not multiple_history.empty else pd.DataFrame())
            if series.empty:
                continue
            add_multiple_series(
                perf_figure, series["quarter"], series["value"],
                current_group_median(analytics_frame, perf_group_type, group,
                                     chart_multiple),
                name=group, colour=PALETTE[index % len(PALETTE)], width=2)
        perf_figure.update_layout(
            yaxis_title=f"Median {MULTIPLES[chart_multiple]}")

    perf_figure.update_layout(height=430, hovermode="x unified",
                              margin=dict(t=20, b=0),
                              legend=dict(orientation="h", y=1.1))
    st.plotly_chart(perf_figure, width="stretch", key="chart_group_performance")
    if perf_display != "Cumulative return":
        st.caption(
            ":grey[Each line runs to today's price — the ◆ is the group's "
            "current median multiple, with the market cap at the last close "
            "and the reported earnings held fixed.]")

# ==========================================================================
# 6. valuation screen
# ==========================================================================
st.container(key="section_valuation_screen")
st.header("Valuation screen")
st.caption(
    "A composite percentile rank across the multiples, computed **within each "
    "sector** - a utility's EV/EBITDA is not comparable to a software firm's. "
    "Cheap means a low rank on price multiples and a high one on quality."
)

screen_controls = st.columns([1, 1, 1, 2])
direction = screen_controls[0].radio("Show", ["Cheapest", "Most expensive"],
                                     key="screen_direction")
require_profit = screen_controls[1].checkbox("Profitable only", value=True,
                                             key="screen_profit")
require_dcf = screen_controls[2].checkbox("Has DCF", value=True, key="screen_dcf")
screen_controls[3].caption(
    "Ranks are percentiles within sector, so every score is relative to peers."
)

candidates = filtered.copy()
if require_profit:
    candidates = candidates[(candidates["netinc_ttm"] > 0)
                            & (candidates["ebitda_ttm"] > 0)]
if require_dcf:
    candidates = candidates[candidates["implied_share_price_fcff"].notna()]

cheap_metrics = ["pe_ttm", "ps_ttm", "ev_ebitda_ttm", "ev_ebit_ttm", "p_fcf_ttm"]
quality_metrics = ["roic_ttm", "fcf_yield", "ebit_margin_ttm"]
scores = pd.DataFrame(index=candidates.index)
for metric in cheap_metrics:
    #values = candidates[metric].where(candidates[metric] > 0)
    values = candidates[metric]
    scores[metric] = values.groupby(candidates["sector"]).rank(pct=True)
for metric in quality_metrics:
    scores[metric] = (1.0 - candidates[metric]
                      .groupby(candidates["sector"]).rank(pct=True))
candidates["value_score"] = scores.mean(axis=1, skipna=True)
#candidates = candidates.dropna(subset=["value_score"])

if candidates.empty:
    st.info("No companies match those screen settings.")
    ranked = candidates
else:
    ranked = (candidates.nsmallest(25, "value_score") if direction == "Cheapest"
              else candidates.nlargest(25, "value_score"))
    screen_columns = ["ticker", "name", "sector", "industry", "profile",
                      "marketcap", "last_close", "value_score", "pe_ttm",
                      "ev_ebitda_ttm", "ps_ttm", "fcf_yield", "roic_ttm",
                      "fair_value", "fair_value_upside", "confidence",
                      "implied_share_price_fcff", "upside_fcff"]
    st.dataframe(
        format_table(ranked[screen_columns], money_cols=["marketcap"],
                     percent_cols=["fcf_yield", "roic_ttm", "upside_fcff",
                                   "fair_value_upside", "confidence"],
                     number_cols=["pe_ttm", "ev_ebitda_ttm", "ps_ttm", "value_score"],
                     price_cols=["last_close", "fair_value",
                                 "implied_share_price_fcff"]),
        hide_index=True, width="stretch",
        column_config={
            "ticker": st.column_config.TextColumn("Ticker", width=70),
            "name": st.column_config.TextColumn("Company", width=175),
            "sector": st.column_config.TextColumn("Sector", width=135),
            "industry": st.column_config.TextColumn("Industry", width=150),
            "marketcap": st.column_config.TextColumn("Market cap", width=105),
            "last_close": st.column_config.TextColumn("Price", width=90),
            "value_score": st.column_config.TextColumn("Score", width=75),
            "pe_ttm": st.column_config.TextColumn("P/E", width=75),
            "ev_ebitda_ttm": st.column_config.TextColumn("EV/EBITDA", width=95),
            "ps_ttm": st.column_config.TextColumn("P/S", width=75),
            "fcf_yield": st.column_config.TextColumn("FCF yld", width=85),
            "roic_ttm": st.column_config.TextColumn("ROIC", width=80),
            "profile": st.column_config.TextColumn(
                "Profile", width=130,
                help="How the valuation engine classified the company, which "
                     "sets which models it is valued on."),
            "fair_value": st.column_config.TextColumn("Fair value", width=100),
            "fair_value_upside": st.column_config.TextColumn("FV upside", width=92),
            "confidence": st.column_config.TextColumn("Confidence", width=95),
            "implied_share_price_fcff": st.column_config.TextColumn("DCF FCFF",
                                                                    width=98),
            "upside_fcff": st.column_config.TextColumn("Upside", width=88),
        }, key="table_valuation_screen")

if not ranked.empty:
    expand_ticker = st.selectbox("Expand a company", ranked["ticker"].tolist(),
                                 key="expand_ticker")
    # The same card the profile section renders, so the two cannot drift apart
    # and a company reads identically wherever you reach it from.
    profile_card(candidates[candidates["ticker"] == expand_ticker].iloc[0],
                 universe, key="screen")

# ==========================================================================
# 7. DCF constructor
# ==========================================================================
st.container(key="section_dcf_constructor")
st.header("DCF constructor")
st.caption(
    "Sliders start at the model's own fitted values, so the first thing you "
    "see is the base case. Move one and the model reruns end to end - the same "
    "code that produced the stored numbers."
)

drivers = ui.load_drivers()
dcf_options = [t for t in ticker_options if t in set(drivers["ticker"])]
if not dcf_options:
    st.info("No DCF available for the current filter.")
    st.stop()

default_index = dcf_options.index(chart_ticker) if chart_ticker in dcf_options else 0
dcf_ticker = st.selectbox("Company", dcf_options, index=default_index, key="dcf_ticker")
base_row = drivers[drivers["ticker"] == dcf_ticker].iloc[0]

# Slider keys carry the ticker, so switching company resets every slider to
# that company's fitted values instead of carrying the previous one's edits.
slider_column, chart_column = st.columns([1, 3])
overrides = {}
with slider_column:
    st.markdown("**Drivers** (start at fitted values)")
    stored_summary = ui.load_summary().set_index("ticker")
    for field, label in dcf.EDITABLE_DRIVERS.items():
        if field not in base_row.index:
            continue
        if pd.isna(base_row[field]):
            # terminal_growth_override is deliberately null in the driver table
            # - null means "use the model's own cap". Seed the slider with the
            # terminal growth the model actually applied, so it opens on the
            # base case like every other driver rather than disappearing.
            if field == "terminal_growth_override" and dcf_ticker in stored_summary.index:
                fitted = float(stored_summary.loc[dcf_ticker, "terminal_growth"])
            else:
                continue
        else:
            fitted = float(base_row[field])
        bound_key = dcf.DRIVER_BOUND_KEY.get(field)
        low, high = dcf.BOUNDS[bound_key] if bound_key else (0.01, 0.35)
        overrides[field] = st.slider(
            label, float(low), float(high),
            value=float(np.clip(fitted, low, high)), step=0.005, format="%.3f",
            key=f"driver_{dcf_ticker}_{field}", help=f"Fitted value: {fitted:.3f}")
    if st.button("Reset to fitted values", width="stretch", key="dcf_reset"):
        for field in list(overrides):
            st.session_state.pop(f"driver_{dcf_ticker}_{field}", None)
        st.rerun()

try:
    whatif_forecast, whatif_summary = dcf.revalue(dcf_ticker, overrides, drivers)
    base_forecast = ui.load_forecast(dcf_ticker)
    base_summary = ui.load_summary().set_index("ticker").loc[dcf_ticker]
except (KeyError, IndexError) as error:
    st.error(f"Could not revalue {dcf_ticker}: {error}")
    st.stop()

with chart_column:
    cash_history = ui.load_cashflow_history(dcf_ticker)
    metrics_history = ui.load_metrics_history(dcf_ticker)
    driver_history = ui.load_driver_history(dcf_ticker)
    wacc_history = ui.load_wacc_history(dcf_ticker)

    # EPS history is net income over the current share count, matching the
    # forecast line, which also holds shares constant.
    eps_history = metrics_history.copy()
    share_count = base_summary["shares"]
    if not eps_history.empty and pd.notna(share_count) and float(share_count) > 0:
        eps_history["eps_ttm"] = eps_history["netinc_ttm"] / float(share_count)
    elif not eps_history.empty:
        eps_history["eps_ttm"] = np.nan

    metric_choice = st.radio(
        "Series", ["Revenue", "FCFF", "FCFE", "EPS", "EBIT margin",
                   "Capex / revenue", "Change in WC / revenue", "D&A",
                   "Discount rates"],
        horizontal=True, key="dcf_series")

    # Every series is drawn against its own reported history. A driver path is
    # only judgeable against where the company has actually been - a capex
    # ratio fading to 3% means nothing until you can see it ran at 8%.
    series_map = {
        "Revenue": ("revenue", "revenue_ttm", cash_history, "$"),
        "FCFF": ("fcff", "fcff_ttm", cash_history, "$"),
        "FCFE": ("fcfe", "fcfe_ttm", cash_history, "$"),
        "EPS": ("eps", "eps_ttm", eps_history, "$"),
        "EBIT margin": ("ebit_margin", "ebit_margin_ttm", metrics_history, "%"),
        "Capex / revenue": ("capex_to_revenue", "capex_to_revenue",
                            driver_history, "%"),
        "Change in WC / revenue": ("dwc_to_revenue", "dwc_to_revenue",
                                   driver_history, "%"),
        "D&A": ("da", "da", driver_history, "$"),
        "Discount rates": ("wacc", "wacc", wacc_history, "%"),
    }
    forecast_column, history_column, source, unit = series_map[metric_choice]

    figure = go.Figure()
    if source is not None and not source.empty and history_column in source.columns:
        figure.add_trace(go.Scatter(
            x=pd.to_datetime(source["date"]), y=source[history_column],
            name="WACC (reported)" if metric_choice == "Discount rates"
            else "Reported", line=dict(color=MUTED, width=2)))

    base_dates = pd.to_datetime(base_summary["base_date"]) + pd.to_timedelta(
        base_forecast["year"] * 365.25, unit="D")

    if metric_choice == "Discount rates":
        # WACC and cost of equity are single rates, not projected paths, so the
        # "forecast" is a flat segment across the forecast window. Drawn as
        # traces rather than add_hline: hline annotations pin to the plot edge
        # and were landing on top of each other and on the historical series.
        # As traces they get legend entries and hover values instead.
        if not wacc_history.empty:
            figure.add_trace(go.Scatter(
                x=pd.to_datetime(wacc_history["date"]),
                y=wacc_history["cost_of_equity"], name="Cost of equity (reported)",
                line=dict(color="#6B4C8A", width=2, dash="dot")))
        span = [base_dates.iloc[0], base_dates.iloc[-1]]
        for label, value, colour, dash in (
                ("WACC - base", float(base_summary["wacc"]), ACCENT, "dash"),
                ("WACC - what-if", float(whatif_summary["wacc"]), UP, None),
                ("Cost of equity - what-if",
                 float(whatif_summary["cost_of_equity"]), "#6B4C8A", None)):
            figure.add_trace(go.Scatter(
                x=span, y=[value, value], name=f"{label} ({value:.2%})",
                mode="lines", line=dict(color=colour, width=2.5, dash=dash)))
    else:
        figure.add_trace(go.Scatter(
            x=base_dates, y=base_forecast[forecast_column], name="Base forecast",
            line=dict(color=ACCENT, width=2, dash="dash")))
        figure.add_trace(go.Scatter(
            x=base_dates, y=whatif_forecast[forecast_column], name="What-if",
            line=dict(color=UP, width=2.5)))

    if unit == "%":
        figure.add_hline(y=0, line_width=1, line_color=MUTED, opacity=0.35)

    figure.update_layout(
        height=400, hovermode="x unified", margin=dict(t=30, b=0),
        yaxis_tickformat=".1%" if unit == "%" else ",.0f",
        legend=dict(orientation="h", y=1.1))
    st.plotly_chart(figure, width="stretch", key="chart_dcf_series")

    market_price = float(base_summary["market_price"])
    results = st.columns(4)
    for position, (label, key) in enumerate(
            [("Implied FCFF", "implied_share_price_fcff"),
             ("Implied FCFE", "implied_share_price_fcfe"),
             ("Implied EPS", "implied_share_price_eps")]):
        new = float(whatif_summary[key])
        old = float(base_summary[key]) if pd.notna(base_summary[key]) else np.nan
        results[position].metric(
            label, f"${new:,.2f}" if np.isfinite(new) else "-",
            f"{new - old:+,.2f} vs base" if np.isfinite(old) and np.isfinite(new)
            else None)
    results[3].metric("Market price", f"${market_price:,.2f}")

    # Two dates matter here and they are not the same one. The valuation is
    # built from the last filing; the comparison is against the last close.
    filing_price = base_summary.get("price_at_filing")
    st.caption(
        f"Valuation basis: filing of "
        f"**{pd.Timestamp(base_summary['base_date']).date()}**"
        + (f" (price then ${filing_price:,.2f})" if pd.notna(filing_price) else "")
        + f" · Market price: last close of "
        f"**{pd.Timestamp(base_summary['market_price_date']).date()}**. "
        "Upside and P(above market) are measured against the last close, not "
        "the filing-date price."
    )

    band = st.columns(4)
    band[0].metric("p15", f"${whatif_summary['price_fcff_p15']:,.2f}")
    band[1].metric("p50", f"${whatif_summary['price_fcff_p50']:,.2f}")
    band[2].metric("p85", f"${whatif_summary['price_fcff_p85']:,.2f}")
    band[3].metric("P(above market)",
                   ui.percent(whatif_summary["prob_above_market_fcff"], 0))

    if bool(base_summary.get("dcf_caution_sector", False)):
        st.warning(
            f"{dcf_ticker} is in **{base_summary['sector']}**. An FCFF DCF is "
            "structurally the wrong model for banks, insurers and REITs - debt "
            "is raw material rather than financing, and capex and working "
            "capital carry no operating meaning. Indicative only.")

dcf_peers = ui.competitors(universe, dcf_ticker, count=6)
if not dcf_peers.empty:
    st.container(key="section_dcf_peers")
    st.subheader("How the competitors are valued")
    st.caption("Same sector and industry, closest market cap. A DCF that "
               "disagrees with every peer is usually the model, not the market.")
    peer_columns = ["ticker", "name", "peer_match", "marketcap", "last_close",
                    "fair_value", "fair_value_upside", "confidence",
                    "implied_share_price_fcff", "implied_share_price_eps",
                    "pe_ttm", "ps_ttm", "ev_ebitda_ttm", "wacc"]
    st.dataframe(
        format_table(dcf_peers[peer_columns], money_cols=["marketcap"],
                     percent_cols=["fair_value_upside", "confidence", "wacc"],
                     number_cols=["pe_ttm", "ps_ttm", "ev_ebitda_ttm"],
                     price_cols=["last_close", "fair_value",
                                 "implied_share_price_fcff",
                                 "implied_share_price_eps"]),
        hide_index=True, width="stretch",
        column_config={
            "ticker": st.column_config.TextColumn("Ticker", width=70),
            "name": st.column_config.TextColumn("Company", width=200),
            "peer_match": st.column_config.TextColumn("Match", width=135),
            "marketcap": st.column_config.TextColumn("Market cap", width=105),
            "last_close": st.column_config.TextColumn("Price", width=90),
            "implied_share_price_fcff": st.column_config.TextColumn("DCF FCFF",
                                                                    width=98),
            "implied_share_price_eps": st.column_config.TextColumn("DCF EPS",
                                                                   width=95),
            "fair_value": st.column_config.TextColumn("Fair value", width=100),
            "fair_value_upside": st.column_config.TextColumn("FV upside", width=92),
            "confidence": st.column_config.TextColumn("Confidence", width=95),
            "pe_ttm": st.column_config.TextColumn("P/E", width=75),
            "ps_ttm": st.column_config.TextColumn("P/S", width=75),
            "ev_ebitda_ttm": st.column_config.TextColumn("EV/EBITDA", width=95),
            "wacc": st.column_config.TextColumn("WACC", width=80),
        }, key="table_dcf_peers")

st.divider()
st.caption(
    "Data: Sharadar (Nasdaq Data Link) Core US Equities Bundle, 10-year tier - "
    "FRED for rates, S&P 500, Nasdaq 100, Brent and VIX - gold via GLD. "
    "Fundamentals are as-reported (point-in-time), indexed to SEC filing dates."
)
