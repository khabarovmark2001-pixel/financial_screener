#!/usr/bin/env python3
"""
Portfolio: enter trades, see what they are worth, and how they have behaved.

Three panels. The blotter takes trades and stores them outside the app so a
restart does not lose them. The positions table values what the blotter
implies, splitting P&L into the part already banked and the part still at
risk, and putting a volatility beside each. The chart plots either the whole
portfolio or any one holding inside it, against a benchmark.

The arithmetic - average cost, split restatement, time-weighted return - is in
portfolio.py, with the reasoning for each choice. This file is presentation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import portfolio as pf
import ui_data as ui

st.set_page_config(page_title="Portfolio", page_icon="briefcase", layout="wide")

UP, DOWN, ACCENT, MUTED = "#1B7F5E", "#B4472F", "#2F5D8A", "#8894A0"
PALETTE = ["#2F5D8A", "#B4472F", "#1B7F5E", "#8A6D2F", "#6B4C8A", "#2F7F8A"]

PORTFOLIO = "PORTFOLIO"
VALUE, RETURN, PNL = "Market value", "Cumulative return", "Profit and loss"
CHART_VIEWS = (RETURN, VALUE, PNL)

BENCHMARKS = {"S&P 500": "sp500", "Nasdaq 100": "nasdaq100",
              "Gold": "gold", "Brent": "brent"}

analytics_frame = ui.load_analytics()
universe = analytics_frame.sort_values("marketcap", ascending=False)
names = dict(zip(universe["ticker"], universe["name"].fillna("")))

st.title("Portfolio")

with st.sidebar:
    st.header("Display")
    ui.currency_selector()
    st.caption(
        "Everything on this page is in US dollars whichever way that is set: "
        "a trade is a dollar cash flow on a US exchange, so there is no "
        "as-reported version of a position's value."
    )


@st.cache_data(ttl=600, show_spinner=False)
def price_panel(tickers: tuple[str, ...]) -> dict[str, pd.DataFrame]:
    """Price history for every ticker the blotter mentions, keyed by ticker."""
    return {t: ui.load_prices(t) for t in tickers}


def close_on(prices: pd.DataFrame | None, when) -> float:
    """The close on or before a date - what a trade that day was priced near."""
    if prices is None or prices.empty:
        return float("nan")
    series = prices[["date", "close"]].dropna().copy()
    series["date"] = pd.to_datetime(series["date"])
    series = series[series["date"] <= pd.Timestamp(when)]
    return float(series["close"].iloc[-1]) if not series.empty else float("nan")


trades = pf.load_trades()
held = tuple(sorted(set(trades["ticker"]))) if not trades.empty else ()
panel = price_panel(held) if held else {}

# ==========================================================================
# 1. enter a trade
# ==========================================================================
st.container(key="section_trade_entry")
st.subheader("Enter a trade")

ticker_options = universe["ticker"].tolist()
entry = st.columns([2, 1.2, 1, 1, 1, 1])

# Outside the form, so choosing a company or a date can refresh the suggested
# price before the trade is submitted. A form batches its widgets until the
# button is pressed, which would make the suggestion always one trade stale.
entry_ticker = entry[0].selectbox(
    "Company", ticker_options, key="trade_ticker",
    format_func=lambda t: f"{t} — {names.get(t, '')}"[:55])
entry_date = entry[1].date_input("Trade date", value=pd.Timestamp.today().date(),
                                 key="trade_date")
entry_side = entry[2].radio("Side", list(pf.SIDES), key="trade_side",
                            horizontal=True)

suggested = close_on(ui.load_prices(entry_ticker), entry_date)
entry_quantity = entry[3].number_input("Quantity", min_value=0.0, value=100.0,
                                       step=1.0, key="trade_quantity")
entry_price = entry[4].number_input(
    "Price ($)", min_value=0.0,
    value=float(suggested) if np.isfinite(suggested) else 0.0,
    step=0.01, format="%.4f", key=f"trade_price_{entry_ticker}_{entry_date}",
    help="Defaults to the close on the trade date. Overwrite it with the "
         "price actually paid - the fill, not the close.")
entry_fees = entry[5].number_input("Fees ($)", min_value=0.0, value=0.0,
                                   step=0.01, key="trade_fees")

note_column, button_column = st.columns([4, 1])
entry_note = note_column.text_input("Note (optional)", key="trade_note",
                                    placeholder="Why, or where it was filled")

if np.isfinite(suggested):
    note_column.caption(
        f"Close on or before {entry_date}: **${suggested:,.2f}**. "
        f"Consideration at the price above: "
        f"**${entry_quantity * entry_price:,.2f}** plus ${entry_fees:,.2f} fees. "
        f"Enter the trade as it was done - 10 shares at $1,200, not 100 at "
        f"$120. Splits since are restated automatically.")
else:
    note_column.warning(
        f"No stored price history for {entry_ticker} on or before {entry_date}. "
        f"The trade can still be entered, but it will not be valued or charted "
        f"until that ticker has prices in `data/stocks.parquet`.")

if button_column.button("Add trade", type="primary", width="stretch",
                        key="trade_submit",
                        disabled=entry_quantity <= 0 or entry_price <= 0):
    pf.add_trade(entry_ticker, entry_date, entry_side, entry_quantity,
                 entry_price, entry_fees, entry_note)
    price_panel.clear()
    st.success(f"{entry_side} {entry_quantity:,.4g} {entry_ticker} "
               f"at ${entry_price:,.2f} recorded.")
    st.rerun()

if trades.empty:
    st.info("No trades yet. Enter one above and the positions, P&L and chart "
            "appear here.")
    st.stop()

# ==========================================================================
# 2. positions
# ==========================================================================
holdings = pf.positions(trades, panel, analytics_frame)
history = pf.holdings_history(trades, panel)

st.container(key="section_positions")
st.subheader("Positions")

totals = pf.summary(history, PORTFOLIO) if not history.empty else {}
open_positions = holdings[holdings["is_open"]]

if totals:
    strip = st.columns(6)
    strip[0].metric("Market value", ui.money(totals["market_value"]))
    strip[1].metric("Net invested", ui.money(totals["invested"]),
                    help="Cash paid in less cash taken out, fees included. "
                         "Not a cost basis: a position sold at a profit can "
                         "leave this negative while the holding is still open.")
    strip[2].metric("Total P&L", ui.money(totals["pnl"]),
                    f"{totals['cumulative_return']:+.1%} time-weighted")
    strip[3].metric("Realised P&L", ui.money(holdings["realised_pnl"].sum()))
    strip[4].metric("Volatility 1Y",
                    ui.percent(totals["volatility"], 1)
                    if np.isfinite(totals.get("volatility", np.nan)) else "—",
                    help="Annualised standard deviation of the portfolio's own "
                         "daily time-weighted return over the last year. It is "
                         "the volatility of the holdings, not of the deposit "
                         "schedule - trade cash flows are removed before the "
                         "return is measured.")
    strip[5].metric("Max drawdown", ui.percent(totals["max_drawdown"], 1)
                    if np.isfinite(totals.get("max_drawdown", np.nan)) else "—")

short = holdings[holdings["quantity"] < -1e-9]
if not short.empty:
    st.warning(
        f"Sold more than was held in {', '.join(short['ticker'])}. The blotter "
        f"records it as entered and the position shows short; check for a "
        f"missing buy.")

position_columns = ["ticker", "name", "sector", "quantity", "avg_cost",
                    "last_close", "market_value", "weight", "return_pct",
                    "unrealised_pnl", "realised_pnl", "total_pnl",
                    "volatility_1y", "trades"]
view = holdings[[c for c in position_columns if c in holdings.columns]].copy()

formatted = view.copy()
for column in ("market_value", "unrealised_pnl", "realised_pnl", "total_pnl"):
    formatted[column] = formatted[column].map(ui.money)
for column in ("avg_cost", "last_close"):
    formatted[column] = formatted[column].map(
        lambda v: f"${v:,.2f}" if pd.notna(v) and np.isfinite(v) else "—")
for column in ("weight", "return_pct", "volatility_1y"):
    formatted[column] = formatted[column].map(lambda v: ui.percent(v, 1))
formatted["quantity"] = formatted["quantity"].map(lambda v: f"{v:,.4g}")

st.dataframe(
    formatted, hide_index=True, width="stretch",
    column_config={
        "ticker": st.column_config.TextColumn("Ticker", width=70),
        "name": st.column_config.TextColumn("Company", width=190),
        "sector": st.column_config.TextColumn("Sector", width=140),
        "quantity": st.column_config.TextColumn("Shares", width=85),
        "avg_cost": st.column_config.TextColumn(
            "Avg cost", width=90,
            help="Average cost of what is still held, fees capitalised, "
                 "restated for any split since the trade."),
        "last_close": st.column_config.TextColumn("Last", width=85),
        "market_value": st.column_config.TextColumn("Value", width=95),
        "weight": st.column_config.TextColumn("Weight", width=80),
        "return_pct": st.column_config.TextColumn(
            "Return", width=85,
            help="Unrealised P&L over the cost of what is still held."),
        "unrealised_pnl": st.column_config.TextColumn("Unrealised", width=100),
        "realised_pnl": st.column_config.TextColumn(
            "Realised", width=95,
            help="Banked on sales, average-cost matched, net of fees."),
        "total_pnl": st.column_config.TextColumn("Total P&L", width=100),
        "volatility_1y": st.column_config.TextColumn(
            "Vol 1Y", width=80,
            help="Annualised standard deviation of the stock's own daily "
                 "total return over the last year - a property of the stock, "
                 "not of your position size."),
        "trades": st.column_config.NumberColumn("Trades", format="%d", width=75),
    }, key="table_positions")

if not open_positions.empty and open_positions["weight"].notna().any():
    st.caption(
        f"{len(open_positions)} open position(s), "
        f"{len(holdings) - len(open_positions)} closed. Largest weight: "
        f"**{open_positions.iloc[0]['ticker']}** at "
        f"{open_positions.iloc[0]['weight']:.1%}. Weights are of market value, "
        f"so they sum to 100% across open positions only.")

# ==========================================================================
# 3. chart
# ==========================================================================
st.container(key="section_portfolio_chart")
st.subheader("Performance")

if history.empty:
    st.info("Nothing to chart: none of the traded tickers has stored prices.")
    st.stop()

chartable = [PORTFOLIO] + sorted(set(history["ticker"]) - {PORTFOLIO})
chart_controls = st.columns([2, 1.6, 1.4, 2])
chosen = chart_controls[0].multiselect(
    "Show", chartable, default=[PORTFOLIO], key="pf_chart_series",
    format_func=lambda t: ("Whole portfolio" if t == PORTFOLIO
                           else f"{t} — {names.get(t, '')}"[:40]),
    help="The portfolio, any holding inside it, or several at once.")
chart_view = chart_controls[1].radio("View", list(CHART_VIEWS), key="pf_chart_view")
timeframe = chart_controls[2].selectbox(
    "Timeframe", ["Max", "1W", "1M", "3M", "6M", "YTD", "1Y", "3Y", "5Y"],
    key="pf_chart_tf")
overlay = chart_controls[3].multiselect(
    "Benchmark", list(BENCHMARKS), default=["S&P 500"], key="pf_chart_benchmark",
    help="Only meaningful on cumulative return, where everything is rebased "
         "to 0% at the start of the window.")

if not chosen:
    st.info("Pick at least one series to plot.")
    st.stop()

last_date = pd.to_datetime(history["date"]).max()
window_start = ui.timeframe_start(timeframe, last_date)

figure = go.Figure()
for index, ticker in enumerate(chosen):
    series = pf.performance(history, ticker)
    if series.empty:
        continue
    series = series[pd.to_datetime(series["date"]) >= window_start]
    if series.empty:
        continue

    label = "Portfolio" if ticker == PORTFOLIO else ticker
    if chart_view == RETURN:
        # Rebased inside the window: a 3-month view should start at 0%, not
        # inherit whatever the position had already made before it.
        level = 1.0 + series["cumulative_return"].to_numpy()
        values = level / level[0] - 1.0
        hover = "%{y:.2%}<extra>" + label + "</extra>"
    elif chart_view == VALUE:
        values = series["market_value"]
        hover = "$%{y:,.0f}<extra>" + label + "</extra>"
    else:
        values = series["pnl"]
        hover = "$%{y:,.0f}<extra>" + label + "</extra>"

    figure.add_trace(go.Scatter(
        x=series["date"], y=values, name=label,
        line=dict(color=PALETTE[index % len(PALETTE)],
                  width=2.5 if ticker == PORTFOLIO else 1.7),
        hovertemplate=hover))

if overlay and chart_view == RETURN:
    market = ui.load_market_data()
    market["date"] = pd.to_datetime(market["date"])
    market = market[(market["date"] >= window_start) & (market["date"] <= last_date)]
    for offset, label in enumerate(overlay):
        series = market[["date", BENCHMARKS[label]]].dropna()
        if series.empty:
            continue
        values = series[BENCHMARKS[label]].to_numpy()
        figure.add_trace(go.Scatter(
            x=series["date"], y=values / values[0] - 1.0, name=label,
            line=dict(width=1.5, dash="dash",
                      color=PALETTE[(offset + len(chosen)) % len(PALETTE)]),
            hovertemplate="%{y:.2%}<extra>" + label + "</extra>"))
elif overlay:
    st.caption(f":grey[Benchmarks are drawn on {RETURN.lower()} only - an "
               f"index level has no dollar value to compare against.]")

if chart_view in (RETURN, PNL):
    figure.add_hline(y=0, line_width=1, line_color=MUTED, opacity=0.5)
figure.update_layout(
    height=440, hovermode="x unified", margin=dict(t=30, b=0),
    legend=dict(orientation="h", y=1.1),
    yaxis=dict(title=chart_view,
               tickformat=".0%" if chart_view == RETURN else "$,.0f"))
st.plotly_chart(figure, width="stretch", key="chart_portfolio_performance")

st.caption(
    "**Cumulative return** is time-weighted: each day's return is measured "
    "after removing that day's trade cash flow, so paying money in is not a "
    "gain and the line is comparable with the benchmark beside it. "
    "**Market value** and **Profit and loss** are in dollars - P&L is market "
    "value less net cash invested, so it moves when you trade as well as when "
    "the market does. Every series is valued on split-adjusted closes with "
    "dividends left in the price, which is what the position is actually "
    "worth; the volatility columns use total-return prices instead, because a "
    "dividend ex-date is a price fall that was never a loss."
)

# ==========================================================================
# 4. blotter
# ==========================================================================
st.container(key="section_blotter")
st.subheader("Trade blotter")

blotter = pf.restate_for_splits(trades, panel).sort_values(
    ["date", "trade_id"], ascending=[False, False])
blotter["consideration"] = (blotter["quantity"] * blotter["price"]
                            + np.where(blotter["side"] == pf.BUY, 1, -1)
                            * blotter["fees"])

display = pd.DataFrame({
    "Remove": False,
    "ID": blotter["trade_id"],
    "Date": pd.to_datetime(blotter["date"]).dt.date,
    "Ticker": blotter["ticker"],
    "Side": blotter["side"],
    "Shares": blotter["quantity"],
    "Price": blotter["price"],
    "Fees": blotter["fees"],
    "Consideration": blotter["consideration"],
    "Split since": blotter["split_factor"],
    "Note": blotter["note"],
})

edited = st.data_editor(
    display, hide_index=True, width="stretch", key="editor_blotter",
    disabled=[c for c in display.columns if c != "Remove"],
    column_config={
        "Remove": st.column_config.CheckboxColumn("Remove", width=80),
        "ID": st.column_config.NumberColumn("ID", format="%d", width=55),
        "Date": st.column_config.DateColumn("Date", width=105),
        "Ticker": st.column_config.TextColumn("Ticker", width=75),
        "Side": st.column_config.TextColumn("Side", width=65),
        "Shares": st.column_config.NumberColumn("Shares", format="%.4g", width=90),
        "Price": st.column_config.NumberColumn("Price", format="$%.4f", width=95),
        "Fees": st.column_config.NumberColumn("Fees", format="$%.2f", width=80),
        "Consideration": st.column_config.NumberColumn(
            "Consideration", format="$%.2f", width=120,
            help="Shares x price, plus fees on a buy and less fees on a sale."),
        "Split since": st.column_config.NumberColumn(
            "Split since", format="%.4g", width=100,
            help="How many of today's shares one share bought that day became. "
                 "1 means no split. Quantities and prices are restated by this "
                 "before anything is valued."),
        "Note": st.column_config.TextColumn("Note", width=220),
    })

actions = st.columns([1, 1, 3])
marked = edited.loc[edited["Remove"], "ID"].tolist()
if actions[0].button(f"Delete {len(marked)} selected", key="blotter_delete",
                     width="stretch", disabled=not marked, type="primary"):
    pf.delete_trades(marked)
    price_panel.clear()
    st.rerun()
actions[1].download_button(
    "Download blotter (CSV)",
    trades.to_csv(index=False).encode(), width="stretch",
    file_name="portfolio_trades.csv", mime="text/csv", key="blotter_download")
actions[2].caption(
    f"Stored in `{pf.TRADES_CSV.name}` at the project root — a plain file no "
    f"stage of the data refresh touches, so the blotter survives a restart "
    f"and can be edited or backed up by hand. Cost basis is **average cost**: "
    f"a sale realises against the running average of what is still held, which "
    f"is what a broker's average-price column shows. The split between "
    f"realised and unrealised depends on that choice; the total P&L does not."
)
