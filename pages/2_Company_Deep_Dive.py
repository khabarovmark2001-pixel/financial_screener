#!/usr/bin/env python3
"""
Company deep dive: historical multiples, DCF financials, calendar, ownership.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

import ui_data as ui

st.set_page_config(page_title="Company Deep Dive", page_icon="mag",
                   layout="wide")

UP, DOWN, ACCENT, MUTED = "#1B7F5E", "#B4472F", "#2F5D8A", "#8894A0"
PALETTE = ["#2F5D8A", "#B4472F", "#1B7F5E", "#8A6D2F", "#6B4C8A", "#2F7F8A",
           "#A5541F", "#4B6F44"]

analytics_frame = ui.load_analytics()
universe = analytics_frame[analytics_frame["in_universe"]].copy()
universe = universe.sort_values("marketcap", ascending=False)

st.title("Company Deep Dive")

with st.sidebar:
    st.header("Display")
    money_basis = ui.currency_selector()
    st.caption(
        "Every figure on this page is built from the metrics and cash-flow "
        "tables, which are restated in US dollars when they are built — so "
        "this page reads the same either way. The setting governs the "
        "statements page, where the filed figures are shown line by line."
    )

options = universe["ticker"].tolist()
labels = dict(zip(universe["ticker"], universe["name"].fillna("")))
selected = st.selectbox(
    "Company", options, key="deep_dive_ticker",
    format_func=lambda t: f"{t} — {labels.get(t, '')}"[:60])

row = universe[universe["ticker"] == selected].iloc[0]
st.caption(f"**{row['name']}** - {row['sector']} > **{row['industry']}** - "
           f"{row['exchange']} - market cap {ui.money(row['marketcap'])}")

header = st.columns(7)
header[0].metric("Price", f"${row.last_close:,.2f}",
                 f"{row.return_1d:+.2%}" if pd.notna(row.return_1d) else None)
header[1].metric("1y", f"{row.return_1y:+.1%}" if pd.notna(row.return_1y) else "-")
header[2].metric("Vol 1y",
                 f"{row.volatility_1y:.1%}" if pd.notna(row.volatility_1y) else "-")
header[3].metric("P/E", f"{row.pe_ttm:.1f}" if pd.notna(row.pe_ttm) else "-",
                 help="At the latest close, against the last reported "
                      "trailing twelve months.")
header[4].metric("P/B", f"{row.pb:.2f}" if pd.notna(row.pb) else "-",
                 help="At the latest close, against book equity on the last "
                      "reported balance sheet.")
header[5].metric("WACC", f"{row.wacc:.2%}" if pd.notna(row.wacc) else "-")
header[6].metric("DCF FCFF",
                 f"${row.implied_share_price_fcff:,.2f}"
                 if pd.notna(row.implied_share_price_fcff) else "-",
                 f"{row.upside_fcff:+.1%}" if pd.notna(row.upside_fcff) else None)

valuations = st.columns(4)
for position, (label, price_key, band_low, band_high, probability) in enumerate([
        ("FCFF", "implied_share_price_fcff", "price_fcff_p15", "price_fcff_p85",
         "prob_above_market_fcff"),
        ("FCFE", "implied_share_price_fcfe", "price_fcfe_p15", "price_fcfe_p85",
         "prob_above_market_fcfe"),
        ("EPS", "implied_share_price_eps", "price_eps_p15", "price_eps_p85",
         "prob_above_market_eps")]):
    value = row.get(price_key)
    valuations[position].metric(
        f"{label} model", f"${value:,.2f}" if pd.notna(value) else "-",
        f"p15 ${row.get(band_low):,.0f} - p85 ${row.get(band_high):,.0f}"
        if pd.notna(row.get(band_low)) else None)
valuations[3].metric("Industry", str(row["industry"]))

metrics_history = ui.load_metrics_history(selected)
cashflow_history = ui.load_cashflow_history(selected)
forecast = ui.load_forecast(selected)
summary_all = ui.load_summary().set_index("ticker")
summary = summary_all.loc[selected] if selected in summary_all.index else None

tabs = st.tabs(["Historical multiples", "Financials & DCF", "Competitors",
                "Reporting calendar", "Corporate actions", "Ownership",
                "Risk & quality"])

# ==========================================================================
with tabs[0]:
    st.container(key="section_multiples_history")
    st.subheader("Multiples through time")
    st.caption(
        "Each filed point is point-in-time: the multiple as it could have been "
        "computed on that filing date, never restated. The ◆ at the right-hand "
        "end of each line is the same multiple at the **latest close** — the "
        "number this page's header and every table on the site show."
    )
    if metrics_history.empty:
        st.info("No filing history stored for this ticker.")
    else:
        choices = st.multiselect(
            "Multiples", ["pe_ttm", "ps_ttm", "pb", "ev_ebitda_ttm", "ev_ebit_ttm",
                          "ev_sales_ttm", "p_fcf_ttm", "p_ebitda_ttm"],
            default=["pe_ttm", "ev_ebitda_ttm", "ps_ttm"],
            format_func=lambda c: {"pe_ttm": "P/E", "ps_ttm": "P/S", "pb": "P/B",
                                   "ev_ebitda_ttm": "EV/EBITDA",
                                   "ev_ebit_ttm": "EV/EBIT",
                                   "ev_sales_ttm": "EV/Sales",
                                   "p_fcf_ttm": "P/FCF",
                                   "p_ebitda_ttm": "P/EBITDA"}[c])
        clip = st.checkbox(
            "Hide negative and extreme values", value=True,
            help="Negative multiples come from loss-making periods and compress "
                 "the axis so the rest of the series becomes unreadable.")

        # The stored series is point-in-time - each value is the multiple at
        # the price quoted on that filing date - so on its own it stops
        # wherever the last filing landed and disagrees with every current
        # figure on the page. Each line is carried one segment further to the
        # latest close, holding the reported earnings fixed and moving only
        # the market cap, and the diamond marks where the filings stop.
        #
        # Dated to the last session in the database rather than to the wall
        # clock: on a weekend, `today` is a date no price exists for and the
        # point would float off the end of the axis away from the bar it was
        # computed from.
        last_session = pd.Timestamp(row["as_of"])
        figure = go.Figure()
        dates = pd.to_datetime(metrics_history["date"])
        for index, column in enumerate(choices):
            series = metrics_history[column].astype(float)
            if clip:
                series = series.where((series > 0) & (series < series.quantile(0.98)))

            current = row.get(column)
            current = float(current) if pd.notna(current) else np.nan
            if clip and np.isfinite(current) and current <= 0:
                current = np.nan

            colour = PALETTE[index % len(PALETTE)]
            line_x, line_y = list(dates), list(series)
            if np.isfinite(current):
                line_x, line_y = line_x + [last_session], line_y + [current]
            figure.add_trace(go.Scatter(
                x=line_x, y=line_y, name=column, mode="lines",
                line=dict(color=colour, width=2)))
            if np.isfinite(current):
                figure.add_trace(go.Scatter(
                    x=[last_session], y=[current], mode="markers", showlegend=False,
                    marker=dict(color=colour, size=9, symbol="diamond",
                                line=dict(width=1, color="white")),
                    hovertemplate="Now: %{y:,.2f}<extra>" + column + "</extra>"))
        figure.update_layout(height=420, hovermode="x unified",
                             margin=dict(t=20, b=0),
                             legend=dict(orientation="h", y=1.1))
        st.plotly_chart(figure, width='stretch', key="chart_multiples_history")
        st.caption(
            ":grey[The ◆ on each line is the multiple at the latest close, "
            "against the same reported twelve months as the last filed point "
            "beside it. Only the market cap moves between the two — the "
            "distance between them is the price move since the filing.]")

        st.subheader("Where it trades now versus its own history")
        st.caption(
            "**Now** is the multiple at the latest close, against the last "
            "reported twelve months — not the last point on the line above, "
            "which is the multiple on the day that filing landed. Where the "
            "price has moved since, the two differ, and it is the current one "
            "that says whether the stock is cheap today."
        )
        percentiles = []
        for column in choices:
            series = metrics_history[column].astype(float)
            series = series.where(series > 0).dropna()
            if len(series) < 8:
                continue
            # The repriced value from the analytics row where there is one;
            # `p_ebitda_ttm` is not carried there, so it falls back to the last
            # filed point rather than being dropped from the table.
            current = row.get(column)
            if pd.isna(current):
                current = series.iloc[-1]
            current = float(current)
            at_filing = row.get(f"{column}_at_filing")
            percentiles.append({
                "Multiple": column, "Now": current,
                "At last filing": (float(at_filing) if pd.notna(at_filing)
                                   else series.iloc[-1]),
                "Median": series.median(), "Min": series.min(), "Max": series.max(),
                "Percentile": (series < current).mean(),
            })
        if percentiles:
            st.dataframe(
                pd.DataFrame(percentiles), hide_index=True, width='stretch',
                column_config={
                    "Now": st.column_config.NumberColumn(format="%.2f"),
                    "At last filing": st.column_config.NumberColumn(
                        format="%.2f",
                        help="The same multiple at the price quoted on the "
                             "filing date — the last point on the chart above."),
                    "Median": st.column_config.NumberColumn(format="%.2f"),
                    "Min": st.column_config.NumberColumn(format="%.2f"),
                    "Max": st.column_config.NumberColumn(format="%.2f"),
                    "Percentile": st.column_config.ProgressColumn(
                        "Percentile of own history", format="percent",
                        min_value=0.0, max_value=1.0),
                }, key="table_multiple_percentiles")

# ==========================================================================
with tabs[1]:
    st.container(key="section_financials_dcf")
    st.subheader("Reported history and 15-year forecast")
    if forecast.empty or summary is None:
        st.info("No DCF for this ticker — it needs 6+ years of filing history.")
    else:
        base_date = pd.to_datetime(summary["base_date"])
        forecast_dates = base_date + pd.to_timedelta(forecast["year"] * 365.25,
                                                     unit="D")
        # EPS history is derived from reported net income over the current
        # share count, so it is comparable to the forecast line, which also
        # holds shares constant.
        eps_history = metrics_history.copy()
        shares = summary["shares"]
        if pd.notna(shares) and float(shares) > 0:
            eps_history["eps_ttm"] = eps_history["netinc_ttm"] / float(shares)
        else:
            eps_history["eps_ttm"] = np.nan

        figure = make_subplots(
            rows=3, cols=2, subplot_titles=(
                "Revenue", "FCFF", "FCFE", "EPS", "EBIT margin", "Net income"),
            vertical_spacing=0.10)

        panels = [
            (1, 1, "revenue", "revenue_ttm", cashflow_history),
            (1, 2, "fcff", "fcff_ttm", cashflow_history),
            (2, 1, "fcfe", "fcfe_ttm", cashflow_history),
            (2, 2, "eps", "eps_ttm", eps_history),
            (3, 1, "ebit_margin", "ebit_margin_ttm", metrics_history),
            (3, 2, "net_income", "netinc_ttm", metrics_history),
        ]
        for r, c, forecast_column, history_column, source in panels:
            if not source.empty and history_column in source.columns:
                figure.add_trace(go.Scatter(
                    x=pd.to_datetime(source["date"]), y=source[history_column],
                    name="Reported", legendgroup="rep", showlegend=(r == 1 and c == 1),
                    line=dict(color=MUTED, width=2)), row=r, col=c)
            figure.add_trace(go.Scatter(
                x=forecast_dates, y=forecast[forecast_column], name="Forecast",
                legendgroup="fc", showlegend=(r == 1 and c == 1),
                line=dict(color=ACCENT, width=2, dash="dash")), row=r, col=c)
        figure.update_yaxes(tickformat=".1%", row=3, col=1)
        figure.update_layout(height=860, hovermode="x unified",
                             margin=dict(t=50, b=0),
                             legend=dict(orientation="h", y=1.08))
        st.plotly_chart(figure, width='stretch', key="chart_financials_grid")

        st.subheader("Valuation bridge")
        bridge = st.columns(5)
        bridge[0].metric("PV of 15y FCFF", ui.money(summary["sum_pv_fcff"]))
        bridge[1].metric("PV of terminal", ui.money(summary["pv_terminal_fcff"]),
                         f"{summary['terminal_share_of_ev']:.0%} of EV"
                         if pd.notna(summary["terminal_share_of_ev"]) else None)
        bridge[2].metric("Enterprise value",
                         ui.money(summary["enterprise_value_dcf"]))
        bridge[3].metric("Less net debt", ui.money(-summary["net_debt"]))
        bridge[4].metric("Equity value", ui.money(summary["equity_value_dcf"]))

        st.subheader("Assumptions behind this valuation")
        assumptions = pd.DataFrame([
            {"Driver": "Revenue growth, year 1",
             "Value": summary["base_revenue_growth"], "Format": "pct"},
            {"Driver": "Revenue growth, terminal",
             "Value": summary["terminal_growth"], "Format": "pct"},
            {"Driver": "EBIT margin, year 1",
             "Value": summary["base_ebit_margin"], "Format": "pct"},
            {"Driver": "EBIT margin, terminal",
             "Value": summary["terminal_ebit_margin"], "Format": "pct"},
            {"Driver": "Sector median EBIT margin",
             "Value": summary["sector_median_ebit_margin"], "Format": "pct"},
            {"Driver": "Capex / revenue, year 1",
             "Value": summary["base_capex_to_revenue"], "Format": "pct"},
            {"Driver": "Capex / revenue, terminal",
             "Value": summary["terminal_capex_to_revenue"], "Format": "pct"},
            {"Driver": "WACC", "Value": summary["wacc"], "Format": "pct"},
            {"Driver": "Cost of equity",
             "Value": summary["cost_of_equity"], "Format": "pct"},
            {"Driver": "Cost of debt", "Value": summary["cost_of_debt"], "Format": "pct"},
            {"Driver": "Tax rate", "Value": summary["tax_rate"], "Format": "pct"},
            {"Driver": "Cash-flow sigma (FCFF)",
             "Value": summary["sigma_fcff"], "Format": "num"},
        ])
        assumptions["Value"] = [
            f"{v:.2%}" if f == "pct" else f"{v:.2f}"
            for v, f in zip(assumptions["Value"], assumptions["Format"])]
        st.dataframe(assumptions[["Driver", "Value"]], hide_index=True,
                     width='stretch', key="table_assumptions")

        with st.expander("Full year-by-year forecast"):
            st.dataframe(forecast, hide_index=True, width='stretch', key="table_forecast_full")

# ==========================================================================
with tabs[2]:
    st.container(key="section_competitors")
    st.subheader("Nearest competitors")
    st.caption(
        "Same sector and industry, closest market cap on a log scale - a $10bn "
        "and a $100bn company are not comparable even though both sit far from "
        "a $1tn one on a linear scale. Industry matches rank ahead of "
        "sector-only ones, and the Match column says which you are looking at."
    )
    peers = ui.competitors(universe, selected, count=8)
    if peers.empty:
        st.info("No peers found for this ticker.")
    else:
        subject = universe[universe["ticker"] == selected]
        combined = pd.concat([subject.assign(peer_match="this company"), peers],
                             ignore_index=True)
        peer_columns = ["ticker", "name", "peer_match", "marketcap", "last_close",
                        "pe_ttm", "ev_ebitda_ttm", "ps_ttm", "ebit_margin_ttm",
                        "roic_ttm", "return_1y", "implied_share_price_fcff",
                        "implied_share_price_eps", "upside_fcff", "wacc"]
        view = combined[peer_columns].copy()
        for money_column in ["marketcap"]:
            view[money_column] = view[money_column].map(ui.money)
        for percent_column in ["ebit_margin_ttm", "roic_ttm", "return_1y",
                               "upside_fcff", "wacc"]:
            view[percent_column] = view[percent_column].map(lambda v: ui.percent(v, 1))
        for number_column in ["pe_ttm", "ev_ebitda_ttm", "ps_ttm"]:
            view[number_column] = view[number_column].map(lambda v: ui.number(v, 2))
        for price_column in ["last_close", "implied_share_price_fcff",
                             "implied_share_price_eps"]:
            view[price_column] = view[price_column].map(
                lambda v: f"${v:,.2f}" if pd.notna(v) else "-")
        st.dataframe(view, hide_index=True, width="stretch", column_config={
            "ticker": st.column_config.TextColumn("Ticker", width=70),
            "name": st.column_config.TextColumn("Company", width=190),
            "peer_match": st.column_config.TextColumn("Match", width=130),
            "marketcap": st.column_config.TextColumn("Market cap", width=105),
            "last_close": st.column_config.TextColumn("Price", width=90),
            "pe_ttm": st.column_config.TextColumn("P/E", width=75),
            "ev_ebitda_ttm": st.column_config.TextColumn("EV/EBITDA", width=95),
            "ps_ttm": st.column_config.TextColumn("P/S", width=75),
            "ebit_margin_ttm": st.column_config.TextColumn("EBIT %", width=85),
            "roic_ttm": st.column_config.TextColumn("ROIC", width=80),
            "return_1y": st.column_config.TextColumn("1Y", width=80),
            "implied_share_price_fcff": st.column_config.TextColumn("DCF FCFF",
                                                                    width=98),
            "implied_share_price_eps": st.column_config.TextColumn("DCF EPS",
                                                                   width=95),
            "upside_fcff": st.column_config.TextColumn("Upside", width=88),
            "wacc": st.column_config.TextColumn("WACC", width=80),
        }, key="table_peers")

        chart_metric = st.selectbox(
            "Compare on", ["pe_ttm", "ev_ebitda_ttm", "ps_ttm", "ebit_margin_ttm",
                           "roic_ttm", "return_1y"],
            format_func=lambda c: {"pe_ttm": "P/E", "ev_ebitda_ttm": "EV/EBITDA",
                                   "ps_ttm": "P/S", "ebit_margin_ttm": "EBIT margin",
                                   "roic_ttm": "ROIC", "return_1y": "1-year return"}[c],
            key="peer_metric")
        chart_frame = combined.dropna(subset=[chart_metric])
        bar = go.Figure(go.Bar(
            x=chart_frame["ticker"], y=chart_frame[chart_metric],
            marker_color=[ACCENT if t == selected else MUTED
                          for t in chart_frame["ticker"]]))
        percent_metrics = {"ebit_margin_ttm", "roic_ttm", "return_1y"}
        bar.update_layout(
            height=320, margin=dict(t=20, b=0),
            yaxis_tickformat=".1%" if chart_metric in percent_metrics else None)
        st.plotly_chart(bar, width="stretch", key="chart_peer_compare")

# ==========================================================================
with tabs[3]:
    st.container(key="section_reporting_calendar")
    st.subheader("Next reporting date")
    st.warning(
        "**Estimated, not announced.** The Sharadar Bundle carries no forward "
        "earnings calendar, so this is derived from each company's own filing "
        "cadence — the median gap between its recent filings added to the last "
        "one. Accurate to a few days for regular filers; confirm with the "
        "company's investor-relations page before trading around it.",
        icon="⚠️")

    reports = ui.earnings_calendar(tuple(universe["ticker"].head(400)))
    this_row = reports[reports["ticker"] == selected]
    if this_row.empty:
        st.info("No filing cadence available for this ticker.")
    else:
        record = this_row.iloc[0]
        calendar = st.columns(4)
        calendar[0].metric("Last filing",
                           pd.Timestamp(record["last_filing"]).date().isoformat())
        calendar[1].metric("Typical gap", f"{record['median_gap_days']:.0f} days")
        calendar[2].metric("Next report",
                           pd.Timestamp(record["next_report"]).date().isoformat())
        calendar[3].metric("Days away", f"{record['days_away']:.0f}")
        st.caption(f"Source: **{record['source']}**")

    st.subheader("What the model expects next year")
    st.caption(
        "Analyst consensus is a separate Sharadar product not included in this "
        "subscription, so the market-implied column cannot be populated. Shown "
        "instead: the trailing twelve months as reported, against the DCF's "
        "year-1 projection."
    )
    if not forecast.empty and not cashflow_history.empty:
        latest_actual = cashflow_history.iloc[-1]
        year_one = forecast.iloc[0]
        comparison = pd.DataFrame([
            {"Measure": "Revenue", "Reported TTM": latest_actual["revenue_ttm"],
             "DCF year 1": year_one["revenue"]},
            {"Measure": "FCFF", "Reported TTM": latest_actual["fcff_ttm"],
             "DCF year 1": year_one["fcff"]},
            {"Measure": "FCFE", "Reported TTM": latest_actual["fcfe_ttm"],
             "DCF year 1": year_one["fcfe"]},
        ])
        comparison["Change"] = (comparison["DCF year 1"]
                                / comparison["Reported TTM"].replace(0, np.nan) - 1)
        st.dataframe(
            comparison, hide_index=True, width='stretch',
            column_config={
                "Reported TTM": st.column_config.NumberColumn(format="compact"),
                "DCF year 1": st.column_config.NumberColumn(format="compact"),
                "Change": st.column_config.NumberColumn(format="percent"),
            }, key="table_next_vs_dcf")

    st.subheader("Upcoming across the largest companies")
    soon = reports[reports["days_away"].between(-5, 90)].head(30)
    if soon.empty:
        st.info("Nothing estimated in the next 90 days for the largest names.")
    else:
        soon = soon.merge(
            universe[["ticker", "name", "sector", "industry", "marketcap"]],
            on="ticker", how="left")
        st.dataframe(
            soon[["ticker", "name", "sector", "industry", "marketcap",
                  "last_filing", "next_report", "days_away", "source"]],
            hide_index=True, width='stretch',
            column_config={
                "marketcap": st.column_config.NumberColumn("Mkt cap", format="compact"),
                "next_report": st.column_config.DateColumn("Next report"),
                "source": st.column_config.TextColumn("Source", width=210),
                "days_away": st.column_config.NumberColumn("Days", format="%d"),
            }, key="table_upcoming_reports")

# ==========================================================================
with tabs[4]:
    st.container(key="section_corporate_actions")
    st.subheader("Dividends, splits and ticker changes")
    try:
        actions = ui.corporate_actions(selected)
    except Exception as error:  # the live API can refuse; the page should not die
        st.error(f"Could not load corporate actions: {error}")
        actions = pd.DataFrame()

    if actions.empty:
        st.info("No corporate actions in the last five years.")
    else:
        st.dataframe(actions, hide_index=True, width='stretch', key="table_corporate_actions")
        dividends = actions[actions["action"] == "dividend"].copy()
        if not dividends.empty:
            dividends["date"] = pd.to_datetime(dividends["date"])
            dividends = dividends.sort_values("date")
            figure = go.Figure(go.Bar(x=dividends["date"], y=dividends["value"],
                                      marker_color=ACCENT))
            figure.update_layout(height=280, yaxis_title="Dividend per share ($)",
                                 margin=dict(t=20, b=0))
            st.plotly_chart(figure, width='stretch', key="chart_dividends")

            trailing = dividends[dividends["date"]
                                 >= dividends["date"].max() - pd.DateOffset(years=1)]
            annual = trailing["value"].sum()
            if row.last_close > 0:
                st.metric("Trailing dividend yield", f"{annual / row.last_close:.2%}",
                          f"${annual:,.2f} per share over the last year")

# ==========================================================================
with tabs[5]:
    st.container(key="section_ownership")
    st.subheader("Institutional ownership")
    try:
        holdings = ui.institutional_holdings(selected)
    except Exception as error:
        st.error(f"Could not load holdings: {error}")
        holdings = pd.DataFrame()
    if holdings.empty:
        st.info("No 13F summary available for this ticker.")
    else:
        st.dataframe(holdings.head(40), hide_index=True, width='stretch', key="table_holdings")

    st.subheader("Insider transactions")
    st.caption(
        "SEC Form 4 filings. Cluster buying by officers is a stronger signal "
        "than selling, which is often scheduled compensation rather than a view."
    )
    try:
        insiders = ui.insider_transactions(selected)
    except Exception as error:
        st.error(f"Could not load insider transactions: {error}")
        insiders = pd.DataFrame()
    if insiders.empty:
        st.info("No insider transactions in the last two years.")
    else:
        st.dataframe(insiders.head(60), hide_index=True, width='stretch', key="table_insiders")

# ==========================================================================
with tabs[6]:
    st.container(key="section_risk")
    st.subheader("Risk")
    risk = st.columns(5)
    risk[0].metric("Beta", f"{row.beta:.2f}" if pd.notna(row.beta) else "-")
    risk[1].metric("Alpha (annual)",
                   f"{row.alpha_annual:+.1%}" if pd.notna(row.alpha_annual) else "-")
    risk[2].metric("R² of beta",
                   f"{row.beta_r2:.2f}" if pd.notna(row.beta_r2) else "-",
                   help="How much of the stock's move the market explains. Below "
                        "~0.1 the beta is close to noise.")
    risk[3].metric("Max drawdown 1y",
                   f"{row.max_drawdown_1y:.1%}"
                   if pd.notna(row.max_drawdown_1y) else "-")
    risk[4].metric("Avg daily $ volume", ui.money(row.avg_dollar_volume_3m))

    st.subheader("Leverage and coverage")
    leverage = st.columns(4)
    leverage[0].metric("Debt / equity",
                       f"{row.debt_to_equity:.2f}"
                       if pd.notna(row.debt_to_equity) else "-")
    leverage[1].metric("Net debt / EBITDA",
                       f"{row.net_debt_to_ebitda:.2f}"
                       if pd.notna(row.net_debt_to_ebitda) else "-")
    leverage[2].metric("Interest coverage",
                       f"{row.interest_coverage:.1f}"
                       if pd.notna(row.interest_coverage) else "-")
    leverage[3].metric("Cost of debt",
                       f"{row.cost_of_debt:.2%}" if pd.notna(row.cost_of_debt) else "-")

    st.subheader("Returns against the market")
    prices = ui.load_prices(selected)
    market = ui.load_market_data()
    if not prices.empty:
        prices["date"] = pd.to_datetime(prices["date"])
        market["date"] = pd.to_datetime(market["date"])
        merged = prices[["date", "closeadj"]].merge(
            market[["date", "sp500"]], on="date", how="inner").dropna()
        window = merged[merged["date"] >= merged["date"].max()
                        - pd.DateOffset(years=3)]
        if len(window) > 10:
            figure = go.Figure()
            for column, label, colour in [("closeadj", selected, ACCENT),
                                          ("sp500", "S&P 500", MUTED)]:
                figure.add_trace(go.Scatter(
                    x=window["date"],
                    y=window[column] / window[column].iloc[0] * 100,
                    name=label, line=dict(color=colour, width=2)))
            figure.update_layout(height=340, yaxis_title="Rebased to 100",
                                 hovermode="x unified", margin=dict(t=20, b=0),
                                 legend=dict(orientation="h", y=1.1))
            st.plotly_chart(figure, width='stretch', key="chart_vs_market")

    st.subheader("S&P 500 membership")
    try:
        members = ui.sp500_membership()
        record = members[members["ticker"] == selected]
        if record.empty:
            st.info(f"{selected} is not in the S&P 500 index history held here.")
        else:
            st.dataframe(record.sort_values("date", ascending=False).head(10),
                         hide_index=True, width='stretch', key="table_sp500_membership")
    except Exception as error:
        st.error(f"Could not load S&P 500 membership: {error}")
