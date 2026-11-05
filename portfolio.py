#!/usr/bin/env python3
"""
A trade blotter, the positions it implies, and how they have performed.

Trades live in `portfolio_trades.csv` at the project root - a plain file no
stage of `update_data.sh` touches, the same arrangement the sector overrides
use. Session state would lose the blotter on every restart, which for a
portfolio is not a limitation but a different, useless product.

Three things here are decisions rather than arithmetic, so they are stated
rather than buried:

Cost basis is **average cost**. A sale realises the gain against the running
average of what is still held, not against a specific lot. Specific-lot and
FIFO matching give different realised/unrealised splits and different tax
answers; average cost is the one that needs no lot selection from the user and
is what a broker's "average price" column shows. The *total* P&L is identical
under all three - only the split between realised and unrealised moves.

Quantities are **restated for splits**. A trade is entered the way it was
actually done: 10 shares at $1,200, not 100 at $120. Every price series here is
split-adjusted, so a raw quantity times an adjusted price would misvalue the
position by the split factor. Sharadar carries both `close` (adjusted) and
`closeunadj` (as printed), and their ratio on the trade date is exactly the
cumulative split factor since - so the quantity is multiplied by it and the
price divided by it. Cost is unchanged; only the units move.

Return is **time-weighted**, not the change in account value. Money put in is
not a gain, so each day's return is measured after removing that day's trade
cash flow:

    r_t = (V_t - V_(t-1) - net_invested_t) / V_(t-1)

which is what makes the portfolio line comparable to an index line drawn
beside it, and what makes its volatility a property of the holdings rather
than of the deposit schedule.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
TRADES_CSV = HERE / "portfolio_trades.csv"

BUY, SELL = "Buy", "Sell"
SIDES = (BUY, SELL)

TRADE_COLUMNS = ["trade_id", "date", "ticker", "side", "quantity", "price",
                 "fees", "note"]

TRADING_DAYS = 252


# --------------------------------------------------------------------------
# the blotter
# --------------------------------------------------------------------------

def empty_trades() -> pd.DataFrame:
    frame = pd.DataFrame(columns=TRADE_COLUMNS)
    return frame.astype({"trade_id": "int64", "ticker": "object",
                         "side": "object", "quantity": "float64",
                         "price": "float64", "fees": "float64",
                         "note": "object"})


def load_trades() -> pd.DataFrame:
    """The blotter, oldest first. A missing or unreadable file is an empty one.

    Unreadable rather than absent matters here: this file is meant to be
    hand-editable, so a half-saved edit is a realistic state. Losing the page
    to a parse error would make the file harder to fix, not easier, so a bad
    file reads as empty and the page says so.
    """
    if not TRADES_CSV.exists():
        return empty_trades()
    try:
        frame = pd.read_csv(TRADES_CSV)
    except Exception:
        return empty_trades()
    if frame.empty:
        return empty_trades()

    for column in TRADE_COLUMNS:
        if column not in frame.columns:
            frame[column] = np.nan
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame["ticker"] = frame["ticker"].astype(str).str.strip().str.upper()
    frame["side"] = frame["side"].where(frame["side"].isin(SIDES), BUY)
    for column in ("quantity", "price", "fees"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce").fillna(0.0)
    frame["note"] = frame["note"].fillna("")
    frame = frame.dropna(subset=["date"])
    frame = frame[frame["quantity"] > 0]
    return (frame[TRADE_COLUMNS].sort_values(["date", "trade_id"])
            .reset_index(drop=True))


def save_trades(frame: pd.DataFrame) -> None:
    out = frame.copy()
    out["date"] = pd.to_datetime(out["date"]).dt.strftime("%Y-%m-%d")
    out[TRADE_COLUMNS].to_csv(TRADES_CSV, index=False)


def add_trade(ticker: str, date, side: str, quantity: float, price: float,
              fees: float = 0.0, note: str = "") -> pd.DataFrame:
    trades = load_trades()
    next_id = int(trades["trade_id"].max()) + 1 if not trades.empty else 1
    new = pd.DataFrame([{
        "trade_id": next_id, "date": pd.Timestamp(date),
        "ticker": str(ticker).strip().upper(),
        "side": side if side in SIDES else BUY,
        "quantity": float(quantity), "price": float(price),
        "fees": float(fees), "note": note}])
    trades = pd.concat([trades, new], ignore_index=True)
    save_trades(trades)
    return trades


def delete_trades(trade_ids) -> pd.DataFrame:
    trades = load_trades()
    trades = trades[~trades["trade_id"].isin(list(trade_ids))]
    save_trades(trades)
    return trades


# --------------------------------------------------------------------------
# split restatement
# --------------------------------------------------------------------------

def split_factors(trades: pd.DataFrame,
                  price_panel: dict[str, pd.DataFrame]) -> pd.Series:
    """Cumulative split factor between each trade date and today, per trade.

    `closeunadj / close` on a given session is how many of today's shares one
    share of that session became. It is 1.0 for a ticker that has never split,
    10.0 on the day before NVDA's ten-for-one, and it is read off the price
    file rather than off a corporate-actions feed so it cannot disagree with
    the series the position is later valued against.
    """
    factors = pd.Series(1.0, index=trades.index)
    for ticker, group in trades.groupby("ticker"):
        prices = price_panel.get(ticker)
        if prices is None or prices.empty:
            continue
        if not {"close", "closeunadj"} <= set(prices.columns):
            continue
        series = prices[["date", "close", "closeunadj"]].dropna().copy()
        series["date"] = pd.to_datetime(series["date"])
        series = series[series["close"] > 0].sort_values("date")
        if series.empty:
            continue
        series["factor"] = series["closeunadj"] / series["close"]

        wanted = pd.DataFrame({"date": pd.to_datetime(group["date"])},
                              index=group.index).sort_values("date")
        matched = pd.merge_asof(wanted.reset_index(), series[["date", "factor"]],
                                on="date", direction="nearest")
        factors.loc[matched["index"]] = matched["factor"].fillna(1.0).to_numpy()
    return factors.replace(0.0, 1.0).fillna(1.0)


def restate_for_splits(trades: pd.DataFrame,
                       price_panel: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Trades in today's share terms. Cost is preserved exactly."""
    out = trades.copy()
    factor = split_factors(out, price_panel)
    out["split_factor"] = factor
    out["quantity_adj"] = out["quantity"] * factor
    out["price_adj"] = out["price"] / factor
    return out


# --------------------------------------------------------------------------
# positions
# --------------------------------------------------------------------------

def _run_average_cost(group: pd.DataFrame) -> dict:
    """Walk one ticker's trades, keeping a running average cost.

    Fees are capitalised into the book on a buy and deducted from proceeds on
    a sell, which is how a broker reports them and what makes the realised
    figure the cash answer rather than the gross one.

    A sale larger than the holding is not rejected - a blotter that refuses
    the trade you actually did is worse than one that shows it. It is capped
    for the realised calculation and leaves a short position, which the page
    flags.
    """
    quantity = 0.0
    book = 0.0          # total cost of what is still held
    realised = 0.0
    invested = 0.0      # gross cash put in, for a cost-basis line

    for _, trade in group.iterrows():
        size = float(trade["quantity_adj"])
        price = float(trade["price_adj"])
        fees = float(trade["fees"])
        if trade["side"] == BUY:
            quantity += size
            book += size * price + fees
            invested += size * price + fees
        else:
            average = book / quantity if quantity > 0 else price
            closing = min(size, quantity) if quantity > 0 else 0.0
            realised += closing * (price - average) - fees
            book -= closing * average
            quantity -= size
            invested -= size * price - fees

    return {"quantity": quantity,
            "cost_basis": book,
            "avg_cost": book / quantity if quantity > 0 else np.nan,
            "realised_pnl": realised,
            "net_invested": invested}


def positions(trades: pd.DataFrame, price_panel: dict[str, pd.DataFrame],
              analytics: pd.DataFrame | None = None) -> pd.DataFrame:
    """One row per ticker ever traded, valued at the latest close."""
    if trades.empty:
        return pd.DataFrame()

    restated = restate_for_splits(trades, price_panel)
    rows = []
    for ticker, group in restated.sort_values(["date", "trade_id"]).groupby(
            "ticker"):
        row = {"ticker": ticker, "trades": len(group),
               "first_trade": group["date"].min(),
               "last_trade": group["date"].max()}
        row.update(_run_average_cost(group))
        rows.append(row)

    frame = pd.DataFrame(rows)

    last = {t: _last_close(price_panel.get(t)) for t in frame["ticker"]}
    frame["last_close"] = frame["ticker"].map(last)
    frame["market_value"] = frame["quantity"] * frame["last_close"]
    frame["unrealised_pnl"] = frame["market_value"] - frame["cost_basis"]
    frame["total_pnl"] = frame["unrealised_pnl"] + frame["realised_pnl"]
    # Against cost, not against market value: the question a P&L percentage
    # answers is what the money put in has done.
    frame["return_pct"] = np.where(
        frame["cost_basis"] > 0,
        frame["unrealised_pnl"] / frame["cost_basis"].replace(0, np.nan),
        np.nan)
    frame["volatility_1y"] = frame["ticker"].map(
        lambda t: realised_volatility(price_panel.get(t)))

    total_value = frame.loc[frame["quantity"] > 0, "market_value"].sum()
    frame["weight"] = np.where(total_value > 0,
                               frame["market_value"] / total_value, np.nan)
    frame.loc[frame["quantity"] <= 0, "weight"] = np.nan

    if analytics is not None and not analytics.empty:
        labels = analytics[["ticker", "name", "sector", "industry"]]
        frame = frame.merge(labels, on="ticker", how="left")

    frame["is_open"] = frame["quantity"] > 1e-9
    return frame.sort_values(["is_open", "market_value"],
                             ascending=[False, False]).reset_index(drop=True)


def _last_close(prices: pd.DataFrame | None) -> float:
    if prices is None or prices.empty or "close" not in prices.columns:
        return np.nan
    series = prices["close"].dropna()
    return float(series.iloc[-1]) if not series.empty else np.nan


def realised_volatility(prices: pd.DataFrame | None,
                        window: int = TRADING_DAYS) -> float:
    """Annualised standard deviation of daily returns over the last year.

    On `closeadj`, not `close`: a volatility is a property of the total return
    an investor experienced, and a dividend ex-date is a price drop that was
    never a loss. The position *value* series elsewhere uses `close`, because
    the position really is worth less once the cash has left the company.
    """
    if prices is None or prices.empty:
        return np.nan
    column = "closeadj" if "closeadj" in prices.columns else "close"
    series = prices[column].dropna()
    if len(series) < 20:
        return np.nan
    daily = series.pct_change().dropna().tail(window)
    if len(daily) < 20:
        return np.nan
    return float(daily.std() * np.sqrt(TRADING_DAYS))


# --------------------------------------------------------------------------
# history
# --------------------------------------------------------------------------

def holdings_history(trades: pd.DataFrame,
                     price_panel: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Daily market value, cost basis and cash flow, per ticker and in total.

    Long format - one row per (date, ticker) plus a `PORTFOLIO` total - so the
    chart can plot either without a second pass over the trades.
    """
    if trades.empty:
        return pd.DataFrame()

    restated = restate_for_splits(trades, price_panel)
    start = pd.to_datetime(restated["date"]).min()

    frames = []
    for ticker, group in restated.groupby("ticker"):
        prices = price_panel.get(ticker)
        if prices is None or prices.empty:
            continue
        series = prices[["date", "close"]].dropna().copy()
        series["date"] = pd.to_datetime(series["date"])
        series = series[series["date"] >= start].sort_values("date")
        if series.empty:
            continue

        signed = np.where(group["side"] == BUY, 1.0, -1.0)
        moves = pd.DataFrame({
            "date": pd.to_datetime(group["date"]),
            "shares": signed * group["quantity_adj"].to_numpy(),
            # Cash out of pocket: a buy costs the consideration plus fees, a
            # sale returns the consideration less fees.
            "cash": (signed * group["quantity_adj"].to_numpy()
                     * group["price_adj"].to_numpy()) + group["fees"].to_numpy(),
        })
        # A trade dated on a non-session day (a weekend entry, or a date typed
        # from a confirmation) is carried to the next session rather than
        # dropped, so its cash flow still lands in the return series.
        moves = pd.merge_asof(moves.sort_values("date"),
                              series[["date"]].assign(session=series["date"]),
                              on="date", direction="forward")
        moves["session"] = moves["session"].fillna(series["date"].iloc[-1])
        daily = moves.groupby("session")[["shares", "cash"]].sum()

        series = series.set_index("date")
        series["shares_delta"] = daily["shares"].reindex(series.index).fillna(0.0)
        series["cash_flow"] = daily["cash"].reindex(series.index).fillna(0.0)
        series["shares"] = series["shares_delta"].cumsum()
        series["market_value"] = series["shares"] * series["close"]
        series["ticker"] = ticker
        frames.append(series.reset_index()[
            ["date", "ticker", "close", "shares", "market_value", "cash_flow"]])

    if not frames:
        return pd.DataFrame()

    per_ticker = pd.concat(frames, ignore_index=True)
    total = (per_ticker.groupby("date")[["market_value", "cash_flow"]]
             .sum().reset_index())
    total["ticker"] = "PORTFOLIO"
    total["shares"] = np.nan
    total["close"] = np.nan
    return pd.concat([per_ticker, total], ignore_index=True).sort_values(
        ["ticker", "date"]).reset_index(drop=True)


def performance(history: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """One series' value, cumulative time-weighted return and running P&L.

    The daily return removes that day's trade cash flow before dividing, so
    money paid in is not counted as a gain. Days where the position was flat
    at zero - before the first trade, or between a full exit and a re-entry -
    return 0% rather than a division by zero, which keeps the compounded line
    continuous across a gap instead of restarting it.
    """
    series = history[history["ticker"] == ticker].sort_values("date").copy()
    if series.empty:
        return series

    value = series["market_value"].to_numpy(dtype="float64")
    flow = series["cash_flow"].to_numpy(dtype="float64")
    previous = np.concatenate([[0.0], value[:-1]])

    with np.errstate(divide="ignore", invalid="ignore"):
        daily = np.where(previous > 0, (value - previous - flow) / previous, 0.0)
    daily = np.nan_to_num(daily, nan=0.0, posinf=0.0, neginf=0.0)

    series["daily_return"] = daily
    series["cumulative_return"] = np.cumprod(1.0 + daily) - 1.0
    series["invested"] = flow.cumsum()
    series["pnl"] = series["market_value"] - series["invested"]
    return series


def summary(history: pd.DataFrame, ticker: str = "PORTFOLIO") -> dict:
    """Headline numbers for one series, for the metric strip above the chart."""
    series = performance(history, ticker)
    if series.empty:
        return {}
    daily = series["daily_return"].to_numpy()
    live = daily[series["market_value"].to_numpy() > 0]
    level = np.cumprod(1.0 + daily)
    drawdown = level / np.maximum.accumulate(level) - 1.0
    return {
        "market_value": float(series["market_value"].iloc[-1]),
        "invested": float(series["invested"].iloc[-1]),
        "pnl": float(series["pnl"].iloc[-1]),
        "cumulative_return": float(series["cumulative_return"].iloc[-1]),
        "volatility": (float(np.std(live[-TRADING_DAYS:]) * np.sqrt(TRADING_DAYS))
                       if len(live) >= 20 else np.nan),
        "max_drawdown": float(drawdown.min()) if len(drawdown) else np.nan,
        "sessions": int(len(series)),
    }
