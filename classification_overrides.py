#!/usr/bin/env python3
"""
Manual sector / industry overrides.

Sharadar's own classification is what `analytics.parquet` carries, and every
data refresh rewrites that file from the vendor. Anything you correct by hand
therefore has to live somewhere the refresh never touches: this module keeps it
in a single JSON file at the project root, and re-applies it to the analytics
frame every time the app loads it. The vendor label is kept alongside, so an
override can always be undone and you can still see what it replaced.

    {"AAPL": {"sector": "Technology", "industry": "Consumer Hardware"}}

A missing key means "leave the vendor's value alone", so you can override the
sector only, the industry only, or both.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
OVERRIDES_PATH = HERE / "classification_overrides.json"

FIELDS = ("sector", "industry")


def load() -> dict[str, dict[str, str]]:
    """Every stored override, keyed by ticker. Never raises."""
    if not OVERRIDES_PATH.exists():
        return {}
    try:
        raw = json.loads(OVERRIDES_PATH.read_text())
    except (json.JSONDecodeError, OSError):
        return {}
    if not isinstance(raw, dict):
        return {}
    out: dict[str, dict[str, str]] = {}
    for ticker, entry in raw.items():
        if not isinstance(entry, dict):
            continue
        cleaned = {field: str(entry[field]).strip() for field in FIELDS
                   if entry.get(field) and str(entry[field]).strip()}
        if cleaned:
            out[str(ticker).strip().upper()] = cleaned
    return out


def _write(overrides: dict[str, dict[str, str]]) -> None:
    """Atomic write, so an interrupted save cannot truncate the file."""
    payload = json.dumps(dict(sorted(overrides.items())), indent=2) + "\n"
    handle, temporary = tempfile.mkstemp(dir=str(HERE), suffix=".json")
    try:
        with os.fdopen(handle, "w") as file:
            file.write(payload)
        os.replace(temporary, OVERRIDES_PATH)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def set_override(ticker: str, sector: str | None = None,
                 industry: str | None = None) -> dict[str, dict[str, str]]:
    """Store (or clear) one ticker's labels. Blank values drop that field."""
    ticker = str(ticker).strip().upper()
    overrides = load()
    entry = dict(overrides.get(ticker, {}))
    for field, value in (("sector", sector), ("industry", industry)):
        value = (value or "").strip()
        if value:
            entry[field] = value
        else:
            entry.pop(field, None)
    if entry:
        overrides[ticker] = entry
    else:
        overrides.pop(ticker, None)
    _write(overrides)
    return overrides


def clear_override(ticker: str) -> dict[str, dict[str, str]]:
    """Drop every override for one ticker, restoring the vendor's labels."""
    overrides = load()
    if overrides.pop(str(ticker).strip().upper(), None) is not None:
        _write(overrides)
    return overrides


def clear_all() -> None:
    _write({})


def apply(frame: pd.DataFrame,
          overrides: dict[str, dict[str, str]] | None = None) -> pd.DataFrame:
    """Return `frame` with overridden sector / industry labels applied.

    Adds `vendor_sector` / `vendor_industry` (what the data file said) and
    `classification_overridden`, so the UI can mark the rows you have edited
    without having to re-read the JSON per row.
    """
    if frame.empty or "ticker" not in frame:
        return frame
    overrides = load() if overrides is None else overrides
    out = frame.copy()
    for field in FIELDS:
        if field in out:
            out[f"vendor_{field}"] = out[field]
    out["classification_overridden"] = False
    if not overrides:
        return out

    tickers = out["ticker"].astype(str).str.upper()
    for field in FIELDS:
        if field not in out:
            continue
        mapping = {ticker: entry[field] for ticker, entry in overrides.items()
                   if field in entry}
        if not mapping:
            continue
        replacement = tickers.map(mapping)
        touched = replacement.notna()
        out.loc[touched, field] = replacement[touched]
        out.loc[touched, "classification_overridden"] = True
    return out


def describe(ticker: str) -> dict[str, str]:
    """The stored override for one ticker, or an empty dict."""
    return load().get(str(ticker).strip().upper(), {})
