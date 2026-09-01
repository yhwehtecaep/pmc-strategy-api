"""
price_history_service.py

Works around koboterminal.com's Free-plan restriction (confirmed
2026-08-31: historical price requests are capped at days<=7; a single
days=130 call 403s uniformly on every symbol -- see ngx_pulse_service.py's
module docstring for the full incident writeup) by accumulating price
history locally in db.py's price_history table, a small days<=7 pull at a
time, rather than depending on one live call to supply the full 130-day
screening lookback.

THREE JOBS, kept in one module because they share the same table and the
same "raw NGX Pulse/Kobo Terminal prices row -> db row" shape, but each is
independently callable:

  1. seed_from_repo_panel()   -- ONE-TIME backfill from the project's
     existing raw_price_history.json (already has real history back to
     Feb 2026, committed to the cfa-challenge repo), so the accumulated
     window starts near-full instead of empty. Safe to re-run (upserts).

  2. run_daily_price_update() -- pulls days=7 per symbol (safely under
     the Free-plan cap, with 6 days of overlap as a buffer against a
     missed run) and upserts into price_history. Intended to run once a
     day (new GitHub Actions cron, not yet wired -- see module bottom).

  3. load_price_series_by_symbol() -- reads the accumulated table back
     out as {symbol: pd.Series}, the exact shape
     screening_service.screen_universe expects, replacing
     ngx_pulse_service.get_price_series_for_symbols as the caller
     monthly_screen_service.py would use once this is wired in.

NOT YET DONE (deliberately, pending confirmation): wiring #2 into a
scheduled trigger (a daily GitHub Actions workflow calling a new main.py
endpoint, same pattern as monthly-screen.yml) and swapping
monthly_screen_service.run_monthly_screen_and_notify to call #3 instead of
nps.get_price_series_for_symbols directly. Both touch live production
routing/scheduling and are left for a separate, explicit step rather than
guessed at here.
"""

import json
import logging
from datetime import date, datetime, timedelta
from typing import Iterable, Optional

import pandas as pd
import requests

import db
import ngx_pulse_service as nps

logger = logging.getLogger(__name__)

DAILY_FETCH_DAYS = 7   # Free-plan hard cap, confirmed 2026-08-31 -- do not raise without a plan upgrade
SCREENING_LOOKBACK_DAYS = 130   # matches screening_service's momentum/vol window; used to trim on read

REPO_RAW_PRICE_HISTORY_URL = (
    "https://raw.githubusercontent.com/yhwehtecaep/cfa-challenge/main/raw_price_history.json"
)


def get_symbols_to_update() -> list:
    """Union of the current live universe (so new entrants get picked up
    as soon as they qualify) and everything already being tracked in
    price_history (so a symbol that temporarily drops out of the live
    universe filter -- e.g. one quiet trading day pushing it under the
    volume threshold -- doesn't lose its accumulated window and have to
    restart from zero). Costs exactly one extra NGX Pulse /stocks snapshot
    request (not quota-limited -- see ngx_pulse_service.py's docstring),
    used only to decide WHICH symbols to update, not to fetch their price
    history."""
    import universe_service as us
    live_symbols, _sectors, _prices = us.build_live_universe()
    tracked_symbols = db.get_price_history_symbols()
    return sorted(set(live_symbols) | set(tracked_symbols))


# ---------------------------------------------------------------------
# 1. One-time seed from the repo's existing historical panel
# ---------------------------------------------------------------------

def seed_from_repo_panel(url: str = REPO_RAW_PRICE_HISTORY_URL) -> dict:
    """Loads raw_price_history.json (list of {"symbol", "raw_response":
    {"prices": [{"trade_date", "close_price", "volume", ...}, ...]}}) and
    upserts every row into price_history. Idempotent -- safe to re-run;
    existing (symbol, trade_date) rows are just re-confirmed, not
    duplicated (see db.upsert_price_history).

    Returns {"symbols_seeded": int, "rows_upserted": int} for visibility --
    this can take a while for ~74 symbols x ~130 rows each, so the caller
    (a one-off script, not a request-path call) gets something to log.
    """
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    data = resp.json()

    symbols_seeded = 0
    rows_upserted = 0
    for entry in data:
        symbol = entry.get("symbol")
        prices = entry.get("raw_response", {}).get("prices", [])
        if not symbol or not prices:
            continue
        rows = []
        for p in prices:
            if p.get("trade_date") is None or p.get("close_price") is None:
                continue
            rows.append({
                "trade_date": pd.to_datetime(p["trade_date"]).date(),
                "close_price": float(p["close_price"]),
                "volume": float(p["volume"]) if p.get("volume") is not None else None,
            })
        if not rows:
            continue
        db.upsert_price_history(symbol, rows)
        symbols_seeded += 1
        rows_upserted += len(rows)

    logger.info("Seeded price_history: %d symbols, %d rows.", symbols_seeded, rows_upserted)
    return {"symbols_seeded": symbols_seeded, "rows_upserted": rows_upserted}


# ---------------------------------------------------------------------
# 2. Daily accumulation (days<=7 per symbol, meant to run once/day)
# ---------------------------------------------------------------------

def run_daily_price_update(symbols: Iterable[str], days: int = DAILY_FETCH_DAYS) -> dict:
    """Pulls up to `days` (default 7, the Free-plan cap) of recent price
    history per symbol and upserts into price_history. Deliberately does
    NOT use nps.get_price_series_for_symbols (that returns a pd.Series
    with volume already discarded) -- goes one level lower, to
    nps.fetch_price_history_raw, so volume is preserved for the DB row.

    Reuses ngx_pulse_service's existing logging (added 2026-08-31) for
    per-symbol failure visibility -- a 403/network failure on one symbol
    here logs the real reason and just skips that symbol for today,
    consistent with the rest of this project's fail-soft-but-not-silent
    convention.

    Returns {"symbols_updated": int, "symbols_failed": list} for the
    caller (a daily cron job) to log/alert on.
    """
    if days > DAILY_FETCH_DAYS:
        logger.warning(
            "run_daily_price_update called with days=%d > the confirmed Free-plan cap of %d -- "
            "this will 403 on every symbol unless the plan has since been upgraded.",
            days, DAILY_FETCH_DAYS,
        )

    updated = []
    failed = []
    for symbol in symbols:
        try:
            raw = nps.fetch_price_history_raw(symbol, days=days)
        except Exception as e:  # noqa: BLE001 -- one bad symbol must not stop the daily run
            logger.warning("Daily price update failed for %s: %s", symbol, e)
            failed.append(symbol)
            continue

        if not raw.get("success") or not raw.get("prices"):
            logger.warning("Daily price update for %s returned no usable prices.", symbol)
            failed.append(symbol)
            continue

        rows = []
        for p in raw["prices"]:
            if p.get("trade_date") is None or p.get("close_price") is None:
                continue
            rows.append({
                "trade_date": pd.to_datetime(p["trade_date"]).date(),
                "close_price": float(p["close_price"]),
                "volume": float(p["volume"]) if p.get("volume") is not None else None,
            })
        if not rows:
            failed.append(symbol)
            continue

        db.upsert_price_history(symbol, rows)
        updated.append(symbol)

    logger.info("Daily price update: %d updated, %d failed.", len(updated), len(failed))
    return {"symbols_updated": updated, "symbols_failed": failed}


# ---------------------------------------------------------------------
# 3. Assemble accumulated rows back into screening_service's input shape
# ---------------------------------------------------------------------

def load_price_series_by_symbol(
    symbols: Iterable[str],
    lookback_days: int = SCREENING_LOOKBACK_DAYS,
    min_points: int = 100,
) -> tuple:
    """Reads price_history for each symbol and returns
    {symbol: pd.Series} in exactly the shape
    screening_service.compute_momentum_vol_from_price_series expects
    (indexed by date, most recent last) -- this is the drop-in replacement
    for nps.get_price_series_for_symbols once the daily accumulation job
    has been running long enough to have real coverage.

    min_points mirrors compute_momentum_vol_from_price_series's own
    internal >=100 requirement -- symbols with fewer accumulated points
    than that are reported separately (as "insufficient") rather than
    silently included with a too-short series, so the caller can tell
    "still accumulating, check back later" apart from "genuinely no data".

    Returns (price_series_by_symbol, insufficient_symbols).
    """
    since = date.today() - timedelta(days=int(lookback_days * 1.6))
    # *1.6 buffer: lookback_days is a TRADING-day target but trade_date
    # rows are calendar days (weekends/holidays have no rows) -- pulling a
    # wider calendar window and letting compute_momentum_vol_from_price_series's
    # own >=100 check be the real gate avoids under-fetching near NGX
    # holidays, at the cost of occasionally reading a few extra old rows.

    price_series_by_symbol = {}
    insufficient = []
    for symbol in symbols:
        rows = db.get_price_history(symbol, since=since)
        if len(rows) < min_points:
            insufficient.append(symbol)
            continue
        idx = pd.to_datetime([r["trade_date"] for r in rows])
        series = pd.Series([float(r["close_price"]) for r in rows], index=idx, name=symbol)
        series = series.sort_index()
        price_series_by_symbol[symbol] = series

    return price_series_by_symbol, insufficient
