"""
ngx_pulse_service.py

Live market-data client for NGX Pulse (ngxpulse.ng) -- the deployed bot/
API's live price source (project decision, confirmed: NGX Pulse stays the
live source, stockanalysis.com scraping in fundamentals_service.py is
unrelated and unchanged). Updates every ~20 minutes during the trading
session per project history.

Unreachable from this sandbox (not in the network egress allowlist -- same
constraint as stockanalysis.com, api.telegram.org, supabase.co). Tested via
replay of real captured responses in real_data_helpers.py, same discipline
as test_fundamentals_real_data.py and test_screen_real_data.py: only
requests.get() is mocked, all parsing/business logic below runs for real.

AUTH -- CONFIRMED (2026-08-29, against a real prior successful live call):
base URL is https://ngxpulse.ng (unchanged), and auth is an `X-API-Key`
header, NOT `Authorization: Bearer` -- the previous default in this file
was wrong and would have 401'd on first live deployment. NGX_PULSE_BASE_URL
and NGX_PULSE_API_KEY are still read from environment variables so this
can be pointed at the real endpoint without a code change; set
NGX_PULSE_API_KEY on Render (and in Colab for testing) to a real key.
The specific key used to confirm this was pasted in a chat session and
should be rotated before relying on it in production.

SCOPE DECISION (confirmed with the user, changes what "quota-aware" means
here): the bot's live price needs are for symbols actually HELD across
active portfolios -- in practice a small, overlapping set (~10 symbols),
not the full ~150-300 name NGX universe. This matters because:

  - Current prices for ANY set of symbols cost exactly ONE NGX Pulse
    request: /api/ngxdata/stocks returns the FULL market snapshot in a
    single call (confirmed real: 147 unique symbols in one distinct real
    pull, per the data captured this session), and the caller filters
    down to whatever symbols it actually needs client-side. There is no
    per-symbol cost for "current price" -- get_current_prices_and_sectors()
    below is the workhorse for live portfolio valuation and can run as
    often as needed, well within the 100-request/day quota, regardless of
    how many portfolios or holdings exist.

  - Historical price SERIES (needed for momentum/vol screening, 130-day
    lookback) DOES cost one request per symbol
    (/api/ngxdata/prices/{symbol}), and IS genuinely quota-sensitive -- but
    only for full-universe screening runs (periodic, e.g. the monthly
    rebalance review per project policy), not the day-to-day live price
    refresh loop. Fetching history for just a portfolio's ~10 held names
    is cheap regardless (well under the 100/day cap even run daily).

Given this, get_price_series_for_symbols() below uses a simple in-memory
per-call request budget (stop after max_requests, report which symbols
were skipped so the caller can resume later), NOT a persistent DB-backed
quota tracker. This is intentionally simpler than the fundamentals_cache
pattern in db.py -- revisit only if full-universe screening needs to run
more than once/day in practice, which isn't the case today.

Documented API shape (per project history + confirmed against real
captured responses this session):
  - GET /api/ngxdata/stocks  -- {"stocks": [...]}, one row per symbol:
    symbol, name, current_price, previous_close, change_percent, volume,
    market_cap, shares_outstanding, sector, market, trade_date.
  - GET /api/ngxdata/market  -- {"data": {...}}, broader market stats.
  - GET /api/ngxdata/prices/{symbol}?days=N -- {"success": bool, "symbol":
    str, "prices": [{"trade_date", "close_price", ...}, ...], "count": int}.
    days is HARD-CAPPED at 1,000 by NGX Pulse itself; oldest data ~May 2022.
  - GET /health -- discovery endpoint, lists valid route patterns.

TODO (confirm with the user / against a real successful Colab call before
first live deployment): the exact base URL and whether the Personal tier
requires an API key/auth header at all -- NGX_PULSE_BASE_URL defaults to
"https://ngxpulse.ng" and NGX_PULSE_API_KEY defaults to unset (no auth
header sent) below, both overridable via environment variables, but neither
was independently re-verified in this session; project history documents
the *response shapes* (confirmed real, used throughout this module) but not
a live-tested base URL/auth combination from within this codebase.
"""

import os
from typing import Iterable, Optional

import pandas as pd
import requests

NGX_PULSE_BASE_URL = os.environ.get("NGX_PULSE_BASE_URL", "https://ngxpulse.ng").rstrip("/")
NGX_PULSE_API_KEY = os.environ.get("NGX_PULSE_API_KEY", "")
REQUEST_TIMEOUT = 15
MAX_DAYS_PER_REQUEST = 1000   # NGX Pulse's own documented hard cap on ?days=
DAILY_REQUEST_QUOTA = 100     # NGX Pulse Personal tier's documented daily cap
DEFAULT_LOOKBACK_DAYS = 130   # matches screening_service's momentum/vol window


def _headers() -> dict:
    """NGX Pulse expects the key on the X-API-Key header, confirmed
    against a real successful live call -- NOT Authorization: Bearer,
    which was this function's incorrect prior default and would have
    401'd on first live deployment."""
    headers = {"User-Agent": "Mozilla/5.0 (compatible; PMCStrategyBot/1.0)"}
    if NGX_PULSE_API_KEY:
        headers["X-API-Key"] = NGX_PULSE_API_KEY
    return headers


def fetch_stocks_snapshot() -> pd.DataFrame:
    """One request, full market snapshot -- current_price, sector, market,
    etc. for every listed symbol. Raises requests.RequestException on
    network failure (caller decides how to handle; this is the raw fetch,
    not the fail-soft convenience wrapper -- see get_current_prices_and_sectors)."""
    url = f"{NGX_PULSE_BASE_URL}/api/ngxdata/stocks"
    resp = requests.get(url, headers=_headers(), timeout=REQUEST_TIMEOUT)
    resp.raise_for_status()
    return pd.DataFrame(resp.json()["stocks"])


def fetch_market_snapshot() -> dict:
    """/api/ngxdata/market -- data nested under 'data'. Not per-symbol,
    also a single request regardless of caller needs."""
    url = f"{NGX_PULSE_BASE_URL}/api/ngxdata/market"
    resp = requests.get(url, headers=_headers(), timeout=REQUEST_TIMEOUT)
    resp.raise_for_status()
    return resp.json()["data"]


def fetch_price_history_raw(symbol: str, days: int = DEFAULT_LOOKBACK_DAYS) -> dict:
    """Raw /api/ngxdata/prices/{symbol} call. Raises ValueError BEFORE
    making the request if days exceeds NGX Pulse's hard cap, rather than
    silently truncating -- a silent truncation could look to
    screening_service like "not enough history" for a reason the caller
    never sees, which is worse than failing loudly here."""
    if days > MAX_DAYS_PER_REQUEST:
        raise ValueError(f"days={days} exceeds NGX Pulse's hard cap of {MAX_DAYS_PER_REQUEST}")
    url = f"{NGX_PULSE_BASE_URL}/api/ngxdata/prices/{symbol}"
    resp = requests.get(url, headers=_headers(), params={"days": days}, timeout=REQUEST_TIMEOUT)
    resp.raise_for_status()
    return resp.json()


def get_price_series(symbol: str, days: int = DEFAULT_LOOKBACK_DAYS) -> Optional[pd.Series]:
    """Fail-soft wrapper around fetch_price_history_raw: returns a
    pd.Series indexed by trade_date (ascending, oldest -> newest), values
    = close_price -- exactly the shape
    screening_service.compute_momentum_vol_from_price_series expects.
    Returns None (not an exception) if the symbol has no usable data
    (network failure, success=False, or an empty/malformed prices list) --
    a caller screening many symbols shouldn't have one bad symbol kill the
    whole run, matching fundamentals_service's fail-soft convention at the
    top-level function."""
    try:
        raw = fetch_price_history_raw(symbol, days=days)
    except (requests.RequestException, ValueError):
        return None
    if not raw.get("success") or not raw.get("prices"):
        return None
    df = pd.DataFrame(raw["prices"])
    if "trade_date" not in df.columns or "close_price" not in df.columns:
        return None
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    df = df.sort_values("trade_date")
    if df["close_price"].isna().all():
        return None
    return pd.Series(df["close_price"].values, index=df["trade_date"].values, name=symbol)


def get_current_prices_and_sectors(symbols: Optional[Iterable[str]] = None) -> tuple:
    """THE function for live portfolio valuation. One NGX Pulse request
    regardless of how many symbols are requested -- fetches the full
    market snapshot and filters client-side. Pass the union of all
    currently-held symbols across active portfolios (typically ~10) to
    value live holdings, or omit `symbols` entirely to get the full
    universe (used when screening). Returns (current_price_by_symbol,
    sector_by_symbol) -- the exact two dicts main.py's UniverseInput (and
    the stateful /rebalance-check path) need. Symbols not found in the
    live snapshot (delisted, typo, etc.) are simply absent from both
    returned dicts rather than raising -- the caller can detect this by
    checking which requested symbols are missing."""
    df = fetch_stocks_snapshot()
    if symbols is not None:
        wanted = set(symbols)
        df = df[df["symbol"].isin(wanted)]
    current_price_by_symbol = dict(zip(df["symbol"], df["current_price"]))
    sector_by_symbol = dict(zip(df["symbol"], df["sector"]))
    return current_price_by_symbol, sector_by_symbol


def get_price_series_for_symbols(
    symbols: Iterable[str],
    days: int = DEFAULT_LOOKBACK_DAYS,
    max_requests: int = DAILY_REQUEST_QUOTA,
) -> tuple:
    """Quota-aware batch fetch of price HISTORY for a set of symbols --
    genuinely per-symbol cost, unlike current prices above. Used for
    screening (full universe -- the case this quota guard actually
    protects) or for a portfolio's held names specifically (~10, cheap
    regardless). Stops once max_requests calls have been made rather than
    running until NGX Pulse returns 429 -- per documented project
    convention, a 429 here means the DAILY cap is exhausted (not a
    transient rate limit), so the correct response is to stop, not retry.
    Returns (price_series_by_symbol, skipped_symbols): skipped_symbols
    covers BOTH symbols skipped due to the request budget AND symbols that
    returned no usable data, so a caller can distinguish "try again later"
    from "genuinely no data" only by re-checking count vs. budget if
    needed -- kept simple deliberately since both cases currently resolve
    the same way (retry on a later run)."""
    price_series_by_symbol = {}
    skipped = []
    requests_made = 0
    for symbol in symbols:
        if requests_made >= max_requests:
            skipped.append(symbol)
            continue
        series = get_price_series(symbol, days=days)
        requests_made += 1
        if series is not None:
            price_series_by_symbol[symbol] = series
        else:
            skipped.append(symbol)
    return price_series_by_symbol, skipped
