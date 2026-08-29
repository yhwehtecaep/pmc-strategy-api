"""
test_ngx_pulse_service.py

Validates ngx_pulse_service.py's actual, unmodified parsing and business
logic against real captured NGX Pulse data:
  - /api/ngxdata/stocks -- reconstructed from ngx_daily_prices.csv's real
    2026-08-04 pull (147 real unique symbols, real prices/sectors),
    re-wrapped in the {"stocks": [...]} envelope a real response would
    have. (Filtered on the file's 'Date' pull-timestamp column, not
    'trade_date' -- see real_data_helpers.build_mock_stocks_response's
    docstring for why: trade_date repeats across pulls on non-trading
    days, which would otherwise double-count every symbol.)
  - /api/ngxdata/prices/{symbol} -- replayed directly from
    raw_price_history.json, real captured responses for 73 real symbols.
  - /api/ngxdata/market -- NOT backed by a real captured response (none was
    pulled this session); the one test for fetch_market_snapshot() uses a
    hand-built mock consistent with the documented {"data": {...}} shape
    only, flagged explicitly below as the one exception to "real data only"
    in this file.

Only requests.get() is patched (real ngxpulse.ng is unreachable from this
sandbox, same constraint as stockanalysis.com/api.telegram.org/
supabase.co); a URL-aware dispatcher routes each call to the right replay,
also counting calls so tests can assert on the "one request regardless of
symbol count" and quota-cutoff claims made in ngx_pulse_service.py's
docstring.
"""

import sys
from unittest.mock import patch

sys.path.insert(0, "/home/claude/pmc_api")

import pandas as pd

import ngx_pulse_service as nps
import real_data_helpers as rdh

PASS, FAIL = [], []


def check(name, cond, detail=""):
    if cond:
        PASS.append(name)
        print(f"PASS: {name}")
    else:
        FAIL.append((name, detail))
        print(f"FAIL: {name} -- {detail}")


# =======================================================================
# Fake requests.get dispatcher -- routes by URL, replays real data, counts calls.
# =======================================================================

class _FakeResponse:
    def __init__(self, json_data, status_code=200):
        self._json = json_data
        self.status_code = status_code

    def json(self):
        return self._json

    def raise_for_status(self):
        if self.status_code >= 400:
            import requests
            raise requests.HTTPError(f"{self.status_code} error")


_MOCK_STOCKS = rdh.build_mock_stocks_response()
_MOCK_PRICE_HISTORY = rdh.load_raw_price_history()
call_log = []


def _fake_get(url, headers=None, params=None, timeout=None):
    call_log.append((url, params))
    if url.endswith("/api/ngxdata/stocks"):
        return _FakeResponse(_MOCK_STOCKS)
    if url.endswith("/api/ngxdata/market"):
        return _FakeResponse({"data": {"asi": 145230.55, "market_cap": 9.1e13}})
    if "/api/ngxdata/prices/" in url:
        symbol = url.rsplit("/", 1)[-1]
        if symbol in _MOCK_PRICE_HISTORY:
            return _FakeResponse(_MOCK_PRICE_HISTORY[symbol])
        return _FakeResponse({"success": False, "symbol": symbol, "prices": [], "count": 0})
    return _FakeResponse({}, status_code=404)


# =======================================================================
print("\n=== fetch_stocks_snapshot: parses the real 294-symbol snapshot ===")
# =======================================================================

call_log.clear()
with patch.object(nps.requests, "get", side_effect=_fake_get):
    df = nps.fetch_stocks_snapshot()

check("fetch_stocks_snapshot returns all 147 real unique symbols (one distinct real pull, not double-counted)",
      len(df) == 147, len(df))
check("fetch_stocks_snapshot returns a DataFrame with real columns (current_price, sector)",
      "current_price" in df.columns and "sector" in df.columns, list(df.columns))
check("fetch_stocks_snapshot made exactly one request", len(call_log) == 1, call_log)


# =======================================================================
print("\n=== get_current_prices_and_sectors: ONE request regardless of symbol count ===")
# =======================================================================

call_log.clear()
with patch.object(nps.requests, "get", side_effect=_fake_get):
    prices_full, sectors_full = nps.get_current_prices_and_sectors()
check("full-universe call returns 147 real current prices (one distinct real pull)",
      len(prices_full) == 147, len(prices_full))
check("full-universe call made exactly one request", len(call_log) == 1, call_log)

# Simulate the actual scoped use case: ~10 real held symbols
held_symbols = ["MTNN", "DANGCEM", "GTCO", "ZENITHBANK", "SEPLAT", "ARADEL", "NB", "UBA", "ACCESSCORP", "FIDELITYBK"]
call_log.clear()
with patch.object(nps.requests, "get", side_effect=_fake_get):
    prices_held, sectors_held = nps.get_current_prices_and_sectors(symbols=held_symbols)
check("filtering to 10 held symbols still returns exactly those (found in real data)",
      set(prices_held.keys()) == set(held_symbols), prices_held.keys())
check("filtering to 10 held symbols STILL made exactly ONE request (the core scope decision)",
      len(call_log) == 1, call_log)
check("held-symbol prices match the real full-universe values (correct filtering, not corrupted)",
      all(prices_held[s] == prices_full[s] for s in held_symbols),
      {s: (prices_held[s], prices_full[s]) for s in held_symbols if prices_held[s] != prices_full[s]})

# Symbol not in the live snapshot (e.g. delisted/typo) -- absent, not a crash
call_log.clear()
with patch.object(nps.requests, "get", side_effect=_fake_get):
    prices_missing, sectors_missing = nps.get_current_prices_and_sectors(symbols=["MTNN", "NOTAREALSYMBOL"])
check("unknown symbol is simply absent from the result, not a crash",
      "NOTAREALSYMBOL" not in prices_missing and "MTNN" in prices_missing, prices_missing)


# =======================================================================
print("\n=== get_price_series: real captured price history, correctly parsed ===")
# =======================================================================

with patch.object(nps.requests, "get", side_effect=_fake_get):
    series = nps.get_price_series("AIRTELAFRI", days=130)

check("get_price_series returns a pd.Series for a real symbol", isinstance(series, pd.Series), type(series))
check("get_price_series returns all 130 real observations (real count from raw_price_history.json)",
      len(series) == 130, len(series))
check("get_price_series is sorted ascending (oldest -> newest, matches screening_service's expectation)",
      list(series.index) == sorted(series.index), series.index[:5])
check("get_price_series values are real close_price numbers (first real value is 2270.0)",
      float(series.iloc[0]) == 2270.0, series.iloc[0])
check("get_price_series last real value matches raw_price_history.json (6300.0)",
      float(series.iloc[-1]) == 6300.0, series.iloc[-1])


# =======================================================================
print("\n=== get_price_series: fail-soft on missing/bad data, doesn't crash ===")
# =======================================================================

with patch.object(nps.requests, "get", side_effect=_fake_get):
    missing_series = nps.get_price_series("NOTAREALSYMBOL", days=130)
check("get_price_series returns None (not a crash) for a symbol with no real data",
      missing_series is None, missing_series)

import requests as _requests_module


def _raising_get(*a, **kw):
    raise _requests_module.ConnectionError("simulated network failure")


with patch.object(nps.requests, "get", side_effect=_raising_get):
    network_fail_series = nps.get_price_series("AIRTELAFRI", days=130)
check("get_price_series returns None (not a raised exception) on a network failure",
      network_fail_series is None, network_fail_series)


# =======================================================================
print("\n=== fetch_price_history_raw: hard cap on days, fails BEFORE the request ===")
# =======================================================================

call_log.clear()
raised = False
try:
    with patch.object(nps.requests, "get", side_effect=_fake_get):
        nps.fetch_price_history_raw("AIRTELAFRI", days=5000)
except ValueError:
    raised = True
check("days > MAX_DAYS_PER_REQUEST raises ValueError", raised)
check("no request was made when days exceeded the hard cap (fails before, not silently truncates)",
      len(call_log) == 0, call_log)


# =======================================================================
print("\n=== get_price_series_for_symbols: quota-aware batch fetch over real symbols ===")
# =======================================================================

real_symbols_available = list(_MOCK_PRICE_HISTORY.keys())[:10]  # 10 real symbols with real history

call_log.clear()
with patch.object(nps.requests, "get", side_effect=_fake_get):
    series_dict, skipped = nps.get_price_series_for_symbols(real_symbols_available, days=130, max_requests=100)
check("batch fetch (under budget) gets all 10 real symbols", len(series_dict) == 10, len(series_dict))
check("batch fetch (under budget) skips none", skipped == [], skipped)
check("batch fetch made exactly 10 requests (one per symbol, real per-symbol cost confirmed)",
      len(call_log) == 10, call_log)

# Quota cutoff: budget of 3 against 10 symbols -- exactly 3 real fetches, 7 skipped
call_log.clear()
with patch.object(nps.requests, "get", side_effect=_fake_get):
    series_dict2, skipped2 = nps.get_price_series_for_symbols(real_symbols_available, days=130, max_requests=3)
check("quota cutoff: exactly 3 symbols fetched when max_requests=3", len(series_dict2) == 3, len(series_dict2))
check("quota cutoff: exactly 7 symbols skipped (budget-limited, not data-missing)", len(skipped2) == 7, skipped2)
check("quota cutoff: NO MORE than 3 real requests were actually made (the whole point of the guard)",
      len(call_log) == 3, call_log)
check("quota cutoff: skipped symbols are exactly the ones not fetched",
      set(series_dict2.keys()) | set(skipped2) == set(real_symbols_available),
      (set(series_dict2.keys()), set(skipped2)))


# =======================================================================
print("\n=== fetch_market_snapshot: shape-consistent mock (NOT backed by real captured data) ===")
# =======================================================================
# Unlike every other test in this file, no real /api/ngxdata/market
# response was captured this session -- this is the one exception, flagged
# explicitly per this project's standard of not silently treating a
# hand-built mock as if it were real data.

with patch.object(nps.requests, "get", side_effect=_fake_get):
    market = nps.fetch_market_snapshot()
check("fetch_market_snapshot correctly unwraps the 'data' key (shape-only mock, not real)",
      "asi" in market, market)


# =======================================================================
print("\n" + "=" * 60)
print(f"TOTAL: {len(PASS)} passed, {len(FAIL)} failed")
if FAIL:
    print("\nFAILURES:")
    for name, detail in FAIL:
        print(f"  - {name}: {detail}")
    sys.exit(1)
else:
    print("ALL TESTS PASSED")
    sys.exit(0)
