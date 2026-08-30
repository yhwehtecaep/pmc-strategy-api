"""
test_universe_service.py

Verifies universe_service.py's filtering logic. UNLIKE
test_ngx_pulse_service.py (which replays real captured NGX Pulse
responses from real_data/), this file uses a HAND-BUILT, clearly
synthetic mock -- real_data/ was not available in the session this was
built in. The numbers below are invented specifically to exercise each
filter boundary, NOT real market data. If real_data/ is later uploaded,
this file should be revisited to additionally validate against a real
/stocks snapshot, same discipline as test_ngx_pulse_service.py.

Only requests.get() is patched; universe_service.build_live_universe's
actual filtering logic runs for real against this synthetic snapshot.
"""

import sys
from unittest.mock import patch

sys.path.insert(0, "/home/claude/pmc_api")

import ngx_pulse_service as nps
import universe_service as us

PASS, FAIL = [], []


def check(name, cond, detail=""):
    if cond:
        PASS.append(name)
        print(f"PASS: {name}")
    else:
        FAIL.append((name, detail))
        print(f"FAIL: {name} -- {detail}")


# =======================================================================
# SYNTHETIC mock -- explicitly NOT real captured data (see module docstring)
# =======================================================================

_SYNTHETIC_STOCKS = {
    "stocks": [
        # Passes both filters cleanly
        {"symbol": "BIGCO", "name": "Big Company Plc", "current_price": 100.0,
         "market_cap": 50_000_000_000, "volume": 200_000, "sector": "Industrial Goods"},
        # Exactly AT both thresholds -- boundary case, must PASS (>=, not >)
        {"symbol": "EXACT", "name": "Exactly At Threshold Plc", "current_price": 10.0,
         "market_cap": 20_000_000_000, "volume": 50_000, "sector": "Consumer Goods"},
        # Fails market cap only (just under)
        {"symbol": "SMALLCAP", "name": "Small Cap Plc", "current_price": 5.0,
         "market_cap": 19_999_999_999, "volume": 500_000, "sector": "Agriculture"},
        # Fails volume only (just under)
        {"symbol": "ILLIQUID", "name": "Illiquid Plc", "current_price": 50.0,
         "market_cap": 100_000_000_000, "volume": 49_999, "sector": "Oil and Gas"},
        # Fails BOTH filters
        {"symbol": "TINYCO", "name": "Tiny Company Plc", "current_price": 1.0,
         "market_cap": 500_000_000, "volume": 1_000, "sector": "Conglomerates"},
        # Passes both filters, but is a documented non-equity exclusion
        {"symbol": "NIDF", "name": "Nigeria Infrastructure Debt Fund", "current_price": 100.0,
         "market_cap": 100_000_000_000, "volume": 1_000_000, "sector": "Debt Fund"},
        {"symbol": "NREIT", "name": "Nigeria REIT", "current_price": 20.0,
         "market_cap": 30_000_000_000, "volume": 300_000, "sector": "Real Estate"},
        # Missing/malformed market_cap -- must be excluded, not crash or silently pass
        {"symbol": "MESSYDATA", "name": "Messy Data Plc", "current_price": 15.0,
         "market_cap": None, "volume": 500_000, "sector": "Banking"},
        {"symbol": "BADVOLUME", "name": "Bad Volume Plc", "current_price": 25.0,
         "market_cap": 40_000_000_000, "volume": "not_a_number", "sector": "Banking"},
    ]
}


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


def _fake_get(url, headers=None, params=None, timeout=None):
    if url.endswith("/api/ngxdata/stocks"):
        return _FakeResponse(_SYNTHETIC_STOCKS)
    return _FakeResponse({}, status_code=404)


# =======================================================================
print("\n=== build_live_universe: filtering logic ===")
# =======================================================================

with patch.object(nps.requests, "get", side_effect=_fake_get):
    symbols, sector_by_symbol, current_price_by_symbol = us.build_live_universe()

check("BIGCO (clearly passes both filters) is included", "BIGCO" in symbols, symbols)
check("EXACT (exactly AT both thresholds) is included -- filter is >=, not >",
      "EXACT" in symbols, symbols)
check("SMALLCAP (market cap just under threshold) is excluded", "SMALLCAP" not in symbols, symbols)
check("ILLIQUID (volume just under threshold) is excluded", "ILLIQUID" not in symbols, symbols)
check("TINYCO (fails both filters) is excluded", "TINYCO" not in symbols, symbols)
check("NIDF is excluded despite passing both size/liquidity filters (non-equity, per doc Section 2)",
      "NIDF" not in symbols, symbols)
check("NREIT is excluded despite passing both size/liquidity filters (non-equity, per doc Section 2)",
      "NREIT" not in symbols, symbols)
check("MESSYDATA (missing market_cap) is excluded, not crashed on or silently passed",
      "MESSYDATA" not in symbols, symbols)
check("BADVOLUME (non-numeric volume) is excluded, not crashed on or silently passed",
      "BADVOLUME" not in symbols, symbols)

check("Exactly 2 symbols pass all filters (BIGCO, EXACT)", sorted(symbols) == ["BIGCO", "EXACT"], symbols)

check("sector_by_symbol is correctly scoped to only the passing symbols",
      set(sector_by_symbol.keys()) == {"BIGCO", "EXACT"}, sector_by_symbol)
check("sector_by_symbol has the correct sector values",
      sector_by_symbol["BIGCO"] == "Industrial Goods" and sector_by_symbol["EXACT"] == "Consumer Goods",
      sector_by_symbol)
check("current_price_by_symbol is correctly scoped and valued",
      current_price_by_symbol == {"BIGCO": 100.0, "EXACT": 10.0}, current_price_by_symbol)


# =======================================================================
print("\n=== build_live_universe: caller-supplied additional exclusions ===")
# =======================================================================

with patch.object(nps.requests, "get", side_effect=_fake_get):
    symbols2, _, _ = us.build_live_universe(excluded_symbols={"BIGCO"})

check("caller-supplied excluded_symbols is ADDITIVE to the built-in NIDF/NREIT exclusion, not a replacement",
      "BIGCO" not in symbols2 and "EXACT" in symbols2, symbols2)


# =======================================================================
print("\n=== build_live_universe: custom thresholds (parameterization) ===")
# =======================================================================

with patch.object(nps.requests, "get", side_effect=_fake_get):
    symbols3, _, _ = us.build_live_universe(min_market_cap=0, min_daily_volume=0)

check("min_market_cap=0, min_daily_volume=0 lets size/liquidity-failing names back in "
      "(SMALLCAP, ILLIQUID, TINYCO now included), while NIDF/NREIT are still excluded (non-equity, not size-based)",
      "SMALLCAP" in symbols3 and "ILLIQUID" in symbols3 and "TINYCO" in symbols3
      and "NIDF" not in symbols3 and "NREIT" not in symbols3,
      symbols3)


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
