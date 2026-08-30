"""
test_monthly_screen_service.py

Verification suite for monthly_screen_service.py's _compare_against_holdings
-- the piece that reuses portfolio_service.check_drift/check_hard_breaches
(the exact same engine main.py's /rebalance-check already uses) plus the
new identify_diversification_additions, to compare a fresh monthly-screen
result against the chat's actual current holdings.

Runs against a fresh local SQLite file, same pattern as test_db.py/
test_bot.py. Only nps.get_current_prices_and_sectors is mocked (the one
live network call this logic makes); everything else -- db writes/reads,
portfolio_service's comparison logic, signal persistence -- runs for real.
"""

import os
import sys
from unittest.mock import patch

DB_PATH = "/home/claude/pmc_api/test_monthly_screen_service.db"
if os.path.exists(DB_PATH):
    os.remove(DB_PATH)
os.environ["DATABASE_URL"] = f"sqlite:///{DB_PATH}"

sys.path.insert(0, "/home/claude/pmc_api")

import pandas as pd
from datetime import date

import db  # noqa: E402
db.init_db()

import monthly_screen_service as mss  # noqa: E402
import ngx_pulse_service as nps  # noqa: E402
import portfolio_service as ps  # noqa: E402

PASS, FAIL = [], []


def check(name, cond, detail=""):
    if cond:
        PASS.append(name)
        print(f"PASS: {name}")
    else:
        FAIL.append((name, detail))
        print(f"FAIL: {name} -- {detail}")


def _escape_md(v):
    return str(v)  # identity escape -- Markdown-safety itself is tested in test_bot.py


# =======================================================================
print("\n=== _compare_against_holdings: no active portfolio linked ===")
# =======================================================================

result = mss._compare_against_holdings(
    "chat_no_portfolio", pd.Series({"GTCO": 0.1}), ["GTCO"], {"GTCO": "Banking"}, _escape_md,
)
check("no active portfolio: returns the 'skipping comparison' note", "No active portfolio linked" in result, result)


# =======================================================================
print("\n=== _compare_against_holdings: active portfolio, zero holdings ===")
# =======================================================================

pid_empty = db.start_new_portfolio_for_chat(
    "chat_empty", "Empty Portfolio", "Ticker", "NGN", 1_000_000, date(2026, 1, 1),
)
result = mss._compare_against_holdings(
    "chat_empty", pd.Series({"GTCO": 0.1}), ["GTCO"], {"GTCO": "Banking"}, _escape_md,
)
check("zero holdings: returns the 'no holdings yet' note", "has no holdings yet" in result, result)


# =======================================================================
print("\n=== _compare_against_holdings: NGX Pulse price fetch fails ===")
# =======================================================================

pid_pricefail = db.start_new_portfolio_for_chat(
    "chat_pricefail", "Price Fail Portfolio", "Ticker", "NGN", 1_000_000, date(2026, 1, 1),
)
db.upsert_holdings(pid_pricefail, [{"symbol": "GTCO", "shares": 100, "avg_cost": 45.0, "sector": "Banking"}])

with patch.object(nps, "get_current_prices_and_sectors", side_effect=RuntimeError("NGX Pulse down")):
    result = mss._compare_against_holdings(
        "chat_pricefail", pd.Series({"GTCO": 0.1}), ["GTCO"], {"GTCO": "Banking"}, _escape_md,
    )
check("live price fetch failure: comparison degrades gracefully with a clear note, doesn't crash",
      "Could not fetch live prices" in result, result)


# =======================================================================
print("\n=== _compare_against_holdings: clean portfolio, no flags ===")
# =======================================================================

pid_clean = db.start_new_portfolio_for_chat(
    "chat_clean", "Clean Portfolio", "Ticker", "NGN", 1_000_000, date(2026, 1, 1),
)
# 10 holdings (at the diversification floor), evenly weighted across 5 sectors
# (2 each = 20% per sector, safely under the 25% cap) -- matching the fresh
# target closely so nothing drifts either.
_CLEAN_SECTORS_CYCLE = ["Banking", "Consumer Goods", "Industrial Goods", "Oil and Gas", "Agriculture"]
clean_holdings = [
    {"symbol": f"SYM{i}", "shares": 100, "avg_cost": 10.0, "sector": _CLEAN_SECTORS_CYCLE[i % 5]}
    for i in range(10)
]
db.upsert_holdings(pid_clean, clean_holdings)
clean_prices = {f"SYM{i}": 10.0 for i in range(10)}  # unchanged from cost -> weights match target closely
clean_target = pd.Series({f"SYM{i}": 0.10 for i in range(10)})  # each exactly 10%, matches actual exactly
clean_sectors = {f"SYM{i}": _CLEAN_SECTORS_CYCLE[i % 5] for i in range(10)}

with patch.object(nps, "get_current_prices_and_sectors", return_value=(clean_prices, clean_sectors)):
    result = mss._compare_against_holdings(
        "chat_clean", clean_target, [f"SYM{i}" for i in range(10)], clean_sectors, _escape_md,
    )
check("clean portfolio: reports no flags", "No hard breaches, no meaningful drift" in result, result)
check("clean portfolio: still notes cash-cap isn't checked automatically",
      "Cash-cap breach not checked automatically" in result, result)
check("clean portfolio: does NOT show a HARD BREACHES section", "HARD BREACHES" not in result, result)
check("clean portfolio: does NOT show a DRIFT section", "DRIFT" not in result, result)
check("clean portfolio: does NOT show a DIVERSIFICATION FLOOR section (exactly at the floor)",
      "DIVERSIFICATION FLOOR" not in result, result)

signals_clean = db.get_unacknowledged_signals(pid_clean)
check("clean portfolio: no signals persisted when nothing was flagged", signals_clean == [], signals_clean)


# =======================================================================
print("\n=== _compare_against_holdings: stock cap hard breach ===")
# =======================================================================

pid_breach = db.start_new_portfolio_for_chat(
    "chat_breach", "Breach Portfolio", "Ticker", "NGN", 1_000_000, date(2026, 1, 1),
)
# One massively overweight position (30%+ of the book) plus enough others for MIN_HOLDINGS
breach_holdings = [{"symbol": "BIGPOS", "shares": 1000, "avg_cost": 100.0, "sector": "Oil and Gas"}] + [
    {"symbol": f"SMALL{i}", "shares": 10, "avg_cost": 10.0, "sector": "Consumer Goods"} for i in range(6)
]
db.upsert_holdings(pid_breach, breach_holdings)
breach_prices = {"BIGPOS": 100.0, **{f"SMALL{i}": 10.0 for i in range(6)}}
breach_sectors = {"BIGPOS": "Oil and Gas", **{f"SMALL{i}": "Consumer Goods" for i in range(6)}}
breach_target = pd.Series({"BIGPOS": 0.15, **{f"SMALL{i}": 0.10 for i in range(6)}})

with patch.object(nps, "get_current_prices_and_sectors", return_value=(breach_prices, breach_sectors)):
    result = mss._compare_against_holdings(
        "chat_breach", breach_target, ["BIGPOS"] + [f"SMALL{i}" for i in range(6)], breach_sectors, _escape_md,
    )
check("stock breach: BIGPOS flagged in HARD BREACHES", "HARD BREACHES" in result and "BIGPOS" in result, result)
check("stock breach: shows the actual stock cap threshold in the message", "25.00%" in result, result)

signals_breach = db.get_unacknowledged_signals(pid_breach)
check("stock breach: a stock_breach signal was persisted",
      any(s["signal_type"] == "stock_breach" and s["symbol"] == "BIGPOS" for s in signals_breach), signals_breach)

# Dedup: running the comparison AGAIN with the same breach must not create a duplicate signal
with patch.object(nps, "get_current_prices_and_sectors", return_value=(breach_prices, breach_sectors)):
    mss._compare_against_holdings(
        "chat_breach", breach_target, ["BIGPOS"] + [f"SMALL{i}" for i in range(6)], breach_sectors, _escape_md,
    )
signals_breach_again = db.get_unacknowledged_signals(pid_breach)
check("stock breach: re-running with the same unresolved breach does NOT create a duplicate signal "
      "(reuses db.log_signal_if_new's existing dedup, same as /rebalance-check)",
      len([s for s in signals_breach_again if s["signal_type"] == "stock_breach"]) == 1, signals_breach_again)


# =======================================================================
print("\n=== _compare_against_holdings: sector cap hard breach ===")
# =======================================================================

pid_sector = db.start_new_portfolio_for_chat(
    "chat_sector", "Sector Breach Portfolio", "Ticker", "NGN", 1_000_000, date(2026, 1, 1),
)
sector_holdings = [
    {"symbol": "BANK1", "shares": 150, "avg_cost": 10.0, "sector": "Banking"},
    {"symbol": "BANK2", "shares": 150, "avg_cost": 10.0, "sector": "Banking"},
    {"symbol": "BANK3", "shares": 100, "avg_cost": 10.0, "sector": "Banking"},
] + [{"symbol": f"OTHER{i}", "shares": 10, "avg_cost": 10.0, "sector": "Consumer Goods"} for i in range(4)]
db.upsert_holdings(pid_sector, sector_holdings)
sector_prices = {"BANK1": 10.0, "BANK2": 10.0, "BANK3": 10.0, **{f"OTHER{i}": 10.0 for i in range(4)}}
sector_sectors = {"BANK1": "Banking", "BANK2": "Banking", "BANK3": "Banking", **{f"OTHER{i}": "Consumer Goods" for i in range(4)}}
sector_target = pd.Series({"BANK1": 0.15, "BANK2": 0.15, "BANK3": 0.10, **{f"OTHER{i}": 0.10 for i in range(4)}})

with patch.object(nps, "get_current_prices_and_sectors", return_value=(sector_prices, sector_sectors)):
    result = mss._compare_against_holdings(
        "chat_sector", sector_target, list(sector_prices.keys()), sector_sectors, _escape_md,
    )
check("sector breach: Banking sector flagged in HARD BREACHES", "HARD BREACHES" in result and "Banking" in result, result)

signals_sector = db.get_unacknowledged_signals(pid_sector)
check("sector breach: a sector_breach signal was persisted",
      any(s["signal_type"] == "sector_breach" and s["symbol"] == "Banking" for s in signals_sector), signals_sector)


# =======================================================================
print("\n=== _compare_against_holdings: drift trigger ===")
# =======================================================================

pid_drift = db.start_new_portfolio_for_chat(
    "chat_drift", "Drift Portfolio", "Ticker", "NGN", 1_000_000, date(2026, 1, 1),
)
drift_holdings = [{"symbol": f"D{i}", "shares": 100, "avg_cost": 10.0, "sector": "Banking"} for i in range(10)]
db.upsert_holdings(pid_drift, drift_holdings)
drift_prices = {f"D{i}": 10.0 for i in range(10)}
drift_sectors = {f"D{i}": "Banking" for i in range(10)}
# D0 actual=10%, fresh target=2% -> 8pp drift, well over the 5pp threshold
drift_target = pd.Series({"D0": 0.02, **{f"D{i}": (0.98 / 9) for i in range(1, 10)}})

with patch.object(nps, "get_current_prices_and_sectors", return_value=(drift_prices, drift_sectors)):
    result = mss._compare_against_holdings(
        "chat_drift", drift_target, [f"D{i}" for i in range(10)], drift_sectors, _escape_md,
    )
check("drift trigger: D0 flagged in DRIFT section", "DRIFT" in result and "D0" in result, result)

signals_drift = db.get_unacknowledged_signals(pid_drift)
check("drift trigger: a drift signal was persisted for D0",
      any(s["signal_type"] == "drift" and s["symbol"] == "D0" for s in signals_drift), signals_drift)


# =======================================================================
print("\n=== _compare_against_holdings: held symbol not in the fresh target at all ===")
# =======================================================================

pid_dropped = db.start_new_portfolio_for_chat(
    "chat_dropped", "Dropped Symbol Portfolio", "Ticker", "NGN", 1_000_000, date(2026, 1, 1),
)
# Spread across 5 sectors (2 each = 20%, under the cap) so this test isolates
# the drift check specifically, without an incidental sector breach.
_DROPPED_SECTORS_CYCLE = ["Banking", "Consumer Goods", "Industrial Goods", "Oil and Gas", "Agriculture"]
dropped_holdings = [{"symbol": "DROPPED", "shares": 100, "avg_cost": 10.0, "sector": "Telecom"}] + [
    {"symbol": f"KEEP{i}", "shares": 100, "avg_cost": 10.0, "sector": _DROPPED_SECTORS_CYCLE[i % 5]}
    for i in range(9)
]
db.upsert_holdings(pid_dropped, dropped_holdings)
dropped_prices = {"DROPPED": 10.0, **{f"KEEP{i}": 10.0 for i in range(9)}}
dropped_sectors = {"DROPPED": "Telecom", **{f"KEEP{i}": _DROPPED_SECTORS_CYCLE[i % 5] for i in range(9)}}
# DROPPED isn't in the fresh target at all (fell out of the ranked pool entirely)
dropped_target = pd.Series({f"KEEP{i}": 0.10 for i in range(9)})

with patch.object(nps, "get_current_prices_and_sectors", return_value=(dropped_prices, dropped_sectors)):
    result = mss._compare_against_holdings(
        "chat_dropped", dropped_target, [f"KEEP{i}" for i in range(9)], dropped_sectors, _escape_md,
    )
check("symbol entirely absent from fresh target: still flagged as drift (large gap from 0)",
      "DROPPED" in result, result)
check("symbol entirely absent from fresh target: shown as 'not in fresh pool', not a crash or blank value",
      "not in fresh pool" in result, result)


# =======================================================================
print("\n=== _compare_against_holdings: diversification floor ===")
# =======================================================================

pid_floor = db.start_new_portfolio_for_chat(
    "chat_floor", "Floor Portfolio", "Ticker", "NGN", 1_000_000, date(2026, 1, 1),
)
# Only 6 holdings -- below the floor of 10, above the hard minimum of 5
floor_holdings = [{"symbol": f"F{i}", "shares": 100, "avg_cost": 10.0, "sector": "Banking"} for i in range(6)]
db.upsert_holdings(pid_floor, floor_holdings)
floor_prices = {f"F{i}": 10.0 for i in range(6)}
floor_sectors = {f"F{i}": "Banking" for i in range(6)}
floor_target = pd.Series({f"F{i}": (1 / 6) for i in range(6)})  # matches actual closely -- isolate the floor signal
# Ranked pool includes the 6 held plus 4 new candidates the floor logic should surface
floor_ranked = [f"F{i}" for i in range(6)] + ["NEWA", "NEWB", "NEWC", "NEWD"]

with patch.object(nps, "get_current_prices_and_sectors", return_value=(floor_prices, floor_sectors)):
    result = mss._compare_against_holdings(
        "chat_floor", floor_target, floor_ranked, floor_sectors, _escape_md,
    )
check("diversification floor: section appears when below the floor", "DIVERSIFICATION FLOOR" in result, result)
check("diversification floor: correctly names the top-ranked NEW candidates (NEWA, NEWB, NEWC, NEWD)",
      all(s in result for s in ["NEWA", "NEWB", "NEWC", "NEWD"]), result)
check("diversification floor: does NOT suggest already-held symbols as 'new'",
      not any(f"F{i}" in result.split("DIVERSIFICATION FLOOR")[1] for i in range(6)) if "DIVERSIFICATION FLOOR" in result else False,
      result)

signals_floor = db.get_unacknowledged_signals(pid_floor)
check("diversification floor: a diversification_floor signal was persisted",
      any(s["signal_type"] == "diversification_floor" for s in signals_floor), signals_floor)


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
