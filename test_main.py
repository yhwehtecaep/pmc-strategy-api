"""
test_main.py

Verification suite for main.py, focused on the db.py wiring (the gap the
handoff identified). Uses TestClient + a fresh SQLite DATABASE_URL override,
same pattern as test_db.py, and mocks fundamentals_service's live scrape
(this sandbox has no network route to stockanalysis.com) so tests are
self-contained and fast.

Sections:
  1. Regression -- pre-existing endpoints (/health, /screen, /portfolio/construct,
     /rebalance-check stateless mode) still work exactly as before.
  2. Portfolio / chat CRUD (new).
  3. Holdings CRUD (new).
  4. Trades (new), including invalid-trade rejection.
  5. Signals + rebalance-check STATEFUL mode (new) -- pulls holdings from DB,
     persists signals, dedups on repeated polls.
  6. Fundamentals caching -- verifies the live scraper is only called once
     per (symbol, as_of_date) even across repeated /fundamentals and /screen calls.
  7. End-to-end: construct a portfolio via the API, persist it, then hit
     /rebalance-check with just a portfolio_id and confirm it pulls the
     right holdings -- the exact check the handoff asked for.
"""

import os
import sys

DB_PATH = "/home/claude/pmc_api/test_main.db"
if os.path.exists(DB_PATH):
    os.remove(DB_PATH)
os.environ["DATABASE_URL"] = f"sqlite:///{DB_PATH}"

sys.path.insert(0, "/home/claude/pmc_api")

from datetime import date, datetime
from unittest.mock import patch

import fundamentals_service as fs
import ngx_pulse_service as nps
import requests
from fastapi.testclient import TestClient

import main
import db

client = TestClient(main.app)
# TestClient triggers FastAPI's startup event (db.init_db()) on first request /
# context-manager use, but call it explicitly too so tests don't depend on that.
db.init_db()

PASS, FAIL = [], []


def check(name, cond, detail=""):
    if cond:
        PASS.append(name)
        print(f"PASS: {name}")
    else:
        FAIL.append((name, detail))
        print(f"FAIL: {name} -- {detail}")


_MOCK_ROE_BY_SYMBOL = {
    "DANGCEM": 42.3, "GTCO": 28.1, "ZENITHBANK": 24.7, "MTNN": 55.2,
    "FIDELITYBK": 18.9, "SEPLAT": 33.4,
}


def _mock_fundamentals(symbol, as_of_date, current_price=None):
    """Deterministic stand-in for the live stockanalysis.com scrape. ROE
    varies per symbol (not a constant) -- a constant ROE across the whole
    universe collapses screening_service's np.nanstd(roe_vals) to 0,
    producing a NaN roe_z; that's a real edge case in the untouched
    scoring code, not something this test data should trigger."""
    eps = 10.0
    roe = _MOCK_ROE_BY_SYMBOL.get(symbol, 25.0)
    return fs.PointInTimeFundamentals(
        symbol=symbol, as_of_date=as_of_date, roe=roe,
        pe=(current_price / eps if current_price else None),
        eps_used=eps, eps_period_ending=date(2025, 12, 31), data_available=True,
    )


# =======================================================================
print("\n=== SECTION 1: Regression -- pre-existing endpoints unchanged ===")
# =======================================================================

r = client.get("/health")
check("GET /health still returns ok", r.status_code == 200 and r.json() == {"status": "ok"})

with patch.object(fs, "get_point_in_time_fundamentals", side_effect=_mock_fundamentals):
    universe_payload = {
        "price_series_by_symbol": {
            sym: [100 + i + (0 if sym != "DANGCEM" else i * 2) for i in range(130)]
            for sym in ["DANGCEM", "GTCO", "ZENITHBANK", "MTNN", "FIDELITYBK", "SEPLAT"]
        },
        "current_price_by_symbol": {
            "DANGCEM": 500.0, "GTCO": 45.0, "ZENITHBANK": 40.0,
            "MTNN": 200.0, "FIDELITYBK": 15.0, "SEPLAT": 3000.0,
        },
        "sector_by_symbol": {
            "DANGCEM": "Industrial", "GTCO": "Banking", "ZENITHBANK": "Banking",
            "MTNN": "Telecom", "FIDELITYBK": "Banking", "SEPLAT": "Energy",
        },
        "as_of_date": "2026-08-20",
        "excluded_symbols": ["NIDF", "NREIT"],
    }
    r = client.post("/screen", json=universe_payload)
    check("POST /screen still returns 200", r.status_code == 200, r.text)
    scored = r.json()
    check("POST /screen returns all 6 symbols", len(scored) == 6, scored)
    eligible = [s for s in scored if s["eligible"]]
    check("POST /screen: eligible stocks have composite_score set",
          all(s["composite_score"] is not None for s in eligible), eligible)

    construct_payload = dict(universe_payload, pool_size=6, cash_buffer=0.03)
    r = client.post("/portfolio/construct", json=construct_payload)
    check("POST /portfolio/construct still returns 200", r.status_code == 200, r.text)
    pc = r.json()
    check("POST /portfolio/construct returns weights", len(pc["weights"]) > 0, pc)
    check("POST /portfolio/construct cash_pct is sane (>=3%)", pc["cash_pct"] >= 0.03 - 1e-9, pc)

# Stateless rebalance-check (original contract: actual_weights + sectors, no portfolio_id)
stateless_payload = {
    "actual_weights": {"GTCO": 0.30, "ZENITHBANK": 0.20},
    "fresh_target_weights": {"GTCO": 0.20, "ZENITHBANK": 0.20},
    "sectors": {"GTCO": "Banking", "ZENITHBANK": "Banking"},
    "cash": 0.02,
}
r = client.post("/rebalance-check", json=stateless_payload)
check("POST /rebalance-check stateless mode still returns 200", r.status_code == 200, r.text)
rc = r.json()
check("stateless mode: GTCO drift triggered (10pp > 5pp threshold)",
      any(row["symbol"] == "GTCO" and row["triggered"] for row in rc["drift_rows"]), rc)
check("stateless mode: hard_breaches.stock_breach fires on GTCO (0.30 > 0.25 cap)",
      "GTCO" in rc["hard_breaches"]["stock_breach"], rc)
check("stateless mode: signals_logged empty (no portfolio_id supplied)",
      rc["signals_logged"] == [], rc)


# =======================================================================
print("\n=== SECTION 2: Portfolio / chat CRUD ===")
# =======================================================================

r = client.post("/portfolios", json={
    "name": "CFA PMC 2026", "broker": "Ticker (Cowrywise)", "currency": "NGN",
    "initial_capital": 10_000_000, "inception_date": "2026-01-01",
})
check("POST /portfolios returns 200", r.status_code == 200, r.text)
p1 = r.json()
check("POST /portfolios returns an id", isinstance(p1["id"], str) and len(p1["id"]) > 0, p1)
check("POST /portfolios defaults status=active", p1["status"] == "active", p1)
pid1 = p1["id"]

r = client.get(f"/portfolios/{pid1}")
check("GET /portfolios/{id} returns the right portfolio", r.json()["name"] == "CFA PMC 2026", r.json())

r = client.get("/portfolios/nonexistent-id")
check("GET /portfolios/{missing_id} returns 404", r.status_code == 404, r.text)

r = client.post(f"/portfolios/{pid1}/close")
check("POST /portfolios/{id}/close returns 200", r.status_code == 200, r.text)
check("close sets status=closed", r.json()["status"] == "closed", r.json())

r = client.post("/portfolios/nonexistent-id/close")
check("POST /portfolios/{missing_id}/close returns 404", r.status_code == 404, r.text)

# Chat <-> active portfolio, including the reusability workflow (switch-portfolio)
r = client.post("/portfolios", json={
    "name": "Chat Portfolio A", "broker": "Ticker", "currency": "NGN",
    "initial_capital": 1_000_000, "inception_date": "2026-01-01",
})
pid_chat_a = r.json()["id"]
r = client.post("/chats/chat_test_1/active-portfolio", json={"portfolio_id": pid_chat_a})
check("POST /chats/{id}/active-portfolio returns 200", r.status_code == 200, r.text)

r = client.get("/chats/chat_test_1/active-portfolio")
check("GET /chats/{id}/active-portfolio returns the set portfolio", r.json()["id"] == pid_chat_a, r.json())

r = client.get("/chats/chat_unregistered/active-portfolio")
check("GET /chats/{unregistered}/active-portfolio returns null", r.json() is None, r.json())

r = client.post("/chats/chat_test_1/switch-portfolio", json={
    "name": "Chat Portfolio B (post-competition)", "broker": "Ticker", "currency": "NGN",
    "initial_capital": 2_000_000, "inception_date": "2026-09-01",
})
check("POST /chats/{id}/switch-portfolio returns 200", r.status_code == 200, r.text)
pid_chat_b = r.json()["id"]
check("switch-portfolio returns a new distinct id", pid_chat_b != pid_chat_a, (pid_chat_a, pid_chat_b))

r = client.get(f"/portfolios/{pid_chat_a}")
check("switch-portfolio auto-closed the old one", r.json()["status"] == "closed", r.json())

r = client.get("/chats/chat_test_1/active-portfolio")
check("chat now points at the new portfolio", r.json()["id"] == pid_chat_b, r.json())

r = client.post("/chats/chat_test_1/active-portfolio", json={"portfolio_id": "nonexistent-id"})
check("setting active-portfolio to a nonexistent id returns 404", r.status_code == 404, r.text)


# =======================================================================
print("\n=== SECTION 3: Holdings CRUD ===")
# =======================================================================

r = client.post("/portfolios", json={
    "name": "Holdings API Test", "broker": "Ticker", "currency": "NGN",
    "initial_capital": 1_000_000, "inception_date": "2026-01-01",
})
pid_h = r.json()["id"]

r = client.get(f"/portfolios/{pid_h}/holdings")
check("GET holdings on a fresh portfolio returns empty list", r.json() == [], r.json())

r = client.put(f"/portfolios/{pid_h}/holdings", json=[
    {"symbol": "GTCO", "shares": 100, "avg_cost": 45.0, "sector": "Banking"},
    {"symbol": "ZENITHBANK", "shares": 50, "avg_cost": 40.0, "sector": "Banking"},
])
check("PUT holdings returns 200 with both rows", r.status_code == 200 and len(r.json()) == 2, r.text)

r = client.put(f"/portfolios/{pid_h}/holdings", json=[
    {"symbol": "MTNN", "shares": 20, "avg_cost": 200.0, "sector": "Telecom"},
])
h_after = r.json()
check("second PUT holdings fully replaces (only MTNN remains)",
      len(h_after) == 1 and h_after[0]["symbol"] == "MTNN", h_after)

r = client.put("/portfolios/nonexistent-id/holdings", json=[])
check("PUT holdings on nonexistent portfolio returns 404", r.status_code == 404, r.text)


# =======================================================================
print("\n=== SECTION 4: Trades ===")
# =======================================================================

r = client.post("/portfolios", json={
    "name": "Trade API Test", "broker": "Ticker", "currency": "NGN",
    "initial_capital": 1_000_000, "inception_date": "2026-01-01",
})
pid_t = r.json()["id"]

r = client.post(f"/portfolios/{pid_t}/trades", json={
    "symbol": "GTCO", "side": "buy", "shares": 100, "price": 45.0, "fee": 68.85,
    "reason": "initial_construction", "sector": "Banking",
})
check("POST trades (buy) returns 200", r.status_code == 200, r.text)
t1 = r.json()
check("logged trade has correct side/shares/price", t1["side"] == "buy" and t1["shares"] == 100 and t1["price"] == 45.0, t1)

r = client.get(f"/portfolios/{pid_t}/holdings")
check("buy created the holding via the trade endpoint", len(r.json()) == 1 and r.json()[0]["symbol"] == "GTCO", r.json())

r = client.get(f"/portfolios/{pid_t}/trades")
check("GET trades returns the logged trade", len(r.json()) == 1, r.json())

r = client.post(f"/portfolios/{pid_t}/trades", json={
    "symbol": "GTCO", "side": "sell", "shares": 500, "price": 50.0, "fee": 1.0,
})
check("overselling via the API returns 422 (not 500, not silently accepted)", r.status_code == 422, r.text)

r = client.get(f"/portfolios/{pid_t}/holdings")
gtco = next(h for h in r.json() if h["symbol"] == "GTCO")
check("failed oversell via API did not mutate holdings (still 100 shares)", gtco["shares"] == 100, gtco)

r = client.post(f"/portfolios/{pid_t}/trades", json={
    "symbol": "GTCO", "side": "yeet", "shares": 10, "price": 45.0, "fee": 1.0,
})
check("invalid side via API returns 422", r.status_code == 422, r.text)


# =======================================================================
print("\n=== SECTION 5: Signals + rebalance-check STATEFUL mode ===")
# =======================================================================

r = client.post("/portfolios", json={
    "name": "Stateful Rebalance Test", "broker": "Ticker", "currency": "NGN",
    "initial_capital": 1_000_000, "inception_date": "2026-01-01",
})
pid_r = r.json()["id"]

client.put(f"/portfolios/{pid_r}/holdings", json=[
    {"symbol": "GTCO", "shares": 100, "avg_cost": 45.0, "sector": "Banking"},
    {"symbol": "ZENITHBANK", "shares": 100, "avg_cost": 40.0, "sector": "Banking"},
])
# GTCO: 100 * 60 = 6000; ZENITHBANK: 100 * 20 = 2000; cash 200 -> total 8200
# GTCO weight = 6000/8200 = 0.7317 (way over 25% cap -> hard breach)
prices = {"GTCO": 60.0, "ZENITHBANK": 20.0}

rc_payload = {
    "portfolio_id": pid_r,
    "current_price_by_symbol": prices,
    "fresh_target_weights": {"GTCO": 0.25, "ZENITHBANK": 0.25},
    "cash": 200.0 / 8200.0,
}
r = client.post("/rebalance-check", json=rc_payload)
check("stateful rebalance-check returns 200", r.status_code == 200, r.text)
rc = r.json()
gtco_row = next(row for row in rc["drift_rows"] if row["symbol"] == "GTCO")
check("stateful mode pulled correct actual_weight from DB holdings (GTCO ~0.7317)",
      abs(gtco_row["actual_weight"] - (6000 / 8200)) < 1e-6, gtco_row)
check("stateful mode: stock_breach fires on GTCO (pulled from DB, over 25% cap)",
      "GTCO" in rc["hard_breaches"]["stock_breach"], rc)
check("stateful mode: signals_logged non-empty (breach + drift persisted)",
      len(rc["signals_logged"]) > 0, rc)

r = client.get(f"/portfolios/{pid_r}/signals")
signals_after_first = r.json()
check("signals were actually persisted to the DB", len(signals_after_first) > 0, signals_after_first)

# Poll again with the SAME situation -- dedup should suppress re-alerting
r2 = client.post("/rebalance-check", json=rc_payload)
rc2 = r2.json()
check("second identical poll logs NO new signals (dedup working through the API)",
      rc2["signals_logged"] == [], rc2)

r = client.get(f"/portfolios/{pid_r}/signals")
signals_after_second_poll = r.json()
check("signal count unchanged after the duplicate poll",
      len(signals_after_second_poll) == len(signals_after_first), (signals_after_first, signals_after_second_poll))

# Acknowledge and confirm it drops off the unacknowledged list
sid_to_ack = signals_after_first[0]["id"]
r = client.post(f"/signals/{sid_to_ack}/acknowledge")
check("POST acknowledge returns 200", r.status_code == 200, r.text)
r = client.get(f"/portfolios/{pid_r}/signals")
check("acknowledged signal no longer appears in unacknowledged list",
      all(s["id"] != sid_to_ack for s in r.json()), r.json())

# rebalance-check with neither portfolio_id nor actual_weights -> 422
r = client.post("/rebalance-check", json={"fresh_target_weights": {}, "cash": 0.0})
check("rebalance-check with neither portfolio_id nor actual_weights returns 422", r.status_code == 422, r.text)

# rebalance-check with portfolio_id and NO current_price_by_symbol: current_price_by_symbol
# is no longer required (behavior change) -- it should attempt a LIVE NGX Pulse
# fetch scoped to exactly this portfolio's held symbols.
requested_symbols = {}


def _mock_live_prices(symbols=None):
    requested_symbols["value"] = set(symbols) if symbols is not None else None
    return {"GTCO": 60.0, "ZENITHBANK": 20.0}, {"GTCO": "Banking", "ZENITHBANK": "Banking"}


with patch.object(nps, "get_current_prices_and_sectors", side_effect=_mock_live_prices):
    r = client.post("/rebalance-check", json={
        "portfolio_id": pid_r, "fresh_target_weights": {"GTCO": 0.25, "ZENITHBANK": 0.25}, "cash": 0.05,
    })
check("rebalance-check with portfolio_id + no prices: live fetch succeeds, returns 200 (not 422)",
      r.status_code == 200, r.text)
check("live fetch was scoped to EXACTLY this portfolio's held symbols (GTCO, ZENITHBANK), "
      "not the full universe -- confirms the 'only fetch held symbols' scope decision",
      requested_symbols["value"] == {"GTCO", "ZENITHBANK"}, requested_symbols)
live_rc = r.json()
check("weights computed from the live-fetched prices are structurally sane (both symbols present)",
      {row["symbol"] for row in live_rc["drift_rows"]} == {"GTCO", "ZENITHBANK"}, live_rc)
check("no missing_prices when the live fetch covers every held symbol",
      live_rc["missing_prices"] == [], live_rc)

# Live fetch failure (e.g. NGX Pulse outage) -> 502 with a clear, actionable message,
# NOT a 500 crash and NOT a silent fallback to garbage data.
with patch.object(nps, "get_current_prices_and_sectors", side_effect=requests.RequestException("simulated outage")):
    r = client.post("/rebalance-check", json={
        "portfolio_id": pid_r, "fresh_target_weights": {"GTCO": 0.25, "ZENITHBANK": 0.25}, "cash": 0.05,
    })
check("live fetch failure returns 502, not a 500 crash", r.status_code == 502, r.text)
check("502 error message is actionable (mentions the manual-override escape hatch)",
      "current_price_by_symbol manually" in r.json()["detail"], r.text)

# Live fetch succeeds but is MISSING one held symbol (e.g. delisted) -> surfaced via
# missing_prices, not silently swallowed (that symbol gets valued at 0 -- a real risk
# of masking a breach, so the caller must be told explicitly).
with patch.object(nps, "get_current_prices_and_sectors", return_value=({"GTCO": 60.0}, {"GTCO": "Banking"})):
    r = client.post("/rebalance-check", json={
        "portfolio_id": pid_r, "fresh_target_weights": {"GTCO": 0.25, "ZENITHBANK": 0.25}, "cash": 0.05,
    })
check("partial live fetch (ZENITHBANK missing) still returns 200, doesn't crash", r.status_code == 200, r.text)
check("ZENITHBANK correctly surfaced in missing_prices (not silently dropped)",
      r.json()["missing_prices"] == ["ZENITHBANK"], r.json())

# A portfolio with zero holdings and no current_price_by_symbol should NOT attempt a
# live fetch at all (nothing to price) -- confirms the "nothing held, nothing to
# price" short-circuit, not an unnecessary network call.
r_empty = client.post("/portfolios", json={
    "name": "Empty Portfolio", "broker": "Ticker", "currency": "NGN",
    "initial_capital": 1_000_000, "inception_date": "2026-01-01",
})
pid_empty = r_empty.json()["id"]
with patch.object(nps, "get_current_prices_and_sectors") as mock_live:
    r = client.post("/rebalance-check", json={"portfolio_id": pid_empty, "fresh_target_weights": {}, "cash": 0.0})
check("empty-holdings portfolio with no prices supplied still returns 200", r.status_code == 200, r.text)
check("empty-holdings portfolio does NOT trigger a live fetch (nothing to price)",
      mock_live.call_count == 0, mock_live.call_count)

# Explicit current_price_by_symbol still takes precedence over live fetch (manual
# override / backtesting path, unchanged from before this session's change).
with patch.object(nps, "get_current_prices_and_sectors") as mock_live_unused:
    r = client.post("/rebalance-check", json={
        "portfolio_id": pid_r, "current_price_by_symbol": {"GTCO": 60.0, "ZENITHBANK": 20.0},
        "fresh_target_weights": {"GTCO": 0.25, "ZENITHBANK": 0.25}, "cash": 0.05,
    })
check("explicit current_price_by_symbol bypasses live fetch entirely (manual override still works)",
      r.status_code == 200 and mock_live_unused.call_count == 0, (r.status_code, mock_live_unused.call_count))


# =======================================================================
print("\n=== SECTION 6: Fundamentals caching ===")
# =======================================================================

call_count = {"n": 0}


def _counting_mock_fundamentals(symbol, as_of_date, current_price=None):
    call_count["n"] += 1
    return _mock_fundamentals(symbol, as_of_date, current_price=current_price)


with patch.object(fs, "get_point_in_time_fundamentals", side_effect=_counting_mock_fundamentals):
    call_count["n"] = 0
    r1 = client.get("/fundamentals/DANGCEM", params={"as_of_date": "2026-08-25", "current_price": 500.0})
    check("GET /fundamentals first call succeeds (200)", r1.status_code == 200, r1.text)
    check("GET /fundamentals first call hits the live scraper (cache miss)", call_count["n"] == 1, call_count)

    r2 = client.get("/fundamentals/DANGCEM", params={"as_of_date": "2026-08-25", "current_price": 550.0})
    check("GET /fundamentals second call (same symbol+as_of_date) is a cache HIT (no new scrape)",
          call_count["n"] == 1, call_count)
    check("cached pe is recomputed against the NEW current_price, not stale",
          abs(r2.json()["pe"] - (550.0 / 10.0)) < 1e-6, r2.json())
    check("cached roe is reused as-is",
          r2.json()["roe"] == _MOCK_ROE_BY_SYMBOL["DANGCEM"], r2.json())

    r3 = client.get("/fundamentals/DANGCEM", params={"as_of_date": "2024-01-15", "current_price": 500.0})
    check("GET /fundamentals for a DIFFERENT as_of_date is a fresh cache miss",
          call_count["n"] == 2, call_count)


# =======================================================================
print("\n=== SECTION 7: End-to-end -- construct via API, persist, rebalance-check by portfolio_id only ===")
# =======================================================================

with patch.object(fs, "get_point_in_time_fundamentals", side_effect=_mock_fundamentals):
    r = client.post("/portfolios", json={
        "name": "E2E Test Portfolio", "broker": "Ticker", "currency": "NGN",
        "initial_capital": 10_000_000, "inception_date": "2026-01-01",
    })
    pid_e2e = r.json()["id"]

    universe = {
        "price_series_by_symbol": {
            sym: [100 + i + idx * 3 for i in range(130)]
            for idx, sym in enumerate(["DANGCEM", "GTCO", "ZENITHBANK", "MTNN", "FIDELITYBK", "SEPLAT"])
        },
        "current_price_by_symbol": {
            "DANGCEM": 500.0, "GTCO": 45.0, "ZENITHBANK": 40.0,
            "MTNN": 200.0, "FIDELITYBK": 15.0, "SEPLAT": 3000.0,
        },
        "sector_by_symbol": {
            "DANGCEM": "Industrial", "GTCO": "Banking", "ZENITHBANK": "Banking",
            "MTNN": "Telecom", "FIDELITYBK": "Banking", "SEPLAT": "Energy",
        },
        "as_of_date": "2026-08-27",
        "pool_size": 6,
        "cash_buffer": 0.03,
    }
    r = client.post("/portfolio/construct", json=universe)
    check("E2E: /portfolio/construct succeeds", r.status_code == 200, r.text)
    constructed = r.json()

    # Persist the constructed target as this portfolio's holdings (share counts
    # derived from weights * capital / price, a plausible real usage pattern).
    capital = 10_000_000
    holdings_payload = []
    for sym, w in constructed["weights"].items():
        price = universe["current_price_by_symbol"][sym]
        shares = (w * capital) / price
        holdings_payload.append({
            "symbol": sym, "shares": shares, "avg_cost": price,
            "sector": universe["sector_by_symbol"][sym],
        })
    r = client.put(f"/portfolios/{pid_e2e}/holdings", json=holdings_payload)
    check("E2E: constructed portfolio persisted via PUT holdings", r.status_code == 200, r.text)

    # Now hit /rebalance-check with JUST portfolio_id (+ prices) and confirm it
    # pulls back weights matching what was constructed.
    r = client.post("/rebalance-check", json={
        "portfolio_id": pid_e2e,
        "current_price_by_symbol": universe["current_price_by_symbol"],
        "fresh_target_weights": constructed["weights"],
        "cash": constructed["cash_pct"],
        "persist_signals": False,
    })
    check("E2E: rebalance-check by portfolio_id alone succeeds", r.status_code == 200, r.text)
    rc = r.json()
    max_drift = max(abs(row["drift"]) for row in rc["drift_rows"] if row["drift"] is not None)
    check("E2E: pulled-back weights match constructed target (drift ~0, same target vs actual)",
          max_drift < 1e-6, (max_drift, rc["drift_rows"]))
    # NOTE: not asserting action_required is False here. This specific universe's
    # rank-weighted construction happens to place some names exactly at the 25%
    # stock cap; the weight -> shares -> weight round trip through the holdings
    # table introduces ~1e-15 floating point noise, which can land a name a
    # hair on either side of the strict `>` cap check in check_hard_breaches
    # (portfolio_service.py's own test suite explicitly validates "exactly at
    # cap is not a breach" -- this is expected, correct, pre-existing behavior,
    # not something this test should assert against). The meaningful assertion
    # here is that the pulled-back weights match the constructed target, above.
    check("E2E: drift_rows cover all constructed symbols (holdings round-trip didn't drop any)",
          {row["symbol"] for row in rc["drift_rows"]} == set(constructed["weights"].keys()), rc)


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
