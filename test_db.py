"""
test_db.py

Verification suite for db.py. Runs against a fresh local SQLite file
(deleted and recreated each run) since this sandbox has no network route
to Supabase -- the schema/queries are dialect-portable SQLAlchemy Core,
so this exercises the exact same code path Postgres will use in
production; only the DATABASE_URL differs.
"""

import os
import sys

DB_PATH = "/home/claude/pmc_api/test_pmc.db"
if os.path.exists(DB_PATH):
    os.remove(DB_PATH)
os.environ["DATABASE_URL"] = f"sqlite:///{DB_PATH}"

sys.path.insert(0, "/home/claude/pmc_api")
import db  # noqa: E402

db.init_db()

from datetime import date, timedelta

PASS, FAIL = [], []


def check(name, cond, detail=""):
    if cond:
        PASS.append(name)
        print(f"PASS: {name}")
    else:
        FAIL.append((name, detail))
        print(f"FAIL: {name} -- {detail}")


# ---------------------------------------------------------------------
print("\n=== Portfolio CRUD ===")
pid1 = db.create_portfolio("CFA PMC 2026", "Ticker (Cowrywise)", "NGN", 10_000_000, date(2026, 1, 1))
check("create_portfolio returns a string id", isinstance(pid1, str) and len(pid1) > 0, pid1)

p1 = db.get_portfolio(pid1)
check("get_portfolio returns correct name", p1["name"] == "CFA PMC 2026", p1)
check("get_portfolio defaults status=active", p1["status"] == "active", p1)
check("get_portfolio preserves initial_capital", float(p1["initial_capital"]) == 10_000_000, p1)

missing = db.get_portfolio("nonexistent-id")
check("get_portfolio returns None for missing id", missing is None, missing)

db.close_portfolio(pid1)
p1_closed = db.get_portfolio(pid1)
check("close_portfolio sets status=closed", p1_closed["status"] == "closed", p1_closed)
check("close_portfolio sets closed_date", p1_closed["closed_date"] is not None, p1_closed)


# ---------------------------------------------------------------------
print("\n=== Telegram chat <-> active portfolio ===")
pid2 = db.create_portfolio("Test Personal Account", "Ticker", "NGN", 500_000, date(2026, 6, 1))
db.set_active_portfolio("chat_100", pid2)
active = db.get_active_portfolio("chat_100")
check("get_active_portfolio returns the right portfolio", active is not None and active["id"] == pid2, active)

no_chat = db.get_active_portfolio("chat_does_not_exist")
check("get_active_portfolio returns None for unregistered chat", no_chat is None, no_chat)

db.register_chat("chat_200")  # registered but no active portfolio set yet
active_none = db.get_active_portfolio("chat_200")
check("get_active_portfolio returns None when registered but unset", active_none is None, active_none)

db.register_chat("chat_200")  # calling again must not error or duplicate
check("register_chat is idempotent (no error on repeat call)", True)


# ---------------------------------------------------------------------
print("\n=== Portfolio switch (the reusability requirement) ===")
pid_a = db.start_new_portfolio_for_chat(
    "chat_300", "CFA PMC 2026", "Ticker", "NGN", 10_000_000, date(2026, 1, 1),
)
db.upsert_holdings(pid_a, [{"symbol": "DANGCEM", "shares": 100, "avg_cost": 350.0, "sector": "Industrial"}])
db.log_trade(pid_a, "DANGCEM", "buy", 100, 350.0, 82.62, reason="initial_construction")

pid_b = db.start_new_portfolio_for_chat(
    "chat_300", "Post-Competition Personal Account", "Ticker", "NGN", 2_000_000, date(2026, 9, 1),
)
check("start_new_portfolio_for_chat returns a new distinct id", pid_b != pid_a, (pid_a, pid_b))

pa_after = db.get_portfolio(pid_a)
check("old portfolio auto-closed on switch", pa_after["status"] == "closed", pa_after)

active_after_switch = db.get_active_portfolio("chat_300")
check("chat now points at new portfolio", active_after_switch["id"] == pid_b, active_after_switch)

old_holdings_intact = db.get_holdings(pid_a)
check("old portfolio's holdings NOT deleted (history preserved)",
      len(old_holdings_intact) == 1 and old_holdings_intact[0]["symbol"] == "DANGCEM",
      old_holdings_intact)

new_holdings_empty = db.get_holdings(pid_b)
check("new portfolio starts with zero holdings (no bleed-through)",
      len(new_holdings_empty) == 0, new_holdings_empty)

old_trades_intact = db.get_trades(pid_a)
check("old portfolio's trade history preserved after switch", len(old_trades_intact) == 1, old_trades_intact)


# ---------------------------------------------------------------------
print("\n=== Portfolio isolation (two active portfolios, same symbol) ===")
pid_x = db.create_portfolio("Portfolio X", "Ticker", "NGN", 1_000_000, date(2026, 1, 1))
pid_y = db.create_portfolio("Portfolio Y", "Ticker", "NGN", 1_000_000, date(2026, 1, 1))
db.upsert_holdings(pid_x, [{"symbol": "ZENITHBANK", "shares": 50, "avg_cost": 40.0, "sector": "Banking"}])
db.upsert_holdings(pid_y, [{"symbol": "ZENITHBANK", "shares": 999, "avg_cost": 41.0, "sector": "Banking"}])
hx = db.get_holdings(pid_x)
hy = db.get_holdings(pid_y)
check("Portfolio X holding unaffected by Portfolio Y holding same symbol",
      len(hx) == 1 and float(hx[0]["shares"]) == 50, hx)
check("Portfolio Y holding correct and isolated", len(hy) == 1 and float(hy[0]["shares"]) == 999, hy)


# ---------------------------------------------------------------------
print("\n=== Holdings CRUD (full-replace semantics) ===")
pid_h = db.create_portfolio("Holdings Test", "Ticker", "NGN", 1_000_000, date(2026, 1, 1))
db.upsert_holdings(pid_h, [
    {"symbol": "A", "shares": 10, "avg_cost": 5.0, "sector": "Banking"},
    {"symbol": "B", "shares": 20, "avg_cost": 8.0, "sector": "Consumer"},
])
h1 = db.get_holdings(pid_h)
check("initial upsert_holdings creates both rows", len(h1) == 2, h1)

db.upsert_holdings(pid_h, [{"symbol": "C", "shares": 5, "avg_cost": 2.0, "sector": "Industrial"}])
h2 = db.get_holdings(pid_h)
check("second upsert_holdings fully replaces (A, B gone, only C remains)",
      len(h2) == 1 and h2[0]["symbol"] == "C", h2)


# ---------------------------------------------------------------------
print("\n=== holdings_to_weights ===")
holdings_list = [
    {"symbol": "A", "shares": 100}, {"symbol": "B", "shares": 50},
]
prices = {"A": 10.0, "B": 20.0}  # A market value=1000, B market value=1000
weights = db.holdings_to_weights(holdings_list, prices, cash=0.0)
check("holdings_to_weights: equal market value -> equal weights",
      abs(weights["A"] - 0.5) < 1e-9 and abs(weights["B"] - 0.5) < 1e-9, weights)

weights_with_cash = db.holdings_to_weights(holdings_list, prices, cash=500.0)  # total = 2500
check("holdings_to_weights: cash correctly dilutes weights",
      abs(weights_with_cash["A"] - (1000 / 2500)) < 1e-9, weights_with_cash)

weights_missing_price = db.holdings_to_weights([{"symbol": "Z", "shares": 10}], {}, cash=0.0)
check("holdings_to_weights: missing price treated as 0 value, not a crash",
      weights_missing_price["Z"] == 0.0, weights_missing_price)

weights_zero_total = db.holdings_to_weights([], {}, cash=0.0)
check("holdings_to_weights: zero total portfolio doesn't divide-by-zero crash",
      weights_zero_total == {}, weights_zero_total)


# ---------------------------------------------------------------------
print("\n=== log_trade: buy (new position) ===")
pid_t = db.create_portfolio("Trade Test", "Ticker", "NGN", 1_000_000, date(2026, 1, 1))
db.log_trade(pid_t, "GTCO", "buy", 100, 45.0, 68.85, reason="initial_construction", sector="Banking")
th = db.get_holdings(pid_t)
check("buy on empty portfolio creates a new holding", len(th) == 1 and th[0]["symbol"] == "GTCO", th)
check("new holding avg_cost = buy price", float(th[0]["avg_cost"]) == 45.0, th)
check("new holding shares = buy shares", float(th[0]["shares"]) == 100, th)

trades_t = db.get_trades(pid_t)
check("trade logged in trades table", len(trades_t) == 1 and trades_t[0]["side"] == "buy", trades_t)


print("\n=== log_trade: buy (add to existing position, weighted avg cost) ===")
db.log_trade(pid_t, "GTCO", "buy", 100, 55.0, 84.15, reason="drift")
th2 = db.get_holdings(pid_t)
gtco = next(h for h in th2 if h["symbol"] == "GTCO")
expected_avg_cost = ((100 * 45.0) + (100 * 55.0)) / 200  # = 50.0
check("second buy: shares summed correctly", float(gtco["shares"]) == 200, gtco)
check("second buy: avg_cost is correctly weighted", abs(float(gtco["avg_cost"]) - expected_avg_cost) < 1e-9,
      f"got={gtco['avg_cost']}, expected={expected_avg_cost}")


print("\n=== log_trade: sell (partial, avg_cost unchanged) ===")
db.log_trade(pid_t, "GTCO", "sell", 50, 60.0, 78.48, reason="drift")
th3 = db.get_holdings(pid_t)
gtco2 = next(h for h in th3 if h["symbol"] == "GTCO")
check("partial sell: shares reduced correctly", float(gtco2["shares"]) == 150, gtco2)
check("partial sell: avg_cost unchanged by a sell", abs(float(gtco2["avg_cost"]) - 50.0) < 1e-9, gtco2)


print("\n=== log_trade: sell (full position, holding removed) ===")
db.log_trade(pid_t, "GTCO", "sell", 150, 62.0, 100.28, reason="hard_breach")
th4 = db.get_holdings(pid_t)
check("full sell removes the holding row entirely", len(th4) == 0, th4)

all_trades = db.get_trades(pid_t)
check("all 4 trades present in trade history despite holding being closed out",
      len(all_trades) == 4, len(all_trades))


print("\n=== log_trade: invalid operations rejected, root state unchanged ===")
sell_nonexistent_raised = False
try:
    db.log_trade(pid_t, "NEVERBOUGHT", "sell", 10, 20.0, 1.0)
except ValueError:
    sell_nonexistent_raised = True
check("selling a symbol never held raises ValueError", sell_nonexistent_raised)

db.log_trade(pid_t, "MTNN", "buy", 100, 200.0, 306.0, reason="initial_construction")
oversell_raised = False
try:
    db.log_trade(pid_t, "MTNN", "sell", 150, 210.0, 1.0)  # only 100 held
except ValueError:
    oversell_raised = True
check("overselling beyond held shares raises ValueError", oversell_raised)

mtnn_after_failed_oversell = next(h for h in db.get_holdings(pid_t) if h["symbol"] == "MTNN")
check("failed oversell did not mutate holdings (still 100 shares)",
      float(mtnn_after_failed_oversell["shares"]) == 100, mtnn_after_failed_oversell)

bad_side_raised = False
try:
    db.log_trade(pid_t, "MTNN", "yeet", 10, 200.0, 1.0)
except ValueError:
    bad_side_raised = True
check("invalid side value raises ValueError (not silently accepted)", bad_side_raised)


# ---------------------------------------------------------------------
print("\n=== delete_trade: undo via full replay, not delta-reversal ===")
pid_d = db.create_portfolio("Delete Trade Test", "Ticker", "NGN", 1_000_000, date(2026, 1, 1))

# Case 1: only trade for a symbol -- deleting it fully removes the holding
tid_only = db.log_trade(pid_d, "SOLO", "buy", 100, 10.0, 1.53, sector="Industrial")
result_solo = db.delete_trade(pid_d, tid_only)
check("delete_trade: deleting the only trade for a symbol returns None (position fully closed)",
      result_solo is None, result_solo)
check("delete_trade: holding row actually removed from DB", db.get_holdings(pid_d) == [], db.get_holdings(pid_d))
nonexistent_raised = False
try:
    db.delete_trade(pid_d, tid_only)  # already deleted -- second delete should fail
except ValueError:
    nonexistent_raised = True
check("delete_trade: deleting an already-deleted (nonexistent) trade id raises ValueError", nonexistent_raised)

# Case 2: delete the LAST of several buys -- straightforward case
tid_b1 = db.log_trade(pid_d, "MULTI", "buy", 100, 10.0, 1.53, sector="Banking")
tid_b2 = db.log_trade(pid_d, "MULTI", "buy", 100, 20.0, 3.06)
# avg_cost here = (100*10 + 100*20)/200 = 15.0
result_del_last = db.delete_trade(pid_d, tid_b2)
check("delete_trade (delete last buy): shares correctly reverted to 100",
      float(result_del_last["shares"]) == 100, result_del_last)
check("delete_trade (delete last buy): avg_cost correctly reverted to 10.0 (the first buy's price)",
      abs(float(result_del_last["avg_cost"]) - 10.0) < 1e-9, result_del_last)
check("delete_trade (delete last buy): sector preserved across the replay",
      result_del_last["sector"] == "Banking", result_del_last)

# Case 3: delete an EARLIER buy among three -- the case naive delta-reversal gets wrong,
# replay gets right (this is the actual correctness test for the chosen approach).
tid_c1 = db.log_trade(pid_d, "LAYER", "buy", 100, 10.0, 1.53, sector="Consumer")
tid_c2 = db.log_trade(pid_d, "LAYER", "buy", 100, 20.0, 3.06)
tid_c3 = db.log_trade(pid_d, "LAYER", "buy", 100, 30.0, 4.59)
# avg_cost now = (100*10 + 100*20 + 100*30)/300 = 20.0
# Deleting the FIRST buy (tid_c1) should leave exactly what two buys at 20 and 30 produce:
# avg_cost = (100*20 + 100*30)/200 = 25.0, shares = 200 -- NOT a naive "subtract tid_c1's
# contribution from the current avg_cost", which would not correctly reconstruct this.
result_del_middle = db.delete_trade(pid_d, tid_c1)
check("delete_trade (delete an EARLIER buy, not the last): shares correctly = 200 (300-100)",
      float(result_del_middle["shares"]) == 200, result_del_middle)
check("delete_trade (delete an EARLIER buy): avg_cost correctly replayed to 25.0, "
      "the actual result of a fresh 200@20/30 buy sequence -- proves replay, not delta-reversal",
      abs(float(result_del_middle["avg_cost"]) - 25.0) < 1e-9,
      f"got={result_del_middle['avg_cost']}, expected=25.0")

# Case 4: delete a sell -- shares added back, avg_cost untouched (sells never affect avg_cost)
tid_s1 = db.log_trade(pid_d, "SELLBACK", "buy", 100, 10.0, 1.53, sector="Oil & Gas")
tid_s2 = db.log_trade(pid_d, "SELLBACK", "sell", 40, 15.0, 1.31)
holding_before_undo_sell = next(h for h in db.get_holdings(pid_d) if h["symbol"] == "SELLBACK")
check("delete_trade setup: sell reduced shares to 60 first", float(holding_before_undo_sell["shares"]) == 60,
      holding_before_undo_sell)
result_del_sell = db.delete_trade(pid_d, tid_s2)
check("delete_trade (delete a sell): shares correctly restored to 100",
      float(result_del_sell["shares"]) == 100, result_del_sell)
check("delete_trade (delete a sell): avg_cost unaffected (still 10.0, sells never touch avg_cost)",
      abs(float(result_del_sell["avg_cost"]) - 10.0) < 1e-9, result_del_sell)

# Case 5: cross-portfolio isolation -- deleting a trade in one portfolio must not touch another
pid_d2 = db.create_portfolio("Delete Trade Test 2", "Ticker", "NGN", 1_000_000, date(2026, 1, 1))
db.log_trade(pid_d2, "MULTI", "buy", 999, 5.0, 1.0, sector="Banking")  # same symbol, different portfolio
db.delete_trade(pid_d, tid_b1)  # delete the one remaining MULTI trade in pid_d
holding_pid_d2_after = next(h for h in db.get_holdings(pid_d2) if h["symbol"] == "MULTI")
check("delete_trade: deleting a trade in one portfolio does not affect another portfolio's same-symbol holding",
      float(holding_pid_d2_after["shares"]) == 999, holding_pid_d2_after)

wrong_portfolio_raised = False
try:
    db.delete_trade(pid_d, next(t["id"] for t in db.get_trades(pid_d2) if t["symbol"] == "MULTI"))
except ValueError:
    wrong_portfolio_raised = True
check("delete_trade: trade id valid but wrong portfolio_id raises ValueError (not cross-portfolio deletable)",
      wrong_portfolio_raised)

# Case 6: the oversell guard -- undoing an EARLIER buy that a LATER sell already depended
# on must be refused outright, not silently produce an inconsistent/wrong holding.
tid_g1 = db.log_trade(pid_d, "GUARD", "buy", 100, 10.0, 1.53, sector="Banking")
tid_g2 = db.log_trade(pid_d, "GUARD", "sell", 80, 12.0, 1.31)  # depends on tid_g1's 100 shares
guard_raised, guard_msg = False, None
try:
    db.delete_trade(pid_d, tid_g1)  # would leave the sell of 80 with only 0 shares available
except ValueError as e:
    guard_raised, guard_msg = True, str(e)
check("delete_trade: undoing a buy that a later sell depends on is REFUSED (oversell guard)",
      guard_raised, guard_msg)
check("delete_trade oversell guard: neither the trade nor the holding was touched (transaction fully rolled back)",
      len(db.get_trades(pid_d)) >= 2 and any(t["id"] == tid_g1 for t in db.get_trades(pid_d)),
      db.get_trades(pid_d))
holding_guard_untouched = next(h for h in db.get_holdings(pid_d) if h["symbol"] == "GUARD")
check("delete_trade oversell guard: holding state completely unchanged after the refused delete",
      float(holding_guard_untouched["shares"]) == 20 and abs(float(holding_guard_untouched["avg_cost"]) - 10.0) < 1e-9,
      holding_guard_untouched)

# But undoing the LATER sell first, THEN the earlier buy, is perfectly valid (order matters,
# correctly so) -- confirms the guard is about actual consistency, not a blanket restriction.
db.delete_trade(pid_d, tid_g2)  # undo the sell first -- now safe to undo the buy
result_guard_after_sell_undone = db.delete_trade(pid_d, tid_g1)
check("delete_trade: undoing the dependent sell FIRST then the earlier buy succeeds cleanly",
      result_guard_after_sell_undone is None, result_guard_after_sell_undone)


# ---------------------------------------------------------------------
print("\n=== Unique constraint on holdings(portfolio_id, symbol) ===")
pid_u = db.create_portfolio("Uniqueness Test", "Ticker", "NGN", 1_000_000, date(2026, 1, 1))
constraint_raised = False
try:
    with db.engine.begin() as conn:
        conn.execute(db.holdings.insert().values(
            id=db._uuid(), portfolio_id=pid_u, symbol="DUP", shares=10, avg_cost=1.0,
        ))
        conn.execute(db.holdings.insert().values(
            id=db._uuid(), portfolio_id=pid_u, symbol="DUP", shares=20, avg_cost=1.0,
        ))
except Exception:
    constraint_raised = True
check("DB rejects a duplicate (portfolio_id, symbol) row at the schema level",
      constraint_raised, "expected IntegrityError-class exception on duplicate insert")


# ---------------------------------------------------------------------
print("\n=== Signals: dedup on unacknowledged, re-fires after acknowledge ===")
pid_s = db.create_portfolio("Signal Test", "Ticker", "NGN", 1_000_000, date(2026, 1, 1))
sid1 = db.log_signal_if_new(pid_s, "drift", "GTCO", {"drift": 0.08})
check("first signal insert returns an id", sid1 is not None, sid1)

sid2 = db.log_signal_if_new(pid_s, "drift", "GTCO", {"drift": 0.09})
check("duplicate unacknowledged signal (same type+symbol) is suppressed", sid2 is None, sid2)

unacked = db.get_unacknowledged_signals(pid_s)
check("still only one unacknowledged signal on record", len(unacked) == 1, unacked)

sid_diff_symbol = db.log_signal_if_new(pid_s, "drift", "ZENITHBANK", {"drift": 0.07})
check("different symbol, same type -> new signal allowed (not deduped)", sid_diff_symbol is not None, sid_diff_symbol)

db.acknowledge_signal(sid1)
unacked_after_ack = db.get_unacknowledged_signals(pid_s)
check("acknowledged signal no longer in unacknowledged list",
      all(s["id"] != sid1 for s in unacked_after_ack), unacked_after_ack)

sid3 = db.log_signal_if_new(pid_s, "drift", "GTCO", {"drift": 0.10})
check("after acknowledging, the same signal type+symbol CAN fire again", sid3 is not None, sid3)


print("\n=== Signals: cross-portfolio isolation ===")
pid_s2 = db.create_portfolio("Signal Test 2", "Ticker", "NGN", 1_000_000, date(2026, 1, 1))
sid_other_portfolio = db.log_signal_if_new(pid_s2, "drift", "GTCO", {"drift": 0.08})
check("same type+symbol but different portfolio is NOT deduped against portfolio 1",
      sid_other_portfolio is not None, sid_other_portfolio)


# ---------------------------------------------------------------------
print("\n=== list_active_portfolios (supports bot.py's /list_portfolios, /use_portfolio) ===")
before_count = len(db.list_active_portfolios())

pid_la1 = db.create_portfolio("List Test A", "Ticker", "NGN", 1_000_000, date(2026, 1, 1))
pid_la2 = db.create_portfolio("List Test B", "Ticker", "NGN", 500_000, date(2026, 2, 1))
pid_la3 = db.create_portfolio("List Test C (will be closed)", "Ticker", "NGN", 250_000, date(2026, 3, 1))
db.close_portfolio(pid_la3)

active_list = db.list_active_portfolios()
check("list_active_portfolios includes newly-created active portfolios",
      pid_la1 in [p["id"] for p in active_list] and pid_la2 in [p["id"] for p in active_list], active_list)
check("list_active_portfolios EXCLUDES a closed portfolio",
      pid_la3 not in [p["id"] for p in active_list], active_list)
check("list_active_portfolios grew by exactly 2 (the two active ones, not the closed one)",
      len(active_list) == before_count + 2, (before_count, len(active_list)))

names_in_list = {p["id"]: p["name"] for p in active_list}
check("list_active_portfolios returns full portfolio dicts (name matches)",
      names_in_list[pid_la1] == "List Test A" and names_in_list[pid_la2] == "List Test B", names_in_list)


# ---------------------------------------------------------------------
print("\n=== Fundamentals cache (shared across portfolios, keyed by symbol+as_of_date) ===")
d1 = date(2026, 8, 1)
cached_before = db.get_cached_fundamentals("DANGCEM", d1)
check("no cache entry before first write -> None", cached_before is None, cached_before)

db.cache_fundamentals("DANGCEM", d1, roe=42.33, pe=17.54, eps_used=59.86, eps_period_ending=date(2025, 12, 31))
cached_after = db.get_cached_fundamentals("DANGCEM", d1)
check("cache read-back matches what was written", cached_after is not None and abs(float(cached_after["pe"]) - 17.54) < 1e-6, cached_after)

# different as_of_date for the same symbol is a DIFFERENT cache entry (point-in-time correctness)
d2 = date(2024, 1, 15)
cached_d2 = db.get_cached_fundamentals("DANGCEM", d2)
check("different as_of_date for same symbol is a separate cache miss (no cross-date bleed)",
      cached_d2 is None, cached_d2)

db.cache_fundamentals("DANGCEM", d2, roe=37.07, pe=47.15, eps_used=22.27, eps_period_ending=date(2022, 12, 31))
cached_d2_after = db.get_cached_fundamentals("DANGCEM", d2)
check("second as_of_date cached independently, doesn't overwrite the first",
      cached_d2_after is not None and abs(float(cached_d2_after["pe"]) - 47.15) < 1e-6, cached_d2_after)
cached_d1_still_intact = db.get_cached_fundamentals("DANGCEM", d1)
check("original d1 cache entry untouched by writing d2", abs(float(cached_d1_still_intact["pe"]) - 17.54) < 1e-6, cached_d1_still_intact)

# staleness
db.cache_fundamentals("STALETEST", d1, roe=10.0, pe=5.0, eps_used=1.0, eps_period_ending=d1)
fresh_read = db.get_cached_fundamentals("STALETEST", d1, max_age_days=30)
check("freshly-written cache entry is NOT considered stale", fresh_read is not None, fresh_read)
stale_read = db.get_cached_fundamentals("STALETEST", d1, max_age_days=0)
check("max_age_days=0 correctly treats even a same-instant write as stale (boundary case)",
      stale_read is None, stale_read)

# upsert overwrite behavior
db.cache_fundamentals("DANGCEM", d1, roe=99.0, pe=1.0, eps_used=1.0, eps_period_ending=d1)
overwritten = db.get_cached_fundamentals("DANGCEM", d1)
check("re-caching same (symbol, as_of_date) overwrites rather than duplicating",
      abs(float(overwritten["pe"]) - 1.0) < 1e-6, overwritten)


# ---------------------------------------------------------------------
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
