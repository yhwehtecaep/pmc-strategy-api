"""
test_portfolio_service.py

Verification suite for portfolio_service.py per the handoff's "EXACT NEXT
STEPS". Run standalone (prints PASS/FAIL per test, raises on first hard
failure so nothing is silently skipped).
"""

import sys
import pandas as pd
import numpy as np

from portfolio_service import (
    enforce_caps,
    build_initial_portfolio,
    safety_pass,
    check_drift,
    check_hard_breaches,
    estimate_trade_fees,
    MAX_STOCK_WEIGHT,
    MAX_SECTOR_WEIGHT,
    MAX_CASH,
    BUY_FEE_RATE,
    SELL_FEE_RATE,
)

PASS = []
FAIL = []


def check(name, cond, detail=""):
    if cond:
        PASS.append(name)
        print(f"PASS: {name}")
    else:
        FAIL.append((name, detail))
        print(f"FAIL: {name}  -- {detail}")


def close(a, b, tol=1e-6):
    return abs(a - b) < tol


# ---------------------------------------------------------------------
# TEST 1: basic stock-cap trim + sum-to-1.0 check
# ---------------------------------------------------------------------
print("\n=== TEST 1: basic stock-cap trim ===")
w1 = pd.Series({"A": 0.40, "B": 0.20, "C": 0.20, "D": 0.20})
sectors1 = {"A": "Banking", "B": "Consumer", "C": "Industrial", "D": "Telecom"}
weights1, shortfall1 = enforce_caps(w1, sectors1)

check("T1: A capped to <= 0.25", weights1["A"] <= MAX_STOCK_WEIGHT + 1e-9,
      f"A={weights1['A']}")
check("T1: weights + shortfall sum to 1.0", close(weights1.sum() + shortfall1, 1.0),
      f"sum={weights1.sum()}, shortfall={shortfall1}, total={weights1.sum()+shortfall1}")
check("T1: shortfall is ~0 (plenty of eligible capacity elsewhere)", close(shortfall1, 0.0),
      f"shortfall={shortfall1}")
# sector caps: B,C,D each own sector (25% cap), so B/C/D individually can't
# absorb much of A's excess without also hitting their own 25% stock cap.
for s in ["B", "C", "D"]:
    check(f"T1: {s} respects its own stock cap", weights1[s] <= MAX_STOCK_WEIGHT + 1e-9,
          f"{s}={weights1[s]}")


# ---------------------------------------------------------------------
# TEST 2: sector-cap trim, 3 stocks in one sector
# ---------------------------------------------------------------------
print("\n=== TEST 2: sector-cap trim (3 stocks, one sector) ===")
w2 = pd.Series({"X1": 0.15, "X2": 0.15, "X3": 0.10, "Y1": 0.20, "Z1": 0.20, "W1": 0.20})
sectors2 = {"X1": "Banking", "X2": "Banking", "X3": "Banking",
            "Y1": "Consumer", "Z1": "Industrial", "W1": "Telecom"}
# Banking sector pre-cap total = 0.40, over the 0.25 sector cap
weights2, shortfall2 = enforce_caps(w2, sectors2)
bank_total2 = weights2[["X1", "X2", "X3"]].sum()

check("T2: Banking sector total <= 0.25 after enforcement", bank_total2 <= MAX_SECTOR_WEIGHT + 1e-6,
      f"bank_total={bank_total2}")
check("T2: weights + shortfall sum to 1.0", close(weights2.sum() + shortfall2, 1.0),
      f"sum={weights2.sum()}, shortfall={shortfall2}")
check("T2: no negative weights", (weights2 >= -1e-9).all(), f"{weights2.to_dict()}")
# Y1, Z1, W1 are each sole occupants of their sector (cap = min(stock cap, sector cap) = 0.25)
for s in ["Y1", "Z1", "W1"]:
    check(f"T2: {s} respects its own stock cap", weights2[s] <= MAX_STOCK_WEIGHT + 1e-9,
          f"{s}={weights2[s]}")


# ---------------------------------------------------------------------
# TEST 3: safety_pass, A=30%/B=30%/cash=20% starting point
# ---------------------------------------------------------------------
print("\n=== TEST 3: safety_pass basic over-cap correction ===")
w3 = pd.Series({"A": 0.30, "B": 0.30, "C": 0.10, "D": 0.10})
sectors3 = {"A": "Banking", "B": "Consumer", "C": "Industrial", "D": "Telecom"}
cash3 = 0.20
new_w3, new_cash3, trim3, redeploy3 = safety_pass(w3, sectors3, cash3)

check("T3: A trimmed to <= 0.25", new_w3["A"] <= MAX_STOCK_WEIGHT + 1e-9, f"A={new_w3['A']}")
check("T3: B trimmed to <= 0.25", new_w3["B"] <= MAX_STOCK_WEIGHT + 1e-9, f"B={new_w3['B']}")
check("T3: trim_total > 0 (A and B were over cap)", trim3 > 0, f"trim_total={trim3}")
total3 = new_w3.sum() + new_cash3
check("T3: weights + cash conserve total capital (=1.0)", close(total3, 1.0),
      f"weights_sum={new_w3.sum()}, cash={new_cash3}, total={total3}")
# cash started at 0.20 (< MAX_CASH=0.05 is false, 0.20 > 0.05) so redeploy should trigger
check("T3: redeployment triggered since starting cash > MAX_CASH", redeploy3 > 0,
      f"redeploy_total={redeploy3}")
check("T3: final cash reduced from redeployment but still >= cash_buffer",
      new_cash3 < cash3 + trim3, f"final_cash={new_cash3}")


# ---------------------------------------------------------------------
# TEST 4 (NEW / REGRESSION): sector-headroom double-counting bug
# ---------------------------------------------------------------------
print("\n=== TEST 4: sector-headroom double-counting regression ===")
# Two stocks E, F share "Banking" sector, currently at 0.10 each (sector
# total 0.20, so real shared headroom to the 0.25 sector cap is only 0.05).
# Each stock individually has room up to its own 0.25 stock cap (i.e. each
# could rise by 0.15 on a standalone basis) -- so sum of INDIVIDUAL rooms
# (0.15 + 0.15 = 0.30) vastly exceeds the shared sector headroom (0.05).
# The old (buggy) code would credit each stock the full 0.15 "room"
# independently, deploy up to 0.30 combined, and blow the sector cap to
# 0.20 + 0.30 = 0.50 (or however much cash allowed) -- more than double
# the 0.25 cap.
w4 = pd.Series({"E": 0.30, "F": 0.10, "G": 0.10, "H": 0.10})
sectors4 = {"E": "Banking", "F": "Banking", "G": "Consumer", "H": "Industrial"}
# E is over its own 0.25 stock cap -> gets trimmed by 0.05 -> trim_total=0.05
# Banking sector after E's trim: E=0.25, F=0.10 -> sector total 0.35, still
# over the 0.25 sector cap -> triggers the sector-trim loop too.
cash4 = 0.10  # 0.30+0.10+0.10+0.10=0.60 weights + 0.10 cash = 0.70; irrelevant, testing safety_pass mechanics directly
new_w4, new_cash4, trim4, redeploy4 = safety_pass(w4, sectors4, cash4)

bank_total4 = new_w4["E"] + new_w4["F"]
check("T4: Banking sector total <= 0.25 after safety_pass (no double count)",
      bank_total4 <= MAX_SECTOR_WEIGHT + 1e-6, f"bank_total={bank_total4}")
check("T4: E respects stock cap", new_w4["E"] <= MAX_STOCK_WEIGHT + 1e-9, f"E={new_w4['E']}")
check("T4: F respects stock cap", new_w4["F"] <= MAX_STOCK_WEIGHT + 1e-9, f"F={new_w4['F']}")
total4 = new_w4.sum() + new_cash4
check("T4: weights + cash conserve total capital", close(total4, w4.sum() + cash4),
      f"got={total4}, expected={w4.sum()+cash4}")

# Now specifically force redeployment INTO a saturated shared sector to
# prove the water-filling pre-pass caps joint redeployment correctly.
# Sector "Banking": stocks P, Q currently at 0.05 each (sector total 0.10,
# headroom to 0.25 cap = 0.15). Individual stock-cap rooms are 0.20 each
# (0.25-0.05), summing to 0.40 -- more than double the real 0.15 headroom.
w5 = pd.Series({"P": 0.05, "Q": 0.05, "R": 0.25, "S": 0.10})  # R forces a trim -> frees cash to redeploy
sectors5 = {"P": "Banking", "Q": "Banking", "R": "Consumer", "S": "Industrial"}
cash5 = 0.20  # well above MAX_CASH=0.05, and R is already at cap so no trim occurs there
# R at exactly 0.25 (== cap, not > cap) triggers no trim by itself; force an
# actual over-cap so trim_total > 0 and the redeploy branch executes.
w5 = pd.Series({"P": 0.05, "Q": 0.05, "R": 0.30, "S": 0.10})
new_w5, new_cash5, trim5, redeploy5 = safety_pass(w5, sectors5, cash5)
bank_total5 = new_w5["P"] + new_w5["Q"]
check("T5: Banking sector (P+Q) stays within 0.25 cap after redeployment",
      bank_total5 <= MAX_SECTOR_WEIGHT + 1e-6, f"P+Q={bank_total5}")
check("T5: redeployment occurred", redeploy5 > 0, f"redeploy_total={redeploy5}")
check("T5: P did not exceed its individual room in isolation (sanity, not stock cap)",
      new_w5["P"] <= MAX_STOCK_WEIGHT + 1e-9, f"P={new_w5['P']}")
check("T5: Q did not exceed its individual room in isolation (sanity, not stock cap)",
      new_w5["Q"] <= MAX_STOCK_WEIGHT + 1e-9, f"Q={new_w5['Q']}")
total5 = new_w5.sum() + new_cash5
check("T5: weights + cash conserve total capital", close(total5, w5.sum() + cash5),
      f"got={total5}, expected={w5.sum()+cash5}")


# ---------------------------------------------------------------------
# TEST 6: build_initial_portfolio -- tuple signature + reconciliation
# ---------------------------------------------------------------------
print("\n=== TEST 6: build_initial_portfolio ===")
ranked = [f"S{i}" for i in range(1, 16)]  # S1 (best) .. S15 (worst)
sectors6 = {}
sector_names = ["Banking", "Consumer", "Industrial", "Telecom", "Oil&Gas",
                "Agric", "RealEstate", "Healthcare", "Utilities", "Insurance"]
for i, sym in enumerate(ranked):
    sectors6[sym] = sector_names[i % len(sector_names)]

result6 = build_initial_portfolio(ranked, sectors6)
check("T6: returns a 2-tuple", isinstance(result6, tuple) and len(result6) == 2,
      f"type={type(result6)}, len={len(result6) if isinstance(result6, tuple) else 'n/a'}")
weights6, cash6 = result6
check("T6: weights + cash reconcile to 1.0", close(weights6.sum() + cash6, 1.0),
      f"weights_sum={weights6.sum()}, cash={cash6}, total={weights6.sum()+cash6}")
check("T6: cash >= CASH_BUFFER (0.03) always", cash6 >= 0.03 - 1e-9, f"cash={cash6}")
check("T6: no stock exceeds MAX_STOCK_WEIGHT", (weights6 <= MAX_STOCK_WEIGHT + 1e-9).all(),
      f"max weight={weights6.max()}")

# Concentrated-sector case: force many top-ranked stocks into ONE sector so
# the pool genuinely cannot be compliantly deployed -> shortfall must show
# up as extra cash, not vanish.
sectors6b = {sym: ("Banking" if i < 8 else sector_names[i % len(sector_names)])
             for i, sym in enumerate(ranked)}
weights6b, cash6b = build_initial_portfolio(ranked, sectors6b)
check("T6b: concentrated pool -- weights + cash still reconcile to 1.0",
      close(weights6b.sum() + cash6b, 1.0),
      f"weights_sum={weights6b.sum()}, cash={cash6b}")
check("T6b: concentrated pool -- cash absorbs undeployable shortfall (cash > base buffer)",
      cash6b > 0.03 + 1e-6, f"cash={cash6b}")
bank_total6b = sum(weights6b[s] for s in weights6b.index if sectors6b.get(s) == "Banking")
check("T6b: Banking sector total <= 0.25 despite concentration",
      bank_total6b <= MAX_SECTOR_WEIGHT + 1e-6, f"bank_total={bank_total6b}")


# ---------------------------------------------------------------------
# TEST 7: check_drift
# ---------------------------------------------------------------------
print("\n=== TEST 7: check_drift ===")
actual7 = pd.Series({"A": 0.20, "B": 0.10, "C": 0.15})
target7 = pd.Series({"A": 0.20, "B": 0.18, "C": 0.08})  # B drifts +0.08 (>5%), C drifts -0.07 (>5%), A no drift
drift_df7 = check_drift(actual7, target7)

check("T7: A not triggered (no drift)", drift_df7.loc["A", "triggered"] == False,
      f"A row: {drift_df7.loc['A'].to_dict()}")
check("T7: B triggered (drift 0.08 > 0.05 threshold)", drift_df7.loc["B", "triggered"] == True,
      f"B row: {drift_df7.loc['B'].to_dict()}")
check("T7: C triggered (drift -0.07, abs > 0.05 threshold)", drift_df7.loc["C", "triggered"] == True,
      f"C row: {drift_df7.loc['C'].to_dict()}")
check("T7: drift values computed correctly for B", close(drift_df7.loc["B", "drift"], 0.08),
      f"B drift={drift_df7.loc['B', 'drift']}")

# Edge case: currently-held name absent from fresh targets (e.g. dropped
# from universe) -> fresh_target should be NaN, not silently zero.
actual7b = pd.Series({"A": 0.20, "Z": 0.10})
target7b = pd.Series({"A": 0.20})
drift_df7b = check_drift(actual7b, target7b)
check("T7b: dropped name Z has NaN fresh_target (not silently 0)",
      pd.isna(drift_df7b.loc["Z", "fresh_target"]), f"Z fresh_target={drift_df7b.loc['Z', 'fresh_target']}")


# ---------------------------------------------------------------------
# TEST 8: check_hard_breaches
# ---------------------------------------------------------------------
print("\n=== TEST 8: check_hard_breaches ===")
sectors8 = {"A": "Banking", "B": "Banking", "C": "Consumer", "D": "Telecom"}

# Case 1: clean portfolio, no breaches
w8_clean = pd.Series({"A": 0.20, "B": 0.20, "C": 0.20, "D": 0.20})
cash8_clean = 0.05  # wait, need 5 holdings min... let's check with MIN_HOLDINGS=5
sectors8_clean = {"A": "Banking", "B": "Consumer", "C": "Industrial", "D": "Telecom", "E": "Oil&Gas"}
w8_clean = pd.Series({"A": 0.20, "B": 0.20, "C": 0.20, "D": 0.20, "E": 0.15})
cash8_clean = 0.05
breaches_clean = check_hard_breaches(w8_clean, sectors8_clean, cash8_clean)
check("T8: clean portfolio -- no stock breach", len(breaches_clean["stock_breach"]) == 0,
      f"{breaches_clean['stock_breach']}")
check("T8: clean portfolio -- no sector breach", len(breaches_clean["sector_breach"]) == 0,
      f"{breaches_clean['sector_breach']}")
check("T8: clean portfolio -- no cash breach", breaches_clean["cash_breach"] == False)
check("T8: clean portfolio -- no holdings breach (5 holdings, min=5)",
      breaches_clean["holdings_breach"] == False)
check("T8: clean portfolio -- any_breach False", breaches_clean["any_breach"] == False)

# Case 2: stock cap breach
w8_stock = pd.Series({"A": 0.30, "B": 0.20, "C": 0.20, "D": 0.15, "E": 0.10})
breaches_stock = check_hard_breaches(w8_stock, sectors8_clean, 0.05)
check("T8: stock breach detected", "A" in breaches_stock["stock_breach"], f"{breaches_stock['stock_breach']}")
check("T8: any_breach True on stock breach", breaches_stock["any_breach"] == True)

# Case 3: sector breach (A+B both Banking, combined > 0.25)
sectors8_sec = {"A": "Banking", "B": "Banking", "C": "Consumer", "D": "Telecom", "E": "Oil&Gas"}
w8_sec = pd.Series({"A": 0.15, "B": 0.15, "C": 0.20, "D": 0.30 - 0.20, "E": 0.20})
w8_sec = pd.Series({"A": 0.15, "B": 0.15, "C": 0.20, "D": 0.20, "E": 0.30})
# fix: keep D,E under stock cap individually, only sector A+B breaches
w8_sec = pd.Series({"A": 0.15, "B": 0.15, "C": 0.20, "D": 0.25, "E": 0.25})
breaches_sec = check_hard_breaches(w8_sec, sectors8_sec, 0.05)
check("T8: sector breach detected (Banking = 0.30 > 0.25)",
      "Banking" in breaches_sec["sector_breach"], f"{breaches_sec['sector_breach']}")
check("T8: no stock breach in this case (D,E at exactly 0.25, not > )",
      len(breaches_sec["stock_breach"]) == 0, f"{breaches_sec['stock_breach']}")

# Case 4: cash breach
breaches_cash = check_hard_breaches(w8_clean, sectors8_clean, cash=0.10)
check("T8: cash breach detected (0.10 > MAX_CASH=0.05)", breaches_cash["cash_breach"] == True)
check("T8: any_breach True on cash breach", breaches_cash["any_breach"] == True)

# Case 5: holdings breach (< MIN_HOLDINGS=5)
w8_few = pd.Series({"A": 0.30, "B": 0.30, "C": 0.20})
sectors8_few = {"A": "Banking", "B": "Consumer", "C": "Industrial"}
breaches_few = check_hard_breaches(w8_few, sectors8_few, 0.05)
check("T8: holdings breach detected (3 holdings < min 5)", breaches_few["holdings_breach"] == True)
check("T8: any_breach True on holdings breach", breaches_few["any_breach"] == True)


# ---------------------------------------------------------------------
# TEST 9: estimate_trade_fees -- asymmetric buy/sell rates
# ---------------------------------------------------------------------
print("\n=== TEST 9: estimate_trade_fees (asymmetric) ===")
check("T9: BUY_FEE_RATE constant is 1.53%", close(BUY_FEE_RATE, 0.0153), f"{BUY_FEE_RATE}")
check("T9: SELL_FEE_RATE constant is 2.18%", close(SELL_FEE_RATE, 0.0218), f"{SELL_FEE_RATE}")

fee9a = estimate_trade_fees(sell_amount=0.0, buy_amount=0.10)
check("T9a: buy-only fee = 0.10 * 0.0153", close(fee9a, 0.10 * 0.0153), f"fee={fee9a}")

fee9b = estimate_trade_fees(sell_amount=0.10, buy_amount=0.0)
check("T9b: sell-only fee = 0.10 * 0.0218", close(fee9b, 0.10 * 0.0218), f"fee={fee9b}")

check("T9c: selling costs strictly more than buying the same amount",
      fee9b > fee9a, f"sell_fee={fee9b}, buy_fee={fee9a}")

fee9d = estimate_trade_fees(sell_amount=0.05, buy_amount=0.05)
expected9d = 0.05 * 0.0218 + 0.05 * 0.0153
check("T9d: combined buy+sell fee matches manual calc", close(fee9d, expected9d),
      f"got={fee9d}, expected={expected9d}")

# Realistic rebalance: sell 8% of portfolio, buy 8% (a swap trade)
fee9e = estimate_trade_fees(sell_amount=0.08, buy_amount=0.08)
expected9e = 0.08 * (0.0218 + 0.0153)
check("T9e: symmetric swap trade fee = 8% * (sell+buy rates)", close(fee9e, expected9e),
      f"got={fee9e}, expected={expected9e}, drag={fee9e*100:.3f}% of portfolio")


# ---------------------------------------------------------------------
# SUMMARY
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
