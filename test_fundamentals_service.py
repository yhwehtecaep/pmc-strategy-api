"""
Run this in Colab to verify fundamentals_service.py actually works with
real network access. Upload fundamentals_service.py to the same Colab
session first (or paste its contents into a cell above this one).
"""

from datetime import datetime
from fundamentals_service import get_point_in_time_fundamentals

# Test 1: a stock we KNOW has good data (DANGCEM, confirmed working all day)
print("TEST 1: DANGCEM (should have real ROE and P/E)")
result = get_point_in_time_fundamentals("DANGCEM", datetime(2026, 8, 20), current_price=1050.0)
print(result)
print()

# Test 2: ETI, known to have a non-standard income statement (no EPS row)
print("TEST 2: ETI (should have ROE but pe=None, since it has no standard EPS row)")
result = get_point_in_time_fundamentals("ETI", datetime(2026, 8, 20), current_price=65.0)
print(result)
print()

# Test 3: a stock not in today's manually-tested list, to prove this
# works for ANY symbol, not a hardcoded set
print("TEST 3: FIDELITYBK (not specifically tested by hand today -- proves generality)")
result = get_point_in_time_fundamentals("FIDELITYBK", datetime(2026, 8, 20), current_price=15.0)
print(result)
print()

# Test 4: a historical date, to confirm the reporting-lag logic works
# (should return OLDER fundamentals than Test 1, since less would have
# been publicly reported yet)
print("TEST 4: DANGCEM as of an EARLIER date (Jan 2024) -- should show older EPS")
result = get_point_in_time_fundamentals("DANGCEM", datetime(2024, 1, 15), current_price=1050.0)
print(result)

print("\n" + "="*70)
print("If all four tests returned data_available=True with sensible values,")
print("the module is working correctly and is ready to build the rest of")
print("the API on top of.")
print("="*70)
