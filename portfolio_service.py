"""
portfolio_service.py

The validated portfolio construction algorithm: rank-based initial
weighting, iterative "water-filling" cap enforcement, a diversification
floor, and a final cap-verification safety pass -- the exact methodology
proven across all of today's backtesting (Section "Portfolio Construction
Algorithm" and the cap-verification fix in "Walk-Forward Backtest
Validation" of the strategy documentation).
"""

from dataclasses import dataclass, field
from typing import Optional
import pandas as pd
import numpy as np

MAX_STOCK_WEIGHT = 0.25
MAX_SECTOR_WEIGHT = 0.25
SECTOR_WARNING_THRESHOLD = 0.22  # soft early-warning line below the 25% hard cap -- flagged
                                  # for review, never blocks/forces action the way a real
                                  # sector_breach does. Item #5 from the original review-
                                  # list recommendations: Financial Services sat at 22.13%
                                  # per the strategy doc with no code surfacing that it was
                                  # approaching the cap until now.
CASH_BUFFER = 0.03
MAX_CASH = 0.05
MIN_HOLDINGS = 5
DIVERSIFICATION_FLOOR = 10
DRIFT_THRESHOLD = 0.05
BUY_FEE_RATE = 0.0153   # confirmed from live executed trades
SELL_FEE_RATE = 0.0218  # confirmed from live executed trades, NOT symmetric with buy


def enforce_caps(weights: pd.Series, sectors: dict, max_stock=MAX_STOCK_WEIGHT,
                  max_sector=MAX_SECTOR_WEIGHT, max_iterations=30) -> tuple:
    """Iterative water-filling cap enforcement. Caps any stock/sector over
    its limit and redistributes the freed weight proportionally among
    everything still under BOTH its own cap and its sector's cap.

    Returns (weights, undeployable_shortfall). undeployable_shortfall is
    the portion of capital that genuinely cannot be compliantly deployed
    (e.g. because every remaining candidate is in an already-saturated
    sector) -- this must be treated as EXTRA CASH by the caller, never
    silently discarded. This explicit handling is the fix for a bug found
    via testing: the earlier version let this shortfall vanish (weights
    summing to less than 1.0) rather than accounting for it.

    IMPORTANT: this function preserves whatever total the input `weights`
    sum to -- it does NOT assume that total is 1.0. Callers that reserve
    a cash buffer up front (e.g. build_initial_portfolio, which passes in
    weights summing to `1 - cash_buffer`) rely on this: capping only ever
    redistributes weight that already exists in the input, never invents
    new weight to top the total back up to 100%. (Bug found via testing:
    an earlier version hardcoded the redistribution target as `1.0`,
    which silently absorbed any pre-reserved cash buffer into the stock
    weights before the caller had a chance to add it back -- inflating
    the final weights+cash total above 1.0.)
    """
    target_total = weights.sum()
    df = pd.DataFrame({"weight": weights.copy()})
    df["sector"] = df.index.map(sectors)
    df["locked"] = False  # stock-level lock: hit its own 25% cap
    df["sector_locked"] = False  # this stock's sector is already at/over its cap

    for _ in range(max_iterations):
        violated = False

        over_stock = df[(df["weight"] > max_stock) & (~df["locked"])]
        if len(over_stock) > 0:
            df.loc[over_stock.index, "weight"] = max_stock
            df.loc[over_stock.index, "locked"] = True
            violated = True

        sector_totals = df.groupby("sector")["weight"].sum()
        for sector, total in sector_totals.items():
            if total > max_sector + 1e-9:
                mask = (df["sector"] == sector) & (~df["locked"])
                unlocked_weight = df.loc[mask, "weight"].sum()
                if unlocked_weight > 0:
                    locked_weight = df.loc[(df["sector"] == sector) & (df["locked"]), "weight"].sum()
                    target_unlocked = max(max_sector - locked_weight, 0)
                    scale = target_unlocked / unlocked_weight
                    df.loc[mask, "weight"] *= scale
                    violated = True
            # Mark ANY stock in a sector that's now at (or essentially at) its
            # cap as sector_locked, so redistribution never pushes it back over.
            new_total = df[df["sector"] == sector]["weight"].sum()
            if new_total >= max_sector - 1e-9:
                df.loc[df["sector"] == sector, "sector_locked"] = True

        # Redistribute any shortfall ONLY among positions free of BOTH locks.
        shortfall = target_total - df["weight"].sum()
        eligible_mask = (~df["locked"]) & (~df["sector_locked"])
        if shortfall > 1e-9 and eligible_mask.sum() > 0:
            eligible_sum = df.loc[eligible_mask, "weight"].sum()
            if eligible_sum > 0:
                df.loc[eligible_mask, "weight"] += df.loc[eligible_mask, "weight"] / eligible_sum * shortfall
                violated = True  # redistribution can create new cap breaches -- re-check next pass

        if not violated:
            break

    final_shortfall = max(target_total - df["weight"].sum(), 0.0)
    return df["weight"], final_shortfall


def build_initial_portfolio(ranked_symbols: list, sectors: dict, pool_size: int = 15,
                              cash_buffer: float = CASH_BUFFER) -> tuple:
    """ranked_symbols: list of symbols already sorted best-to-worst by
    composite score (e.g. from screening_service.screen_universe).
    Returns (target_weights, actual_cash_pct) for a fresh (first-time)
    deployment. actual_cash_pct may exceed cash_buffer if the candidate
    pool is too sector-concentrated to fully deploy compliantly, or if
    small positions get dropped after capping -- both cases are reported
    explicitly, never silently discarded."""
    top = ranked_symbols[:pool_size]
    n = len(top)
    rank_weights = pd.Series({sym: (n - i) for i, sym in enumerate(top)})
    weights = rank_weights / rank_weights.sum() * (1 - cash_buffer)

    weights, shortfall = enforce_caps(weights, sectors)

    negligible_mask = weights < 0.02
    negligible_weight = weights[negligible_mask].sum()
    weights = weights[~negligible_mask]

    actual_cash = cash_buffer + shortfall + negligible_weight
    return weights, actual_cash


def safety_pass(weights: pd.Series, sectors: dict, cash: float,
                 max_stock=MAX_STOCK_WEIGHT, max_sector=MAX_SECTOR_WEIGHT,
                 cash_buffer=CASH_BUFFER, max_cash=MAX_CASH) -> tuple:
    """Final cap re-verification, applied AFTER all other adjustments each
    period. This is the fix discovered today: earlier steps (drift
    correction, diversification-floor additions) can push a stock or
    sector back over its cap as a side effect, even though each step
    individually checked caps against the weights it started with. This
    catches anything that slipped through and redeploys the freed cash
    properly, rather than leaving it idle (which would itself create a
    cash-limit violation). Returns (corrected_weights, corrected_cash,
    amount_trimmed, amount_redeployed).
    """
    df = pd.DataFrame({"weight": weights.copy()})
    df["sector"] = df.index.map(sectors)
    trim_total = 0.0
    violated, iterations = True, 0

    while violated and iterations < 10:
        violated = False
        iterations += 1
        over_stock = df[df["weight"] > max_stock]
        for s in over_stock.index:
            trim = df.loc[s, "weight"] - max_stock
            df.loc[s, "weight"] = max_stock
            trim_total += trim
            violated = True
        sector_totals = df.groupby("sector")["weight"].sum()
        for sector, total in sector_totals.items():
            if total > max_sector:
                syms = df[df["sector"] == sector].index
                excess = total - max_sector
                sec_sum = df.loc[syms, "weight"].sum()
                for s in syms:
                    trim = excess * (df.loc[s, "weight"] / sec_sum)
                    df.loc[s, "weight"] -= trim
                    trim_total += trim
                violated = True

    new_weights = df["weight"].copy()
    cash += trim_total

    redeploy_total = 0.0
    if cash > max_cash and trim_total > 0:
        excess_cash = cash - cash_buffer
        sector_totals_now = new_weights.groupby(new_weights.index.map(sectors)).sum()

        # Compute each stock's SAFE room, respecting sector capacity as a
        # shared pool (not double-counted per stock). Within each sector,
        # do a mini water-fill: if the sum of individual stock-cap rooms
        # exceeds the sector's remaining headroom, scale all of them down
        # proportionally so they jointly never exceed it.
        safe_room = {}
        for sec in sector_totals_now.index:
            sec_syms = [s for s in new_weights.index if sectors.get(s) == sec]
            sector_headroom = max(max_sector - sector_totals_now[sec], 0)
            stock_rooms = {s: max(max_stock - new_weights[s], 0) for s in sec_syms}
            total_stock_room = sum(stock_rooms.values())
            if total_stock_room <= sector_headroom or total_stock_room == 0:
                safe_room.update(stock_rooms)  # sector headroom isn't the binding constraint
            else:
                scale = sector_headroom / total_stock_room
                safe_room.update({s: r * scale for s, r in stock_rooms.items()})

        total_safe_room = sum(safe_room.values())
        if total_safe_room > 0:
            deploy_total = min(excess_cash, total_safe_room)
            fill_ratio = deploy_total / total_safe_room  # <=1.0, scales safely if cash-constrained
            for s, r in safe_room.items():
                if r <= 0:
                    continue
                deploy = r * fill_ratio
                new_weights[s] += deploy
                cash -= deploy
                redeploy_total += deploy

    return new_weights, cash, trim_total, redeploy_total


def identify_diversification_additions(current_symbols, ranked_pool: list,
                                          floor: int = DIVERSIFICATION_FLOOR) -> list:
    """The diversification-floor logic described in the strategy
    documentation's walk-forward backtest section (Section 10: "Adding a
    practical diversification floor (replenish below 10 holdings, distinct
    from the hard compliance minimum of 5)") but never previously
    implemented anywhere in this codebase -- DIVERSIFICATION_FLOOR was
    declared above but unused until now.

    If len(current_symbols) is already >= floor, returns [] -- no action
    needed, matching the documented policy that new names are added ONLY
    when holdings have shrunk below the floor, never as a routine "top up
    to more names" action every month regardless of current count.

    Otherwise, returns up to (floor - len(current_symbols)) symbols from
    ranked_pool, in rank order (best first), EXCLUDING anything already
    held -- these are recommendations for NEW positions to open, not a
    target-weight computation (the caller decides how to size them,
    typically by re-running build_initial_portfolio with these names
    folded into the held set, or by simple equal-split of whatever cash
    is being deployed -- this function only answers "which names", not
    "how much of each").

    current_symbols: iterable of symbols currently held (order doesn't
    matter -- converted to a set internally).
    ranked_pool: symbols already sorted best-to-worst by composite score
    (e.g. from a fresh screen_universe() + eligibility filter), NOT
    already limited to a construction pool_size -- pass the fuller ranked
    list so there's actually room to find replacements beyond whatever
    pool_size names were used for weighting."""
    held = set(current_symbols)
    needed = floor - len(held)
    if needed <= 0:
        return []
    candidates = [s for s in ranked_pool if s not in held]
    return candidates[:needed]


def check_drift(actual_weights: pd.Series, fresh_target_weights: pd.Series,
                 drift_threshold: float = DRIFT_THRESHOLD) -> pd.DataFrame:
    """Compares actual current weights to freshly-computed targets (for
    currently-held names only -- new names are never added purely on
    drift, only via the diversification floor or hard breach). Returns
    a DataFrame flagging which positions cross the drift threshold."""
    df = pd.DataFrame({"actual_weight": actual_weights})
    df["fresh_target"] = fresh_target_weights.reindex(df.index)
    df["drift"] = df["fresh_target"] - df["actual_weight"]
    df["triggered"] = df["drift"].abs() > drift_threshold
    return df


def check_hard_breaches(weights: pd.Series, sectors: dict, cash: float,
                          max_stock=MAX_STOCK_WEIGHT, max_sector=MAX_SECTOR_WEIGHT,
                          max_cash=MAX_CASH, min_holdings=MIN_HOLDINGS) -> dict:
    """Checks the four hard compliance rules that require IMMEDIATE action
    regardless of fee cost, per the rebalance policy: stock cap, sector
    cap, cash cap, minimum holdings."""
    sector_totals = weights.groupby(weights.index.map(sectors)).sum()
    return {
        "stock_breach": weights[weights > max_stock].to_dict(),
        "sector_breach": sector_totals[sector_totals > max_sector].to_dict(),
        "cash_breach": cash > max_cash,
        "holdings_breach": len(weights) < min_holdings,
        "any_breach": (
            len(weights[weights > max_stock]) > 0
            or len(sector_totals[sector_totals > max_sector]) > 0
            or cash > max_cash
            or len(weights) < min_holdings
        ),
    }


def check_sector_warnings(weights: pd.Series, sectors: dict,
                            warning_threshold: float = SECTOR_WARNING_THRESHOLD,
                            hard_cap: float = MAX_SECTOR_WEIGHT) -> dict:
    """Soft early-warning check, separate from check_hard_breaches: flags
    any sector sitting AT OR ABOVE warning_threshold (22%) but still
    UNDER hard_cap (25%) -- i.e. approaching the hard cap without having
    crossed it yet. A sector already over hard_cap is a sector_breach
    (check_hard_breaches' job, "act immediately"); this function
    deliberately does NOT re-flag those here, to keep the two signal
    types non-overlapping -- a caller wanting "everything at/above 22%
    regardless of whether it's also over 25%" should just compare
    sector_totals directly rather than combining these two dicts.

    Returns {"sector_warning": {sector: weight, ...}, "any_warning": bool}
    -- same dict-of-offenders shape as check_hard_breaches' stock_breach/
    sector_breach keys, for the caller to format identically."""
    sector_totals = weights.groupby(weights.index.map(sectors)).sum()
    warned = sector_totals[
        (sector_totals >= warning_threshold) & (sector_totals < hard_cap)
    ]
    return {
        "sector_warning": warned.to_dict(),
        "any_warning": len(warned) > 0,
    }


def estimate_trade_fees(sell_amount: float, buy_amount: float,
                          sell_rate: float = SELL_FEE_RATE, buy_rate: float = BUY_FEE_RATE) -> float:
    """Estimated fee drag for a given amount of portfolio weight traded.
    Note the confirmed asymmetry: selling costs more than buying."""
    return sell_amount * sell_rate + buy_amount * buy_rate
