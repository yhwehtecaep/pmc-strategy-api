"""
screening_service.py

Combines momentum, low-volatility, and point-in-time fundamentals
(via fundamentals_service.py) into the same validated composite scoring
methodology as backtest_stage8_improved_pe.py -- the best-supported
version tested to date (Section "Improved Value Factor" in the strategy
documentation).

Price data is expected to come from the caller (typically NGX Pulse's
live /stocks and /prices endpoints for real-time use, or the saved
historical panel for backtesting) -- this module doesn't fetch prices
itself, keeping it agnostic to the data source and easy to test.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional
import numpy as np
import pandas as pd

from fundamentals_service import get_point_in_time_fundamentals

ROE_WEIGHT = 1.0
PE_WEIGHT = 1.0
MOMENTUM_WEIGHT = 1.0
VOL_WEIGHT = 1.0


@dataclass
class ScoredStock:
    symbol: str
    momentum_return: Optional[float] = None
    annualized_vol: Optional[float] = None
    roe: Optional[float] = None
    pe: Optional[float] = None
    momentum_z: Optional[float] = None
    vol_z: Optional[float] = None
    roe_z: Optional[float] = None
    pe_z: Optional[float] = None
    composite_score: Optional[float] = None
    sector: Optional[str] = None
    eligible: bool = True
    exclusion_reason: Optional[str] = None


def compute_momentum_vol_from_price_series(price_series: pd.Series) -> tuple:
    """price_series: a pandas Series indexed by date, most recent last,
    covering the lookback window (typically 130 trading days).
    Returns (momentum_return, annualized_vol), or (None, None) if there's
    not enough data."""
    series = price_series.dropna()
    if len(series) < 100:  # require most of a 130-day window to trust the reading
        return None, None
    momentum = (series.iloc[-1] / series.iloc[0]) - 1
    daily_returns = series.pct_change().dropna()
    ann_vol = daily_returns.std() * np.sqrt(252)
    if pd.isna(momentum) or pd.isna(ann_vol) or ann_vol == 0:
        return None, None
    return float(momentum), float(ann_vol)


def screen_universe(
    price_series_by_symbol: dict,       # {symbol: pd.Series of prices, lookback window}
    current_price_by_symbol: dict,      # {symbol: float}
    sector_by_symbol: dict,             # {symbol: str}
    as_of_date: datetime,
    excluded_symbols: Optional[set] = None,  # e.g. {"NIDF", "NREIT"} -- non-equity instruments
    fundamentals_fetcher: Optional[callable] = None,  # optional injection point: defaults to
    # the live get_point_in_time_fundamentals call below. Callers (e.g. main.py) can pass a
    # cache-checking wrapper with the same signature (symbol, as_of_date, current_price=...)
    # -> PointInTimeFundamentals to avoid re-scraping on every screen. Default behavior is
    # unchanged when this is omitted.
) -> list:
    """
    Screens every symbol provided and returns a list of ScoredStock objects,
    ranked by composite_score descending. This is the same methodology as
    the validated v8 backtest: momentum_z - vol_z - pe_z + roe_z, with
    graceful neutral fallback for any missing fundamentals data.

    NOTE: this makes one fundamentals_service call per symbol, which is a
    real, rate-limited network operation -- for a ~75-name universe this
    will take a few minutes, same as every fundamentals pull today. The
    caller is responsible for deciding how often to refresh (e.g. cache
    fundamentals results and only re-screen momentum/vol daily, refreshing
    fundamentals monthly at the actual rebalance review).
    """
    excluded_symbols = excluded_symbols or set()
    fetch_fundamentals = fundamentals_fetcher or get_point_in_time_fundamentals
    results = []

    raw_rows = []
    for symbol, price_series in price_series_by_symbol.items():
        stock = ScoredStock(symbol=symbol, sector=sector_by_symbol.get(symbol))

        if symbol in excluded_symbols:
            stock.eligible = False
            stock.exclusion_reason = "Excluded (non-equity instrument, e.g. debt fund or REIT)"
            results.append(stock)
            continue

        momentum, vol = compute_momentum_vol_from_price_series(price_series)
        if momentum is None:
            stock.eligible = False
            stock.exclusion_reason = "Insufficient price history for the lookback window"
            results.append(stock)
            continue
        stock.momentum_return = momentum
        stock.annualized_vol = vol

        current_price = current_price_by_symbol.get(symbol)
        fund = fetch_fundamentals(symbol, as_of_date, current_price=current_price)
        stock.roe = fund.roe
        stock.pe = fund.pe

        raw_rows.append(stock)

    if len(raw_rows) < 5:
        # Not enough eligible stocks to compute meaningful z-scores
        return results + raw_rows

    momentum_vals = np.array([s.momentum_return for s in raw_rows])
    vol_vals = np.array([s.annualized_vol for s in raw_rows])
    roe_vals = np.array([s.roe if s.roe is not None else np.nan for s in raw_rows])
    pe_vals = np.array([s.pe if s.pe is not None else np.nan for s in raw_rows])

    momentum_z = (momentum_vals - momentum_vals.mean()) / momentum_vals.std()
    vol_z = (vol_vals - vol_vals.mean()) / vol_vals.std()

    roe_valid = ~np.isnan(roe_vals)
    if roe_valid.sum() >= 3:
        roe_z_raw = (roe_vals - np.nanmean(roe_vals)) / np.nanstd(roe_vals)
        roe_z = np.where(roe_valid, roe_z_raw, 0.0)
    else:
        roe_z = np.zeros(len(raw_rows))

    pe_valid = ~np.isnan(pe_vals)
    if pe_valid.sum() >= 3:
        pe_z_raw = (pe_vals - np.nanmean(pe_vals)) / np.nanstd(pe_vals)
        pe_z = np.where(pe_valid, pe_z_raw, 0.0)
    else:
        pe_z = np.zeros(len(raw_rows))

    for i, stock in enumerate(raw_rows):
        stock.momentum_z = float(momentum_z[i])
        stock.vol_z = float(vol_z[i])
        stock.roe_z = float(roe_z[i])
        stock.pe_z = float(pe_z[i])
        stock.composite_score = (
            MOMENTUM_WEIGHT * stock.momentum_z
            - VOL_WEIGHT * stock.vol_z
            - PE_WEIGHT * stock.pe_z
            + ROE_WEIGHT * stock.roe_z
        )

    raw_rows.sort(key=lambda s: s.composite_score, reverse=True)
    return results + raw_rows
