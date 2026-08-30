"""
universe_service.py

Live investable-universe construction from NGX Pulse's full market
snapshot -- the piece described in the strategy documentation (Section 3,
"Data Infrastructure -- Universe Construction") but never actually
implemented anywhere in the codebase until now. screening_service.
screen_universe() takes an already-filtered candidate set as input (price
series, current prices, sectors); this module is what produces that input
from one live NGX Pulse pull, applying exactly the two documented filters:

    - Minimum market capitalization: NGN 20 billion
    - Minimum daily traded volume: 50,000 shares

...plus the explicit non-equity exclusion confirmed in Section 2
(Eligibility Interpretation): NIDF (a debt fund) and NREIT (a REIT unit),
neither of which is ordinary equity despite being NGX-listed.

NOTE ON VOLUME: the strategy documentation's Section 3 states this filter
as a flat "minimum daily traded volume" threshold, distinct from the
TRAILING 20-DAY AVERAGE volume specifically used in the walk-forward
BACKTEST's point-in-time universe reconstruction (Section 10) -- that
refinement was needed there specifically to avoid retroactively applying
today's liquidity to historical dates using only historical daily data
available at each date. For this LIVE monthly screen, NGX Pulse's
/api/ngxdata/stocks snapshot only exposes a single day's volume figure
(the most recent trading day), not a rolling average -- computing a true
20-day average live would cost one additional NGX Pulse request per
candidate symbol just to check eligibility, before any symbol has even
been screened. Given the modest 100-request/day quota (ngx_pulse_service.
DAILY_REQUEST_QUOTA), that trade-off isn't worth it for a threshold this
documentation itself describes as a simple flat minimum, not an average --
so this uses the live snapshot's single-day volume directly, consistent
with the documented text. This is a known simplification, not an oversight:
a stock that's normally liquid but had one unusually quiet trading day
right before the monthly review could be excluded this cycle that
wouldn't be under a rolling-average version of the same rule.
"""

from typing import Optional

import ngx_pulse_service as nps

MIN_MARKET_CAP = 20_000_000_000  # NGN 20 billion, strategy documentation Section 3
MIN_DAILY_VOLUME = 50_000        # shares, strategy documentation Section 3
EXCLUDED_NON_EQUITY = {"NIDF", "NREIT"}  # strategy documentation Section 2, Eligibility Interpretation


def build_live_universe(min_market_cap: float = MIN_MARKET_CAP,
                          min_daily_volume: float = MIN_DAILY_VOLUME,
                          excluded_symbols: Optional[set] = None) -> tuple:
    """One live NGX Pulse request (the full /stocks snapshot -- same
    single-request-regardless-of-symbol-count cost as
    get_current_prices_and_sectors, but this function does its own fetch
    rather than reusing that one, since it needs the market_cap and
    volume columns that function doesn't return).

    Returns (symbols, sector_by_symbol, current_price_by_symbol):
      - symbols: list of tickers passing BOTH the market-cap and volume
        filters, with NIDF/NREIT (or any caller-supplied excluded_symbols)
        removed regardless of whether they'd otherwise pass.
      - sector_by_symbol / current_price_by_symbol: exactly the two dicts
        screening_service.screen_universe and portfolio_service.
        build_initial_portfolio need, scoped to just the returned symbols.

    Rows with missing/non-numeric market_cap or volume are excluded
    (treated as failing the filter, not as passing it) -- a symbol NGX
    Pulse can't currently size or measure liquidity for shouldn't be
    silently let into a live portfolio construction run.
    """
    excluded = EXCLUDED_NON_EQUITY | (excluded_symbols or set())

    df = nps.fetch_stocks_snapshot()
    df = df[~df["symbol"].isin(excluded)]

    market_cap = pd_to_numeric_safe(df["market_cap"])
    volume = pd_to_numeric_safe(df["volume"])
    df = df[(market_cap >= min_market_cap) & (volume >= min_daily_volume)]

    symbols = df["symbol"].tolist()
    sector_by_symbol = dict(zip(df["symbol"], df["sector"]))
    current_price_by_symbol = dict(zip(df["symbol"], df["current_price"]))
    return symbols, sector_by_symbol, current_price_by_symbol


def pd_to_numeric_safe(series):
    """Coerces to numeric, turning anything unparseable into NaN (which
    then correctly fails the >= comparison, i.e. is excluded) rather than
    raising -- NGX Pulse's snapshot is real, external, occasionally messy
    data, and a single malformed row must not crash universe construction
    for every other symbol."""
    import pandas as pd
    return pd.to_numeric(series, errors="coerce")
