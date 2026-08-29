"""
fundamentals_service.py

Reusable fundamentals module for the PMC Strategy API.

This is the piece the whole strategy's robustness improvement depended on:
a computed, continuously-updating point-in-time P/E (price / most-recently-
reported EPS), not a static scraped snapshot. It works for ANY stock symbol,
not a hardcoded list -- new listings, newly-liquid names, or any future
addition to the screened universe all go through the same pipeline.

Real, permanent limitations (not bugs, not fixable by more scraping):
  - A stock with negative or zero trailing earnings has no meaningful P/E.
    This function correctly returns None for pe in that case, and callers
    should treat it as neutral (not drop the stock), same as the live
    strategy's existing missing-data convention.
  - A stock whose income statement page doesn't expose a standard EPS row
    (observed for ETI specifically) will also return None for pe.
  - ROE has near-universal coverage; P/E does not, by economic necessity,
    not by omission.
"""

import re
import time
from io import StringIO
from datetime import datetime, timedelta
from dataclasses import dataclass
from typing import Optional

import requests
import pandas as pd

HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
REPORTING_LAG_DAYS = 120  # confirmed safely conservative vs. NGX's actual 90-day filing rule
ROE_CAP = 100.0           # caps thin-equity-base distortions (e.g. MTNN, PZ historically)
REQUEST_TIMEOUT = 15
POLITE_DELAY_SECONDS = 2  # rate limiting for stockanalysis.com


@dataclass
class PointInTimeFundamentals:
    symbol: str
    as_of_date: datetime
    roe: Optional[float]        # % , most recently available as of as_of_date
    pe: Optional[float]         # computed price/EPS, None if undefined (negative earnings, missing data)
    eps_used: Optional[float]   # the EPS figure the P/E was computed from, for auditability
    eps_period_ending: Optional[datetime]  # which fiscal year's EPS was used
    data_available: bool        # False if nothing could be found for this symbol at all


def _parse_column_header(col_str: str):
    """Extract (label, period_ending_date) from stockanalysis.com's column
    header format, e.g. "('FY 2025', \"Dec '25 Dec 31, 2025\")" """
    year_match = re.search(r"'(FY \d{4}|TTM|Current)'", col_str)
    date_match = re.search(r"(\w{3} \d{1,2}, \d{4})\"?\)?$", col_str)
    if not year_match or not date_match:
        return None, None
    label = year_match.group(1)
    try:
        period_ending = pd.to_datetime(date_match.group(1))
    except Exception:
        return None, None
    return label, period_ending


def _clean_value(val) -> Optional[float]:
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return None
    s = str(val).strip()
    if s in ("-", "", "nan", "NaN"):
        return None
    s = s.replace("%", "").replace(",", "")
    try:
        return float(s)
    except ValueError:
        return None


def fetch_raw_ratios_table(symbol: str) -> Optional[pd.DataFrame]:
    """Fetch the ratios page (ROE, Debt/Equity, etc.) for one symbol."""
    url = f"https://stockanalysis.com/quote/ngx/{symbol}/financials/ratios/"
    try:
        resp = requests.get(url, headers=HEADERS, timeout=REQUEST_TIMEOUT)
        if resp.status_code != 200:
            return None
        tables = pd.read_html(StringIO(resp.text))
        return max(tables, key=len)
    except Exception:
        return None


def fetch_raw_income_statement(symbol: str) -> Optional[pd.DataFrame]:
    """Fetch the income statement page (EPS) for one symbol."""
    url = f"https://stockanalysis.com/quote/ngx/{symbol}/financials/"
    try:
        resp = requests.get(url, headers=HEADERS, timeout=REQUEST_TIMEOUT)
        if resp.status_code != 200:
            return None
        tables = pd.read_html(StringIO(resp.text))
        return max(tables, key=len)
    except Exception:
        return None


def _extract_series(df: pd.DataFrame, row_label_pattern: str, exclude_pattern: Optional[str] = None):
    """Given a raw stockanalysis.com table, return {available_date: value}
    for the row matching row_label_pattern, across all confirmed fiscal-year
    columns (skips 'Current' and 'TTM', which aren't tied to a clean single
    disclosure date)."""
    if df is None or len(df.columns) < 2:
        return {}
    label_col = df.columns[0]
    matches = df[df[label_col].astype(str).str.contains(row_label_pattern, case=False, na=False)]
    if exclude_pattern:
        matches = matches[~df[label_col].astype(str).str.fullmatch(exclude_pattern, case=False, na=False)]
    if len(matches) == 0:
        return {}
    row = matches.iloc[0]

    series = {}
    for col in df.columns[1:]:
        fy_label, period_ending = _parse_column_header(str(col))
        if fy_label is None or fy_label in ("Current", "TTM"):
            continue
        val = _clean_value(row[col])
        if val is None:
            continue
        available_date = period_ending + timedelta(days=REPORTING_LAG_DAYS)
        series[period_ending] = {"available_date": available_date, "value": val}
    return series


def get_point_in_time_fundamentals(
    symbol: str,
    as_of_date: datetime,
    current_price: Optional[float] = None,
) -> PointInTimeFundamentals:
    """
    The main entry point. Returns point-in-time ROE and computed P/E for any
    symbol, as of any date, respecting the reporting lag (no lookahead bias).

    If current_price is provided, P/E is computed as current_price / most
    recently available EPS. If current_price is None, P/E cannot be computed
    (this function does not fetch live prices itself -- that's the caller's
    responsibility, since price data should come from the shared price feed,
    not be re-fetched per fundamentals call).
    """
    ratios_table = fetch_raw_ratios_table(symbol)
    time.sleep(POLITE_DELAY_SECONDS)
    income_table = fetch_raw_income_statement(symbol)
    time.sleep(POLITE_DELAY_SECONDS)

    roe_series = _extract_series(ratios_table, "Return on Equity") if ratios_table is not None else {}
    eps_series = _extract_series(income_table, "Earnings Per Share") if income_table is not None else {}

    if not roe_series and not eps_series:
        return PointInTimeFundamentals(
            symbol=symbol, as_of_date=as_of_date, roe=None, pe=None,
            eps_used=None, eps_period_ending=None, data_available=False,
        )

    # Most recent ROE actually available as of as_of_date
    roe_available = {k: v for k, v in roe_series.items() if v["available_date"] <= as_of_date}
    roe_val = None
    if roe_available:
        latest_period = max(roe_available.keys())
        roe_val = min(roe_available[latest_period]["value"], ROE_CAP)

    # Most recent EPS actually available as of as_of_date
    eps_available = {k: v for k, v in eps_series.items() if v["available_date"] <= as_of_date}
    eps_val, eps_period = None, None
    if eps_available:
        eps_period = max(eps_available.keys())
        eps_val = eps_available[eps_period]["value"]

    pe_val = None
    if eps_val is not None and eps_val > 0 and current_price is not None:
        pe_val = current_price / eps_val
    # eps_val <= 0 (loss-making) or missing -> pe_val stays None, correctly
    # signaling "undefined", not an error -- callers should treat this as
    # neutral in scoring, matching the live strategy's convention.

    return PointInTimeFundamentals(
        symbol=symbol, as_of_date=as_of_date, roe=roe_val, pe=pe_val,
        eps_used=eps_val, eps_period_ending=eps_period,
        data_available=True,
    )
