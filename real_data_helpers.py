"""
real_data_helpers.py

Shared helpers for turning the raw JSON/CSV files pulled from
github.com/yhwehtecaep/cfa-challenge (the actual Colab-session outputs,
not synthetic test fixtures) back into the exact shapes
fundamentals_service.py, screening_service.py, and main.py expect.

Used by test_fundamentals_real_data.py, test_screen_real_data.py, and
test_ngx_pulse_service.py so all three exercise REAL parsing/business
logic against REAL scraped/API data, with only the outbound network call
itself (stockanalysis.com and ngxpulse.ng -- both unreachable from this
sandbox, same constraint as api.telegram.org) replaced by a replay of
what those calls actually returned when the data was collected.
"""

import json
import os
import pandas as pd

REAL_DATA_DIR = os.path.join(os.path.dirname(__file__), "real_data")


def _records_to_df(records: list) -> pd.DataFrame:
    """The JSON files store each stockanalysis.com table as a list of
    row-dicts (one dict per row, column names as dict keys) -- exactly
    what pd.read_html() + DataFrame.to_dict('records') produces. Column
    order is preserved via dict insertion order (Python 3.7+), consistent
    across rows in this dataset."""
    return pd.DataFrame.from_records(records)


def load_raw_ratios_table(symbol: str):
    """Real equivalent of fundamentals_service.fetch_raw_ratios_table(),
    replayed from full_universe_fundamentals_raw.json instead of a live
    stockanalysis.com fetch. Returns None if the symbol wasn't scraped
    (matches fetch_raw_ratios_table's None-on-failure contract)."""
    path = os.path.join(REAL_DATA_DIR, "full_universe_fundamentals_raw.json")
    with open(path) as f:
        data = json.load(f)
    if symbol not in data:
        return None
    return _records_to_df(data[symbol])


def load_raw_income_statement(symbol: str):
    """Real equivalent of fundamentals_service.fetch_raw_income_statement()."""
    path = os.path.join(REAL_DATA_DIR, "full_universe_income_statement_raw.json")
    with open(path) as f:
        data = json.load(f)
    if symbol not in data:
        return None
    return _records_to_df(data[symbol])


def available_symbols_with_fundamentals() -> list:
    path = os.path.join(REAL_DATA_DIR, "full_universe_fundamentals_raw.json")
    with open(path) as f:
        data = json.load(f)
    return sorted(data.keys())


def load_price_panel() -> pd.DataFrame:
    """backtest_price_panel_cleaned.csv: wide panel, trade_date index,
    one column per symbol, Apr 2022 - Jul 2026, ~13% NaN (illiquid names
    / late listings -- real, not injected)."""
    path = os.path.join(REAL_DATA_DIR, "backtest_price_panel_cleaned.csv")
    df = pd.read_csv(path, parse_dates=["trade_date"])
    return df.set_index("trade_date").sort_index()


def load_shortlist_universe() -> pd.DataFrame:
    """shortlist_universe.csv: single NGX Pulse /stocks-shaped snapshot
    (2026-07-31) for the 76-symbol screened universe -- symbol, name,
    current_price, sector, market, etc."""
    return pd.read_csv(os.path.join(REAL_DATA_DIR, "shortlist_universe.csv"))


def load_ngx_daily_prices() -> pd.DataFrame:
    """ngx_daily_prices.csv: the same /stocks-shaped schema across three
    separate daily snapshots (147 symbols x 3 dates) -- broader than
    shortlist_universe.csv, useful for the NGX Pulse service tests."""
    return pd.read_csv(os.path.join(REAL_DATA_DIR, "ngx_daily_prices.csv"))


def build_mock_stocks_response(pull_date: str = None) -> dict:
    """Reconstructs a realistic /api/ngxdata/stocks response envelope
    ({"stocks": [...]}) from a real snapshot in ngx_daily_prices.csv --
    same real numbers, just re-wrapped in the envelope shape a real
    response would have.

    IMPORTANT: filters on the 'Date' column (the actual day this Colab
    pull was made), NOT 'trade_date' (the last NGX trading date the price
    reflects). trade_date repeats across multiple pulls whenever a pull
    lands on a non-trading day (confirmed by inspection: filtering on
    trade_date alone produces 2 rows per symbol for the same nominal
    date, which no real single API response would ever contain -- 'Date'
    is what actually identifies one distinct real API call). Defaults to
    the most recent pull in the file (2026-08-04, 147 unique symbols)."""
    df = load_ngx_daily_prices()
    date_to_use = pull_date or df["Date"].max()
    snapshot = df[df["Date"] == date_to_use].drop(columns=["Date"], errors="ignore")
    return {"stocks": snapshot.to_dict(orient="records")}


def load_raw_price_history() -> dict:
    """raw_price_history.json: list of {"symbol": ..., "raw_response": {
    ...real NGX Pulse /api/ngxdata/prices/{symbol} response...}}.
    Returns {symbol: raw_response_dict}, matching exactly what
    ngx_pulse_service.fetch_price_history_raw() will receive from
    requests.get(...).json() in production. Confirmed real shape:
    {"success": bool, "symbol": str, "prices": [{"trade_date",
    "close_price", ...}, ...], "count": int}."""
    path = os.path.join(REAL_DATA_DIR, "raw_price_history.json")
    with open(path) as f:
        records = json.load(f)
    return {r["symbol"]: r["raw_response"] for r in records}
