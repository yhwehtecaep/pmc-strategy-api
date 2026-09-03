"""
main.py

FastAPI app for the CFA Society Nigeria PMC strategy service. Originally
wired together fundamentals_service.py, screening_service.py, and
portfolio_service.py behind four stateless endpoints:

    GET  /fundamentals/{symbol}   -- point-in-time fundamentals for one symbol
    POST /screen                  -- rank the full universe by composite score
    POST /portfolio/construct     -- screen + build a fresh compliant portfolio
    POST /rebalance-check         -- drift + hard-breach check on a live portfolio

Since then, also wired in: db.py (persistent portfolios/holdings/trades/
signals -- see the Portfolios/Holdings/Trades/Signals sections below, and
/rebalance-check's optional portfolio_id path), bot.py (POST
/telegram/webhook), ngx_pulse_service.py (live current-price fetch for
/rebalance-check's stateful path when current_price_by_symbol is omitted),
and universe_service.py (POST /monthly-screen -- builds the live
investable universe, screens it, constructs a fresh portfolio, and sends
the result via Telegram; designed to be triggered automatically by a
scheduled job, e.g. GitHub Actions, rather than run by hand each month).
This module docstring covers only the original four; search each section
header below for the rest.

IMPORTANT INTEGRATION NOTE (found while wiring this up, not a bug in either
module individually): screening_service.screen_universe() returns ineligible
stocks FIRST in its list, then eligible stocks sorted best-to-worst by
composite_score -- the list as a whole is NOT "rank order" until you filter
to eligible==True. /portfolio/construct filters on eligible before handing
symbols to portfolio_service.build_initial_portfolio, which expects a
purely best-to-worst ranked_symbols list. Any future caller reusing
screen_universe() directly must do the same filter -- see _ranked_eligible_symbols().
"""

from datetime import datetime, date
from typing import Dict, List, Optional
import os

import pandas as pd
from fastapi import FastAPI, HTTPException, Request, Header, BackgroundTasks
from pydantic import BaseModel, Field

import fundamentals_service as fs
import screening_service as ss
import portfolio_service as ps
import db
import bot as tg_bot
import ngx_pulse_service as nps
import monthly_screen_service as mss
import price_history_service as phs
import requests

app = FastAPI(
    title="CFA Society Nigeria PMC Strategy API",
    description="Systematic momentum + low-vol + value + quality NGX equity strategy.",
    version="0.1.0",
)


@app.on_event("startup")
def _on_startup():
    db.init_db()


# ---------------------------------------------------------------------
# Shared request/response models
# ---------------------------------------------------------------------

class UniverseInput(BaseModel):
    """Common inputs needed to screen a universe. Price series are plain
    ordered lists (oldest -> newest) since only order matters for momentum
    and volatility -- dates are supplied separately via as_of_date for the
    fundamentals reporting-lag check."""
    price_series_by_symbol: Dict[str, List[float]] = Field(
        ..., description="symbol -> ordered list of prices, oldest to newest, "
                          "covering the lookback window (typically 130 trading days)."
    )
    current_price_by_symbol: Dict[str, float] = Field(
        ..., description="symbol -> latest traded price, used to compute point-in-time P/E."
    )
    sector_by_symbol: Dict[str, str] = Field(..., description="symbol -> sector name.")
    as_of_date: date = Field(..., description="Screening date; fundamentals respect the 120-day reporting lag as of this date.")
    excluded_symbols: Optional[List[str]] = Field(
        default=None, description="Non-equity instruments to exclude, e.g. ['NIDF', 'NREIT']."
    )


class ScoredStockOut(BaseModel):
    symbol: str
    sector: Optional[str] = None
    eligible: bool
    exclusion_reason: Optional[str] = None
    momentum_return: Optional[float] = None
    annualized_vol: Optional[float] = None
    roe: Optional[float] = None
    pe: Optional[float] = None
    momentum_z: Optional[float] = None
    vol_z: Optional[float] = None
    roe_z: Optional[float] = None
    pe_z: Optional[float] = None
    composite_score: Optional[float] = None


class FundamentalsOut(BaseModel):
    symbol: str
    as_of_date: datetime
    roe: Optional[float]
    pe: Optional[float]
    eps_used: Optional[float]
    eps_period_ending: Optional[datetime]
    data_available: bool


FUNDAMENTALS_CACHE_MAX_AGE_DAYS = mss.FUNDAMENTALS_CACHE_MAX_AGE_DAYS  # re-exported for backward compat
_get_cached_or_live_fundamentals = mss.get_cached_or_live_fundamentals  # single shared implementation


def _run_screen(payload: UniverseInput) -> List[ss.ScoredStock]:
    price_series_by_symbol = {
        sym: pd.Series(vals)
        for sym, vals in payload.price_series_by_symbol.items()
    }
    excluded = set(payload.excluded_symbols) if payload.excluded_symbols else set()
    as_of_dt = datetime.combine(payload.as_of_date, datetime.min.time())
    return ss.screen_universe(
        price_series_by_symbol=price_series_by_symbol,
        current_price_by_symbol=payload.current_price_by_symbol,
        sector_by_symbol=payload.sector_by_symbol,
        as_of_date=as_of_dt,
        excluded_symbols=excluded,
        fundamentals_fetcher=_get_cached_or_live_fundamentals,
    )


_ranked_eligible_symbols = mss.ranked_eligible_symbols  # single shared implementation, see monthly_screen_service.py


def _scored_stock_to_out(s: ss.ScoredStock) -> ScoredStockOut:
    return ScoredStockOut(
        symbol=s.symbol, sector=s.sector, eligible=s.eligible,
        exclusion_reason=s.exclusion_reason, momentum_return=s.momentum_return,
        annualized_vol=s.annualized_vol, roe=s.roe, pe=s.pe,
        momentum_z=s.momentum_z, vol_z=s.vol_z, roe_z=s.roe_z, pe_z=s.pe_z,
        composite_score=s.composite_score,
    )


# ---------------------------------------------------------------------
# GET /fundamentals/{symbol}
# ---------------------------------------------------------------------

@app.get("/fundamentals/{symbol}", response_model=FundamentalsOut)
def get_fundamentals(
    symbol: str,
    as_of_date: Optional[date] = None,
    current_price: Optional[float] = None,
):
    """Point-in-time ROE and computed P/E for one symbol. Defaults
    as_of_date to today if not supplied. current_price is required to get
    a non-null pe (see fundamentals_service docstring: this module never
    fetches prices itself)."""
    as_of_dt = datetime.combine(as_of_date, datetime.min.time()) if as_of_date else datetime.utcnow()
    result = _get_cached_or_live_fundamentals(symbol, as_of_dt, current_price=current_price)
    return FundamentalsOut(
        symbol=result.symbol, as_of_date=result.as_of_date, roe=result.roe,
        pe=result.pe, eps_used=result.eps_used,
        eps_period_ending=result.eps_period_ending, data_available=result.data_available,
    )


# ---------------------------------------------------------------------
# POST /screen
# ---------------------------------------------------------------------

@app.post("/screen", response_model=List[ScoredStockOut])
def screen(payload: UniverseInput):
    """Rank the supplied universe by composite score. NOTE: the returned
    list has ineligible stocks first, then eligible stocks sorted
    best-to-worst -- it is NOT a single rank-ordered list end to end.
    Filter on `eligible` before treating list order as a ranking."""
    scored = _run_screen(payload)
    return [_scored_stock_to_out(s) for s in scored]


# ---------------------------------------------------------------------
# POST /portfolio/construct
# ---------------------------------------------------------------------

class PortfolioConstructRequest(UniverseInput):
    pool_size: int = Field(default=15, description="Top-N eligible stocks considered for the portfolio.")
    cash_buffer: float = Field(default=ps.CASH_BUFFER, description="Target cash reserve, e.g. 0.03 for 3%.")


class PortfolioConstructResponse(BaseModel):
    weights: Dict[str, float]
    cash_pct: float
    holdings_count: int
    pool_considered: List[str] = Field(description="Ranked-eligible symbols the pool was drawn from, best-to-worst.")
    ineligible_symbols: Dict[str, str] = Field(description="symbol -> exclusion_reason for anything screened out.")


@app.post("/portfolio/construct", response_model=PortfolioConstructResponse)
def construct_portfolio(payload: PortfolioConstructRequest):
    """Screens the universe, then builds a fresh compliant portfolio from
    the top `pool_size` ELIGIBLE names (ineligible/insufficient-history
    stocks are excluded from consideration entirely, never ranked)."""
    scored = _run_screen(payload)
    ranked_symbols = _ranked_eligible_symbols(scored)

    if len(ranked_symbols) < ps.MIN_HOLDINGS:
        raise HTTPException(
            status_code=422,
            detail=f"Only {len(ranked_symbols)} eligible symbols after screening; "
                   f"need at least {ps.MIN_HOLDINGS} to build a compliant portfolio.",
        )

    weights, cash_pct = ps.build_initial_portfolio(
        ranked_symbols, payload.sector_by_symbol,
        pool_size=payload.pool_size, cash_buffer=payload.cash_buffer,
    )

    ineligible = {s.symbol: (s.exclusion_reason or "unspecified") for s in scored if not s.eligible}

    return PortfolioConstructResponse(
        weights=weights.to_dict(),
        cash_pct=cash_pct,
        holdings_count=len(weights),
        pool_considered=ranked_symbols[:payload.pool_size],
        ineligible_symbols=ineligible,
    )


# ---------------------------------------------------------------------
# POST /rebalance-check
# ---------------------------------------------------------------------

class RebalanceCheckRequest(BaseModel):
    portfolio_id: Optional[str] = Field(
        default=None, description="If supplied, actual_weights and sectors are pulled from "
                                   "the database instead of being required directly (see "
                                   "current_price_by_symbol, needed to value the holdings)."
    )
    current_price_by_symbol: Optional[Dict[str, float]] = Field(
        default=None, description="symbol -> current price, used to convert stored share "
                                   "counts into weights. Optional even when portfolio_id is "
                                   "supplied: if omitted, current prices for the portfolio's "
                                   "held symbols are fetched LIVE from NGX Pulse (one request "
                                   "regardless of how many symbols are held -- see "
                                   "ngx_pulse_service.py). Supply this manually to override "
                                   "with specific prices (e.g. backtesting a historical date, "
                                   "or bypassing a live NGX Pulse outage)."
    )
    persist_signals: bool = Field(
        default=True, description="When portfolio_id is supplied, whether newly-triggered "
                                   "drift/breach issues are persisted via db.log_signal_if_new "
                                   "(deduped) so repeated polls don't re-alert on the same "
                                   "unresolved issue."
    )
    actual_weights: Optional[Dict[str, float]] = Field(
        default=None, description="Currently-held weights. Required if portfolio_id is not supplied."
    )
    fresh_target_weights: Dict[str, float] = Field(
        ..., description="Freshly-computed target weights for currently-held names "
                          "(typically from a fresh /portfolio/construct call)."
    )
    sectors: Optional[Dict[str, str]] = Field(
        default=None, description="symbol -> sector, for hard-breach sector checks. Required "
                                   "if portfolio_id is not supplied; if portfolio_id IS "
                                   "supplied, sectors from stored holdings are used, with any "
                                   "symbols in this dict overriding/filling in gaps."
    )
    cash: float = Field(..., description="Current cash fraction of the portfolio.")
    drift_threshold: float = Field(default=ps.DRIFT_THRESHOLD)


class DriftRow(BaseModel):
    symbol: str
    actual_weight: float
    fresh_target: Optional[float]
    drift: Optional[float]
    triggered: bool


class RebalanceCheckResponse(BaseModel):
    drift_rows: List[DriftRow]
    any_drift_triggered: bool
    hard_breaches: dict
    sector_warnings: dict = Field(
        default_factory=dict,
        description="Soft early-warning check (portfolio_service.check_sector_warnings): "
                     "sectors at/above SECTOR_WARNING_THRESHOLD (22%) but still under the "
                     "25% hard cap. Deliberately NOT included in action_required -- this is "
                     "advance notice to watch a sector, not a compliance breach requiring "
                     "immediate action the way hard_breaches does.",
    )
    action_required: bool = Field(
        description="True if either a hard breach exists (must act immediately, "
                     "any fee cost) or a drift threshold was crossed (act at next "
                     "monthly review per the drift-triggered rebalance policy)."
    )
    signals_logged: List[str] = Field(
        default_factory=list,
        description="IDs of newly-persisted signals (empty unless portfolio_id and "
                     "persist_signals were both supplied, and something new triggered).",
    )
    missing_prices: List[str] = Field(
        default_factory=list,
        description="Held symbols that could not be priced (absent from the live NGX Pulse "
                     "snapshot, or absent from a manually-supplied current_price_by_symbol). "
                     "These are valued at 0 in the weight calculation (db.holdings_to_weights' "
                     "documented convention), which can UNDERSTATE that position's weight and "
                     "mask a real cap breach -- always check this list is empty before trusting "
                     "action_required=False.",
    )


@app.post("/rebalance-check", response_model=RebalanceCheckResponse)
def rebalance_check(payload: RebalanceCheckRequest):
    """Runs both required checks per the rebalance policy: drift-triggered
    review candidates (>5pp from a fresh target) and the four hard
    compliance breaches that require immediate action regardless of fee
    cost (stock cap, sector cap, cash cap, minimum holdings).

    Two ways to supply the current portfolio state:
    - Stateless (original contract, unchanged): pass actual_weights + sectors directly.
    - Stateful: pass portfolio_id. Holdings and their sectors are pulled from
      the database via db.get_holdings(); current prices come from
      current_price_by_symbol if supplied, otherwise are fetched LIVE from
      NGX Pulse for exactly the portfolio's held symbols (see
      ngx_pulse_service.py -- one request regardless of holding count). The
      check logic below is identical either way.
    """
    if payload.portfolio_id:
        if not (0 <= payload.cash < 1):
            raise HTTPException(
                status_code=422,
                detail=f"cash must be a fraction in [0, 1) when portfolio_id is supplied, got {payload.cash}.",
            )
        holdings_list = db.get_holdings(payload.portfolio_id)
        held_symbols = [h["symbol"] for h in holdings_list]

        if payload.current_price_by_symbol:
            current_prices = payload.current_price_by_symbol
        elif not held_symbols:
            current_prices = {}  # nothing held, nothing to price -- not an error
        else:
            # SCOPE DECISION (confirmed with the user): live price fetches only ever
            # need to cover symbols actually HELD in this one portfolio -- in
            # practice a small set (~10). ngx_pulse_service's
            # get_current_prices_and_sectors() costs exactly ONE NGX Pulse request
            # regardless of how many symbols are passed (the full market snapshot
            # is fetched once and filtered client-side), so this is cheap
            # regardless of portfolio size -- see ngx_pulse_service.py's module
            # docstring for the full reasoning.
            try:
                current_prices, _live_sectors = nps.get_current_prices_and_sectors(symbols=held_symbols)
            except requests.RequestException as e:
                raise HTTPException(
                    status_code=502,
                    detail=f"Could not fetch live prices from NGX Pulse: {e}. "
                           "Supply current_price_by_symbol manually to bypass live fetch "
                           "(e.g. during an NGX Pulse outage).",
                )

        missing_prices = [sym for sym in held_symbols if sym not in current_prices]

        # IMPORTANT UNIT NOTE: `cash` throughout this API (and portfolio_service's
        # check_hard_breaches, which this feeds into below) is a FRACTION of the
        # total portfolio, 0-1. db.holdings_to_weights, however, expects cash as an
        # ABSOLUTE currency amount to sum against holdings' market value (see its
        # docstring and test_db.py's cash=500.0-style usage). Converting the
        # supplied fraction f into the equivalent absolute amount, given market
        # value M, via cash = f*M/(1-f) (derived from f = cash/(M+cash)) preserves
        # the API's existing fraction-based contract while feeding db.py what it
        # actually expects -- rather than silently passing a 0-1 fraction into a
        # function that treats it as a raw currency amount, which was the actual
        # bug caught by test_main.py (see HANDOFF for the general policy of
        # root-causing rather than patching around test failures).
        market_value = sum(
            float(h["shares"]) * current_prices.get(h["symbol"], 0.0)
            for h in holdings_list
        )
        cash_amount = (payload.cash * market_value / (1 - payload.cash)) if market_value > 0 else 0.0
        weights_dict = db.holdings_to_weights(holdings_list, current_prices, cash_amount)
        sectors = {h["symbol"]: h["sector"] for h in holdings_list}
        if payload.sectors:
            sectors.update(payload.sectors)
    else:
        missing_prices = []
        if payload.actual_weights is None or payload.sectors is None:
            raise HTTPException(
                status_code=422,
                detail="Either portfolio_id, or both actual_weights and sectors, must be supplied.",
            )
        weights_dict = payload.actual_weights
        sectors = payload.sectors

    actual = pd.Series(weights_dict)
    target = pd.Series(payload.fresh_target_weights)

    drift_df = ps.check_drift(actual, target, drift_threshold=payload.drift_threshold)
    drift_rows = [
        DriftRow(
            symbol=sym,
            actual_weight=row["actual_weight"],
            fresh_target=(None if pd.isna(row["fresh_target"]) else row["fresh_target"]),
            drift=(None if pd.isna(row["drift"]) else row["drift"]),
            triggered=bool(row["triggered"]),
        )
        for sym, row in drift_df.iterrows()
    ]
    any_drift = any(r.triggered for r in drift_rows)

    breaches = ps.check_hard_breaches(actual, sectors, payload.cash)
    sector_warnings = ps.check_sector_warnings(actual, sectors)

    signals_logged: List[str] = []
    if payload.portfolio_id and payload.persist_signals:
        for r in drift_rows:
            if r.triggered:
                sid = db.log_signal_if_new(
                    payload.portfolio_id, "drift", r.symbol,
                    {"actual_weight": r.actual_weight, "fresh_target": r.fresh_target, "drift": r.drift},
                )
                if sid:
                    signals_logged.append(sid)
        for sym, w in breaches["stock_breach"].items():
            sid = db.log_signal_if_new(payload.portfolio_id, "stock_breach", sym, {"weight": w})
            if sid:
                signals_logged.append(sid)
        for sector, w in breaches["sector_breach"].items():
            sid = db.log_signal_if_new(payload.portfolio_id, "sector_breach", sector, {"weight": w})
            if sid:
                signals_logged.append(sid)
        if breaches["cash_breach"]:
            sid = db.log_signal_if_new(payload.portfolio_id, "cash_breach", None, {"cash": payload.cash})
            if sid:
                signals_logged.append(sid)
        if breaches["holdings_breach"]:
            sid = db.log_signal_if_new(
                payload.portfolio_id, "holdings_breach", None, {"holdings_count": len(actual)}
            )
            if sid:
                signals_logged.append(sid)
        for sector, w in sector_warnings["sector_warning"].items():
            sid = db.log_signal_if_new(payload.portfolio_id, "sector_warning", sector, {"weight": w})
            if sid:
                signals_logged.append(sid)

    return RebalanceCheckResponse(
        drift_rows=drift_rows,
        any_drift_triggered=any_drift,
        hard_breaches=breaches,
        sector_warnings=sector_warnings,
        action_required=bool(breaches["any_breach"]) or any_drift,
        signals_logged=signals_logged,
        missing_prices=missing_prices,
    )


# ---------------------------------------------------------------------
# Portfolios (thin wrappers over db.py; see db.py docstring for the
# reusability design: every portfolio is its own row/uuid, nothing global)
# ---------------------------------------------------------------------

class PortfolioCreateRequest(BaseModel):
    name: str
    broker: str
    currency: str = "NGN"
    initial_capital: float
    inception_date: date


class PortfolioOut(BaseModel):
    id: str
    name: str
    broker: Optional[str] = None
    currency: Optional[str] = None
    initial_capital: Optional[float] = None
    inception_date: date
    status: str
    closed_date: Optional[date] = None
    created_at: Optional[datetime] = None


def _portfolio_to_out(p: dict) -> PortfolioOut:
    return PortfolioOut(
        id=p["id"], name=p["name"], broker=p["broker"], currency=p["currency"],
        initial_capital=(float(p["initial_capital"]) if p["initial_capital"] is not None else None),
        inception_date=p["inception_date"], status=p["status"],
        closed_date=p["closed_date"], created_at=p["created_at"],
    )


@app.post("/portfolios", response_model=PortfolioOut)
def create_portfolio(payload: PortfolioCreateRequest):
    pid = db.create_portfolio(
        payload.name, payload.broker, payload.currency,
        payload.initial_capital, payload.inception_date,
    )
    return _portfolio_to_out(db.get_portfolio(pid))


@app.get("/portfolios/{portfolio_id}", response_model=PortfolioOut)
def get_portfolio(portfolio_id: str):
    p = db.get_portfolio(portfolio_id)
    if p is None:
        raise HTTPException(status_code=404, detail=f"No portfolio with id {portfolio_id}")
    return _portfolio_to_out(p)


class PortfolioCloseRequest(BaseModel):
    closed_date: Optional[date] = None


@app.post("/portfolios/{portfolio_id}/close", response_model=PortfolioOut)
def close_portfolio(portfolio_id: str, payload: PortfolioCloseRequest = PortfolioCloseRequest()):
    if db.get_portfolio(portfolio_id) is None:
        raise HTTPException(status_code=404, detail=f"No portfolio with id {portfolio_id}")
    db.close_portfolio(portfolio_id, closed_date=payload.closed_date)
    return _portfolio_to_out(db.get_portfolio(portfolio_id))


# ---------------------------------------------------------------------
# Telegram chat <-> active portfolio
# ---------------------------------------------------------------------

class SwitchPortfolioRequest(BaseModel):
    name: str
    broker: str
    currency: str = "NGN"
    initial_capital: float
    inception_date: date
    close_previous: bool = True


@app.post("/chats/{chat_id}/switch-portfolio", response_model=PortfolioOut)
def switch_portfolio(chat_id: str, payload: SwitchPortfolioRequest):
    """The one-call 'competition ended, here's the new portfolio' action:
    closes whatever this chat is currently pointed at, creates the new
    portfolio, and points the chat at it. See db.start_new_portfolio_for_chat."""
    new_id = db.start_new_portfolio_for_chat(
        chat_id, payload.name, payload.broker, payload.currency,
        payload.initial_capital, payload.inception_date,
        close_previous=payload.close_previous,
    )
    return _portfolio_to_out(db.get_portfolio(new_id))


class SetActivePortfolioRequest(BaseModel):
    portfolio_id: str


@app.get("/chats/{chat_id}/active-portfolio", response_model=Optional[PortfolioOut])
def get_active_portfolio(chat_id: str):
    active = db.get_active_portfolio(chat_id)
    return _portfolio_to_out(active) if active else None


@app.post("/chats/{chat_id}/active-portfolio", response_model=PortfolioOut)
def set_active_portfolio(chat_id: str, payload: SetActivePortfolioRequest):
    if db.get_portfolio(payload.portfolio_id) is None:
        raise HTTPException(status_code=404, detail=f"No portfolio with id {payload.portfolio_id}")
    db.set_active_portfolio(chat_id, payload.portfolio_id)
    return _portfolio_to_out(db.get_portfolio(payload.portfolio_id))


# ---------------------------------------------------------------------
# Holdings
# ---------------------------------------------------------------------

class HoldingIn(BaseModel):
    symbol: str
    shares: float
    avg_cost: Optional[float] = None
    sector: Optional[str] = None


class HoldingOut(BaseModel):
    id: str
    portfolio_id: str
    symbol: str
    sector: Optional[str] = None
    shares: float
    avg_cost: Optional[float] = None
    last_updated: Optional[datetime] = None


def _holding_to_out(h: dict) -> HoldingOut:
    return HoldingOut(
        id=h["id"], portfolio_id=h["portfolio_id"], symbol=h["symbol"], sector=h["sector"],
        shares=float(h["shares"]), avg_cost=(float(h["avg_cost"]) if h["avg_cost"] is not None else None),
        last_updated=h["last_updated"],
    )


@app.get("/portfolios/{portfolio_id}/holdings", response_model=List[HoldingOut])
def get_holdings(portfolio_id: str):
    return [_holding_to_out(h) for h in db.get_holdings(portfolio_id)]


@app.put("/portfolios/{portfolio_id}/holdings", response_model=List[HoldingOut])
def upsert_holdings(portfolio_id: str, payload: List[HoldingIn]):
    """Full replace, not a merge -- pass the complete current holdings set
    (matches db.upsert_holdings semantics, documented there)."""
    if db.get_portfolio(portfolio_id) is None:
        raise HTTPException(status_code=404, detail=f"No portfolio with id {portfolio_id}")
    db.upsert_holdings(portfolio_id, [h.dict() for h in payload])
    return [_holding_to_out(h) for h in db.get_holdings(portfolio_id)]


# ---------------------------------------------------------------------
# Trades
# ---------------------------------------------------------------------

class TradeLogRequest(BaseModel):
    symbol: str
    side: str
    shares: float
    price: float
    fee: float
    reason: Optional[str] = None
    sector: Optional[str] = None


class TradeOut(BaseModel):
    id: str
    portfolio_id: str
    symbol: str
    side: str
    shares: float
    price: float
    fee: float
    reason: Optional[str] = None
    executed_at: Optional[datetime] = None


def _trade_to_out(t: dict) -> TradeOut:
    return TradeOut(
        id=t["id"], portfolio_id=t["portfolio_id"], symbol=t["symbol"], side=t["side"],
        shares=float(t["shares"]), price=float(t["price"]), fee=float(t["fee"]),
        reason=t["reason"], executed_at=t["executed_at"],
    )


@app.post("/portfolios/{portfolio_id}/trades", response_model=TradeOut)
def log_trade(portfolio_id: str, payload: TradeLogRequest):
    """Logs the trade AND updates holdings in the same transaction (see
    db.log_trade). Raises 422 on an invalid trade (bad side, oversell, or
    selling a symbol never held) -- holdings are left unchanged on failure."""
    if db.get_portfolio(portfolio_id) is None:
        raise HTTPException(status_code=404, detail=f"No portfolio with id {portfolio_id}")
    try:
        tid = db.log_trade(
            portfolio_id, payload.symbol, payload.side, payload.shares,
            payload.price, payload.fee, reason=payload.reason, sector=payload.sector,
        )
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    trade = next(t for t in db.get_trades(portfolio_id) if t["id"] == tid)
    return _trade_to_out(trade)


@app.get("/portfolios/{portfolio_id}/trades", response_model=List[TradeOut])
def get_trades(portfolio_id: str):
    return [_trade_to_out(t) for t in db.get_trades(portfolio_id)]


# ---------------------------------------------------------------------
# Signals
# ---------------------------------------------------------------------

class SignalOut(BaseModel):
    id: str
    portfolio_id: str
    signal_type: str
    symbol: Optional[str] = None
    detail: Optional[dict] = None
    fired_at: Optional[datetime] = None
    acknowledged: bool


def _signal_to_out(s: dict) -> SignalOut:
    return SignalOut(
        id=s["id"], portfolio_id=s["portfolio_id"], signal_type=s["signal_type"],
        symbol=s["symbol"], detail=s["detail"], fired_at=s["fired_at"], acknowledged=s["acknowledged"],
    )


@app.get("/portfolios/{portfolio_id}/signals", response_model=List[SignalOut])
def get_unacknowledged_signals(portfolio_id: str):
    return [_signal_to_out(s) for s in db.get_unacknowledged_signals(portfolio_id)]


@app.post("/signals/{signal_id}/acknowledge")
def acknowledge_signal(signal_id: str):
    db.acknowledge_signal(signal_id)
    return {"acknowledged": True, "signal_id": signal_id}


# ---------------------------------------------------------------------
# Telegram bot webhook
# ---------------------------------------------------------------------

@app.post("/telegram/webhook")
async def telegram_webhook(
    request: Request,
    x_telegram_bot_api_secret_token: Optional[str] = Header(default=None),
):
    """Receives Telegram Update objects (per Telegram's webhook model,
    set up via setWebhook -- see deployment notes). Verifies the secret
    token header if TELEGRAM_WEBHOOK_SECRET is configured, then hands off
    to bot.handle_update(), which parses the command, dispatches it, and
    sends the reply back to Telegram directly (this endpoint's response
    to Telegram itself doesn't carry the reply -- Telegram's webhook
    contract is fire-and-forget, replies go via a separate sendMessage
    call). Always returns 200 on a recognized-but-failed command (the
    error goes to the user as a chat message instead) so Telegram doesn't
    interpret a failure as a delivery problem and retry indefinitely."""
    if not tg_bot.verify_webhook_secret(x_telegram_bot_api_secret_token):
        raise HTTPException(status_code=401, detail="invalid webhook secret")
    update = await request.json()
    tg_bot.handle_update(update)
    return {"ok": True}


@app.get("/health")
def health():
    return {"status": "ok"}


# ---------------------------------------------------------------------
# Monthly screen (automated end-of-month trigger, e.g. via a scheduled
# GitHub Actions workflow -- see the deployment notes for the workflow
# file). Builds the live investable universe from NGX Pulse
# (universe_service.py), screens it (screening_service.py), constructs a
# fresh compliant portfolio (portfolio_service.py), and sends the
# resulting target weights to the given Telegram chat -- it does NOT
# execute trades or touch any stored portfolio/holdings. Per project
# workflow, the person reviews the weights, executes the real trades on
# Ticker manually, and logs them via the bot's /log_trade -- this endpoint
# only replaces the "run the screen" step, not the "decide and trade" step.
# ---------------------------------------------------------------------

MONTHLY_SCREEN_SECRET = os.environ.get("MONTHLY_SCREEN_SECRET", "")


class MonthlyScreenRequest(BaseModel):
    chat_id: str = Field(..., description="Telegram chat id to send the resulting target weights to.")
    pool_size: int = Field(default=15, description="Top-N eligible stocks considered for the portfolio.")
    cash_buffer: float = Field(default=ps.CASH_BUFFER)
    lookback_days: int = Field(default=nps.DEFAULT_LOOKBACK_DAYS)
    max_ngx_pulse_requests: int = Field(
        default=90, description="Budget for the price-history fetch loop (one request per "
                                 "candidate symbol). Left under NGX Pulse's 100/day cap on "
                                 "purpose -- this endpoint's own universe-snapshot call and "
                                 "any same-day /value checks from the bot also draw on the "
                                 "same real daily quota, which ngx_pulse_service.py tracks "
                                 "only per-call, not across the whole day (see its docstring)."
    )


@app.post("/monthly-screen")
def monthly_screen(
    payload: MonthlyScreenRequest,
    background_tasks: BackgroundTasks,
    x_monthly_screen_secret: Optional[str] = Header(default=None),
):
    """Kicks off a live universe screen + portfolio construction in the
    background and returns immediately; results arrive via Telegram, not
    in this response (see monthly_screen_service.run_monthly_screen_and_notify's
    docstring for why). Requires X-Monthly-Screen-Secret to match
    MONTHLY_SCREEN_SECRET when that env var is configured -- same pattern
    as the Telegram webhook secret, since this endpoint triggers real
    NGX Pulse quota usage and several minutes of scraping and shouldn't
    be triggerable by anyone who finds the URL.

    The actual work lives in monthly_screen_service.py, shared with
    bot.py's /monthly_screen Telegram command -- this endpoint and that
    command are two different TRIGGERS for the exact same implementation,
    not two implementations."""
    if MONTHLY_SCREEN_SECRET and x_monthly_screen_secret != MONTHLY_SCREEN_SECRET:
        raise HTTPException(status_code=401, detail="invalid monthly screen secret")
    background_tasks.add_task(
        mss.run_monthly_screen_and_notify,
        payload.chat_id, payload.pool_size, payload.cash_buffer,
        payload.lookback_days, payload.max_ngx_pulse_requests,
        tg_bot.send_message, tg_bot._escape_md,
    )
    return {
        "status": "accepted",
        "detail": "Monthly screen started in the background; results will be sent via Telegram.",
    }


# ---------------------------------------------------------------------
# Daily price history accumulation
# ---------------------------------------------------------------------
# Works around koboterminal.com's Free-plan restriction (confirmed
# 2026-08-31: a single days=130 request 403s uniformly on every symbol --
# see ngx_pulse_service.py's module docstring for the incident writeup)
# by pulling a small days<=7 window daily and accumulating it in db.py's
# price_history table via price_history_service.py. This is genuinely
# routine infrastructure (unlike /monthly-screen, which represents a real
# portfolio decision point) -- failures are logged via
# price_history_service's existing per-symbol logging and visible in
# Render's logs, not pushed to Telegram, to avoid a daily notification for
# something that isn't actionable most days.

DAILY_PRICE_UPDATE_SECRET = os.environ.get("DAILY_PRICE_UPDATE_SECRET", "")


def _run_daily_price_update_task():
    """Background task body: resolves which symbols to update (live
    universe union already-tracked symbols -- see
    price_history_service.get_symbols_to_update), then pulls days<=7 per
    symbol and upserts. Intentionally synchronous/no return value used --
    this runs off the request/response path via BackgroundTasks, same as
    monthly_screen; the result is logged, not returned anywhere, since
    nothing is waiting on this response."""
    try:
        symbols = phs.get_symbols_to_update()
        result = phs.run_daily_price_update(symbols)
        print(
            f"Daily price update: {len(result['symbols_updated'])} updated, "
            f"{len(result['symbols_failed'])} failed, out of {len(symbols)} symbols."
        )
    except Exception as e:  # noqa: BLE001 -- background task, nothing else observes this
        print(f"Daily price update failed entirely: {e}")


@app.post("/daily-price-update")
def daily_price_update(
    background_tasks: BackgroundTasks,
    x_daily_price_update_secret: Optional[str] = Header(default=None),
):
    """Kicks off the daily days<=7 price-history accumulation pull in the
    background and returns immediately. Requires
    X-Daily-Price-Update-Secret to match DAILY_PRICE_UPDATE_SECRET when
    that env var is configured -- same secret-header pattern as
    /monthly-screen, since this still triggers real NGX Pulse/Kobo
    Terminal requests and shouldn't be triggerable by anyone who finds the
    URL. Intended to be called once a day by a scheduled GitHub Actions
    workflow (see .github/workflows/daily-price-update.yml), not by hand."""
    if DAILY_PRICE_UPDATE_SECRET and x_daily_price_update_secret != DAILY_PRICE_UPDATE_SECRET:
        raise HTTPException(status_code=401, detail="invalid daily price update secret")
    background_tasks.add_task(_run_daily_price_update_task)
    return {
        "status": "accepted",
        "detail": "Daily price update started in the background; check Render logs for the result.",
    }


@app.post("/price-history/seed")
def price_history_seed(
    background_tasks: BackgroundTasks,
    x_daily_price_update_secret: Optional[str] = Header(default=None),
):
    """ONE-TIME manual trigger for price_history_service.seed_from_repo_panel()
    -- backfills price_history from the project's existing
    raw_price_history.json (real history back to Feb 2026) so the daily
    accumulation job starts from a near-full window instead of empty.
    Idempotent (safe to call more than once -- upserts, never duplicates),
    but there's normally no reason to call it after the first time. Reuses
    DAILY_PRICE_UPDATE_SECRET rather than a third secret -- this endpoint
    is closely related to /daily-price-update and doesn't need its own."""
    if DAILY_PRICE_UPDATE_SECRET and x_daily_price_update_secret != DAILY_PRICE_UPDATE_SECRET:
        raise HTTPException(status_code=401, detail="invalid daily price update secret")
    background_tasks.add_task(phs.seed_from_repo_panel)
    return {
        "status": "accepted",
        "detail": "Price history seed started in the background; check Render logs for the result.",
    }
