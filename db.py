"""
db.py

Persistence layer for the PMC strategy API, built to make the bot
reusable across portfolios (this competition, a future one, multiple
users) rather than hardcoded to a single live account.

DESIGN: every portfolio gets its own row and its own uuid. Holdings,
trades, and signals all belong to a portfolio_id -- never global. A
telegram_chats table maps a chat to whichever portfolio it's CURRENTLY
pointed at, via active_portfolio_id. Switching to a new portfolio
(e.g. after this competition ends) is one explicit action -- there is
no way for the bot to auto-detect this, since there's no live broker
feed; closing the old portfolio and activating a new one is deliberate,
never inferred.

fundamentals_cache is NOT portfolio-scoped -- ROE/P/E belong to the
stock, not to who holds it, so every portfolio using this bot shares
one cache and avoids redundant stockanalysis.com scraping.

PORTABILITY: built on SQLAlchemy Core (not raw dialect-specific SQL)
so the exact same schema and queries run against SQLite (used for all
testing in this session, since this sandbox has no network route to
Supabase) and Postgres/Supabase in production -- swap DATABASE_URL,
no code changes. uuids are generated in Python (uuid.uuid4()) rather
than relying on a DB-side default, since SQLite has no native uuid
type and this keeps behavior identical across both engines.
"""

import os
import json
import uuid
from datetime import datetime, date, timedelta
from typing import Optional

from sqlalchemy import (
    create_engine, MetaData, Table, Column, String, Numeric, Boolean,
    DateTime, Date, Text, ForeignKey, JSON, UniqueConstraint, select, update, delete, and_,
)

DATABASE_URL = os.environ.get("DATABASE_URL", "sqlite:///./pmc_local.db")

# Supabase gives connection strings starting postgres://, but SQLAlchemy's
# psycopg2 driver expects postgresql://. Normalize so either form works.
_normalized_url = DATABASE_URL
if _normalized_url.startswith("postgres://"):
    _normalized_url = _normalized_url.replace("postgres://", "postgresql://", 1)

engine = create_engine(_normalized_url, future=True)
metadata = MetaData()


def _uuid() -> str:
    return str(uuid.uuid4())


# ---------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------

portfolios = Table(
    "portfolios", metadata,
    Column("id", String, primary_key=True),
    Column("name", String, nullable=False),
    Column("broker", String),
    Column("currency", String, default="NGN"),
    Column("initial_capital", Numeric),
    Column("inception_date", Date, nullable=False),
    Column("status", String, nullable=False, default="active"),  # 'active' | 'closed'
    Column("closed_date", Date),
    Column("created_at", DateTime, default=datetime.utcnow),
)

holdings = Table(
    "holdings", metadata,
    Column("id", String, primary_key=True),
    Column("portfolio_id", String, ForeignKey("portfolios.id"), nullable=False),
    Column("symbol", String, nullable=False),
    Column("sector", String),
    Column("shares", Numeric, nullable=False),
    Column("avg_cost", Numeric),
    Column("last_updated", DateTime, default=datetime.utcnow),
    UniqueConstraint("portfolio_id", "symbol", name="uq_holdings_portfolio_symbol"),
)

trades = Table(
    "trades", metadata,
    Column("id", String, primary_key=True),
    Column("portfolio_id", String, ForeignKey("portfolios.id"), nullable=False),
    Column("symbol", String, nullable=False),
    Column("side", String, nullable=False),  # 'buy' | 'sell'
    Column("shares", Numeric, nullable=False),
    Column("price", Numeric, nullable=False),
    Column("fee", Numeric, nullable=False),
    Column("reason", String),  # 'initial_construction' | 'drift' | 'hard_breach' | ...
    Column("executed_at", DateTime, default=datetime.utcnow),
)

signals = Table(
    "signals", metadata,
    Column("id", String, primary_key=True),
    Column("portfolio_id", String, ForeignKey("portfolios.id"), nullable=False),
    Column("signal_type", String, nullable=False),  # 'drift' | 'stock_breach' | 'sector_breach' | 'cash_breach' | 'holdings_breach'
    Column("symbol", String),
    Column("detail", JSON),
    Column("fired_at", DateTime, default=datetime.utcnow),
    Column("acknowledged", Boolean, default=False),
)

telegram_chats = Table(
    "telegram_chats", metadata,
    Column("chat_id", String, primary_key=True),  # stored as string; Telegram chat_ids fit in bigint but string is dialect-safe
    Column("active_portfolio_id", String, ForeignKey("portfolios.id")),
    Column("created_at", DateTime, default=datetime.utcnow),
)

fundamentals_cache = Table(
    "fundamentals_cache", metadata,
    Column("symbol", String, primary_key=True),
    Column("as_of_date", Date, primary_key=True),
    Column("roe", Numeric),
    Column("pe", Numeric),
    Column("eps_used", Numeric),
    Column("eps_period_ending", Date),
    Column("fetched_at", DateTime, default=datetime.utcnow),
)

price_history = Table(
    "price_history", metadata,
    Column("symbol", String, primary_key=True),
    Column("trade_date", Date, primary_key=True),
    Column("close_price", Numeric, nullable=False),
    Column("volume", Numeric),
    Column("fetched_at", DateTime, default=datetime.utcnow),
)
# NOT portfolio-scoped, same reasoning as fundamentals_cache -- a price on
# a given date is a fact about the stock, not about who holds it, so every
# portfolio using this bot shares one accumulated history.
#
# WHY THIS TABLE EXISTS: koboterminal.com's Free plan (the account this
# project is currently on, confirmed 2026-08-31) caps historical price
# requests at days<=7 -- screening_service needs >=100 data points per
# symbol for momentum/vol, which a single live call can no longer provide
# on this plan. This table lets a small days<=7 pull, run daily, accumulate
# into a real usable window over time via upsert (see upsert_price_history),
# the same way the existing raw_price_history.json panel in the project's
# GitHub repo was almost certainly built -- see ngx_pulse_service.py's
# module docstring for the full incident writeup.


def init_db():
    """Creates all tables if they don't already exist. Safe to call on
    every app startup -- idempotent."""
    metadata.create_all(engine)


# ---------------------------------------------------------------------
# Portfolios
# ---------------------------------------------------------------------

def create_portfolio(name: str, broker: str, currency: str,
                      initial_capital: float, inception_date: date) -> str:
    pid = _uuid()
    with engine.begin() as conn:
        conn.execute(portfolios.insert().values(
            id=pid, name=name, broker=broker, currency=currency,
            initial_capital=initial_capital, inception_date=inception_date,
            status="active", created_at=datetime.utcnow(),
        ))
    return pid


def close_portfolio(portfolio_id: str, closed_date: Optional[date] = None):
    with engine.begin() as conn:
        conn.execute(
            update(portfolios)
            .where(portfolios.c.id == portfolio_id)
            .values(status="closed", closed_date=closed_date or date.today())
        )


def get_portfolio(portfolio_id: str) -> Optional[dict]:
    with engine.connect() as conn:
        row = conn.execute(select(portfolios).where(portfolios.c.id == portfolio_id)).mappings().first()
        return dict(row) if row else None


def list_active_portfolios() -> list:
    """Every portfolio with status='active', across the whole system --
    NOT scoped per chat, because the portfolios table has no chat_id
    column at all (only telegram_chats.active_portfolio_id tracks the
    CURRENT single pointer per chat, never history). This system is
    single-user in practice (one competitor, one DB), so that's fine --
    lets a chat managing multiple simultaneously-open portfolios (e.g.
    the competition portfolio and a personal account) see all of them
    via bot.py's /list_portfolios and switch between them via
    /use_portfolio, without either ever needing to be closed just to
    check on the other."""
    with engine.connect() as conn:
        rows = conn.execute(
            select(portfolios).where(portfolios.c.status == "active")
            .order_by(portfolios.c.created_at)
        ).mappings().all()
        return [dict(r) for r in rows]


# ---------------------------------------------------------------------
# Telegram chat <-> active portfolio
# ---------------------------------------------------------------------

def register_chat(chat_id: str):
    with engine.begin() as conn:
        existing = conn.execute(
            select(telegram_chats).where(telegram_chats.c.chat_id == str(chat_id))
        ).first()
        if not existing:
            conn.execute(telegram_chats.insert().values(
                chat_id=str(chat_id), active_portfolio_id=None, created_at=datetime.utcnow(),
            ))


def set_active_portfolio(chat_id: str, portfolio_id: Optional[str]):
    """Explicitly points a chat at a (possibly different) portfolio, or
    clears the pointer entirely if portfolio_id is None (used when a
    portfolio is closed with no immediate replacement -- see bot.py's
    /close_portfolio). This is the ONLY way the active portfolio changes
    -- never inferred."""
    register_chat(chat_id)
    with engine.begin() as conn:
        conn.execute(
            update(telegram_chats)
            .where(telegram_chats.c.chat_id == str(chat_id))
            .values(active_portfolio_id=portfolio_id)
        )


def get_active_portfolio(chat_id: str) -> Optional[dict]:
    with engine.connect() as conn:
        row = conn.execute(
            select(telegram_chats).where(telegram_chats.c.chat_id == str(chat_id))
        ).mappings().first()
        if not row or not row["active_portfolio_id"]:
            return None
        return get_portfolio(row["active_portfolio_id"])


def start_new_portfolio_for_chat(chat_id: str, name: str, broker: str, currency: str,
                                  initial_capital: float, inception_date: date,
                                  close_previous: bool = True) -> str:
    """The one-call version of 'competition ended, here's the new
    portfolio': closes whatever this chat is currently pointed at (if
    close_previous, the default), creates the new portfolio, and points
    the chat at it. Nothing about the previous portfolio's data is
    deleted -- it just stops being the active one."""
    previous = get_active_portfolio(chat_id)
    if close_previous and previous and previous["status"] == "active":
        close_portfolio(previous["id"])
    new_id = create_portfolio(name, broker, currency, initial_capital, inception_date)
    set_active_portfolio(chat_id, new_id)
    return new_id


# ---------------------------------------------------------------------
# Holdings
# ---------------------------------------------------------------------

def upsert_holdings(portfolio_id: str, holdings_list: list):
    """Replaces the full holdings set for a portfolio in one call.
    holdings_list: [{"symbol": str, "shares": float, "avg_cost": float, "sector": str}, ...]
    This is a full replace (delete + insert), not a merge -- callers
    should pass the complete current holdings set each time, matching
    how build_initial_portfolio/rebalance produce a full target, not a
    diff."""
    with engine.begin() as conn:
        conn.execute(delete(holdings).where(holdings.c.portfolio_id == portfolio_id))
        now = datetime.utcnow()
        for h in holdings_list:
            conn.execute(holdings.insert().values(
                id=_uuid(), portfolio_id=portfolio_id, symbol=h["symbol"],
                sector=h.get("sector"), shares=h["shares"], avg_cost=h.get("avg_cost"),
                last_updated=now,
            ))


def get_holdings(portfolio_id: str) -> list:
    with engine.connect() as conn:
        rows = conn.execute(
            select(holdings).where(holdings.c.portfolio_id == portfolio_id)
        ).mappings().all()
        return [dict(r) for r in rows]


def holdings_to_weights(holdings_list: list, current_price_by_symbol: dict, cash: float) -> dict:
    """Converts stored share counts into portfolio weights, since
    portfolio_service.py operates on weights, not share counts. Positions
    with no supplied current price are skipped with their value treated
    as 0 -- callers should ensure current_price_by_symbol covers every
    held symbol, or the resulting weights will understate true exposure."""
    market_values = {
        h["symbol"]: float(h["shares"]) * current_price_by_symbol.get(h["symbol"], 0.0)
        for h in holdings_list
    }
    total = sum(market_values.values()) + cash
    if total <= 0:
        return {sym: 0.0 for sym in market_values}
    return {sym: mv / total for sym, mv in market_values.items()}


# ---------------------------------------------------------------------
# Trades
# ---------------------------------------------------------------------

def log_trade(portfolio_id: str, symbol: str, side: str, shares: float,
              price: float, fee: float, reason: Optional[str] = None,
              sector: Optional[str] = None) -> str:
    """Logs the trade AND updates the holdings table to reflect it --
    incrementing/decrementing shares and recomputing a weighted-average
    cost on buys. This is the one function that should be called whenever
    a trade is actually executed on the broker, so holdings never drift
    out of sync with the trade log."""
    if side not in ("buy", "sell"):
        raise ValueError(f"side must be 'buy' or 'sell', got {side!r}")

    tid = _uuid()
    with engine.begin() as conn:
        conn.execute(trades.insert().values(
            id=tid, portfolio_id=portfolio_id, symbol=symbol, side=side,
            shares=shares, price=price, fee=fee, reason=reason,
            executed_at=datetime.utcnow(),
        ))

        existing = conn.execute(
            select(holdings).where(and_(
                holdings.c.portfolio_id == portfolio_id, holdings.c.symbol == symbol,
            ))
        ).mappings().first()

        if side == "buy":
            if existing:
                old_shares = float(existing["shares"])
                old_cost = float(existing["avg_cost"] or 0.0)
                new_shares = old_shares + shares
                new_avg_cost = (
                    ((old_shares * old_cost) + (shares * price)) / new_shares
                    if new_shares > 0 else 0.0
                )
                conn.execute(
                    update(holdings).where(holdings.c.id == existing["id"]).values(
                        shares=new_shares, avg_cost=new_avg_cost, last_updated=datetime.utcnow(),
                    )
                )
            else:
                conn.execute(holdings.insert().values(
                    id=_uuid(), portfolio_id=portfolio_id, symbol=symbol, sector=sector,
                    shares=shares, avg_cost=price, last_updated=datetime.utcnow(),
                ))
        else:  # sell
            if not existing:
                raise ValueError(f"Cannot sell {symbol}: no existing holding in portfolio {portfolio_id}")
            new_shares = float(existing["shares"]) - shares
            if new_shares < -1e-9:
                raise ValueError(
                    f"Cannot sell {shares} shares of {symbol}: only {existing['shares']} held."
                )
            if new_shares <= 1e-9:
                conn.execute(delete(holdings).where(holdings.c.id == existing["id"]))
            else:
                # avg_cost is unchanged by a partial sell (standard convention)
                conn.execute(
                    update(holdings).where(holdings.c.id == existing["id"]).values(
                        shares=new_shares, last_updated=datetime.utcnow(),
                    )
                )
    return tid


def delete_trade(portfolio_id: str, trade_id: str) -> dict:
    """Deletes one trade and correctly recomputes the affected symbol's
    holding by REPLAYING all of that symbol's remaining trades from
    scratch, in execution order -- rather than trying to reverse just
    that trade's delta. Delta-reversal only works cleanly if the deleted
    trade is provably the LAST one for that symbol: avg_cost is a
    running weighted average across every buy, so undoing an EARLIER buy
    by simple subtraction would leave avg_cost wrong for any buy that
    layered on after it. Replay sidesteps this entirely -- the resulting
    holding is always exactly what it would be if the deleted trade had
    never happened, regardless of its position in the symbol's history.

    sector is preserved from the current holding row (trades has no
    sector column of its own -- only set on a holding's initial INSERT,
    same convention as log_trade), so undoing every trade for a symbol
    still remembers its sector if the position is later re-opened via a
    fresh /log_trade buy... except when the delete itself fully closes
    the position, in which case the holding row (and its sector) is
    removed, same as log_trade's own full-sell convention. This is a
    real, accepted trade-off: re-opening a fully-undone position later
    re-fetches sector live, same as any other brand-new position.

    Returns the resulting holding dict for the symbol (or None if the
    replay leaves zero shares -- position fully closed). Raises
    ValueError if no such trade exists in this portfolio, OR if undoing
    it would leave the REMAINING trade history internally inconsistent
    -- specifically, a later sell that the deleted trade's shares had
    made possible would now oversell a position that never held enough
    shares at that point in the sequence. That check runs against every
    step of the replay, not just the final total (an intermediate dip
    below zero is just as invalid as a negative final total), and the
    delete is refused outright rather than silently producing a wrong
    or falsely "fully closed" holding -- nothing is written to the DB
    until the full replay is confirmed valid."""
    with engine.begin() as conn:
        trade = conn.execute(
            select(trades).where(and_(trades.c.id == trade_id, trades.c.portfolio_id == portfolio_id))
        ).mappings().first()
        if not trade:
            raise ValueError(f"No trade with id {trade_id} in portfolio {portfolio_id}")
        symbol = trade["symbol"]

        existing_holding = conn.execute(
            select(holdings).where(and_(holdings.c.portfolio_id == portfolio_id, holdings.c.symbol == symbol))
        ).mappings().first()
        preserved_sector = existing_holding["sector"] if existing_holding else None

        # Compute the replay from the trades that would REMAIN (i.e. excluding
        # trade_id) BEFORE touching the DB at all. If undoing an earlier buy
        # would leave a later sell oversold at any point in the sequence --
        # not just in the final total, an intermediate dip below zero is
        # just as invalid -- the remaining trade history is internally
        # inconsistent and this delete must be refused outright, not
        # silently produce a wrong (or falsely "fully closed") holding.
        remaining = conn.execute(
            select(trades).where(and_(
                trades.c.portfolio_id == portfolio_id,
                trades.c.symbol == symbol,
                trades.c.id != trade_id,
            )).order_by(trades.c.executed_at, trades.c.id)
        ).mappings().all()

        shares, avg_cost = 0.0, 0.0
        for t in remaining:
            t_shares, t_price = float(t["shares"]), float(t["price"])
            if t["side"] == "buy":
                new_shares = shares + t_shares
                avg_cost = ((shares * avg_cost) + (t_shares * t_price)) / new_shares if new_shares > 0 else 0.0
                shares = new_shares
            else:  # sell -- avg_cost unchanged, same convention as log_trade
                shares -= t_shares
                if shares < -1e-9:
                    raise ValueError(
                        f"Cannot undo trade {trade_id[:8]}: without it, the {symbol} sell of "
                        f"{t_shares:g} shares on {t['executed_at']:%Y-%m-%d} [{t['id'][:8]}] would "
                        f"oversell a position that never held enough shares at that point. Undo (or "
                        f"correct) that later trade first, or leave this one in place."
                    )

        # Only now, once the replay is confirmed internally consistent, apply it for real.
        conn.execute(delete(trades).where(trades.c.id == trade_id))
        conn.execute(delete(holdings).where(
            and_(holdings.c.portfolio_id == portfolio_id, holdings.c.symbol == symbol)
        ))

        if shares > 1e-9:
            new_holding_id = _uuid()
            conn.execute(holdings.insert().values(
                id=new_holding_id, portfolio_id=portfolio_id, symbol=symbol, sector=preserved_sector,
                shares=shares, avg_cost=avg_cost, last_updated=datetime.utcnow(),
            ))
            row = conn.execute(select(holdings).where(holdings.c.id == new_holding_id)).mappings().first()
            return dict(row)
        return None


def get_trades(portfolio_id: str) -> list:
    with engine.connect() as conn:
        rows = conn.execute(
            select(trades).where(trades.c.portfolio_id == portfolio_id)
            .order_by(trades.c.executed_at)
        ).mappings().all()
        return [dict(r) for r in rows]


# ---------------------------------------------------------------------
# Signals (dedup so the bot doesn't re-alert on the same unresolved issue)
# ---------------------------------------------------------------------

def get_unacknowledged_signal(portfolio_id: str, signal_type: str, symbol: Optional[str]) -> Optional[dict]:
    with engine.connect() as conn:
        conds = [
            signals.c.portfolio_id == portfolio_id,
            signals.c.signal_type == signal_type,
            signals.c.acknowledged == False,  # noqa: E712
        ]
        if symbol is not None:
            conds.append(signals.c.symbol == symbol)
        row = conn.execute(select(signals).where(and_(*conds))).mappings().first()
        return dict(row) if row else None


def log_signal_if_new(portfolio_id: str, signal_type: str, symbol: Optional[str], detail: dict) -> Optional[str]:
    """Only inserts a new signal row if there isn't already an
    UNACKNOWLEDGED one of the same (portfolio, type, symbol) -- this is
    the dedup that stops the bot re-sending the same drift/breach alert
    every time it polls. Returns the new signal id, or None if it was a
    duplicate of an existing unacknowledged alert."""
    if get_unacknowledged_signal(portfolio_id, signal_type, symbol):
        return None
    sid = _uuid()
    with engine.begin() as conn:
        conn.execute(signals.insert().values(
            id=sid, portfolio_id=portfolio_id, signal_type=signal_type, symbol=symbol,
            detail=detail, fired_at=datetime.utcnow(), acknowledged=False,
        ))
    return sid


def get_unacknowledged_signals(portfolio_id: str) -> list:
    with engine.connect() as conn:
        rows = conn.execute(
            select(signals).where(and_(
                signals.c.portfolio_id == portfolio_id,
                signals.c.acknowledged == False,  # noqa: E712
            )).order_by(signals.c.fired_at)
        ).mappings().all()
        return [dict(r) for r in rows]


def acknowledge_signal(signal_id: str):
    with engine.begin() as conn:
        conn.execute(update(signals).where(signals.c.id == signal_id).values(acknowledged=True))


# ---------------------------------------------------------------------
# Fundamentals cache (shared across all portfolios)
# ---------------------------------------------------------------------

def get_cached_fundamentals(symbol: str, as_of_date: date, max_age_days: int = 30) -> Optional[dict]:
    """Returns a cached fundamentals row for this exact (symbol, as_of_date)
    if it exists and was fetched within max_age_days -- otherwise None,
    signaling the caller should do a live fetch. Cache is keyed by
    as_of_date (not just symbol) because point-in-time fundamentals are
    legitimately different for different as_of dates (that's the whole
    no-lookahead design) -- this is a cache of computed results per date,
    not a single "latest known" snapshot."""
    with engine.connect() as conn:
        row = conn.execute(
            select(fundamentals_cache).where(and_(
                fundamentals_cache.c.symbol == symbol,
                fundamentals_cache.c.as_of_date == as_of_date,
            ))
        ).mappings().first()
        if not row:
            return None
        age = datetime.utcnow() - row["fetched_at"]
        if age > timedelta(days=max_age_days):
            return None
        return dict(row)


def cache_fundamentals(symbol: str, as_of_date: date, roe: Optional[float], pe: Optional[float],
                        eps_used: Optional[float], eps_period_ending: Optional[date]):
    with engine.begin() as conn:
        existing = conn.execute(
            select(fundamentals_cache).where(and_(
                fundamentals_cache.c.symbol == symbol,
                fundamentals_cache.c.as_of_date == as_of_date,
            ))
        ).first()
        values = dict(
            roe=roe, pe=pe, eps_used=eps_used, eps_period_ending=eps_period_ending,
            fetched_at=datetime.utcnow(),
        )
        if existing:
            conn.execute(
                update(fundamentals_cache).where(and_(
                    fundamentals_cache.c.symbol == symbol,
                    fundamentals_cache.c.as_of_date == as_of_date,
                )).values(**values)
            )
        else:
            conn.execute(fundamentals_cache.insert().values(
                symbol=symbol, as_of_date=as_of_date, **values,
            ))


# ---------------------------------------------------------------------
# Price history (accumulated daily, shared across all portfolios)
# ---------------------------------------------------------------------

def upsert_price_history(symbol: str, rows: list):
    """rows: [{"trade_date": date, "close_price": float, "volume": float|None}, ...]
    Upserts each (symbol, trade_date) row independently -- same
    select-then-update-or-insert idiom as cache_fundamentals, just looped
    per row, since one daily fetch typically returns several overlapping
    days (days=7 pulled daily means most rows already exist and are just
    re-confirmed, which is intentional: it's the cheap safety net against
    a single missed day silently leaving a gap in the accumulated window).
    Never deletes -- this ACCUMULATES history over time, unlike
    upsert_holdings' full-replace semantics; a caller wanting to wipe and
    reload a symbol's history entirely should do so explicitly, there is
    no delete_price_history here by design (accidental data loss on a
    slow-to-rebuild accumulated window is a worse failure mode than a
    missing convenience function)."""
    with engine.begin() as conn:
        for row in rows:
            trade_date = row["trade_date"]
            existing = conn.execute(
                select(price_history).where(and_(
                    price_history.c.symbol == symbol,
                    price_history.c.trade_date == trade_date,
                ))
            ).first()
            values = dict(
                close_price=row["close_price"], volume=row.get("volume"),
                fetched_at=datetime.utcnow(),
            )
            if existing:
                conn.execute(
                    update(price_history).where(and_(
                        price_history.c.symbol == symbol,
                        price_history.c.trade_date == trade_date,
                    )).values(**values)
                )
            else:
                conn.execute(price_history.insert().values(
                    symbol=symbol, trade_date=trade_date, **values,
                ))


def get_price_history(symbol: str, since: Optional[date] = None) -> list:
    """Returns this symbol's accumulated history as a list of dicts,
    ascending by trade_date. `since` filters to trade_date >= since (e.g.
    for a 130-day screening lookback window) -- omit for the full
    accumulated history. Deliberately returns plain dicts, not a
    pandas.Series -- db.py has no pandas dependency anywhere else and this
    keeps that boundary intact; converting to the Series shape
    screening_service expects is the caller's job (see
    price_history_service.py)."""
    with engine.connect() as conn:
        query = select(price_history).where(price_history.c.symbol == symbol)
        if since is not None:
            query = query.where(price_history.c.trade_date >= since)
        rows = conn.execute(query.order_by(price_history.c.trade_date)).mappings().all()
        return [dict(r) for r in rows]


def get_price_history_symbols() -> list:
    """Distinct symbols with at least one accumulated price row -- lets a
    caller check coverage (e.g. "which universe symbols have we started
    accumulating history for yet?") without pulling every row."""
    with engine.connect() as conn:
        rows = conn.execute(select(price_history.c.symbol).distinct()).all()
        return [r[0] for r in rows]
