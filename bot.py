"""
bot.py

Telegram bot layer for the PMC Strategy API. Lives inside the same
FastAPI service as main.py (one Render deployment, webhook model per
project decision -- not polling), wired in via POST /telegram/webhook in
main.py.

Talks to db.py DIRECTLY, not through main.py's HTTP layer -- these
commands (portfolio create/switch, status, holdings, trades, signals) are
pure reads/writes against the persistence layer and don't need main.py's
request/response shaping. This keeps bot.py at the same architectural
layer as main.py (both sit on db.py), rather than the bot looping back
through its own API over HTTP for no benefit.

LIVE PRICE DATA: /value uses ngx_pulse_service.get_current_prices_and_sectors()
directly (same module main.py's /rebalance-check stateful path uses) --
one NGX Pulse request regardless of how many symbols are held. This bot
still talks to db.py directly for everything else; ngx_pulse_service is
the one exception where it reaches for live current prices, since that's
a read against an external market-data source, not the persistence layer.

EXPLICITLY OUT OF SCOPE THIS PHASE (not built, not faked): /screen,
/portfolio/construct, /rebalance-check -- these need the strategy's full
compute layer (screening_service, portfolio_service, price SERIES not
just current prices, fundamentals) which live in main.py, not here.
Faking these with placeholder data would be worse than not having them.

TRADE LOGGING (/log_trade): per project workflow, all trades happen at
the end-of-month rebalance review, driven by that month's screening
output -- so every bot-logged trade defaults reason="monthly_rebalance"
unless explicitly overridden (e.g. "initial_construction" for a first
deployment). Fee can be typed as the real broker-confirmed amount, or
"auto" to estimate via portfolio_service's confirmed BUY_FEE_RATE/
SELL_FEE_RATE. For a brand-new symbol (first buy, not previously held),
sector is fetched live from NGX Pulse rather than asked of the user --
db.log_trade only sets sector on the INSERT path for a new holding, so
skipping this would silently leave sector=None and break future
sector-cap checks for that name.

Sandbox note: this environment's network egress allowlist has no route to
api.telegram.org (the same constraint documented elsewhere in this
project for stockanalysis.com/supabase.co) -- so outbound sendMessage
calls cannot be live-tested here. test_bot.py mocks send_message and
exercises webhook parsing/dispatch/db-wiring end-to-end via TestClient,
the same pattern test_main.py used for mocked fundamentals_service calls.
Live delivery to a real Telegram chat must be verified by the user after
deployment (see the setWebhook instructions given alongside this file).
"""

import os
import shlex
from datetime import datetime
from typing import Optional

import requests

import db
import ngx_pulse_service as nps
import portfolio_service as psvc

TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_WEBHOOK_SECRET = os.environ.get("TELEGRAM_WEBHOOK_SECRET", "")
TELEGRAM_API_BASE = "https://api.telegram.org"


def send_message(chat_id: str, text: str, parse_mode: str = "Markdown"):
    """Sends a message back to a Telegram chat via the Bot API. Real
    network call to api.telegram.org -- unreachable from this sandbox,
    mocked in test_bot.py. Raises RuntimeError if no token is configured
    rather than silently no-op-ing, so a misconfigured deployment fails
    loudly instead of looking like a bug in the command logic."""
    if not TELEGRAM_BOT_TOKEN:
        raise RuntimeError("TELEGRAM_BOT_TOKEN is not set")
    url = f"{TELEGRAM_API_BASE}/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    return requests.post(
        url, json={"chat_id": chat_id, "text": text, "parse_mode": parse_mode}, timeout=10
    )


def verify_webhook_secret(header_value: Optional[str]) -> bool:
    """Telegram's setWebhook supports a secret_token that it echoes back
    on every update as the X-Telegram-Bot-Api-Secret-Token header -- this
    is the only thing stopping anyone who finds the webhook URL from
    posting fabricated updates (there's no other auth on this endpoint).
    If TELEGRAM_WEBHOOK_SECRET isn't configured, this is a no-op (True)
    so local/test use isn't blocked -- but it should be set in
    production. See the deployment instructions for the setWebhook call
    that registers it."""
    if not TELEGRAM_WEBHOOK_SECRET:
        return True
    return header_value == TELEGRAM_WEBHOOK_SECRET


# ---------------------------------------------------------------------
# Formatting helpers -- plain functions, easy to unit test independent
# of any Telegram/network concerns.
# ---------------------------------------------------------------------

def _fmt_money(x) -> str:
    try:
        return f"{float(x):,.2f}"
    except (TypeError, ValueError):
        return "N/A"


def _fmt_portfolio(p: dict) -> str:
    return (
        f"*{p['name']}*\n"
        f"Status: {p['status']}\n"
        f"Broker: {p['broker']}\n"
        f"Currency: {p['currency']}\n"
        f"Initial capital: {_fmt_money(p['initial_capital'])}\n"
        f"Inception: {p['inception_date']}\n"
        f"ID: `{p['id']}`"
    )


def _fmt_holdings(holdings: list) -> str:
    if not holdings:
        return "No holdings."
    return "\n".join(
        f"{h['symbol']}: {float(h['shares']):,.2f} sh @ {_fmt_money(h['avg_cost'])} ({h['sector'] or 'N/A'})"
        for h in holdings
    )


def _fmt_trades(trades: list, limit: int = 10) -> str:
    if not trades:
        return "No trades."
    recent = trades[-limit:]
    return "\n".join(
        f"[{t['id'][:8]}] {t['executed_at']:%Y-%m-%d} {t['side'].upper()} {float(t['shares']):,.2f} {t['symbol']} "
        f"@ {_fmt_money(t['price'])} (fee {_fmt_money(t['fee'])})"
        for t in recent
    )


def _fmt_value_lines(holdings: list, current_prices: dict) -> tuple:
    """Returns (lines, total_market_value, total_cost_basis, missing_symbols).
    Pure function, no network/db -- easy to unit test independent of the
    live NGX Pulse call."""
    lines = []
    total_market_value = 0.0
    total_cost_basis = 0.0
    missing = []
    for h in holdings:
        sym = h["symbol"]
        shares = float(h["shares"])
        price = current_prices.get(sym)
        cost = float(h["avg_cost"]) if h["avg_cost"] is not None else None
        if price is None:
            lines.append(f"{sym}: {shares:,.2f} sh -- live price unavailable")
            missing.append(sym)
            continue
        market_value = shares * price
        total_market_value += market_value
        gain_str = ""
        if cost is not None and cost > 0:
            total_cost_basis += shares * cost
            pct = (price - cost) / cost * 100
            gain_str = f" ({pct:+.1f}%)"
        lines.append(f"{sym}: {shares:,.2f} sh @ {_fmt_money(price)} = {_fmt_money(market_value)}{gain_str}")
    return lines, total_market_value, total_cost_basis, missing


def _fmt_signals(signals: list) -> str:
    if not signals:
        return "No unacknowledged signals."
    return "\n".join(
        f"[{s['id'][:8]}] {s['signal_type']}" + (f" {s['symbol']}" if s['symbol'] else "")
        + f" -- {s['detail']}"
        for s in signals
    )


# ---------------------------------------------------------------------
# Command handlers: (chat_id: str, args: list[str]) -> reply text
# ---------------------------------------------------------------------

def cmd_start(chat_id: str, args: list) -> str:
    db.register_chat(chat_id)
    return (
        "Welcome to the CFA PMC Strategy bot.\n"
        "Use /new_portfolio to create your first portfolio, or /help to see all commands."
    )


def cmd_help(chat_id: str, args: list) -> str:
    return (
        "*Available commands*\n"
        "/new_portfolio \"name\" broker currency initial_capital YYYY-MM-DD -- "
        "create a portfolio and make it active for this chat\n"
        "/switch_portfolio \"name\" broker currency initial_capital YYYY-MM-DD -- "
        "close the current active portfolio and start a new one\n"
        "/close_portfolio [YYYY-MM-DD] -- close the active portfolio without starting "
        "a new one (defaults closed date to today)\n"
        "/status -- show the active portfolio's details\n"
        "/holdings -- show the active portfolio's current holdings\n"
        "/value -- live market value of current holdings, priced via NGX Pulse "
        "(equity holdings only -- cash isn't tracked here)\n"
        "/log_trade buy|sell symbol shares price fee|auto [reason] -- record an "
        "executed trade (defaults reason to \"monthly_rebalance\"); updates holdings "
        "automatically\n"
        "/undo_trade trade_id -- undo a mistakenly logged trade (full id or its first "
        "8 chars, shown in /trades) and correctly recompute the affected holding\n"
        "/trades -- show the active portfolio's recent trade history (with ids for /undo_trade)\n"
        "/signals -- show unacknowledged drift/breach signals\n"
        "/ack signal_id -- acknowledge a signal (full id or its first 8 chars)\n"
        "/help -- show this message\n\n"
        "Not yet available here: screening, portfolio construction, and drift "
        "checks -- these need the full strategy compute layer (main.py's job, "
        "not this bot's)."
    )


def _require_active_portfolio(chat_id: str) -> Optional[dict]:
    return db.get_active_portfolio(chat_id)


def _parse_new_portfolio_args(args: list):
    """Returns (name, broker, currency, capital, inception_date) or
    raises ValueError with a usage-appropriate message."""
    if len(args) != 5:
        raise ValueError(
            'Usage: /new_portfolio "name" broker currency initial_capital YYYY-MM-DD\n'
            'Example: /new_portfolio "CFA PMC 2026" Ticker NGN 10000000 2026-01-01'
        )
    name, broker, currency, capital_str, inception_str = args
    try:
        capital = float(capital_str)
    except ValueError:
        raise ValueError(f"initial_capital must be a number, got {capital_str!r}")
    try:
        inception = datetime.strptime(inception_str, "%Y-%m-%d").date()
    except ValueError:
        raise ValueError(f"inception date must be YYYY-MM-DD, got {inception_str!r}")
    return name, broker, currency, capital, inception


def cmd_new_portfolio(chat_id: str, args: list) -> str:
    try:
        name, broker, currency, capital, inception = _parse_new_portfolio_args(args)
    except ValueError as e:
        return str(e)
    pid = db.start_new_portfolio_for_chat(chat_id, name, broker, currency, capital, inception)
    return f"New active portfolio created:\n\n{_fmt_portfolio(db.get_portfolio(pid))}"


def cmd_switch_portfolio(chat_id: str, args: list) -> str:
    # Same mechanics as /new_portfolio -- start_new_portfolio_for_chat
    # always closes whatever's currently active first (see db.py docstring
    # on the reusability/portfolio-switching design). Two command names
    # for the same action because "new" and "switch" read differently
    # depending on whether the chat already has an active portfolio.
    return cmd_new_portfolio(chat_id, args)


def cmd_status(chat_id: str, args: list) -> str:
    p = _require_active_portfolio(chat_id)
    if not p:
        return "No active portfolio for this chat. Use /new_portfolio to create one."
    return _fmt_portfolio(p)


def cmd_holdings(chat_id: str, args: list) -> str:
    p = _require_active_portfolio(chat_id)
    if not p:
        return "No active portfolio for this chat. Use /new_portfolio to create one."
    return f"*Holdings -- {p['name']}*\n\n{_fmt_holdings(db.get_holdings(p['id']))}"


def cmd_trades(chat_id: str, args: list) -> str:
    p = _require_active_portfolio(chat_id)
    if not p:
        return "No active portfolio for this chat. Use /new_portfolio to create one."
    return f"*Recent trades -- {p['name']}* (last 10)\n\n{_fmt_trades(db.get_trades(p['id']))}"


def cmd_signals(chat_id: str, args: list) -> str:
    p = _require_active_portfolio(chat_id)
    if not p:
        return "No active portfolio for this chat. Use /new_portfolio to create one."
    return f"*Unacknowledged signals -- {p['name']}*\n\n{_fmt_signals(db.get_unacknowledged_signals(p['id']))}"


def cmd_value(chat_id: str, args: list) -> str:
    """Live market value of the active portfolio's EQUITY HOLDINGS ONLY,
    priced via NGX Pulse. Does not include cash: cash is not a persisted
    field anywhere in db.py's schema (it's only ever supplied by the
    caller at /rebalance-check time), so there is no stored cash figure
    for this command to add in -- stated explicitly in the reply so it's
    never mistaken for total account value."""
    p = _require_active_portfolio(chat_id)
    if not p:
        return "No active portfolio for this chat. Use /new_portfolio to create one."

    holdings_list = db.get_holdings(p["id"])
    if not holdings_list:
        return f"*Live value -- {p['name']}*\n\nNo holdings. (Cash is not tracked by this bot.)"

    held_symbols = [h["symbol"] for h in holdings_list]
    try:
        current_prices, _sectors = nps.get_current_prices_and_sectors(symbols=held_symbols)
    except requests.RequestException as e:
        return f"Could not fetch live prices from NGX Pulse: {e}"

    lines, total_mv, total_cost, missing = _fmt_value_lines(holdings_list, current_prices)

    body = f"*Live holdings value -- {p['name']}*\n\n" + "\n".join(lines)
    body += f"\n\nTotal market value: {_fmt_money(total_mv)}"
    if total_cost > 0:
        gain = total_mv - total_cost
        gain_pct = gain / total_cost * 100
        body += f"\nTotal cost basis: {_fmt_money(total_cost)}"
        body += f"\nUnrealized P/L: {_fmt_money(gain)} ({gain_pct:+.1f}%)"
    if missing:
        body += f"\n\nNo live price available for: {', '.join(missing)} (excluded from total)."
    body += "\n\n_Equity holdings only -- cash is not tracked by this bot._"
    return body


DEFAULT_TRADE_REASON = "monthly_rebalance"


def _parse_log_trade_args(args: list):
    """Returns (side, symbol, shares, price, fee_str, reason) or raises
    ValueError with a usage-appropriate message.
    Usage: /log_trade side symbol shares price fee|auto [reason]"""
    if len(args) not in (5, 6):
        raise ValueError(
            "Usage: /log_trade buy|sell symbol shares price fee|auto [reason]\n"
            "Example: /log_trade buy GTCO 100 45.00 68.85\n"
            "Example (auto-estimated fee): /log_trade sell MTNN 50 210.00 auto\n"
            f'Reason defaults to "{DEFAULT_TRADE_REASON}" if omitted.'
        )
    side_raw, symbol, shares_str, price_str, fee_str = args[:5]
    side = side_raw.lower()
    if side not in ("buy", "sell"):
        raise ValueError(f"side must be 'buy' or 'sell', got {side_raw!r}")
    symbol = symbol.upper()
    try:
        shares = float(shares_str)
    except ValueError:
        raise ValueError(f"shares must be a number, got {shares_str!r}")
    try:
        price = float(price_str)
    except ValueError:
        raise ValueError(f"price must be a number, got {price_str!r}")
    reason = args[5] if len(args) == 6 else DEFAULT_TRADE_REASON
    return side, symbol, shares, price, fee_str, reason


def cmd_log_trade(chat_id: str, args: list) -> str:
    p = _require_active_portfolio(chat_id)
    if not p:
        return "No active portfolio for this chat. Use /new_portfolio to create one."

    try:
        side, symbol, shares, price, fee_str, reason = _parse_log_trade_args(args)
    except ValueError as e:
        return str(e)

    trade_amount = shares * price
    if fee_str.lower() == "auto":
        fee = psvc.estimate_trade_fees(
            sell_amount=(trade_amount if side == "sell" else 0.0),
            buy_amount=(trade_amount if side == "buy" else 0.0),
        )
    else:
        try:
            fee = float(fee_str)
        except ValueError:
            return f"fee must be a number or 'auto', got {fee_str!r}"

    existing = next((h for h in db.get_holdings(p["id"]) if h["symbol"] == symbol), None)
    sector = existing["sector"] if existing else None
    sector_note = ""
    if side == "buy" and existing is None:
        # New position -- db.log_trade only sets sector on the INSERT path,
        # so a missing sector here would silently break future sector-cap
        # checks for this symbol. Fetch it live rather than ask the user.
        try:
            _prices, sectors = nps.get_current_prices_and_sectors(symbols=[symbol])
            sector = sectors.get(symbol)
            if sector is None:
                sector_note = f"\n\n(Could not find {symbol} in the live NGX Pulse snapshot -- sector left unset.)"
        except requests.RequestException as e:
            sector_note = f"\n\n(Could not fetch sector from NGX Pulse: {e} -- sector left unset.)"

    try:
        db.log_trade(p["id"], symbol, side, shares, price, fee, reason=reason, sector=sector)
    except ValueError as e:
        return f"Trade rejected: {e}"

    holding_after = next((h for h in db.get_holdings(p["id"]) if h["symbol"] == symbol), None)
    position_line = (
        f"New position: {float(holding_after['shares']):,.2f} sh @ avg cost {_fmt_money(holding_after['avg_cost'])}"
        if holding_after else "Position fully closed."
    )
    return (
        f"Logged {side.upper()} {shares:,.2f} {symbol} @ {_fmt_money(price)} "
        f"(fee {_fmt_money(fee)}, reason: {reason})\n{position_line}{sector_note}"
    )


def cmd_close_portfolio(chat_id: str, args: list) -> str:
    """Closes the active portfolio WITHOUT creating a replacement (unlike
    /switch_portfolio, which does both). db.close_portfolio alone does
    NOT clear the chat's active-portfolio pointer (get_active_portfolio
    doesn't filter on status) -- explicitly clearing it here via
    set_active_portfolio(chat_id, None) so the chat genuinely has no
    active portfolio afterward, matching what a competitor would expect
    from "close" (no further trades logged against it via this bot)."""
    p = _require_active_portfolio(chat_id)
    if not p:
        return "No active portfolio for this chat."
    closed_date = None
    if args:
        try:
            closed_date = datetime.strptime(args[0], "%Y-%m-%d").date()
        except ValueError:
            return f"closed date must be YYYY-MM-DD, got {args[0]!r}"
    db.close_portfolio(p["id"], closed_date=closed_date)
    db.set_active_portfolio(chat_id, None)
    return f"Closed portfolio *{p['name']}*. Use /new_portfolio to start a new one."


def cmd_undo_trade(chat_id: str, args: list) -> str:
    """Undoes a mistaken /log_trade by full replay (see db.delete_trade's
    docstring for why replay, not delta-reversal). Matches by full trade
    id or its first-8-char prefix shown in /trades, same UX as /ack."""
    if not args:
        return "Usage: /undo_trade trade_id (full id or its first 8 chars, shown in /trades)"
    p = _require_active_portfolio(chat_id)
    if not p:
        return "No active portfolio for this chat."
    target = args[0]
    trades_list = db.get_trades(p["id"])
    match = next((t for t in trades_list if t["id"] == target or t["id"].startswith(target)), None)
    if not match:
        return f"No trade matching '{target}' found in this portfolio's trade history."

    try:
        resulting_holding = db.delete_trade(p["id"], match["id"])
    except ValueError as e:
        return f"Could not undo trade: {e}"

    summary = (
        f"Undone: {match['side'].upper()} {float(match['shares']):,.2f} {match['symbol']} "
        f"@ {_fmt_money(match['price'])} [{match['id'][:8]}]"
    )
    if resulting_holding:
        summary += (
            f"\n{match['symbol']} position now: {float(resulting_holding['shares']):,.2f} sh "
            f"@ avg cost {_fmt_money(resulting_holding['avg_cost'])}"
        )
    else:
        summary += f"\n{match['symbol']} position is now fully closed."
    return summary


def cmd_ack(chat_id: str, args: list) -> str:
    if not args:
        return "Usage: /ack signal_id"
    p = _require_active_portfolio(chat_id)
    if not p:
        return "No active portfolio for this chat."
    target = args[0]
    signals = db.get_unacknowledged_signals(p["id"])
    match = next((s for s in signals if s["id"] == target or s["id"].startswith(target)), None)
    if not match:
        return f"No unacknowledged signal matching '{target}' found."
    db.acknowledge_signal(match["id"])
    return f"Acknowledged signal [{match['id'][:8]}] {match['signal_type']}."


COMMANDS = {
    "/start": cmd_start,
    "/help": cmd_help,
    "/new_portfolio": cmd_new_portfolio,
    "/switch_portfolio": cmd_switch_portfolio,
    "/close_portfolio": cmd_close_portfolio,
    "/status": cmd_status,
    "/holdings": cmd_holdings,
    "/value": cmd_value,
    "/log_trade": cmd_log_trade,
    "/undo_trade": cmd_undo_trade,
    "/trades": cmd_trades,
    "/signals": cmd_signals,
    "/ack": cmd_ack,
}


def parse_command(text: str):
    """Splits a Telegram message into (command, args). Uses shlex so a
    quoted portfolio name with spaces ("CFA PMC 2026") comes through as
    one arg, not three. Telegram appends '@BotUsername' to commands in
    group chats (e.g. '/status@MyBot') -- stripped here. Falls back to a
    naive whitespace split on unbalanced-quote input rather than raising,
    since a webhook handler should never 500 on a malformed message."""
    stripped = text.strip()
    try:
        parts = shlex.split(stripped)
    except ValueError:
        parts = stripped.split()
    if not parts:
        return None, []
    command = parts[0].split("@")[0].lower()
    return command, parts[1:]


def handle_update(update: dict) -> Optional[dict]:
    """Processes one Telegram Update dict end to end: parses the command,
    dispatches to a handler, sends the reply, and returns the outbound
    payload (chat_id + text) so callers/tests can assert on it without
    needing a real network call. Returns None for update types this
    phase doesn't handle (non-message updates, non-command text)."""
    message = update.get("message") or update.get("edited_message")
    if not message:
        return None

    chat_id = str(message["chat"]["id"])
    text = message.get("text", "")
    if not text.startswith("/"):
        return None

    command, args = parse_command(text)
    handler = COMMANDS.get(command)
    if handler is None:
        reply = f"Unknown command: {command}. Send /help for a list of commands."
    else:
        try:
            reply = handler(chat_id, args)
        except Exception as e:  # noqa: BLE001 -- a webhook must never 500 back to Telegram
            reply = f"Something went wrong processing that command: {e}"

    try:
        resp = send_message(chat_id, reply)
        if resp is not None and resp.status_code >= 300:
            # requests.post doesn't raise on a non-2xx response -- without this
            # check, a real delivery failure (bad token, chat blocked the bot,
            # Telegram API outage) would be completely invisible: the webhook
            # still returns 200 to Telegram (correct -- that's about receipt of
            # the update, not delivery of the reply), but nothing would ever
            # show up in logs. Print goes to Render's log stream.
            print(f"telegram sendMessage failed ({resp.status_code}) for chat {chat_id}: {resp.text[:300]}")
    except Exception as e:  # noqa: BLE001 -- network/config errors must not 500 the webhook either
        print(f"telegram sendMessage raised for chat {chat_id}: {e}")

    return {"chat_id": chat_id, "text": reply}
