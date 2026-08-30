"""
test_bot.py

Verification suite for bot.py and main.py's /telegram/webhook endpoint.
Mocks bot.send_message (this sandbox has no network route to
api.telegram.org) and exercises everything else -- webhook JSON parsing,
shlex command parsing, command dispatch, and db.py reads/writes -- for
real, through TestClient against a fresh SQLite DATABASE_URL override.

Live delivery to a real Telegram chat, and the actual setWebhook
registration, cannot be verified in this sandbox and must be checked by
the user after deployment (see the deployment notes shared alongside
this file).
"""

import os
import sys

DB_PATH = "/home/claude/pmc_api/test_bot.db"
if os.path.exists(DB_PATH):
    os.remove(DB_PATH)
os.environ["DATABASE_URL"] = f"sqlite:///{DB_PATH}"
os.environ["TELEGRAM_BOT_TOKEN"] = "test-token-not-real"

sys.path.insert(0, "/home/claude/pmc_api")

from unittest.mock import patch

import requests
from fastapi.testclient import TestClient

import bot
import main
import db
import portfolio_service as psvc

client = TestClient(main.app)
db.init_db()

PASS, FAIL = [], []


def check(name, cond, detail=""):
    if cond:
        PASS.append(name)
        print(f"PASS: {name}")
    else:
        FAIL.append((name, detail))
        print(f"FAIL: {name} -- {detail}")


def _telegram_message_update(chat_id, text, update_id=1):
    """Builds a minimal, realistic Telegram Update JSON payload."""
    return {
        "update_id": update_id,
        "message": {
            "message_id": update_id,
            "chat": {"id": chat_id, "type": "private"},
            "text": text,
            "date": 1735689600,
        },
    }


# =======================================================================
print("\n=== SECTION 1: parse_command (pure function, no network/db) ===")
# =======================================================================

cmd, args = bot.parse_command("/status")
check("parse_command: simple command, no args", (cmd, args) == ("/status", []), (cmd, args))

cmd, args = bot.parse_command('/new_portfolio "CFA PMC 2026" Ticker NGN 10000000 2026-01-01')
check("parse_command: quoted name with spaces stays one arg",
      args == ["CFA PMC 2026", "Ticker", "NGN", "10000000", "2026-01-01"], args)

cmd, args = bot.parse_command("/status@MyPortfolioBot")
check("parse_command: strips @BotUsername suffix (group chat style)", cmd == "/status", cmd)

cmd, args = bot.parse_command('/new_portfolio "unbalanced quote')
check("parse_command: unbalanced quotes falls back to naive split, doesn't raise",
      cmd == "/new_portfolio", (cmd, args))

cmd, args = bot.parse_command("   ")
check("parse_command: blank text returns (None, [])", (cmd, args) == (None, []), (cmd, args))


# =======================================================================
print("\n=== SECTION 2: webhook secret verification ===")
# =======================================================================

with patch.object(bot, "TELEGRAM_WEBHOOK_SECRET", ""):
    check("verify_webhook_secret: no secret configured -> always True",
          bot.verify_webhook_secret(None) and bot.verify_webhook_secret("anything"))

with patch.object(bot, "TELEGRAM_WEBHOOK_SECRET", "s3cr3t"):
    check("verify_webhook_secret: correct token -> True", bot.verify_webhook_secret("s3cr3t"))
    check("verify_webhook_secret: wrong token -> False", not bot.verify_webhook_secret("wrong"))
    check("verify_webhook_secret: missing header -> False", not bot.verify_webhook_secret(None))


# =======================================================================
print("\n=== SECTION 3: /telegram/webhook endpoint -- auth + basic dispatch ===")
# =======================================================================

sent_messages = []


def _fake_send_message(chat_id, text, parse_mode="Markdown"):
    sent_messages.append({"chat_id": chat_id, "text": text})
    return None


with patch.object(bot, "TELEGRAM_WEBHOOK_SECRET", "s3cr3t"):
    r = client.post("/telegram/webhook", json=_telegram_message_update("123", "/help"))
    check("webhook without secret header returns 401 when secret is configured",
          r.status_code == 401, r.text)

with patch.object(bot, "TELEGRAM_WEBHOOK_SECRET", ""), patch.object(bot, "send_message", side_effect=_fake_send_message):
    sent_messages.clear()
    r = client.post("/telegram/webhook", json=_telegram_message_update("chat_help", "/help"))
    check("webhook /help returns 200", r.status_code == 200, r.text)
    check("webhook /help sent exactly one message", len(sent_messages) == 1, sent_messages)
    check("webhook /help mentions /new_portfolio", "/new\\_portfolio" in sent_messages[-1]["text"], sent_messages)

    sent_messages.clear()
    r = client.post("/telegram/webhook", json=_telegram_message_update("chat_help", "just chatting, not a command"))
    check("webhook with non-command text returns 200", r.status_code == 200, r.text)
    check("webhook with non-command text sends nothing", sent_messages == [], sent_messages)

    sent_messages.clear()
    r = client.post("/telegram/webhook", json={"update_id": 99, "callback_query": {"id": "x"}})
    check("webhook with non-message update returns 200 (doesn't crash)", r.status_code == 200, r.text)
    check("webhook with non-message update sends nothing", sent_messages == [], sent_messages)

    sent_messages.clear()
    r = client.post("/telegram/webhook", json=_telegram_message_update("chat_help", "/nonsense"))
    check("webhook with unknown command still returns 200 (not 500)", r.status_code == 200, r.text)
    check("webhook with unknown command sends a friendly 'unknown command' reply",
          "Unknown command" in sent_messages[-1]["text"], sent_messages)


# =======================================================================
print("\n=== SECTION 4: full command flow -- new_portfolio, status, holdings, trades, signals ===")
# =======================================================================

with patch.object(bot, "TELEGRAM_WEBHOOK_SECRET", ""), patch.object(bot, "send_message", side_effect=_fake_send_message):
    chat = "chat_flow_1"

    sent_messages.clear()
    client.post("/telegram/webhook", json=_telegram_message_update(chat, "/status"))
    check("/status with no active portfolio prompts to create one",
          "No active portfolio" in sent_messages[-1]["text"], sent_messages)

    sent_messages.clear()
    client.post("/telegram/webhook", json=_telegram_message_update(
        chat, '/new_portfolio "CFA PMC 2026" "Ticker (Cowrywise)" NGN 10000000 2026-01-01'
    ))
    check("/new_portfolio confirms creation", "New active portfolio created" in sent_messages[-1]["text"], sent_messages)
    check("/new_portfolio reply includes the portfolio name", "CFA PMC 2026" in sent_messages[-1]["text"], sent_messages)

    active = db.get_active_portfolio(chat)
    check("/new_portfolio actually set the chat's active portfolio in the DB",
          active is not None and active["name"] == "CFA PMC 2026", active)
    pid = active["id"]

    sent_messages.clear()
    client.post("/telegram/webhook", json=_telegram_message_update(chat, "/status"))
    check("/status after creation shows the portfolio", "CFA PMC 2026" in sent_messages[-1]["text"], sent_messages)
    check("/status shows status: active", "active" in sent_messages[-1]["text"], sent_messages)

    db.upsert_holdings(pid, [
        {"symbol": "GTCO", "shares": 100, "avg_cost": 45.0, "sector": "Banking"},
    ])
    sent_messages.clear()
    client.post("/telegram/webhook", json=_telegram_message_update(chat, "/holdings"))
    check("/holdings shows the GTCO position", "GTCO" in sent_messages[-1]["text"], sent_messages)
    check("/holdings shows correct share count", "100.00" in sent_messages[-1]["text"], sent_messages)

    db.log_trade(pid, "GTCO", "buy", 100, 45.0, 68.85, reason="initial_construction")
    sent_messages.clear()
    client.post("/telegram/webhook", json=_telegram_message_update(chat, "/trades"))
    check("/trades shows the logged trade", "GTCO" in sent_messages[-1]["text"] and "BUY" in sent_messages[-1]["text"],
          sent_messages)

    sid = db.log_signal_if_new(pid, "drift", "GTCO", {"drift": 0.08})
    sent_messages.clear()
    client.post("/telegram/webhook", json=_telegram_message_update(chat, "/signals"))
    check("/signals shows the unacknowledged signal", "drift" in sent_messages[-1]["text"], sent_messages)
    check("/signals shows the truncated signal id", sid[:8] in sent_messages[-1]["text"], sent_messages)

    sent_messages.clear()
    client.post("/telegram/webhook", json=_telegram_message_update(chat, f"/ack {sid[:8]}"))
    check("/ack with truncated id confirms acknowledgement", "Acknowledged" in sent_messages[-1]["text"], sent_messages)

    unacked = db.get_unacknowledged_signals(pid)
    check("/ack actually acknowledged the signal in the DB", all(s["id"] != sid for s in unacked), unacked)

    sent_messages.clear()
    client.post("/telegram/webhook", json=_telegram_message_update(chat, "/signals"))
    check("/signals after ack shows no unacknowledged signals",
          "No unacknowledged" in sent_messages[-1]["text"], sent_messages)


# =======================================================================
print("\n=== SECTION 5: /switch_portfolio -- old auto-closed, new one active ===")
# =======================================================================

with patch.object(bot, "TELEGRAM_WEBHOOK_SECRET", ""), patch.object(bot, "send_message", side_effect=_fake_send_message):
    chat = "chat_flow_1"
    old_active = db.get_active_portfolio(chat)

    sent_messages.clear()
    client.post("/telegram/webhook", json=_telegram_message_update(
        chat, '/switch_portfolio "Post-Competition Account" Ticker NGN 2000000 2026-09-01'
    ))
    check("/switch_portfolio confirms creation", "New active portfolio created" in sent_messages[-1]["text"], sent_messages)

    old_after = db.get_portfolio(old_active["id"])
    check("/switch_portfolio auto-closed the previous active portfolio",
          old_after["status"] == "closed", old_after)

    new_active = db.get_active_portfolio(chat)
    check("/switch_portfolio: chat now points at the new portfolio",
          new_active["id"] != old_active["id"] and new_active["name"] == "Post-Competition Account", new_active)

    old_holdings = db.get_holdings(old_active["id"])
    check("/switch_portfolio: old portfolio's holdings preserved, not deleted",
          len(old_holdings) == 1 and old_holdings[0]["symbol"] == "GTCO", old_holdings)


# =======================================================================
print("\n=== SECTION 6: /new_portfolio malformed input -- usage message, nothing created ===")
# =======================================================================

with patch.object(bot, "TELEGRAM_WEBHOOK_SECRET", ""), patch.object(bot, "send_message", side_effect=_fake_send_message):
    chat = "chat_malformed"

    sent_messages.clear()
    client.post("/telegram/webhook", json=_telegram_message_update(chat, "/new_portfolio too few args"))
    check("malformed /new_portfolio (wrong arg count) returns a usage message",
          "Usage:" in sent_messages[-1]["text"], sent_messages)
    check("malformed /new_portfolio did not create a portfolio",
          db.get_active_portfolio(chat) is None, db.get_active_portfolio(chat))

    sent_messages.clear()
    client.post("/telegram/webhook", json=_telegram_message_update(
        chat, '/new_portfolio "Test" Ticker NGN not_a_number 2026-01-01'
    ))
    check("malformed /new_portfolio (bad capital) returns a specific error",
          "initial\\_capital must be a number" in sent_messages[-1]["text"], sent_messages)

    sent_messages.clear()
    client.post("/telegram/webhook", json=_telegram_message_update(
        chat, '/new_portfolio "Test" Ticker NGN 1000000 not-a-date'
    ))
    check("malformed /new_portfolio (bad date) returns a specific error",
          "inception date must be" in sent_messages[-1]["text"], sent_messages)

    check("no portfolio was created across any of the malformed attempts",
          db.get_active_portfolio(chat) is None, db.get_active_portfolio(chat))


# =======================================================================
print("\n=== SECTION 7: /ack with no matching signal ===")
# =======================================================================

with patch.object(bot, "TELEGRAM_WEBHOOK_SECRET", ""), patch.object(bot, "send_message", side_effect=_fake_send_message):
    chat = "chat_flow_1"
    sent_messages.clear()
    client.post("/telegram/webhook", json=_telegram_message_update(chat, "/ack nonexistent12345"))
    check("/ack with no matching signal returns a clear 'not found' message",
          "No unacknowledged signal matching" in sent_messages[-1]["text"], sent_messages)


# =======================================================================
print("\n=== SECTION 8: send_message failure handling doesn't crash the webhook ===")
# =======================================================================

class _FakeResponse:
    def __init__(self, status_code, text=""):
        self.status_code = status_code
        self.text = text


with patch.object(bot, "TELEGRAM_WEBHOOK_SECRET", ""):
    with patch.object(bot, "send_message", return_value=_FakeResponse(403, "Forbidden: bot was blocked by the user")):
        r = client.post("/telegram/webhook", json=_telegram_message_update("chat_help", "/help"))
        check("webhook still returns 200 when send_message returns a non-2xx response", r.status_code == 200, r.text)

    with patch.object(bot, "send_message", side_effect=RuntimeError("TELEGRAM_BOT_TOKEN is not set")):
        r = client.post("/telegram/webhook", json=_telegram_message_update("chat_help", "/help"))
        check("webhook still returns 200 when send_message raises", r.status_code == 200, r.text)


# =======================================================================
print("\n=== SECTION 9: /value -- live market value via mocked ngx_pulse_service ===")
# =======================================================================

with patch.object(bot, "TELEGRAM_WEBHOOK_SECRET", ""), patch.object(bot, "send_message", side_effect=_fake_send_message):
    chat = "chat_value_1"

    sent_messages.clear()
    client.post("/telegram/webhook", json=_telegram_message_update(chat, "/value"))
    check("/value with no active portfolio prompts to create one",
          "No active portfolio" in sent_messages[-1]["text"], sent_messages)

    client.post("/telegram/webhook", json=_telegram_message_update(
        chat, '/new_portfolio "Value Test" Ticker NGN 1000000 2026-01-01'
    ))
    pid_v = db.get_active_portfolio(chat)["id"]

    sent_messages.clear()
    client.post("/telegram/webhook", json=_telegram_message_update(chat, "/value"))
    check("/value with zero holdings reports no holdings, not a crash",
          "No holdings" in sent_messages[-1]["text"], sent_messages)

    db.upsert_holdings(pid_v, [
        {"symbol": "GTCO", "shares": 100, "avg_cost": 45.0, "sector": "Banking"},
        {"symbol": "MTNN", "shares": 10, "avg_cost": 200.0, "sector": "Telecom"},
    ])

    sent_messages.clear()
    with patch.object(bot.nps, "get_current_prices_and_sectors",
                       return_value=({"GTCO": 50.0, "MTNN": 190.0}, {"GTCO": "Banking", "MTNN": "Telecom"})):
        client.post("/telegram/webhook", json=_telegram_message_update(chat, "/value"))
    reply = sent_messages[-1]["text"]
    check("/value shows both symbols priced live", "GTCO" in reply and "MTNN" in reply, reply)
    check("/value computes correct total market value (100*50 + 10*190 = 6,900.00)",
          "6,900.00" in reply, reply)
    check("/value shows unrealized P/L (GTCO up, MTNN down -- net cost basis 4,500+2,000=6,500)",
          "6,500.00" in reply, reply)
    check("/value states cash is not tracked, to avoid being mistaken for total account value",
          "cash is not tracked" in reply.lower(), reply)

    sent_messages.clear()
    with patch.object(bot.nps, "get_current_prices_and_sectors",
                       return_value=({"GTCO": 50.0}, {"GTCO": "Banking"})):  # MTNN missing from live snapshot
        client.post("/telegram/webhook", json=_telegram_message_update(chat, "/value"))
    reply_missing = sent_messages[-1]["text"]
    check("/value flags a symbol with no live price rather than silently valuing it at 0",
          "MTNN" in reply_missing and "unavailable" in reply_missing.lower(), reply_missing)

    sent_messages.clear()
    with patch.object(bot.nps, "get_current_prices_and_sectors", side_effect=requests.RequestException("NGX Pulse down")):
        client.post("/telegram/webhook", json=_telegram_message_update(chat, "/value"))
    reply_fail = sent_messages[-1]["text"]
    check("/value surfaces an NGX Pulse network failure as a clear message, not a 500/crash",
          "Could not fetch live prices" in reply_fail, reply_fail)


# =======================================================================
print("\n=== SECTION 10: /log_trade -- records trades, updates holdings ===")
# =======================================================================

with patch.object(bot, "TELEGRAM_WEBHOOK_SECRET", ""), patch.object(bot, "send_message", side_effect=_fake_send_message):
    chat = "chat_trade_1"

    sent_messages.clear()
    client.post("/telegram/webhook", json=_telegram_message_update(chat, "/log_trade buy GTCO 100 45.00 68.85"))
    check("/log_trade with no active portfolio prompts to create one",
          "No active portfolio" in sent_messages[-1]["text"], sent_messages)

    client.post("/telegram/webhook", json=_telegram_message_update(
        chat, '/new_portfolio "Trade Test" Ticker NGN 1000000 2026-01-01'
    ))
    pid_t = db.get_active_portfolio(chat)["id"]

    sent_messages.clear()
    client.post("/telegram/webhook", json=_telegram_message_update(chat, "/log_trade buy GTCO 100"))
    check("/log_trade with too few args returns a usage message",
          "Usage:" in sent_messages[-1]["text"], sent_messages)

    sent_messages.clear()
    client.post("/telegram/webhook", json=_telegram_message_update(chat, "/log_trade hold GTCO 100 45.00 68.85"))
    check("/log_trade with invalid side rejects clearly",
          "side must be" in sent_messages[-1]["text"], sent_messages)

    sent_messages.clear()
    client.post("/telegram/webhook", json=_telegram_message_update(chat, "/log_trade buy GTCO notanumber 45.00 68.85"))
    check("/log_trade with non-numeric shares rejects clearly",
          "shares must be a number" in sent_messages[-1]["text"], sent_messages)

    # New symbol, explicit fee -- sector fetched live from NGX Pulse
    sent_messages.clear()
    with patch.object(bot.nps, "get_current_prices_and_sectors",
                       return_value=({"GTCO": 45.5}, {"GTCO": "Banking"})) as mock_nps:
        client.post("/telegram/webhook", json=_telegram_message_update(chat, "/log_trade buy GTCO 100 45.00 68.85"))
        check("/log_trade (new symbol) calls NGX Pulse exactly once for sector lookup",
              mock_nps.call_count == 1, mock_nps.call_count)
    reply = sent_messages[-1]["text"]
    check("/log_trade buy confirms with default reason 'monthly_rebalance'",
          "monthly\\_rebalance" in reply, reply)
    check("/log_trade buy shows the resulting position", "100.00" in reply, reply)

    holding = next(h for h in db.get_holdings(pid_t) if h["symbol"] == "GTCO")
    check("/log_trade buy actually created the holding in the DB",
          float(holding["shares"]) == 100 and float(holding["avg_cost"]) == 45.0, holding)
    check("/log_trade buy fetched and stored the live sector", holding["sector"] == "Banking", holding)

    # Add to existing position -- sector already known, NGX Pulse should NOT be called again
    sent_messages.clear()
    with patch.object(bot.nps, "get_current_prices_and_sectors") as mock_nps2:
        client.post("/telegram/webhook", json=_telegram_message_update(chat, "/log_trade buy GTCO 100 55.00 84.15"))
        check("/log_trade adding to an EXISTING position does not call NGX Pulse (sector already known)",
              mock_nps2.call_count == 0, mock_nps2.call_count)
    holding2 = next(h for h in db.get_holdings(pid_t) if h["symbol"] == "GTCO")
    check("/log_trade buy (add to position): shares summed correctly (100+100=200)",
          float(holding2["shares"]) == 200, holding2)
    check("/log_trade buy (add to position): weighted avg cost correct ((100*45+100*55)/200=50)",
          abs(float(holding2["avg_cost"]) - 50.0) < 1e-9, holding2)

    # fee="auto" -- estimated via portfolio_service's confirmed rates
    sent_messages.clear()
    client.post("/telegram/webhook", json=_telegram_message_update(chat, "/log_trade sell GTCO 50 60.00 auto"))
    reply_auto = sent_messages[-1]["text"]
    expected_fee = 50 * 60.00 * psvc.SELL_FEE_RATE
    check("/log_trade with fee='auto' estimates using SELL_FEE_RATE",
          f"{expected_fee:,.2f}" in reply_auto, (reply_auto, expected_fee))

    # Custom reason override
    sent_messages.clear()
    client.post("/telegram/webhook", json=_telegram_message_update(
        chat, "/log_trade sell GTCO 50 62.00 100.28 hard_breach"
    ))
    check("/log_trade with explicit reason overrides the default",
          "hard\\_breach" in sent_messages[-1]["text"], sent_messages)

    # Oversell -- rejected, not a crash, holdings unchanged
    holding_before_oversell = next(h for h in db.get_holdings(pid_t) if h["symbol"] == "GTCO")
    sent_messages.clear()
    client.post("/telegram/webhook", json=_telegram_message_update(chat, "/log_trade sell GTCO 999 60.00 auto"))
    check("/log_trade oversell is rejected with a clear message, not a crash",
          "Trade rejected" in sent_messages[-1]["text"], sent_messages)
    holding_after_oversell = next(h for h in db.get_holdings(pid_t) if h["symbol"] == "GTCO")
    check("/log_trade oversell did not mutate holdings",
          float(holding_before_oversell["shares"]) == float(holding_after_oversell["shares"]),
          (holding_before_oversell, holding_after_oversell))

    # Selling a symbol never held
    sent_messages.clear()
    client.post("/telegram/webhook", json=_telegram_message_update(chat, "/log_trade sell NEVERBOUGHT 10 20.00 auto"))
    check("/log_trade selling a never-held symbol is rejected clearly",
          "Trade rejected" in sent_messages[-1]["text"], sent_messages)

    # NGX Pulse failure on a new-symbol buy -- trade still logs, sector left unset with a note
    sent_messages.clear()
    with patch.object(bot.nps, "get_current_prices_and_sectors", side_effect=requests.RequestException("down")):
        client.post("/telegram/webhook", json=_telegram_message_update(chat, "/log_trade buy MTNN 10 200.00 30.60"))
    reply_fail = sent_messages[-1]["text"]
    check("/log_trade still logs the trade even if the live sector lookup fails",
          "Logged BUY" in reply_fail, reply_fail)
    check("/log_trade notes the sector lookup failure rather than silently proceeding",
          "sector left unset" in reply_fail, reply_fail)
    mtnn_holding = next(h for h in db.get_holdings(pid_t) if h["symbol"] == "MTNN")
    check("/log_trade: holding created despite sector-lookup failure, sector is None",
          mtnn_holding is not None and mtnn_holding["sector"] is None, mtnn_holding)


# =======================================================================
print("\n=== SECTION 11: /close_portfolio -- closes without creating a replacement ===")
# =======================================================================

with patch.object(bot, "TELEGRAM_WEBHOOK_SECRET", ""), patch.object(bot, "send_message", side_effect=_fake_send_message):
    chat = "chat_close_1"

    sent_messages.clear()
    client.post("/telegram/webhook", json=_telegram_message_update(chat, "/close_portfolio"))
    check("/close_portfolio with no active portfolio returns a clear message",
          "No active portfolio" in sent_messages[-1]["text"], sent_messages)

    client.post("/telegram/webhook", json=_telegram_message_update(
        chat, '/new_portfolio "Close Test" Ticker NGN 1000000 2026-01-01'
    ))
    pid_c = db.get_active_portfolio(chat)["id"]

    sent_messages.clear()
    client.post("/telegram/webhook", json=_telegram_message_update(chat, "/close_portfolio 2026-08-31"))
    check("/close_portfolio confirms the close", "Closed portfolio" in sent_messages[-1]["text"], sent_messages)

    closed = db.get_portfolio(pid_c)
    check("/close_portfolio actually set status=closed in the DB", closed["status"] == "closed", closed)
    check("/close_portfolio respected the explicit closed date",
          str(closed["closed_date"]) == "2026-08-31", closed)

    check("/close_portfolio cleared the chat's active-portfolio pointer (no replacement created)",
          db.get_active_portfolio(chat) is None, db.get_active_portfolio(chat))

    sent_messages.clear()
    client.post("/telegram/webhook", json=_telegram_message_update(chat, "/status"))
    check("/status after /close_portfolio correctly shows no active portfolio",
          "No active portfolio" in sent_messages[-1]["text"], sent_messages)

    sent_messages.clear()
    client.post("/telegram/webhook", json=_telegram_message_update(chat, "/close_portfolio not-a-date"))
    # (no active portfolio at this point, so this exercises the no-active-portfolio path, not date parsing --
    # covered here for completeness of the malformed-date branch via a portfolio that still has one)


with patch.object(bot, "TELEGRAM_WEBHOOK_SECRET", ""), patch.object(bot, "send_message", side_effect=_fake_send_message):
    chat2 = "chat_close_2"
    client.post("/telegram/webhook", json=_telegram_message_update(
        chat2, '/new_portfolio "Close Test 2" Ticker NGN 1000000 2026-01-01'
    ))
    sent_messages.clear()
    client.post("/telegram/webhook", json=_telegram_message_update(chat2, "/close_portfolio not-a-date"))
    check("/close_portfolio with a malformed date is rejected clearly, portfolio NOT closed",
          "closed date must be" in sent_messages[-1]["text"], sent_messages)
    check("/close_portfolio malformed date: portfolio still active in DB",
          db.get_active_portfolio(chat2)["status"] == "active", db.get_active_portfolio(chat2))


# =======================================================================
print("\n=== SECTION 12: /undo_trade -- undo via webhook, correct replay ===")
# =======================================================================

with patch.object(bot, "TELEGRAM_WEBHOOK_SECRET", ""), patch.object(bot, "send_message", side_effect=_fake_send_message):
    chat = "chat_undo_1"

    sent_messages.clear()
    client.post("/telegram/webhook", json=_telegram_message_update(chat, "/undo_trade abc12345"))
    check("/undo_trade with no active portfolio returns a clear message",
          "No active portfolio" in sent_messages[-1]["text"], sent_messages)

    client.post("/telegram/webhook", json=_telegram_message_update(
        chat, '/new_portfolio "Undo Test" Ticker NGN 1000000 2026-01-01'
    ))
    pid_u = db.get_active_portfolio(chat)["id"]

    sent_messages.clear()
    client.post("/telegram/webhook", json=_telegram_message_update(chat, "/undo_trade"))
    check("/undo_trade with no trade_id arg returns usage", "Usage:" in sent_messages[-1]["text"], sent_messages)

    sent_messages.clear()
    client.post("/telegram/webhook", json=_telegram_message_update(chat, "/undo_trade nonexistent12345"))
    check("/undo_trade with no matching trade returns a clear 'not found' message",
          "No trade matching" in sent_messages[-1]["text"], sent_messages)

    # Log three buys for the same symbol via the actual webhook (mistakenly wrong middle price),
    # then undo the WRONG one via its short id, and confirm the replay is correct end-to-end.
    with patch.object(bot.nps, "get_current_prices_and_sectors",
                       return_value=({"ARADEL": 500.0}, {"ARADEL": "Oil & Gas"})):
        client.post("/telegram/webhook", json=_telegram_message_update(chat, "/log_trade buy ARADEL 10 100.00 15.30"))
    client.post("/telegram/webhook", json=_telegram_message_update(chat, "/log_trade buy ARADEL 10 999.00 152.99"))  # mistake
    client.post("/telegram/webhook", json=_telegram_message_update(chat, "/log_trade buy ARADEL 10 300.00 45.90"))

    trades_now = db.get_trades(pid_u)
    mistaken = next(t for t in trades_now if t["symbol"] == "ARADEL" and float(t["price"]) == 999.00)
    holding_with_mistake = next(h for h in db.get_holdings(pid_u) if h["symbol"] == "ARADEL")
    expected_wrong_avg = (10 * 100 + 10 * 999 + 10 * 300) / 30
    check("/undo_trade setup: avg_cost reflects the mistaken trade before undo",
          abs(float(holding_with_mistake["avg_cost"]) - expected_wrong_avg) < 1e-6, holding_with_mistake)

    sent_messages.clear()
    client.post("/telegram/webhook", json=_telegram_message_update(chat, f"/undo_trade {mistaken['id'][:8]}"))
    reply = sent_messages[-1]["text"]
    check("/undo_trade confirms which trade was undone", "999.00" in reply, reply)
    check("/undo_trade shows the resulting position in its reply", "ARADEL" in reply, reply)

    holding_after_undo = next(h for h in db.get_holdings(pid_u) if h["symbol"] == "ARADEL")
    expected_correct_avg = (10 * 100 + 10 * 300) / 20  # the two GOOD buys only
    check("/undo_trade end-to-end via webhook: shares correct after undo (30-10=20)",
          float(holding_after_undo["shares"]) == 20, holding_after_undo)
    check("/undo_trade end-to-end via webhook: avg_cost correctly replayed from remaining good trades",
          abs(float(holding_after_undo["avg_cost"]) - expected_correct_avg) < 1e-6,
          f"got={holding_after_undo['avg_cost']}, expected={expected_correct_avg}")
    check("/undo_trade end-to-end via webhook: mistaken trade no longer in trade history",
          all(t["id"] != mistaken["id"] for t in db.get_trades(pid_u)), db.get_trades(pid_u))

    # Undoing the same trade id twice -- second attempt correctly reports not found
    sent_messages.clear()
    client.post("/telegram/webhook", json=_telegram_message_update(chat, f"/undo_trade {mistaken['id'][:8]}"))
    check("/undo_trade on an already-undone trade id returns 'not found', not a crash",
          "No trade matching" in sent_messages[-1]["text"], sent_messages)

    # Undo a full-symbol position down to zero -- position fully closed
    sent_messages.clear()
    trades_final = db.get_trades(pid_u)
    good_trade_ids = [t["id"] for t in trades_final if t["symbol"] == "ARADEL"]
    for tid_ in good_trade_ids:
        client.post("/telegram/webhook", json=_telegram_message_update(chat, f"/undo_trade {tid_[:8]}"))
    last_reply = sent_messages[-1]["text"]
    check("/undo_trade down to zero shares reports the position as fully closed",
          "fully closed" in last_reply.lower(), last_reply)
    check("/undo_trade down to zero shares: no ARADEL holding remains in DB",
          all(h["symbol"] != "ARADEL" for h in db.get_holdings(pid_u)), db.get_holdings(pid_u))


# =======================================================================
print("\n=== SECTION 13: Markdown-safety -- the actual live-production bug ===")
# =======================================================================
# Telegram's legacy Markdown parse mode treats a lone, unpaired _ * ` [ as
# a broken formatting entity and returns 400, which previously caused
# TOTAL SILENCE from the bot (e.g. /start itself failed, since its own
# static text says "/new_portfolio"). This section directly tests the fix
# (_escape_md, applied throughout) and the safety-net fallback.

print("\n--- _escape_md unit behavior ---")
check("_escape_md escapes a lone underscore", bot._escape_md("new_portfolio") == "new\\_portfolio",
      bot._escape_md("new_portfolio"))
check("_escape_md escapes asterisk, backtick, and open-bracket",
      bot._escape_md("a*b`c[d") == "a\\*b\\`c\\[d", bot._escape_md("a*b`c[d"))
check("_escape_md escapes a literal backslash FIRST (so it doesn't double-escape its own output)",
      bot._escape_md("a\\_b") == "a\\\\\\_b", bot._escape_md("a\\_b"))
check("_escape_md handles None safely (returns empty string, not a crash)", bot._escape_md(None) == "", bot._escape_md(None))
check("_escape_md handles non-string input via str()", bot._escape_md(404) == "404", bot._escape_md(404))


def _no_unescaped_markdown_specials(text: str) -> bool:
    """Test helper: True if `text` forms VALID Telegram legacy-Markdown
    (won't 400). bot.py intentionally uses REAL unescaped * (bold) and `
    (code span) for formatting -- so the correct check isn't "zero special
    characters", it's "every unescaped special character is part of a
    correctly paired entity": unescaped _, *, and ` counts must each be
    even (open/close pairs), and unescaped [ must not appear at all since
    nothing in bot.py ever intends a real markdown link (an unmatched [
    is exactly the kind of thing _escape_md exists to prevent)."""
    counts = {"_": 0, "*": 0, "`": 0}
    i = 0
    while i < len(text):
        ch = text[i]
        escaped = i > 0 and text[i - 1] == "\\"
        if ch in counts and not escaped:
            counts[ch] += 1
        if ch == "[" and not escaped:
            return False
        i += 1
    return all(c % 2 == 0 for c in counts.values())


print("\n--- Every static reply that broke live production is now safe ---")

with patch.object(bot, "TELEGRAM_WEBHOOK_SECRET", ""), patch.object(bot, "send_message", side_effect=_fake_send_message):
    chat = "chat_mdsafety_1"

    sent_messages.clear()
    client.post("/telegram/webhook", json=_telegram_message_update(chat, "/start"))
    check("/start's reply (THE actual reported production failure) is Markdown-safe",
          _no_unescaped_markdown_specials(sent_messages[-1]["text"]), sent_messages[-1]["text"])

    sent_messages.clear()
    client.post("/telegram/webhook", json=_telegram_message_update(chat, "/help"))
    check("/help's reply is Markdown-safe", _no_unescaped_markdown_specials(sent_messages[-1]["text"]), sent_messages[-1]["text"])

    sent_messages.clear()
    client.post("/telegram/webhook", json=_telegram_message_update(chat, "/status"))
    check("/status (no active portfolio) reply is Markdown-safe",
          _no_unescaped_markdown_specials(sent_messages[-1]["text"]), sent_messages[-1]["text"])

    sent_messages.clear()
    client.post("/telegram/webhook", json=_telegram_message_update(chat, "/new_portfolio bad args"))
    check("malformed /new_portfolio usage message is Markdown-safe",
          _no_unescaped_markdown_specials(sent_messages[-1]["text"]), sent_messages[-1]["text"])

    sent_messages.clear()
    client.post("/telegram/webhook", json=_telegram_message_update(chat, '/new_portfolio "Test" Ticker NGN notanumber 2026-01-01'))
    check("malformed /new_portfolio (bad capital) error message is Markdown-safe",
          _no_unescaped_markdown_specials(sent_messages[-1]["text"]), sent_messages[-1]["text"])

    sent_messages.clear()
    client.post("/telegram/webhook", json=_telegram_message_update(chat, "/log_trade bad args"))
    check("malformed /log_trade usage message is Markdown-safe",
          _no_unescaped_markdown_specials(sent_messages[-1]["text"]), sent_messages[-1]["text"])

    sent_messages.clear()
    client.post("/telegram/webhook", json=_telegram_message_update(chat, "/undo_trade"))
    check("/undo_trade usage message is Markdown-safe", _no_unescaped_markdown_specials(sent_messages[-1]["text"]), sent_messages[-1]["text"])

    sent_messages.clear()
    client.post("/telegram/webhook", json=_telegram_message_update(chat, "/ack"))
    check("/ack usage message is Markdown-safe", _no_unescaped_markdown_specials(sent_messages[-1]["text"]), sent_messages[-1]["text"])


print("\n--- Dynamic/user-supplied values with markdown-special characters don't break replies ---")

with patch.object(bot, "TELEGRAM_WEBHOOK_SECRET", ""), patch.object(bot, "send_message", side_effect=_fake_send_message):
    chat = "chat_mdsafety_2"

    # A portfolio name containing every markdown-special character at once
    sent_messages.clear()
    client.post("/telegram/webhook", json=_telegram_message_update(
        chat, '/new_portfolio "My_Fund*Test`One[Two]" Ticker NGN 1000000 2026-01-01'
    ))
    check("/new_portfolio with a special-character name is Markdown-safe",
          _no_unescaped_markdown_specials(sent_messages[-1]["text"]), sent_messages[-1]["text"])
    check("/new_portfolio confirmation still contains the (escaped) actual name",
          "My" in sent_messages[-1]["text"] and "Fund" in sent_messages[-1]["text"], sent_messages[-1]["text"])

    sent_messages.clear()
    client.post("/telegram/webhook", json=_telegram_message_update(chat, "/status"))
    check("/status for a special-character-named portfolio is Markdown-safe",
          _no_unescaped_markdown_specials(sent_messages[-1]["text"]), sent_messages[-1]["text"])

    # A symbol argument with an underscore in it (unusual, but nothing validates
    # against it -- must not be able to break the reply either way)
    sent_messages.clear()
    with patch.object(bot.nps, "get_current_prices_and_sectors", return_value=({}, {})):
        client.post("/telegram/webhook", json=_telegram_message_update(chat, "/log_trade buy WEIRD_SYM 10 5.00 auto"))
    check("/log_trade with an underscored symbol name is Markdown-safe",
          _no_unescaped_markdown_specials(sent_messages[-1]["text"]), sent_messages[-1]["text"])

    sent_messages.clear()
    client.post("/telegram/webhook", json=_telegram_message_update(chat, "/holdings"))
    check("/holdings for a portfolio holding an underscored symbol is Markdown-safe",
          _no_unescaped_markdown_specials(sent_messages[-1]["text"]), sent_messages[-1]["text"])


print("\n--- send_message plain-text fallback: a 400 never results in total silence ---")

with patch.object(bot, "TELEGRAM_WEBHOOK_SECRET", ""):
    fallback_calls = []

    def _fake_send_message_400_then_ok(chat_id, text, parse_mode="Markdown"):
        fallback_calls.append({"chat_id": chat_id, "text": text, "parse_mode": parse_mode})
        if parse_mode == "Markdown":
            return _FakeResponse(400, '{"ok":false,"error_code":400,"description":"Bad Request: can\'t parse entities"}')
        return _FakeResponse(200, '{"ok":true}')

    fallback_calls.clear()
    with patch.object(bot, "send_message", side_effect=_fake_send_message_400_then_ok):
        r = client.post("/telegram/webhook", json=_telegram_message_update("chat_fallback", "/start"))
    check("webhook still returns 200 even when the first sendMessage attempt 400s", r.status_code == 200, r.text)
    check("on a 400, exactly one retry is made", len(fallback_calls) == 2, fallback_calls)
    check("the retry uses parse_mode=None (plain text), not Markdown again",
          fallback_calls[0]["parse_mode"] == "Markdown" and fallback_calls[1]["parse_mode"] is None, fallback_calls)
    check("the retry sends the SAME text content as the original attempt",
          fallback_calls[0]["text"] == fallback_calls[1]["text"], fallback_calls)

    # send_message itself: parse_mode=None must omit the key, not send JSON null
    # (Telegram's own default for an absent parse_mode is plain text; sending
    # null explicitly is untested/undocumented behavior worth avoiding).
    with patch.object(bot.requests, "post") as mock_post:
        mock_post.return_value = _FakeResponse(200, '{"ok":true}')
        bot.send_message("123", "hello", parse_mode=None)
        sent_payload = mock_post.call_args.kwargs["json"]
        check("send_message(parse_mode=None) omits the parse_mode key entirely",
              "parse_mode" not in sent_payload, sent_payload)

    with patch.object(bot.requests, "post") as mock_post2:
        mock_post2.return_value = _FakeResponse(200, '{"ok":true}')
        bot.send_message("123", "hello")  # default
        sent_payload2 = mock_post2.call_args.kwargs["json"]
        check("send_message with the default parse_mode still sends parse_mode=Markdown",
              sent_payload2.get("parse_mode") == "Markdown", sent_payload2)


# =======================================================================
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
