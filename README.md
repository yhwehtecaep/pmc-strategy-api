# CFA PMC Strategy Bot: User Manual

This is the Telegram interface to your CFA PMC portfolio-tracking system. It
lives inside the same FastAPI service as the strategy engine (`main.py`),
talking directly to the database (`db.py`) for everything except live prices,
which it pulls from NGX Pulse.

**What this bot is for:** recording and checking your *actual* portfolio,
meaning what you hold, what it's worth right now, and your trade history. It
is **not** the screening/construction engine. Screening, ranking, and
portfolio construction happen separately (via the API or your Colab
workflow) at the end-of-month rebalance review; this bot is where you log
the trades that review produces and check on things day-to-day.

---

## 1. Getting started

Message the bot with `/start` to register your chat. Then create your first
portfolio:

```
/new_portfolio "CFA PMC 2026" Ticker NGN 10000000 2026-01-01
```

Arguments, in order: **name** (quote it if it has spaces), **broker**,
**currency**, **initial capital**, **inception date** (`YYYY-MM-DD`). This
becomes your chat's *active portfolio*: every other command operates on it
until you switch or close it.

Each Telegram chat has exactly **one active portfolio at a time**. You can
have multiple portfolios over time (e.g. the competition portfolio, then a
personal account afterward), but only one is "live" for a given chat at any
moment.

---

## 2. Command reference

### Portfolio lifecycle

**`/new_portfolio "name" broker currency initial_capital YYYY-MM-DD`**
Creates a portfolio and makes it your chat's active one. If you already have
an active portfolio, it is automatically closed first (its data is kept, not
deleted; see `/close_portfolio` below for what "closed" means).

**`/switch_portfolio "name" broker currency initial_capital YYYY-MM-DD`**
Identical to `/new_portfolio`: closes whatever's currently active and starts
a new one in the same step. Two command names for the same action, since
"new" and "switch" read more naturally depending on whether you already have
something active.

**`/close_portfolio [YYYY-MM-DD]`**
Closes the active portfolio **without** creating a replacement. The date is
optional and defaults to today. After this, your chat has no active
portfolio: `/status`, `/holdings`, `/log_trade`, etc. will all tell you so
until you run `/new_portfolio` again. Use this at the actual end of the
competition, or any time you want to stop tracking a portfolio without
immediately starting a new one. (Historical data, holdings and trades, is
never deleted by closing; it's still queryable via the API.)

### Checking on things

**`/status`**
Shows the active portfolio's name, status, broker, currency, initial
capital, inception date, and internal ID.

**`/holdings`**
Lists current positions: symbol, shares, average cost, sector. This is your
*recorded* cost-basis view, with no live prices involved.

**`/value`**
Live market value of your current holdings, priced via NGX Pulse (one
request regardless of how many symbols you hold). For each position, shows
shares, live price, market value, and unrealized gain/loss vs. your average
cost. Totals market value and cost basis at the bottom, with overall
unrealized P/L.

> **Cash is not included.** There's no persisted cash figure anywhere in
> this system: cash only ever exists as a number the API's
> `/rebalance-check` endpoint is told, never something stored. `/value`
> covers equity holdings only, and says so explicitly in its own reply so
> it's never mistaken for your total account value.

If NGX Pulse can't be reached, `/value` tells you plainly rather than
failing silently. If a held symbol has no live price available (delisted,
data gap), it's flagged and excluded from the totals rather than silently
treated as worthless.

**`/trades`**
Shows your last 10 trades, each with a short ID in brackets (e.g.
`[a1b2c3d4]`). You'll need this ID if you ever need `/undo_trade`.

**`/signals`**
Shows unacknowledged drift/breach signals: flags raised elsewhere in the
system (by the API's `/rebalance-check`) when a position has drifted from
its target weight or breached a hard compliance rule (stock cap, sector cap,
cash cap, minimum holdings). The bot doesn't generate these itself, only
displays and lets you acknowledge them.

**`/ack signal_id`**
Acknowledges a signal so it stops showing up in `/signals`. Accepts the full
ID or just its first 8 characters.

### Recording trades

**`/log_trade buy|sell symbol shares price fee|auto [reason]`**

Records an executed trade and updates your holdings automatically:
weighted-average cost on buys, share reduction on sells (average cost is
never changed by a sell, per standard convention).

- **fee**: type the real fee your broker charged, or type `auto` to have it
  estimated using the strategy's confirmed rates (1.53% on buys, 2.18% on
  sells; these are asymmetric, and confirmed from real executed trades, not
  a guess).
- **reason**: optional, defaults to `"monthly_rebalance"`, because per your
  workflow, every trade happens at the end-of-month rebalance review, driven
  by that month's screening output. Override it (e.g.
  `initial_construction`) for anything else, like your very first
  deployment.
- **symbol** is automatically upper-cased.

Examples:
```
/log_trade buy GTCO 100 45.00 68.85
/log_trade sell MTNN 50 210.00 auto
/log_trade buy DANGCEM 20 350.00 auto initial_construction
```

If it's the **first time you're buying a symbol**, the bot fetches its
sector live from NGX Pulse automatically, so you never have to type it.
(This matters: sector is what the strategy's cap-enforcement logic groups
by, so a missing sector would silently break future compliance checks for
that name.) If the live lookup fails, the trade still logs; you'll just see
a note that sector was left unset.

Invalid input (bad number, unknown side, overselling more shares than you
hold, selling a symbol you never bought) is rejected with a clear message.
Nothing is ever partially applied.

**`/undo_trade trade_id`**

Undoes a mistakenly logged trade. Accepts the full trade ID or its first 8
characters (shown in `/trades`). This isn't a simple "reverse the last
number": it correctly **replays your entire remaining trade history** for
that symbol from scratch, so it gives the right answer even if you're
undoing a trade that wasn't the most recent one.

There's one thing it will refuse to do. If undoing a trade would leave a
*later* trade internally inconsistent, for example, undoing a buy that a
subsequent sell already depended on for enough shares, the bot will refuse
and tell you which later trade is the problem, rather than quietly
corrupting your position. In that case, undo the later trade first, then
the earlier one.

```
/undo_trade a1b2c3d4
```

---

## 3. Typical monthly workflow

1. **End of month**: run the screening/portfolio-construction step (outside
   this bot, via the API or your Colab workflow) to get that month's
   target names and weights.
2. **Compare** the fresh output to your current `/holdings`.
3. **Execute** the actual trades on Ticker (Cowrywise).
4. **Log each executed trade** here as you go:
   `/log_trade buy GTCO 100 45.00 68.85`
   Reason defaults to `monthly_rebalance`, so you don't need to type it
   every time.
5. Anytime in between: check `/value` for live P&L, `/holdings` for your
   recorded positions, `/signals` for anything flagged by the API.
6. Made a typo logging a trade? `/trades` to find its short ID,
   `/undo_trade <id>` to correct it, then re-log it properly.

---

## 4. What this bot deliberately does NOT do

- **Screening, portfolio construction, drift/breach checks
  (`/screen`, `/portfolio/construct`, `/rebalance-check`)**: these need the
  full strategy compute layer (momentum/vol/fundamentals scoring, price
  *history*, not just current price) and live in `main.py`/the API, not
  here. This is a deliberate scope boundary, not a missing feature.
- **Bulk-loading holdings**: there's no command that lets you set your
  entire holdings list in one shot. That's an API-only operation
  (`PUT /portfolios/{id}/holdings`), on purpose. A bulk full-replace command
  in a chat interface is one fat-fingered message away from wiping your
  portfolio. `/log_trade`'s additive buy/sell model is the safe path for
  everything ongoing, including building up a fresh portfolio one position
  at a time.
- **Editing a trade's fields in place**: there's no "change this trade's
  price" command, only undo-and-relog. This keeps the trade ledger always
  reconstructible from its own history.

---

## 5. Troubleshooting

- **"No active portfolio for this chat"**: you either haven't run
  `/new_portfolio` yet, or you closed your last one with `/close_portfolio`.
  Run `/new_portfolio` to fix it.
- **`/value` says it can't reach NGX Pulse**: this is a live network issue
  on NGX Pulse's end (or an auth/quota problem), not a bug in your data.
  Your recorded `/holdings` are unaffected; try `/value` again shortly.
- **A trade won't undo**: if you get a message naming a later trade as the
  reason, undo that later trade first (see §2, `/undo_trade`).
- **Unknown command**: `/help` lists everything the bot currently supports.

---

*This manual reflects the bot as of the version delivered alongside it.
Command behavior is defined in `bot.py`; if anything here and the running
code ever disagree, the code is authoritative. Flag it and this manual
should be corrected to match.*
