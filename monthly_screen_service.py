"""
monthly_screen_service.py

The actual live-universe-screen-and-notify workflow, extracted into its
own module specifically to be shared between two callers that CANNOT
import each other directly:

  - main.py's POST /monthly-screen endpoint (triggered automatically by
    the scheduled GitHub Actions workflow), and
  - bot.py's /monthly_screen Telegram command (triggered manually,
    restricted to OWNER_CHAT_ID -- see bot.py).

main.py already imports bot.py (for the /telegram/webhook handoff), so
bot.py importing main.py back would be circular. This module depends on
NEITHER of them: send_message and the Markdown-escape function are passed
in as parameters (send_message_fn, escape_md_fn) rather than imported,
the same dependency-injection pattern screening_service.py already uses
for fundamentals_fetcher -- each caller passes in bot.py's real
send_message/_escape_md, but this module itself has no idea bot.py
exists.

Also houses _get_cached_or_live_fundamentals and _ranked_eligible_symbols
-- these aren't monthly-screen-specific (main.py's /fundamentals/{symbol},
/screen, and /portfolio/construct all use them too), but they were
originally defined in main.py and are moved here as the single shared
home now that a second module needs them, rather than duplicated.
"""

from datetime import datetime
from typing import Callable, List, Optional

import pandas as pd

import fundamentals_service as fs
import screening_service as ss
import portfolio_service as ps
import ngx_pulse_service as nps
import universe_service as us
import db

FUNDAMENTALS_CACHE_MAX_AGE_DAYS = 1  # refreshed daily; avoids re-scraping ~75 names per screen


def get_cached_or_live_fundamentals(
    symbol: str, as_of_date: datetime, current_price: Optional[float] = None
) -> fs.PointInTimeFundamentals:
    """Drop-in replacement for fundamentals_service.get_point_in_time_fundamentals
    that checks db.get_cached_fundamentals() first. On a cache hit, pe is
    recomputed from the cached eps_used against the current current_price
    (price is caller-supplied and can move within the cache's validity
    window; eps_used/eps_period_ending are the stable, reporting-lag-gated
    facts and are safe to reuse as-is). On a miss/stale entry, does the
    live scrape and writes the result back via db.cache_fundamentals()."""
    as_of_d = as_of_date.date() if isinstance(as_of_date, datetime) else as_of_date
    cached = db.get_cached_fundamentals(symbol, as_of_d, max_age_days=FUNDAMENTALS_CACHE_MAX_AGE_DAYS)
    if cached is not None:
        eps_used = float(cached["eps_used"]) if cached["eps_used"] is not None else None
        pe = float(cached["pe"]) if cached["pe"] is not None else None
        if current_price is not None and eps_used is not None and eps_used > 0:
            pe = current_price / eps_used
        return fs.PointInTimeFundamentals(
            symbol=symbol,
            as_of_date=as_of_date,
            roe=float(cached["roe"]) if cached["roe"] is not None else None,
            pe=pe,
            eps_used=eps_used,
            eps_period_ending=cached["eps_period_ending"],
            data_available=True,
        )

    result = fs.get_point_in_time_fundamentals(symbol, as_of_date, current_price=current_price)
    if result.data_available:
        eps_period_ending = result.eps_period_ending
        if isinstance(eps_period_ending, datetime):
            eps_period_ending = eps_period_ending.date()
        db.cache_fundamentals(
            symbol, as_of_d, roe=result.roe, pe=result.pe,
            eps_used=result.eps_used, eps_period_ending=eps_period_ending,
        )
    return result


def ranked_eligible_symbols(scored: List[ss.ScoredStock]) -> List[str]:
    """The one correct way to turn a screen_universe() result into the
    ranked_symbols list portfolio_service.build_initial_portfolio expects:
    filter to eligible stocks (which are already sorted best-to-worst),
    discard ineligible ones entirely rather than letting them lead the
    list."""
    return [s.symbol for s in scored if s.eligible]


def _fmt_pct(x: float) -> str:
    return f"{x * 100:.2f}%"


def run_monthly_screen_and_notify(
    chat_id: str,
    pool_size: int,
    cash_buffer: float,
    lookback_days: int,
    max_ngx_pulse_requests: int,
    send_message_fn: Callable[[str, str], None],
    escape_md_fn: Callable[[object], str],
):
    """The actual work: builds the live universe, screens it, constructs a
    portfolio, compares it against the chat's active portfolio (if any --
    see _compare_against_holdings), and sends the result via
    send_message_fn -- run this off the request/response path (a
    background task in main.py, a daemon thread in bot.py), never
    synchronously inside an HTTP or webhook handler. Screening a ~78-name
    universe means one fundamentals scrape per eligible symbol
    (stockanalysis.com, ~4 seconds each including the polite delay -- see
    fundamentals_service.py), which can comfortably take several minutes
    end to end -- longer than either an HTTP reverse proxy or Telegram's
    own webhook delivery is willing to wait, so the result is always
    delivered via send_message_fn directly, never as a return value."""
    try:
        symbols, sector_by_symbol, current_price_by_symbol = us.build_live_universe()
        if len(symbols) < ps.MIN_HOLDINGS:
            send_message_fn(
                chat_id,
                f"Monthly screen: only {len(symbols)} symbols passed the live universe filter "
                f"(need at least {ps.MIN_HOLDINGS}). Aborting -- check NGX Pulse data quality.",
            )
            return

        price_series_by_symbol, skipped = nps.get_price_series_for_symbols(
            symbols, days=lookback_days, max_requests=max_ngx_pulse_requests,
        )

        as_of_dt = datetime.utcnow()
        scored = ss.screen_universe(
            price_series_by_symbol=price_series_by_symbol,
            current_price_by_symbol=current_price_by_symbol,
            sector_by_symbol=sector_by_symbol,
            as_of_date=as_of_dt,
            excluded_symbols=set(),  # universe_service already excludes NIDF/NREIT
            fundamentals_fetcher=get_cached_or_live_fundamentals,
        )
        ranked = ranked_eligible_symbols(scored)

        if len(ranked) < ps.MIN_HOLDINGS:
            send_message_fn(
                chat_id,
                f"Monthly screen: only {len(ranked)} eligible symbols after screening "
                f"(need at least {ps.MIN_HOLDINGS}). Aborting.",
            )
            return

        weights, cash_pct = ps.build_initial_portfolio(
            ranked, sector_by_symbol, pool_size=pool_size, cash_buffer=cash_buffer,
        )

        lines = [
            f"{escape_md_fn(sym)}: {_fmt_pct(w)}"
            for sym, w in weights.sort_values(ascending=False).items()
        ]
        msg = (
            f"*Monthly screen results -- {as_of_dt:%Y-%m-%d}*\n\n"
            + "\n".join(lines)
            + f"\n\nCash: {_fmt_pct(cash_pct)}"
            + f"\n\n{len(price_series_by_symbol)} of {len(symbols)} universe symbols had usable "
              f"price history; {len(ranked)} were eligible after screening."
        )
        if skipped:
            msg += f"\n\nNo usable price history (excluded): {', '.join(escape_md_fn(s) for s in skipped)}"

        msg += _compare_against_holdings(chat_id, weights, ranked, sector_by_symbol, escape_md_fn)

        send_message_fn(chat_id, msg)

    except Exception as e:  # noqa: BLE001 -- this always runs off the request/response path;
        # nothing else observes an exception here except the person via their messaging channel.
        try:
            send_message_fn(chat_id, f"Monthly screen failed: {escape_md_fn(e)}")
        except Exception as notify_err:
            print(f"monthly screen failed AND could not notify chat {chat_id}: {e} / {notify_err}")


def _compare_against_holdings(chat_id: str, fresh_target_weights: pd.Series, ranked: List[str],
                                sector_by_symbol: dict, escape_md_fn: Callable[[object], str]) -> str:
    """The piece the strategy documentation actually specifies as part of
    the monthly review (Section 8: "Compare the resulting target weights
    to current actual holdings; rebalance only positions that trigger a
    hard breach or meaningful drift") but which wasn't wired into
    /monthly-screen initially -- this reuses the EXACT SAME comparison
    engine main.py's /rebalance-check already runs (portfolio_service.
    check_drift, check_hard_breaches), not a separate implementation, plus
    identify_diversification_additions -- the one piece of v8's documented
    backtest methodology (Section 10: "replenish below 10 holdings,
    distinct from the hard compliance minimum of 5") that was described
    but never actually implemented in live code until now.

    Returns a text block to APPEND to the monthly-screen message (empty
    string if there's no active portfolio linked to this chat -- nothing
    to compare against). Also persists any newly-triggered drift/breach
    signals via db.log_signal_if_new, the same deduping mechanism
    /rebalance-check already uses, so they show up in /signals and can be
    /ack'd the same way regardless of which path found them.

    KNOWN APPROXIMATION: cash is valued at 0 here (weights are normalized
    across held positions only), since cash is not a persisted field
    anywhere in db.py's schema -- see db.py's docstring and /rebalance-
    check's own required `cash` parameter, which this automated path has
    no way to supply. This means computed weights run slightly HIGH
    relative to true (cash-diluted) weights -- a conservative bias for
    cap-breach detection (more likely to flag a false positive than miss
    a real breach), never the reverse. Cash-cap breach is therefore never
    checked here; the message says so explicitly rather than silently
    reporting an unverified "no cash breach"."""
    active = db.get_active_portfolio(chat_id)
    if not active:
        return "\n\n_No active portfolio linked to this chat -- skipping holdings comparison._"

    portfolio_id = active["id"]
    holdings_list = db.get_holdings(portfolio_id)
    if not holdings_list:
        return f"\n\n_Active portfolio '{escape_md_fn(active['name'])}' has no holdings yet -- skipping comparison._"

    held_symbols = [h["symbol"] for h in holdings_list]
    try:
        current_prices, _live_sectors = nps.get_current_prices_and_sectors(symbols=held_symbols)
    except Exception as e:  # noqa: BLE001 -- comparison is best-effort; target weights above still stand
        return f"\n\n_Could not fetch live prices to compare against holdings: {escape_md_fn(e)}_"

    actual_weights_dict = db.holdings_to_weights(holdings_list, current_prices, cash=0.0)
    sectors = {h["symbol"]: h["sector"] for h in holdings_list}
    sectors.update(sector_by_symbol)  # fill in sector for any symbol not in current holdings

    actual = pd.Series(actual_weights_dict)
    drift_df = ps.check_drift(actual, fresh_target_weights, drift_threshold=ps.DRIFT_THRESHOLD)
    breaches = ps.check_hard_breaches(actual, sectors, cash=0.0)
    additions = ps.identify_diversification_additions(held_symbols, ranked, floor=ps.DIVERSIFICATION_FLOOR)

    sections = [f"\n\n*Comparison against current holdings -- {escape_md_fn(active['name'])}*"]
    any_flag = False

    if breaches["stock_breach"] or breaches["sector_breach"] or breaches["holdings_breach"]:
        any_flag = True
        breach_lines = []
        for sym, w in breaches["stock_breach"].items():
            breach_lines.append(f"{escape_md_fn(sym)}: {_fmt_pct(w)} (stock cap {_fmt_pct(ps.MAX_STOCK_WEIGHT)})")
        for sector, w in breaches["sector_breach"].items():
            breach_lines.append(f"{escape_md_fn(sector)} sector: {_fmt_pct(w)} (sector cap {_fmt_pct(ps.MAX_SECTOR_WEIGHT)})")
        if breaches["holdings_breach"]:
            breach_lines.append(f"Only {len(holdings_list)} holdings (minimum {ps.MIN_HOLDINGS})")
        sections.append("*HARD BREACHES -- act immediately:*\n" + "\n".join(breach_lines))

    triggered = drift_df[drift_df["triggered"] | drift_df["fresh_target"].isna()]
    # check_drift deliberately leaves fresh_target as NaN (not silently 0)
    # for a held symbol entirely absent from the fresh target -- see its
    # own docstring/tests. But NaN > drift_threshold evaluates False in
    # pandas, so `triggered` alone would silently MISS the single clearest
    # drift case there is: a name that fell out of the ranked pool
    # completely. That's a comparison-semantics quirk of `triggered`, not
    # a deliberate design choice about what deserves flagging -- so this
    # caller applies that judgment explicitly rather than relying on
    # check_drift's own boolean for this specific case.
    if len(triggered) > 0:
        any_flag = True
        drift_lines = []
        for sym, row in triggered.iterrows():
            target = row["fresh_target"]
            target_str = _fmt_pct(target) if pd.notna(target) else "not in fresh pool"
            drift_lines.append(
                f"{escape_md_fn(sym)}: held {_fmt_pct(row['actual_weight'])}, target {target_str}"
            )
        sections.append(f"*DRIFT (>{_fmt_pct(ps.DRIFT_THRESHOLD)} from target) -- review this month:*\n" + "\n".join(drift_lines))

    if additions:
        any_flag = True
        sections.append(
            f"*DIVERSIFICATION FLOOR ({len(holdings_list)} holdings, floor is {ps.DIVERSIFICATION_FLOOR}) "
            f"-- consider adding:*\n" + "\n".join(escape_md_fn(s) for s in additions)
        )

    if not any_flag:
        sections.append("No hard breaches, no meaningful drift, holdings at/above the diversification floor.")

    sections.append("_Cash-cap breach not checked automatically -- cash isn't tracked in this system._")

    # Persist signals via the SAME dedup mechanism /rebalance-check uses, so
    # /signals and /ack work identically regardless of which path found them.
    for sym, w in breaches["stock_breach"].items():
        db.log_signal_if_new(portfolio_id, "stock_breach", sym, {"weight": w})
    for sector, w in breaches["sector_breach"].items():
        db.log_signal_if_new(portfolio_id, "sector_breach", sector, {"weight": w})
    if breaches["holdings_breach"]:
        db.log_signal_if_new(portfolio_id, "holdings_breach", None, {"holdings_count": len(holdings_list)})
    for sym, row in triggered.iterrows():
        db.log_signal_if_new(
            portfolio_id, "drift", sym,
            {"actual_weight": row["actual_weight"], "fresh_target": row["fresh_target"], "drift": row["drift"]},
        )
    if additions:
        db.log_signal_if_new(
            portfolio_id, "diversification_floor", None,
            {"holdings_count": len(holdings_list), "suggested_additions": additions},
        )

    return "\n\n".join(sections)
