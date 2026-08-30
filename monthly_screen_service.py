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
    portfolio, and sends the result via send_message_fn -- run this off
    the request/response path (a background task in main.py, a daemon
    thread in bot.py), never synchronously inside an HTTP or webhook
    handler. Screening a ~78-name universe means one fundamentals scrape
    per eligible symbol (stockanalysis.com, ~4 seconds each including the
    polite delay -- see fundamentals_service.py), which can comfortably
    take several minutes end to end -- longer than either an HTTP reverse
    proxy or Telegram's own webhook delivery is willing to wait, so the
    result is always delivered via send_message_fn directly, never as a
    return value."""
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
        send_message_fn(chat_id, msg)

    except Exception as e:  # noqa: BLE001 -- this always runs off the request/response path;
        # nothing else observes an exception here except the person via their messaging channel.
        try:
            send_message_fn(chat_id, f"Monthly screen failed: {escape_md_fn(e)}")
        except Exception as notify_err:
            print(f"monthly screen failed AND could not notify chat {chat_id}: {e} / {notify_err}")
