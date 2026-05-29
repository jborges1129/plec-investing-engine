"""
Output formatting for stock screener results.

Handles terminal briefing and (future) Telegram message formatting.
The terminal output is designed to give you everything needed to make a
decision in one pass — no need to open a separate chart just to see the
basic signal picture.
"""

from datetime import datetime
from shared.data import ETF_NAMES  # for sector ETF name lookups


# Tier colors/labels for terminal output
_TIER_HEADERS = {
    'HIGH': '🟢 HIGH TIER  (score 8–10) — highest signal confluence',
    'MID':  '🟡 MID TIER   (score 5–7)  — solid setup, fewer confirming signals',
    'LOW':  '🔴 LOW TIER   (score 0–4)  — passes gates, requires strong qualitative case',
}

_SETUP_EMOJI = {
    'Breakout from Base':           '⚡',
    'Earnings Gap / Post-Beat Drift': '📈',
    'Momentum Continuation':        '🚀',
    'First Pullback in Uptrend':    '↩️ ',
    'Pre-Earnings Setup':           '📅',
    'Deep Pullback to Support':     '🎯',
    'Momentum Setup':               '📊',
}


def _fmt_pct(v) -> str:
    if v is None or (isinstance(v, float) and v != v):  # NaN check
        return '  n/a '
    return f'{v*100:+6.1f}%'


def _fmt_price(v) -> str:
    if v is None or (isinstance(v, float) and v != v):
        return 'n/a'
    return f'${v:.2f}'


def _fmt_cap(v) -> str:
    if not v:
        return 'n/a'
    if v >= 1_000_000_000:
        return f'${v/1_000_000_000:.1f}B'
    return f'${v/1_000_000:.0f}M'


def format_candidate(c: dict, show_detail: bool = True) -> str:
    """Format a single candidate as a terminal block."""
    rank    = c.get('rank', '?')
    sym     = c.get('symbol', '?')
    name    = c.get('name', sym)[:35]
    sector  = c.get('sector', 'Unknown')
    score   = c.get('score', 0)
    setup   = c.get('setup_name', 'Unknown')
    tier    = c.get('setup_tier', '?')
    emoji   = _SETUP_EMOJI.get(setup, '📊')

    price   = c.get('price')
    atr_pct = c.get('atr_pct')
    rsi     = c.get('rsi')
    rvol    = c.get('rvol')
    prox    = c.get('prox_52w')
    rs_spy  = c.get('rs_spy')
    mom     = c.get('mom_12_1')
    cap     = c.get('market_cap', 0)

    ret_1d  = c.get('ret_1d')
    ret_5d  = c.get('ret_5d')
    ret_1m  = c.get('ret_1m')
    ret_3m  = c.get('ret_3m')
    ret_6m  = c.get('ret_6m')

    earnings_flag = ' ⚠ EARNINGS THIS WEEK' if c.get('earnings_soon') else ''
    short_flag    = f" ⚠ SHORT: {c.get('short_float',0)*100:.0f}%" if c.get('short_flag') else ''

    lines = [
        f"  #{rank:<3} {sym:<6}  {name:<35}  Score: {score}/10  Tier {tier}",
        f"       {emoji} {setup}{earnings_flag}{short_flag}",
        f"       Sector: {sector:<28}  Mkt Cap: {_fmt_cap(cap)}",
    ]

    if show_detail:
        lines += [
            f"       Price: {_fmt_price(price):<10}"
            f"  RSI: {rsi:5.1f}{'  ⚠ overbought' if rsi and rsi > 70 else '         '}"
            f"  RVOL: {rvol:.1f}x  ATR: {atr_pct*100:.1f}%" if rsi and atr_pct else '',

            f"       Returns —  Today: {_fmt_pct(ret_1d)}"
            f"  1W: {_fmt_pct(ret_5d)}"
            f"  1M: {_fmt_pct(ret_1m)}"
            f"  3M: {_fmt_pct(ret_3m)}"
            f"  6M: {_fmt_pct(ret_6m)}",

            f"       52W High: {_fmt_price(c.get('high_52w'))} "
            f"({'AT HIGH' if prox and prox >= 0.99 else f'{prox*100:.1f}% of high' if prox else 'n/a'})"
            f"  RS vs SPY (3m): {_fmt_pct(rs_spy)}"
            f"  12-1M Momentum: {_fmt_pct(mom)}",
        ]

    return '\n'.join(l for l in lines if l.strip())


def print_stock_briefing(candidates: list[dict], regime: str = 'unknown') -> None:
    """
    Print the full 20-candidate briefing to the terminal.
    Candidates are grouped into HIGH / MID / LOW tiers.
    """
    if not candidates:
        print("\nNo stock candidates returned by screener.")
        return

    today = datetime.now().strftime('%B %d, %Y — %I:%M %p')
    n     = len(candidates)

    print(f"\n{'═'*72}")
    print(f"  STOCK SCREENER  —  {today}")
    print(f"{'═'*72}")
    print(f"  Universe: S&P MidCap 400 + SmallCap 600  ·  Regime: {regime.upper()}")
    print(f"  {n} candidates from {n} ranked by signal confluence\n")

    current_tier = None
    for c in candidates:
        tier = c.get('tier_label', 'LOW')
        if tier != current_tier:
            current_tier = tier
            print(f"\n  {'─'*68}")
            print(f"  {_TIER_HEADERS.get(tier, tier)}")
            print(f"  {'─'*68}")
        print(format_candidate(c))
        print()

    print(f"{'═'*72}")
    print("  Scoring key:")
    print("    RS vs SPY (2pt)  |  12-1M Momentum (2pt)  |  52W Proximity (2pt)")
    print("    Relative Volume (2pt)  |  Setup Tier (2pt: A=2, B=1, C=0)")
    print(f"\n  Run 'python run.py stocks' to refresh.")
    print(f"{'═'*72}\n")


def format_telegram_message(candidates: list[dict], regime: str = 'unknown') -> str:
    """
    Compact Telegram message for the top candidates.
    Sent when the screener surfaces new HIGH-tier setups.
    """
    if not candidates:
        return "📊 Stock screener: no qualified candidates today."

    today   = datetime.now().strftime('%b %d')
    high    = [c for c in candidates if c.get('tier_label') == 'HIGH']
    mid     = [c for c in candidates if c.get('tier_label') == 'MID']
    n_total = len(candidates)

    lines = [
        f"📊 *Stock Screener — {today}* ({regime.upper()} regime)",
        f"{n_total} candidates: {len(high)} HIGH · {len(mid)} MID · {n_total-len(high)-len(mid)} LOW\n",
    ]

    if high:
        lines.append("*🟢 HIGH TIER:*")
        for c in high[:5]:
            sym    = c.get('symbol', '?')
            name   = (c.get('name') or sym)[:25]
            setup  = c.get('setup_name', '')
            ret3m  = c.get('ret_3m')
            rsi    = c.get('rsi')
            rvol   = c.get('rvol', 1)
            score  = c.get('score', 0)
            ret_s  = f"{ret3m*100:+.1f}%" if ret3m is not None else 'n/a'
            rsi_s  = f"{rsi:.0f}" if rsi is not None else 'n/a'
            lines.append(
                f"• *{sym}* ({name}) — {setup}\n"
                f"  Score {score}/10 · 3M {ret_s} · RSI {rsi_s} · RVOL {rvol:.1f}x"
            )

    if mid:
        lines.append("\n*🟡 MID TIER (top 3):*")
        for c in mid[:3]:
            sym   = c.get('symbol', '?')
            setup = c.get('setup_name', '')
            ret3m = c.get('ret_3m')
            score = c.get('score', 0)
            ret_s = f"{ret3m*100:+.1f}%" if ret3m is not None else 'n/a'
            lines.append(f"• *{sym}* — {setup} · Score {score}/10 · 3M {ret_s}")

    lines.append("\n_Review full list: `python run.py stocks`_")
    return '\n'.join(lines)
