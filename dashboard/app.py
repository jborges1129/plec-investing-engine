import sys
import os
import json
import numpy as np
from collections import Counter, defaultdict
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
from datetime import datetime

from shared.alpaca import get_account, get_positions
from shared.database import (
    init_db, get_all_trades, get_snapshots, get_open_trades,
    save_daily_briefing, get_previous_briefing,
)
from shared.data import (
    get_market_regime, get_price_history, get_hourly_history, get_prior_day_summary,
    get_etf_news_yf, ETF_NAMES, ETF_TOP_HOLDINGS, BULL_ETFS, BEAR_ETFS, ETF_UNIVERSE,
)
from etf_module.strategy import build_signal_dashboard, get_position_sizing
from stocks_module.signals import SECTOR_ETF as STOCK_SECTOR_ETF

st.set_page_config(
    page_title="PLEC Trading Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
    .narrative-box {
        background: #1a1d2e;
        border-left: 4px solid #4a90e2;
        padding: 16px 20px;
        border-radius: 0 8px 8px 0;
        margin: 12px 0;
        line-height: 1.8;
        font-size: 0.95rem;
        color: #d8dce8;
    }
    .narrative-box strong, .narrative-box b {
        color: #ffffff;
    }
    .warning-box {
        background: #2e1f1f;
        border-left: 4px solid #ff4b4b;
        padding: 12px 16px;
        border-radius: 0 8px 8px 0;
        margin: 8px 0;
        color: #f0c0c0;
    }
    .positive-box {
        background: #1a2e22;
        border-left: 4px solid #00d4aa;
        padding: 12px 16px;
        border-radius: 0 8px 8px 0;
        margin: 8px 0;
        color: #b0e8d8;
    }
    .section-title {
        font-size: 1.3rem;
        font-weight: 700;
        color: #e0e0e0;
        margin-top: 8px;
    }
</style>
""", unsafe_allow_html=True)

_PLOT = dict(plot_bgcolor='#0e1117', paper_bgcolor='#0e1117', font_color='#ccc')
_GRID = dict(gridcolor='#1e2130')


# ── Cache ────────────────────────────────────────────────────────────────────

@st.cache_data(ttl=300)
def load_regime():
    return get_market_regime()

@st.cache_data(ttl=300)
def load_prior_day():
    return get_prior_day_summary()

@st.cache_data(ttl=300)
def load_dashboard_data(regime):
    return build_signal_dashboard({'regime': regime})

@st.cache_data(ttl=60)
def load_account():
    a = get_account()
    return {'portfolio_value': float(a.portfolio_value), 'cash': float(a.cash), 'equity': float(a.equity)}

@st.cache_data(ttl=300)
def load_daily_history(symbol, period='3mo'):
    df = get_price_history(symbol, period=period)
    df.index = pd.to_datetime(df.index)
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    return df

@st.cache_data(ttl=900)
def load_hourly_history(symbol):
    df = get_hourly_history(symbol, period='5d')
    df.index = pd.to_datetime(df.index)
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    return df

@st.cache_data(ttl=1800)
def load_etf_news(symbol):
    return get_etf_news_yf(symbol, limit=4)

@st.cache_data(ttl=300)
def load_screener_results():
    today_str  = datetime.now().strftime('%Y-%m-%d')
    cache_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        'data', f'screener_cache_{today_str}.json'
    )
    if os.path.exists(cache_path):
        mtime = datetime.fromtimestamp(os.path.getmtime(cache_path)).strftime('%I:%M %p')
        with open(cache_path) as f:
            return json.load(f), mtime
    return [], None

@st.cache_data(ttl=1800)
def load_stock_news(symbol: str):
    return get_etf_news_yf(symbol, limit=5)


# ── Helpers ──────────────────────────────────────────────────────────────────

def _rsi_series(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / loss.replace(0, float('nan'))
    return (100 - (100 / (1 + rs))).fillna(100)

def _fmt(v, pct=True):
    return f"{v*100:+.1f}%" if pct else f"{v:+.2f}"


# ── Narrative generators ─────────────────────────────────────────────────────

def generate_prior_day_narrative(prior: dict) -> str:
    spy = prior.get('SPY', {})
    vix = prior.get('^VIX', {})
    tlt = prior.get('TLT', {})
    gld = prior.get('GLD', {})
    if not spy:
        return "Prior day data unavailable."
    spy_pct = spy['change_pct'] * 100
    spy_word = "gained" if spy_pct > 0 else "fell"
    out = f"Yesterday, the S&P 500 {spy_word} **{spy_pct:+.2f}%**, closing at ${spy['last_close']:.2f}. "
    if vix:
        vix_pct = vix['change_pct'] * 100
        out += f"The VIX {'rose' if vix_pct > 0 else 'dropped'} {abs(vix_pct):.1f}% to {vix['last_close']:.1f}, "
        if vix['last_close'] < 15:
            out += "an unusually calm tape — very low fear. "
        elif vix['last_close'] < 20:
            out += "a relaxed, risk-on environment. "
        elif vix['last_close'] < 25:
            out += "showing moderate caution. "
        else:
            out += "signaling elevated fear. "
    if tlt:
        tlt_pct = tlt['change_pct'] * 100
        if abs(tlt_pct) > 0.3:
            out += f"Long-term bonds (TLT) {'rose' if tlt_pct > 0 else 'fell'} {abs(tlt_pct):.2f}%, "
            if tlt_pct > 0 and spy_pct < 0:
                out += "a classic flight-to-safety rotation — money moved from stocks to bonds. "
            elif tlt_pct < 0 and spy_pct > 0:
                out += "consistent with risk-on behavior — equities took priority over bonds. "
            else:
                out += "moving alongside equities. "
    if gld:
        gld_pct = gld['change_pct'] * 100
        if abs(gld_pct) > 0.5:
            out += f"Gold {'rose' if gld_pct > 0 else 'fell'} {abs(gld_pct):.2f}%. "
    return out


def generate_regime_narrative(rd: dict) -> str:
    regime = rd['regime']
    vix, spy, ma200, ma50 = rd['vix'], rd['spy_price'], rd['ma200'], rd['ma50']
    pct_above = ((spy / ma200) - 1) * 100
    if regime == 'bull':
        return (
            f"The market is in a **bull regime**. Two conditions both hold: "
            f"VIX at {vix:.1f} is below 20 (investors are calm, taking risk), and "
            f"SPY at ${spy:.2f} sits **{pct_above:.1f}% above its 200-day moving average** of ${ma200:.2f}, "
            f"confirming the long-term uptrend. In bull conditions, momentum ETFs in technology, "
            f"energy, and financials historically lead. Defensive assets (bonds, gold, utilities) tend to lag."
        )
    elif regime == 'bear':
        below = ((ma200 / spy) - 1) * 100 if spy < ma200 else 0
        return (
            f"The market is in a **bear/defensive regime**. "
            f"{'VIX at ' + str(round(vix,1)) + ' is above 25 — elevated investor fear. ' if vix > 25 else ''}"
            f"{'SPY has fallen ' + str(round(below,1)) + '% below its 200-day average of $' + str(round(ma200,2)) + ', breaking the long-term uptrend. ' if spy < ma200 else ''}"
            f"The strategy rotates toward assets that preserve value in downturns: "
            f"long-term bonds (TLT), gold (GLD), healthcare (XLV), utilities (XLU), and consumer staples (XLP)."
        )
    else:
        return (
            f"The market is in a **neutral regime** — mixed signals. "
            f"VIX at {vix:.1f} is neither calm nor fearful, and SPY is within normal range of its "
            f"200-day average of ${ma200:.2f}. The strategy considers both growth and defensive ETFs, "
            f"weighting toward whichever shows stronger recent momentum."
        )


def generate_recommendation_narrative(pos: dict, dashboard: pd.DataFrame, news: list) -> str:
    symbol = pos['symbol']
    name = ETF_NAMES.get(symbol, symbol)
    row = dashboard[dashboard['symbol'] == symbol].iloc[0]

    ret_1d  = row.get('return_1d', 0.0) * 100
    ret_5d  = row.get('return_5d', 0.0) * 100
    ret_1m  = row['return_1m'] * 100
    ret_3m  = row['return_3m'] * 100
    ret_6m  = row['return_6m'] * 100
    rsi     = row['rsi']
    vol     = row['volume_trend']
    price   = pos['price']

    # Relative performance vs SPY
    rel_text = ""
    spy_rows = dashboard[dashboard['symbol'] == 'SPY']
    if not spy_rows.empty:
        spy_3m = spy_rows['return_3m'].iloc[0] * 100
        diff = ret_3m - spy_3m
        if abs(diff) > 1:
            word = "outperforming" if diff > 0 else "underperforming"
            rel_text = f", {word} the S&P 500 by **{abs(diff):.1f}%**"

    # Momentum character
    avg_monthly_pace = ret_3m / 3
    if ret_1m > avg_monthly_pace * 1.1:
        mom_char = "momentum is **accelerating** — the past month is running faster than the 3-month average pace"
    elif ret_1m > 0:
        mom_char = "momentum is positive but cooling — the past month is slower than the 3-month average pace"
    else:
        mom_char = "recent momentum has stalled despite positive 3-month performance"

    # Volume interpretation
    vol_text = {
        'rising':  "Volume is rising — more participants are joining the move, which confirms the trend.",
        'falling': "Volume is falling — the move is happening on thin participation, a mild caution flag.",
        'neutral': "Volume is steady — neither confirming nor contradicting the price trend.",
    }.get(vol, "")

    # Top holdings context
    holdings = ETF_TOP_HOLDINGS.get(symbol, [])
    if holdings:
        h_str = ", ".join(f"{h[0]} ({h[1]:.0f}%)" for h in holdings[:3])
        if symbol == 'TLT':
            holdings_text = f"\n\n**What {symbol} holds:** US Treasury bonds with 20+ year maturities. It moves **inversely with interest rates** — if the Fed signals rate cuts, TLT rises. Duration risk is high; a 1% rise in rates drops TLT ~18%."
        elif symbol in ('GLD', 'SLV'):
            holdings_text = f"\n\n**What {symbol} holds:** Physical {'gold' if symbol == 'GLD' else 'silver'}, held in trust. It tends to rise when real yields are low, inflation is high, or equity volatility spikes. No dividends — pure price appreciation play."
        elif symbol == 'IWM':
            holdings_text = f"\n\n**What {symbol} holds:** ~2,000 small-cap US companies, no single holding above 0.5%. Small caps outperform large caps in early economic expansions when credit is easy, and lag in contractions. Higher volatility than SPY."
        elif symbol in ('EFA', 'EEM'):
            holdings_text = f"\n\n**What {symbol} holds:** Top positions — {h_str}. International funds add currency risk: a strong US dollar is a headwind, a weak dollar is a tailwind. {symbol} {'tracks developed markets (Europe, Japan, Australia)' if symbol == 'EFA' else 'is ~30% China, 20% India, 17% Taiwan — political and currency risk is significant'}."
        else:
            holdings_text = f"\n\n**What {symbol} holds:** Top positions — {h_str}. When these stocks move, {symbol} moves."
    else:
        holdings_text = ""

    # News context
    if news:
        news_lines = "\n".join(f"- *{a['title']}* — {a['publisher']}" for a in news[:3] if a.get('title'))
        news_text = f"\n\n**Recent headlines:**\n{news_lines}"
    else:
        news_text = ""

    # RSI flag
    rsi_text = ""
    if rsi > 75:
        rsi_text = f"\n\n**⚠️ RSI {rsi:.1f} — significantly overbought.** The ETF has risen faster than its 14-day average pace. Momentum strategies continue even in overbought territory, but short-term pullback risk is elevated. The stop-loss accounts for this."
    elif rsi > 70:
        rsi_text = f"\n\n**RSI {rsi:.1f} — modestly overbought.** Strong upward momentum, just above the 70 threshold. Not disqualifying, but worth watching."

    # Risk framework
    stop_pct = (price - pos['stop_price']) / price * 100
    tp_pct   = (pos['take_profit_price'] - price) / price * 100
    risk_text = (
        f"\n\n**Risk framework:** Stop at ${pos['stop_price']:.2f} ({stop_pct:.1f}% below entry) "
        f"= 3× the 14-day ATR of ${pos['atr']:.2f} — three days of normal price movement. "
        f"Take-profit at ${pos['take_profit_price']:.2f} ({tp_pct:.1f}% above entry) "
        f"gives a **2.5:1 reward-to-risk ratio**, meaning this trade only needs to be right ~30% of the time to be net profitable over many trades."
    )

    # Projection
    monthly_rate = ret_3m / 3 / 100
    proj_30d = price * (1 + monthly_rate)
    proj_text = (
        f"\n\n**Projection:** If the current 3-month momentum pace continues for one more month, "
        f"{symbol} would reach approximately **${proj_30d:.2f}** — roughly {'+' if proj_30d > price else ''}"
        f"${proj_30d - price:.2f} from today's ${price:.2f}. "
        f"This is a momentum extrapolation, not a forecast — use it to gauge whether the upside justifies the stop distance."
    )

    return (
        f"**{symbol} — {name}** is ranked #{int(row['rank'])} today. "
        f"3-month: **{ret_3m:+.1f}%**{rel_text}. 1-month: {ret_1m:+.1f}%. "
        f"This week: {ret_5d:+.1f}%. Yesterday: {ret_1d:+.2f}%. "
        f"{mom_char.capitalize()}. {vol_text}"
        f"{holdings_text}"
        f"{news_text}"
        f"{risk_text}{rsi_text}"
        f"{proj_text}"
    )


def generate_master_briefing(
    regime_data: dict,
    dashboard: pd.DataFrame,
    sizing: list,
    news_map: dict,
    prev_briefing,
) -> str:
    """
    Single synthesized morning note: argues WHY these picks, compares against alternatives,
    and builds off the previous day's recommendation.
    """
    if not sizing:
        return (
            "No qualified ETF candidates for the current market regime. "
            "All universe members either show negative 3-month momentum or the available budget is insufficient. "
            "The strategy requires positive 3-month returns before deploying capital — this prevents chasing "
            "assets in downtrends. The intraday scan may surface opportunities if conditions shift mid-day."
        )

    regime = regime_data['regime']
    vix = regime_data['vix']
    spy = regime_data['spy_price']
    ma200 = regime_data['ma200']
    today_symbols = [p['symbol'] for p in sizing]
    today_date = datetime.now().strftime('%B %d, %Y')
    paragraphs = []

    # ── Part 1: Opening — regime context with specifics ──────────────────────
    pct_vs_ma200 = ((spy / ma200) - 1) * 100
    if regime == 'bull':
        open_para = (
            f"**{today_date} — Bull regime. Full 50% allocation.**  \n"
            f"VIX at {vix:.1f} (fear is low — below the 20 threshold) and SPY at ${spy:.2f}, "
            f"which is **{pct_vs_ma200:+.1f}% above its 200-day moving average** of ${ma200:.2f}. "
            f"Both conditions hold simultaneously — this is the only configuration the strategy calls 'bull.' "
            f"Historically, momentum ETFs in growth sectors have led in exactly this environment: "
            f"investors are taking risk, institutions are chasing performance, and the underlying trend is up. "
            f"The strategy deploys 50% of portfolio value into the top momentum ETFs."
        )
    elif regime == 'bear':
        below_pct = ((ma200 / spy) - 1) * 100 if spy < ma200 else 0
        open_para = (
            f"**{today_date} — Defensive regime. Reduced 30% allocation.**  \n"
            f"{'VIX at ' + f'{vix:.1f}' + ' — above 25, signaling genuine investor fear. ' if vix > 25 else ''}"
            f"{'SPY at $' + f'{spy:.2f}' + ' has broken ' + f'{below_pct:.1f}%' + ' below its 200-day average ($' + f'{ma200:.2f}' + ') — the long-term uptrend is broken. ' if spy < ma200 else ''}"
            f"Capital preservation takes priority. The strategy rotates to assets that hold or gain "
            f"during equity downturns: long-term bonds, gold, healthcare, utilities, and consumer staples."
        )
    else:
        open_para = (
            f"**{today_date} — Neutral regime. Moderate 40% allocation.**  \n"
            f"VIX at {vix:.1f} and SPY within normal range of its 200-day average (${ma200:.2f}). "
            f"No clear bull or bear signal — the required conditions for either don't both hold. "
            f"The strategy considers the full 20-ETF universe and selects purely by momentum, "
            f"regardless of sector. Allocation is 40% of portfolio value."
        )
    paragraphs.append(open_para)

    # ── Part 2: Day-over-day continuity ─────────────────────────────────────
    if prev_briefing:
        prev_date, prev_symbols, prev_regime = prev_briefing
        added   = [s for s in today_symbols if s not in prev_symbols]
        dropped = [s for s in prev_symbols  if s not in today_symbols]

        if not added and not dropped:
            # Same picks — explain why the thesis still holds
            daily_notes = []
            for s in today_symbols:
                rows = dashboard[dashboard['symbol'] == s]
                if not rows.empty:
                    r1d = rows.iloc[0].get('return_1d', 0.0) * 100
                    daily_notes.append(f"{s} {'+' if r1d >= 0 else ''}{r1d:.2f}% yesterday")
            mo_str = "; ".join(daily_notes) if daily_notes else "signals held steady"

            regime_shift = ""
            if prev_regime != regime:
                regime_shift = (
                    f" **Regime note:** market shifted from {prev_regime.upper()} to {regime.upper()} "
                    f"since {prev_date} — portfolio allocation has been adjusted accordingly."
                )

            paragraphs.append(
                f"**Continuity from {prev_date}:** Same picks as yesterday — {', '.join(today_symbols)}. "
                f"Daily performance: {mo_str}. "
                f"Momentum rankings haven't shifted enough to trigger a rotation. The strategy only churns "
                f"when a meaningful rank change or regime shift occurs — not on a calendar.{regime_shift}"
            )
        else:
            change_parts = []
            for s in dropped:
                rows = dashboard[dashboard['symbol'] == s]
                if not rows.empty:
                    r = rows.iloc[0]
                    if r['return_3m'] <= 0:
                        why = "3-month momentum turned negative — no longer passes the quality floor"
                    elif r.get('return_1m', 0) < 0:
                        why = f"1-month momentum reversed ({r['return_1m']*100:+.1f}%)"
                    else:
                        why = "displaced to 4th or lower in momentum ranking"
                    change_parts.append(f"**{s} exits** ({why})")
                else:
                    change_parts.append(f"**{s} exits** (no longer in top picks)")
            for s in added:
                rows = dashboard[dashboard['symbol'] == s]
                if not rows.empty:
                    r = rows.iloc[0]
                    change_parts.append(
                        f"**{s} enters** — {r['return_3m']*100:+.1f}% over 3 months, "
                        f"{r['return_1m']*100:+.1f}% this month"
                    )
            paragraphs.append(
                f"**Rotation from {prev_date}:** {'. '.join(change_parts)}. "
                f"The strategy rotates when rankings shift materially — this is that moment."
            )

    # ── Part 3: Selection argument — why these over the alternatives ─────────
    qualified_all = dashboard[dashboard['return_3m'] > 0].copy()
    also_ran = qualified_all[~qualified_all['symbol'].isin(today_symbols)].head(4)

    top_str_parts = [
        f"**{p['symbol']}** ({p['return_3m']*100:+.1f}% 3m)" for p in sizing
    ]

    if not also_ran.empty:
        skip_parts = []
        for _, r in also_ran.iterrows():
            s = r['symbol']
            reasons = []
            if r['rsi'] > 72:
                reasons.append(f"RSI {r['rsi']:.0f}—overbought")
            if r['volume_trend'] == 'falling':
                reasons.append("falling volume")
            avg_pace = r['return_3m'] * 100 / 3
            r1m = r.get('return_1m', 0) * 100
            if r1m < avg_pace * 0.6:
                reasons.append(f"decelerating ({r1m:+.1f}%/mo vs {avg_pace:.1f}%/mo avg)")
            reason_str = "; ".join(reasons) if reasons else "ranked just below the cutoff"
            skip_parts.append(f"{s} ({r['return_3m']*100:+.1f}% 3m, {reason_str})")

        paragraphs.append(
            f"**From the full 20-ETF universe, today's top picks: {', '.join(top_str_parts)}.**  \n"
            f"The next-best candidates were considered but did not displace the leaders: "
            f"{'; '.join(skip_parts)}. "
            f"Holding a concentrated set (up to 5) keeps the portfolio exposed to the strongest signals "
            f"rather than diluting into marginal ideas."
        )
    else:
        paragraphs.append(
            f"**From the 20-ETF universe, today's picks: {', '.join(top_str_parts)}.**"
        )

    # ── Part 4: Per-pick bull case ───────────────────────────────────────────
    pick_parts = []
    for pos in sizing:
        s = pos['symbol']
        name = ETF_NAMES.get(s, s)
        ret_3m  = pos['return_3m'] * 100
        ret_1m  = pos['return_1m'] * 100
        ret_1d  = pos.get('return_1d', 0.0) * 100
        rsi     = pos['rsi']
        vol     = pos['volume_trend']

        avg_pace = ret_3m / 3
        if ret_1m > avg_pace * 1.1:
            velocity = f"**accelerating** (+{ret_1m:.1f}%/mo, ahead of the +{avg_pace:.1f}%/mo 3m average)"
        elif ret_1m > 0:
            velocity = f"positive but cooling (+{ret_1m:.1f}%/mo vs +{avg_pace:.1f}%/mo 3m average)"
        else:
            velocity = f"stalling ({ret_1m:+.1f}%/mo) despite positive 3m trend — watch closely"

        vol_note = {
            'rising':  'Volume rising — institutional participation is growing, confirming the move.',
            'falling': 'Volume falling — price moving on thin participation; mild caution flag.',
            'neutral': 'Volume steady.',
        }.get(vol, '')

        rsi_note = (
            f"RSI {rsi:.0f} — **overbought, elevated pullback risk**." if rsi > 70
            else f"RSI {rsi:.0f} — healthy momentum, room to run."
        )

        spy_rows = dashboard[dashboard['symbol'] == 'SPY']
        rel_note = ""
        if not spy_rows.empty:
            diff = ret_3m - spy_rows['return_3m'].iloc[0] * 100
            if abs(diff) > 1.5:
                rel_note = f" {'+' if diff > 0 else ''}{diff:.1f}% vs the S&P 500 over 3 months."

        holdings = ETF_TOP_HOLDINGS.get(s, [])
        h_note = ""
        if holdings:
            h_note = f" Key holdings: {', '.join(h[0] for h in holdings[:2])}."

        news_note = ""
        news = news_map.get(s, [])
        if news and news[0].get('title'):
            headline = news[0]['title'][:90].rstrip()
            pub = news[0].get('publisher', '')
            news_note = f" Latest headline: *\"{headline}...\"* ({pub})"

        pick_parts.append(
            f"**{s} — {name}:** {ret_3m:+.1f}% over 3 months{rel_note} "
            f"Momentum {velocity}. {vol_note} {rsi_note}{h_note}{news_note}"
        )

    if pick_parts:
        paragraphs.append("**The case for each pick:**\n\n" + "\n\n".join(pick_parts))

    # ── Part 5: Invalidation conditions ─────────────────────────────────────
    if regime == 'bull':
        inv = (
            f"**What would flip this recommendation:** VIX breaking above 20 (currently {vix:.1f}) "
            f"**or** SPY closing below its 200-day average at ${ma200:.2f} "
            f"(currently {pct_vs_ma200:+.1f}% above). "
            f"Either condition alone triggers a regime re-evaluation — if both break, "
            f"the system rotates to defensive ETFs."
        )
    elif regime == 'bear':
        inv = (
            f"**What would flip this recommendation:** Both VIX dropping below 25 (currently {vix:.1f}) "
            f"AND SPY recovering above ${ma200:.2f}. "
            f"One condition alone won't trigger a rotation — both must hold."
        )
    else:
        inv = (
            f"**What would change the regime:** Bull — VIX below 20 AND SPY above ${ma200:.2f}. "
            f"Bear — VIX above 25 AND SPY below ${ma200:.2f}. Currently neither threshold is fully met."
        )
    paragraphs.append(inv)

    return "\n\n".join(paragraphs)


# ── Stock screener constants ──────────────────────────────────────────────────

_STOCK_SETUP_EMOJI = {
    'Breakout from Base':             '⚡',
    'Earnings Gap / Post-Beat Drift': '📈',
    'Momentum Continuation':          '🚀',
    'First Pullback in Uptrend':      '↩️',
    'Pre-Earnings Setup':             '📅',
    'Deep Pullback to Support':       '🎯',
    'Momentum Setup':                 '📊',
}

_SETUP_EXPLANATION = {
    'Breakout from Base': (
        "The stock has been coiling in a tight consolidation (ATR contracting below 75% of its 3-month average) "
        "near its 52-week highs, and is now breaking out on elevated volume. "
        "This is the Volatility Contraction Pattern (VCP) — price tightens as sellers exhaust, then surges when buyers "
        "overwhelm remaining supply. The MA stack confirms the underlying uptrend; the breakout is an acceleration, not a new trend."
    ),
    'Earnings Gap / Post-Beat Drift': (
        "The stock gapped up 4%+ on 1.5× average volume within the past 10 days — a strong proxy for an earnings beat "
        "or major positive catalyst. Post-Earnings Announcement Drift (PEAD) is one of the most replicated anomalies in finance: "
        "the market doesn't fully reprice a beat on day one. Institutions spend 2–8 weeks building positions, analysts raise targets, "
        "and each revision adds fuel. The drift continues until the market fully digests the new earnings level."
    ),
    'Momentum Continuation': (
        "The stock is near its 52-week high with RSI in a healthy, non-overbought range — classic trend continuation. "
        "George & Hwang (2004) documented the 52-week high effect: investors anchor to prior highs as reference points "
        "and are slow to revise expectations upward past them. This anchoring creates a systematic underreaction that "
        "momentum traders exploit. Near-high stocks with healthy RSI continue outperforming for 1–12 months."
    ),
    'First Pullback in Uptrend': (
        "The stock made a strong initial thrust and is now digesting those gains — RSI has pulled back from an overbought read "
        "while price holds above the 20-day EMA. The 'first pullback in a new uptrend' is a classic swing entry: after an "
        "initial catalyst-driven move, momentum stocks consolidate before continuing. The MA stack is fully intact, "
        "confirming this is a healthy pause, not a trend break."
    ),
    'Pre-Earnings Setup': (
        "Technically strong with earnings expected within 7 days. If the company beats, this becomes a PEAD setup. "
        "Binary risk: a miss can erase weeks of gains overnight. The reward scenario is entering before the gap-up rather "
        "than chasing it. Requires careful position sizing — this is Tier C specifically because the outcome is 50/50 until earnings."
    ),
    'Deep Pullback to Support': (
        "Well below its 52-week high but still technically intact — the MA stack holds and 3-month return is positive. "
        "High failure rate setup. Only consider with a specific, well-researched fundamental thesis for why the selling is overdone. "
        "Do not enter without identifying a clear catalyst for reversal."
    ),
    'Momentum Setup': (
        "Passes all quality gates with aligned momentum signals across timeframes. Doesn't fit a tighter pattern category, "
        "but the characteristics are right: positive returns at every timeframe, RSI in healthy range, "
        "above-average volume participation, and the full MA stack intact."
    ),
}

_SETUP_RESEARCH_TIPS = {
    'Breakout from Base': [
        "**Base duration on a weekly chart:** How many weeks has this consolidation lasted? Bases of 4+ weeks built on high prior momentum produce the strongest breakouts (VCP stage analysis).",
        "**Volume dry-up before the break:** Look for a stretch of abnormally quiet volume just before today's action — this 'handle' signals sellers have been exhausted. No dry-up = lower conviction breakout.",
        "**Catalyst check:** Search '[company name] news 2026' — did something specific trigger this? A new contract, analyst upgrade, product launch, or insider purchase that explains *why* the base formed here makes the breakout more durable.",
        "**Sector ETF confirmation:** Check the relevant sector ETF on TradingView. A stock breaking out while its sector ETF is also near highs has a 2–3× higher follow-through rate than one bucking a weak sector.",
    ],
    'Earnings Gap / Post-Beat Drift': [
        "**Read the actual earnings release:** Find the press release on the company's IR page or SEC EDGAR (8-K filing). Look specifically at forward guidance vs. prior guidance — upward revisions sustain PEAD; in-line guidance often leads to 'buy the rumor, sell the news.'",
        "**Analyst estimate revisions:** After a beat, sell-side analysts raise price targets. Search '[symbol] analyst price target raised' — each upgrade triggers institutional buying. Multiple upgrades in a week is a strong signal.",
        "**Gap integrity (most important):** Has the stock held above the pre-earnings close for 3+ days? Gaps that get filled quickly are far less reliable than gaps that hold. Check the chart carefully.",
        "**Earnings call transcript:** Available free on Seeking Alpha or Motley Fool. The CEO's specific language about demand, pricing power, and margin trajectory matters more than the EPS number itself.",
    ],
    'Momentum Continuation': [
        "**Sector ETF relative strength:** Find this company's sector ETF and compare its 3-month chart. If both the individual stock AND the sector ETF are in uptrends, the trade has macro wind at its back.",
        "**Institutional ownership trends:** Check the most recent 13F filings on SEC EDGAR or Whale Wisdom (free). Is a major fund increasing its position? Sustained institutional buying is the primary engine of multi-month momentum.",
        "**Peer comparison:** Look up the top 3 direct competitors and compare 3-month returns. Is this stock genuinely leading the sector, or just following a tide that could turn?",
        "**Fundamental acceleration check:** Pull the last two earnings reports. Is revenue growth getting *faster* quarter over quarter? Price momentum backed by accelerating fundamentals is far more durable than price momentum alone.",
    ],
    'First Pullback in Uptrend': [
        "**Volume signature on the pullback:** The ideal pullback has below-average volume on down days and above-average volume on up days. This shows sellers aren't motivated and buyers are waiting. Check each day on the chart.",
        "**20-day EMA as support:** Is price bouncing from, or still above, the 20-day EMA? For momentum stocks, the 20 EMA is the primary support level — a clean touch-and-bounce is the textbook entry.",
        "**Original catalyst still intact?** What triggered the initial move 1–4 weeks ago? If it was an earnings beat, check if the new guidance is still being revised upward. If it was a product launch, is there new adoption data? The catalyst must still be live.",
        "**Daily range compression:** Look for candles getting progressively smaller (tight closes) before the bounce — ATR contracting on the pullback signals the stock is absorbing supply efficiently before its next move.",
    ],
    'Pre-Earnings Setup': [
        "**Implied move from options:** On Yahoo Finance, click the 'Options' tab. The near-term at-the-money straddle price ÷ stock price = the market's expected earnings move. Compare this to your stop distance — if the implied move is 10% and your stop is 6%, you're not fully protected.",
        "**Historical earnings reaction:** Search '[symbol] earnings history' on Macrotrends or Earnings Whispers. Has the stock gapped up on earnings 3 of the last 4 quarters? Consistent positive reactions increase odds.",
        "**Whisper number vs. consensus:** The published consensus EPS (on Yahoo Finance or Seeking Alpha) is often lower than what sophisticated traders expect. Check EarningsWhispers.com for the 'whisper' — if the company needs to beat the whisper, not just consensus, to gap up.",
        "**Half-size rule:** This is Tier C specifically because of binary event risk. Enter at 50% of your normal position size. If the stock gaps up on earnings, you can add to a full position. If it gaps down, your loss is controlled.",
    ],
    'Deep Pullback to Support': [
        "**Reason for the selloff (required before any entry):** Search '[symbol] why is it down' and read the actual news. Structural selling (revenue declining, business model disruption, management scandal) means the support will likely break. Only trade this if you can identify a specific reason the selling was overdone or temporary.",
        "**Multi-year support level:** Pull up a 3-year or 5-year chart. Is price sitting on a level that has held multiple times over multiple years? One-touch support levels often fail. Multi-touch, multi-year support is far more reliable.",
        "**Short interest context:** High short interest on a deep pullback can work both ways — it can cause a squeeze if a catalyst emerges, or it means smart money sees structural problems. Research why the shorts are there.",
        "**Catalyst required:** Deep pullback reversals almost always require a specific, newsworthy catalyst — analyst upgrade, earnings surprise, buyback announcement, or sector rotation catalyst. Don't enter purely on the chart.",
    ],
    'Momentum Setup': [
        "**What's driving it?** Search '[company name] news 2026' and look for any recent catalyst — earnings, contracts, partnerships, product launches. Momentum with a story is more durable than pure price momentum.",
        "**Peer performance comparison:** Find 3 direct competitors and compare 3-month charts. Is this stock genuinely leading, or is the whole sector moving and this stock is just along for the ride?",
        "**Weekly chart trend quality:** View the weekly chart on TradingView. Is this a clean, consistent uptrend across 6–12 months? Or is it a recent spike within a longer choppy range? Clean weekly uptrends are far more reliable.",
        "**Unusual options activity:** Check Barchart.com's 'Unusual Options Activity' tab for this symbol. Large institutional options positions often precede significant price moves and can confirm or contradict the technical setup.",
    ],
}


def _nv(v) -> bool:
    return v is not None and not (isinstance(v, float) and v != v)


# ── Stock screener narrative generators ──────────────────────────────────────

def generate_stock_intro_narrative(candidates: list, regime_data: dict) -> str:
    if not candidates:
        return (
            "No screener results found for today. "
            "Run `python run.py stocks` in the terminal to generate candidates "
            "(3–5 minutes on first run, instant from cache after)."
        )

    regime    = regime_data.get('regime', 'neutral')
    vix       = regime_data.get('vix', 0)
    spy       = regime_data.get('spy_price', 0)
    ma200     = regime_data.get('ma200', 1)
    pct_above = ((spy / ma200) - 1) * 100 if ma200 else 0

    high_cs = [c for c in candidates if c.get('tier_label') == 'HIGH']
    mid_cs  = [c for c in candidates if c.get('tier_label') == 'MID']
    low_cs  = [c for c in candidates if c.get('tier_label') == 'LOW']

    if regime == 'bull':
        regime_para = (
            f"**Bull regime — ideal environment for small/mid cap momentum.** "
            f"VIX at {vix:.1f} (below 20 — investors are calm and risk-seeking) and SPY is "
            f"**{pct_above:+.1f}% above its 200-day moving average**. "
            f"This is the textbook configuration for small and mid-cap momentum strategies: "
            f"institutions are in full risk-on mode, credit is accessible to smaller companies, "
            f"and the broad market tailwind reduces individual downside risk. "
            f"Fama-French research shows the small-cap size premium compounds most powerfully in "
            f"exactly this environment — when risk appetite is high and the macro trend is intact. "
            f"The highest-conviction setups today (Tier A) deserve full position sizing."
        )
    elif regime == 'bear':
        regime_para = (
            f"**Defensive regime — elevated caution on small/mid caps.** "
            f"Current conditions (VIX {vix:.1f}, SPY under pressure) are unfavorable for smaller companies. "
            f"Small and mid-caps carry disproportionate downside in bear markets: less pricing power, "
            f"thinner margins, restricted credit access, and lower liquidity makes exits costly when you need them. "
            f"If you trade these setups: max 3% of portfolio per position, tighter stops, no averaging down. "
            f"Tier A setups (PEAD and Breakout from Base) have the best odds even in rough markets "
            f"because they're driven by company-specific catalysts, not macro tailwinds."
        )
    else:
        regime_para = (
            f"**Neutral regime — be selective on small/mid caps.** "
            f"VIX at {vix:.1f} with SPY near its 200-day average ({pct_above:+.1f}%). Mixed macro signals. "
            f"In a neutral regime, individual stock selection matters far more than normal — "
            f"the broad market isn't going to bail out mediocre setups. "
            f"Focus on Tier A candidates with clear, identifiable catalysts. "
            f"Avoid Tier C setups entirely until the macro picture resolves."
        )

    # Sector concentration in HIGH tier
    high_sectors = Counter(
        c.get('sector', 'Unknown') for c in high_cs
        if c.get('sector') not in ('Unknown', None, '')
    )
    sector_obs = ""
    if high_sectors and high_cs:
        top_sec, top_count = high_sectors.most_common(1)[0]
        sec_etf = STOCK_SECTOR_ETF.get(top_sec, '')
        sector_obs = (
            f" **{top_count} of {len(high_cs)} HIGH-tier candidates are in {top_sec}**"
            + (f" — check {sec_etf} for sector-level confirmation before entering" if sec_etf else "")
            + "."
        )

    # Setup distribution
    top_setups = Counter(c.get('setup_name', '') for c in candidates if c.get('setup_name')).most_common(2)
    setup_obs = ""
    if top_setups:
        parts = [f"**{s}** ({n} stocks)" for s, n in top_setups]
        setup_obs = f" Dominant setups: {' · '.join(parts)}."

    summary_para = (
        f"**{len(candidates)} candidates passed today's screen** — "
        f"{len(high_cs)} HIGH · {len(mid_cs)} MID · {len(low_cs)} LOW.  \n"
        f"Universe: S&P MidCap 400 + SmallCap 600 (~1,000 stocks).  \n"
        f"Hard gates: price ≥ $5, dollar volume ≥ $5M/day, full MA stack "
        f"(price > 20 EMA > 50 EMA > 200 SMA), RSI 35–82, 3-month return > 0, RVOL ≥ 1.0×.  \n"
        f"Scoring (10 pts): RS vs SPY (2pt) + 12-1M momentum (2pt) + 52W proximity (2pt) "
        f"+ relative volume (2pt) + setup tier A/B/C (2pt)."
        f"{sector_obs}{setup_obs}"
    )

    return regime_para + "\n\n" + summary_para


def generate_stock_candidate_narrative(c: dict, news: list) -> str:
    sym      = c.get('symbol', '?')
    name     = (c.get('name') or sym)
    setup    = c.get('setup_name', 'Momentum Setup')
    tier     = c.get('setup_tier', 'B')
    sector   = c.get('sector', 'Unknown')
    cap      = c.get('market_cap') or 0

    rsi      = c.get('rsi') or 0
    rvol     = c.get('rvol') or 1.0
    atr_pct  = c.get('atr_pct') or 0
    prox     = c.get('prox_52w') or 0
    rs_spy   = c.get('rs_spy')
    mom      = c.get('mom_12_1')
    ret_1m   = c.get('ret_1m') or 0
    ret_3m   = c.get('ret_3m') or 0
    vol_trend = c.get('vol_trend', 'neutral')

    earnings_flag = c.get('earnings_soon', False)
    short_pct     = (c.get('short_float') or 0) * 100
    short_flag    = c.get('short_flag', False)

    tier_labels = {
        'A': 'Tier A — highest conviction (institutional-grade structure)',
        'B': 'Tier B — solid momentum setup',
        'C': 'Tier C — lower conviction, requires strong qualitative justification',
    }

    # Setup explanation
    explanation = _SETUP_EXPLANATION.get(setup, _SETUP_EXPLANATION['Momentum Setup'])

    # Signal synthesis
    signal_parts = []

    if _nv(rs_spy):
        if rs_spy > 0.10:
            signal_parts.append(
                f"**RS vs SPY (3m): +{rs_spy*100:.1f}%** — major outperformance of the index. "
                f"This level of relative strength typically indicates institutional accumulation — "
                f"funds are choosing this stock over the S&P 500."
            )
        elif rs_spy > 0.03:
            signal_parts.append(f"**RS vs SPY (3m): +{rs_spy*100:.1f}%** — outperforming the S&P 500, a positive momentum signal.")
        else:
            signal_parts.append(
                f"**RS vs SPY (3m): {rs_spy*100:+.1f}%** — lagging the index slightly despite passing gates. "
                f"Positive momentum but not leading — watch whether this improves."
            )

    if _nv(mom):
        if mom > 0.20:
            signal_parts.append(
                f"**12-1M Momentum: +{mom*100:.1f}%** — strong academic momentum signal. "
                f"Jegadeesh & Titman (1993): stocks in the top quintile of 12-1 month returns "
                f"continue to outperform for the next 3–12 months. This is one of the most replicated factors in finance."
            )
        elif mom > 0.05:
            signal_parts.append(f"**12-1M Momentum: +{mom*100:.1f}%** — positive momentum factor (returns from 12 months ago to 1 month ago).")
        elif _nv(mom) and mom <= 0.05:
            signal_parts.append(f"**12-1M Momentum: {mom*100:+.1f}%** — weak historical momentum. Recent performance is stronger than the 12-month picture suggests.")

    if prox >= 0.97:
        signal_parts.append(
            f"**52W Proximity: {prox*100:.1f}% of high** — essentially at all-time area highs. "
            f"The George & Hwang (2004) anchoring effect: the market systematically underreacts to stocks near highs, "
            f"creating a continuation bias that is strongest within 3% of the 52-week high."
        )
    elif prox >= 0.88:
        signal_parts.append(f"**52W Proximity: {prox*100:.1f}% of high** — within 12% of highs, confirmed uptrend intact.")

    vol_desc = {
        'rising':  f"**RVOL {rvol:.1f}× — volume rising** (above-average participation confirms the move; institutional buying is visible).",
        'neutral': f"**RVOL {rvol:.1f}× — volume neutral** (steady participation; neither confirming nor contradicting the trend).",
        'falling': f"**RVOL {rvol:.1f}× — volume falling** (below-average participation; the move is happening on thin volume — mild caution flag).",
    }
    signal_parts.append(vol_desc.get(vol_trend, f"**RVOL: {rvol:.1f}×**"))

    # Risk flags
    risk_parts = []
    if earnings_flag:
        risk_parts.append(
            "**⚠️ Earnings within 7 days** — binary event risk. "
            "A miss can erase weeks of gains in a single session. "
            "Consider entering at 50% of normal position size, or waiting for the post-earnings reaction to confirm direction."
        )
    if short_flag:
        risk_parts.append(
            f"**⚠️ Short interest: {short_pct:.0f}% of float** — above the 15% warning threshold. "
            f"Heavy short interest means sophisticated investors see problems. "
            f"It can also fuel a short squeeze if a positive catalyst emerges — but understand *why* the shorts are there before trading against them."
        )
    if rsi > 72:
        risk_parts.append(
            f"**⚠️ RSI {rsi:.0f} — overbought.** "
            f"The stock has risen faster than its 14-day average pace. "
            f"Momentum continues in overbought territory, but short-term pullback risk is elevated. "
            f"The ATR-based stop accounts for this by giving the trade room to breathe."
        )
    if atr_pct > 0.055:
        risk_parts.append(
            f"**High daily volatility (ATR {atr_pct*100:.1f}%/day).** "
            f"Normal daily swings of this magnitude require wider stops. "
            f"Position size down proportionally so your maximum loss on a stop-out is within your normal risk budget."
        )

    # News
    news_lines = [f"- *{a['title']}* — {a.get('publisher', '')}" for a in (news or [])[:3] if a.get('title')]

    # Research tips
    tips = _SETUP_RESEARCH_TIPS.get(setup, _SETUP_RESEARCH_TIPS['Momentum Setup'])

    # Qualitative investigation questions
    first_name = name.split()[0] if name and name != sym else sym
    cap_label  = f"${cap/1e9:.1f}B market cap" if cap >= 1e9 else f"${cap/1e6:.0f}M market cap" if cap > 0 else "market cap unknown"
    qual_questions = [
        f"**Leadership:** Who is {sym}'s CEO and how long have they been in the role? "
        f"CEOs 1–4 years into a tenure with a clear growth thesis often drive the strongest multi-quarter momentum. "
        f"Search '{name} CEO' and read their most recent investor conference presentation.",
        f"**Competitive moat:** What does {sym} do that competitors can't easily replicate? "
        f"For a {cap_label} company in {sector}, look for pricing power, switching costs, or proprietary technology. "
        f"The company's 10-K 'Business' section (free on SEC EDGAR) is the best source.",
        f"**Sector macro tailwind:** Is **{sector}** currently in a favorable macro environment? "
        f"For example: technology benefits from AI capex; energy from oil prices; financials from rates; "
        f"healthcare is defensive in any regime. Check if the macro environment is accelerating or decelerating for this sector.",
        f"**Revenue acceleration:** Pull the last 3 earnings reports from the IR page or Seeking Alpha. "
        f"Is quarterly revenue growth *speeding up* (20% → 25% → 30%), steady, or slowing down? "
        f"Accelerating revenue growth is the single strongest fundamental predictor of continued price momentum.",
        f"**Competitor comparison:** Name {sym}'s top 3 direct competitors and check their stock performance over 3 months. "
        f"If {sym} is dramatically outperforming its peer group, that signals company-specific alpha, not just sector tailwind — "
        f"company-specific momentum is more durable.",
    ]

    lines = [
        f"### {setup} — {tier_labels.get(tier, tier)}",
        "",
        explanation,
        "",
        "**Signal breakdown:**",
    ]
    lines += [f"- {s}" for s in signal_parts]

    if risk_parts:
        lines += ["", "**Risk considerations:**"]
        lines += [f"- {r}" for r in risk_parts]

    if news_lines:
        lines += ["", "**Recent headlines:**"]
        lines += news_lines

    lines += ["", "**Research checklist for this setup:**"]
    lines += [f"{i+1}. {tip}" for i, tip in enumerate(tips)]

    lines += ["", "**Qualitative investigation (do this before trading):**"]
    lines += [f"- {q}" for q in qual_questions]

    return "\n".join(lines)


# ── App ──────────────────────────────────────────────────────────────────────

init_db()
st.title("📈 PLEC Trading Dashboard")
st.caption(f"Last updated: {datetime.now().strftime('%B %d, %Y — %I:%M %p')}")

try:
    acct = load_account()
    positions = get_positions()
    current_symbols = {p.symbol for p in positions}
except Exception as e:
    st.error(f"Could not load account: {e}")
    acct = None
    positions = []
    current_symbols = set()

try:
    regime_data = load_regime()
    regime = regime_data['regime']
except Exception as e:
    st.error(f"Regime error: {e}")
    regime = 'neutral'
    regime_data = {'regime': 'neutral', 'vix': 0, 'spy_price': 0, 'ma50': 1, 'ma200': 1}

try:
    dashboard = load_dashboard_data(regime)
except Exception:
    dashboard = pd.DataFrame()

# Account row — always visible above tabs
if acct:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Portfolio Value", f"${acct['portfolio_value']:,.2f}")
    c2.metric("Cash",            f"${acct['cash']:,.2f}")
    c3.metric("Equity",          f"${acct['equity']:,.2f}")
    c4.metric("Open Positions",  len(positions))

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "🌅 Morning Briefing",
    "📊 ETF Rankings",
    "🔍 ETF Deep Dive",
    "💼 Positions",
    "📜 Trade History",
    "📋 Stock Screener",
])


# ── TAB 1: Morning Briefing ──────────────────────────────────────────────────
with tab1:
    # Prior day
    st.markdown('<div class="section-title">📰 Prior Day Summary</div>', unsafe_allow_html=True)
    try:
        prior = load_prior_day()
        st.markdown(f'<div class="narrative-box">{generate_prior_day_narrative(prior)}</div>', unsafe_allow_html=True)
    except Exception as e:
        st.caption(f"Prior day data unavailable: {e}")

    st.divider()

    # Regime
    st.markdown('<div class="section-title">🌡️ Market Regime</div>', unsafe_allow_html=True)
    regime_colors = {'bull': '#00d4aa', 'bear': '#ff4b4b', 'neutral': '#ffd700'}
    regime_labels = {'bull': '🟢 BULL MARKET', 'bear': '🔴 DEFENSIVE MODE', 'neutral': '🟡 NEUTRAL'}
    color = regime_colors[regime]
    r1, r2, r3, r4 = st.columns([2, 1, 1, 1])
    r1.markdown(f"<div style='font-size:1.6rem;font-weight:bold;color:{color};padding-top:8px'>{regime_labels[regime]}</div>", unsafe_allow_html=True)
    r2.metric("VIX", f"{regime_data['vix']:.1f}")
    r3.metric("SPY vs 200MA", f"{((regime_data['spy_price'] / regime_data['ma200']) - 1) * 100:+.1f}%")
    r4.metric("SPY vs 50MA",  f"{((regime_data['spy_price'] / regime_data['ma50'])  - 1) * 100:+.1f}%")
    st.markdown(f'<div class="narrative-box">{generate_regime_narrative(regime_data)}</div>', unsafe_allow_html=True)

    st.divider()

    # Recommendations
    st.markdown('<div class="section-title">✅ Today\'s Recommendation</div>', unsafe_allow_html=True)
    st.caption("Synthesized from all signals: momentum, RSI, volume, market regime, holdings, and recent news.")
    try:
        if acct and not dashboard.empty:
            sizing = get_position_sizing(dashboard, acct['portfolio_value'], regime)

            # Persist today's picks and load yesterday's for day-over-day comparison
            today_date_str = datetime.now().strftime('%Y-%m-%d')
            today_symbols  = [p['symbol'] for p in sizing]
            prev_briefing  = None
            if sizing:
                try:
                    save_daily_briefing(today_date_str, today_symbols, regime)
                    prev_briefing = get_previous_briefing(today_date_str)
                except Exception:
                    pass

            # Build news map for all picks in one pass
            news_map = {p['symbol']: load_etf_news(p['symbol']) for p in sizing}

            if not sizing:
                st.warning(
                    "No viable candidates — no ETFs with positive 3-month momentum qualify in the current regime. "
                    "Check the ETF Rankings tab for what's in the universe."
                )
            else:
                # ── Master synthesis ────────────────────────────────────────────
                master = generate_master_briefing(regime_data, dashboard, sizing, news_map, prev_briefing)
                st.markdown(f'<div class="narrative-box">{master}</div>', unsafe_allow_html=True)

                st.divider()
                st.markdown("**Position details** — entry, stop-loss, take-profit, and size:")

                # ── Compact per-position cards ──────────────────────────────────
                for pos in sizing:
                    symbol = pos['symbol']
                    name   = ETF_NAMES.get(symbol, symbol)
                    held   = symbol in current_symbols
                    label  = f"✅ {symbol} — {name} *(currently held)*" if held else f"**{symbol} — {name}**"
                    st.markdown(label)

                    stop_pct = (pos['price'] - pos['stop_price']) / pos['price']
                    tp_pct   = (pos['take_profit_price'] - pos['price']) / pos['price']
                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("Entry",       f"~${pos['price']:.2f}")
                    c2.metric("Stop-Loss",   f"${pos['stop_price']:.2f}",        delta=f"-{stop_pct*100:.1f}%", delta_color="inverse")
                    c3.metric("Take-Profit", f"${pos['take_profit_price']:.2f}", delta=f"+{tp_pct*100:.1f}%")
                    c4.metric("Size",        f"{int(pos['qty'])} sh · ${pos['allocation_usd']:,.0f}")

                    # Detailed narrative available on demand
                    with st.expander(f"📊 Full signal breakdown: {symbol}"):
                        narrative = generate_recommendation_narrative(pos, dashboard, news_map.get(symbol, []))
                        st.markdown(narrative)
                    st.markdown("")
    except Exception as e:
        st.error(f"Recommendation error: {e}")


# ── TAB 2: ETF Rankings ──────────────────────────────────────────────────────
with tab2:
    st.markdown('<div class="section-title">📊 ETF Signal Dashboard</div>', unsafe_allow_html=True)
    st.markdown("Each column is an independent signal. Read them together; they are **not** combined into a score.")

    with st.expander("📖 Column guide"):
        st.markdown("""
| Column | What it measures | How to read it |
|--------|-----------------|----------------|
| **Today** | Return since yesterday's close | The immediate picture — what the market thinks right now |
| **1W** | Return over the past 5 trading days | Is the short-term trend aligned with the longer one? |
| **1M / 3M / 6M** | Monthly, quarterly, semi-annual return | Sustained momentum — all three trending up is the strongest signal |
| **RSI** | Speed of recent price moves, 0–100 scale | <30 oversold, 30–70 normal, >70 overbought (risen too fast) |
| **Volume** | Recent vs. 20-day baseline activity | Rising = more buyers joining; falling = thinning participation |
| **Stop** | ATR-derived stop distance | Wider for volatile ETFs — accounts for normal fluctuation |
| **TP** | Take-profit target | 2.5× the stop distance, for a 2.5:1 reward-to-risk ratio |
""")

    try:
        if not dashboard.empty:
            has_short = 'return_1d' in dashboard.columns
            cols = ['rank', 'symbol']
            if has_short:
                cols += ['return_1d', 'return_5d']
            cols += ['return_1m', 'return_3m', 'return_6m', 'rsi', 'volume_trend', 'stop_loss_pct', 'take_profit_pct']
            display = dashboard[cols].copy()
            display.insert(2, 'name', display['symbol'].map(lambda s: ETF_NAMES.get(s, '')))

            pct_cols = [c for c in ['return_1d', 'return_5d', 'return_1m', 'return_3m', 'return_6m', 'stop_loss_pct', 'take_profit_pct'] if c in display.columns]
            for col in pct_cols:
                display[col] = display[col].apply(lambda x: f"{x*100:+.1f}%")
            display['rsi'] = display['rsi'].apply(lambda x: f"{x:.1f}")

            rename = {'rank': 'Rank', 'symbol': 'ETF', 'name': 'Name',
                      'return_1d': 'Today', 'return_5d': '1W', 'return_1m': '1M',
                      'return_3m': '3M', 'return_6m': '6M', 'rsi': 'RSI',
                      'volume_trend': 'Volume', 'stop_loss_pct': 'Stop', 'take_profit_pct': 'TP'}
            display.rename(columns=rename, inplace=True)

            def style_table(row):
                styles = [''] * len(row)
                cols_list = list(row.index)
                try:
                    rsi_i = cols_list.index('RSI')
                    rsi_val = float(row['RSI'])
                    if rsi_val > 70:
                        styles[rsi_i] = 'background-color:#5c1f1f; color:#ff9999; font-weight:bold'
                    elif rsi_val < 30:
                        styles[rsi_i] = 'background-color:#1f3a2a; color:#99ffcc; font-weight:bold'
                except Exception:
                    pass
                for col in ['Today', '1W', '1M', '3M', '6M']:
                    try:
                        i = cols_list.index(col)
                        v = float(row[col].replace('%', '').replace('+', ''))
                        if v > 5:    styles[i] = 'color:#00d4aa; font-weight:bold'
                        elif v > 0:  styles[i] = 'color:#7ecb9a'
                        else:        styles[i] = 'color:#ff6b6b'
                    except Exception:
                        pass
                return styles

            st.dataframe(display.style.apply(style_table, axis=1), use_container_width=True, hide_index=True)

            overbought = dashboard[dashboard['rsi'] > 70].head(3)
            if not overbought.empty:
                st.markdown("**⚠️ RSI flags — overbought territory:**")
                for _, row in overbought.iterrows():
                    rsi = row['rsi']
                    s = row['symbol']
                    n = ETF_NAMES.get(s, s)
                    if rsi > 75:
                        msg = (f"**{s} ({n}) — RSI {rsi:.1f}:** Significantly overbought. The ETF has risen faster "
                               f"than its 14-day average pace. Momentum can persist, but pullback risk is elevated.")
                    else:
                        msg = f"**{s} ({n}) — RSI {rsi:.1f}:** Modestly overbought. Strong momentum, just above the warning threshold."
                    st.markdown(f'<div class="warning-box">{msg}</div>', unsafe_allow_html=True)
        else:
            st.caption("No signal data available.")
    except Exception as e:
        st.error(f"Rankings error: {e}")


# ── TAB 3: ETF Deep Dive ─────────────────────────────────────────────────────
with tab3:
    st.markdown('<div class="section-title">🔍 ETF Deep Dive</div>', unsafe_allow_html=True)

    selected = st.selectbox(
        "Select an ETF",
        options=list(ETF_NAMES.keys()),
        format_func=lambda s: f"{s} — {ETF_NAMES.get(s, s)}",
    )

    try:
        etf_row = None
        if not dashboard.empty and selected in dashboard['symbol'].values:
            etf_row = dashboard[dashboard['symbol'] == selected].iloc[0]

        # Key metrics
        if etf_row is not None:
            has_short = 'return_1d' in dashboard.columns
            m1, m2, m3, m4, m5, m6 = st.columns(6)
            m1.metric("Price",   f"${etf_row['price']:.2f}")
            m2.metric("Today",   f"{etf_row.get('return_1d', 0)*100:+.2f}%")
            m3.metric("1 Week",  f"{etf_row.get('return_5d', 0)*100:+.2f}%")
            m4.metric("1 Month", f"{etf_row['return_1m']*100:+.2f}%")
            m5.metric("RSI",     f"{etf_row['rsi']:.1f}")
            m6.metric("Volume",  etf_row['volume_trend'].capitalize())

        # Load daily history
        df_d = load_daily_history(selected, period='3mo')

        if not df_d.empty:
            close = df_d['Close']
            ma20  = close.rolling(20).mean()
            ma50  = close.rolling(50).mean()
            rsi_s = _rsi_series(close)

            # Linear trend projection (last 20 trading days → next 5)
            last20_close = close.dropna().values[-20:]
            last20_dates = df_d.index[-20:]
            x_fit  = np.arange(20)
            coeffs = np.polyfit(x_fit, last20_close, 1)
            trend_hist_y = np.polyval(coeffs, x_fit)
            proj_dates   = pd.bdate_range(df_d.index[-1], periods=7)[1:]
            proj_y       = np.polyval(coeffs, np.arange(20, 26))

            # Stop / TP levels if this position is held
            open_trades_db = get_open_trades(module='etf')
            trade_map = {t.symbol: t for t in open_trades_db}

            # ─ Daily chart: price + MAs + trend + volume ─
            st.markdown(f"**{selected} — 3-Month Daily** (20d/50d MAs, linear trend projection)")
            fig = make_subplots(rows=2, cols=1, row_heights=[0.75, 0.25],
                                shared_xaxes=True, vertical_spacing=0.04)

            fig.add_trace(go.Scatter(x=df_d.index, y=close, name='Price',
                line=dict(color='#4a90e2', width=2),
                hovertemplate='%{x|%b %d}<br><b>$%{y:.2f}</b><extra></extra>'), row=1, col=1)

            fig.add_trace(go.Scatter(x=df_d.index, y=ma20, name='20d MA',
                line=dict(color='#ffd700', width=1.5, dash='dash'), opacity=0.85), row=1, col=1)

            fig.add_trace(go.Scatter(x=df_d.index, y=ma50, name='50d MA',
                line=dict(color='#ff9500', width=1.5, dash='dot'), opacity=0.85), row=1, col=1)

            # Trend: history + projection joined
            fig.add_trace(go.Scatter(
                x=list(last20_dates) + list(proj_dates),
                y=list(trend_hist_y) + list(proj_y),
                name='20d linear trend',
                line=dict(color='rgba(200,200,255,0.55)', width=1.5, dash='longdash'),
                hovertemplate='Trend: $%{y:.2f}<extra></extra>',
            ), row=1, col=1)

            # Vertical line: today / start of projection
            fig.add_vline(x=df_d.index[-1], line_dash='dash', line_color='rgba(255,255,255,0.12)')

            # Stop / TP if held
            if selected in trade_map:
                t = trade_map[selected]
                if t.stop_loss:
                    fig.add_hline(y=t.stop_loss, line_color='rgba(255,75,75,0.7)', line_dash='dash',
                                  annotation_text=f'Stop ${t.stop_loss:.2f}', row=1, col=1)
                if t.take_profit:
                    fig.add_hline(y=t.take_profit, line_color='rgba(0,212,170,0.7)', line_dash='dash',
                                  annotation_text=f'Target ${t.take_profit:.2f}', row=1, col=1)

            # Volume bars (green = up day, red = down day)
            vol_colors = ['#00d4aa' if c >= o else '#ff4b4b'
                          for c, o in zip(df_d['Close'], df_d['Open'])]
            fig.add_trace(go.Bar(x=df_d.index, y=df_d['Volume'],
                name='Volume', marker_color=vol_colors, showlegend=False,
                hovertemplate='Vol: %{y:,.0f}<extra></extra>'), row=2, col=1)

            fig.update_layout(**_PLOT, height=460, hovermode='x unified',
                yaxis_title='Price ($)', yaxis2_title='Volume',
                legend=dict(orientation='h', yanchor='bottom', y=1.02, x=0),
                margin=dict(t=20, b=10))
            fig.update_xaxes(**_GRID)
            fig.update_yaxes(**_GRID)
            st.plotly_chart(fig, use_container_width=True)

            # ─ RSI chart ─
            st.markdown("**RSI (14-day)** — >70 overbought, <30 oversold. Dashed line shows today.")
            fig_rsi = go.Figure()
            fig_rsi.add_trace(go.Scatter(x=df_d.index, y=rsi_s, name='RSI',
                line=dict(color='#a855f7', width=2),
                hovertemplate='%{x|%b %d}  RSI: %{y:.1f}<extra></extra>'))
            fig_rsi.add_hline(y=70, line_color='rgba(255,75,75,0.55)', line_dash='dash',
                              annotation_text='Overbought (70)', annotation_position='bottom right')
            fig_rsi.add_hline(y=30, line_color='rgba(0,212,170,0.55)', line_dash='dash',
                              annotation_text='Oversold (30)', annotation_position='top right')
            fig_rsi.add_hrect(y0=70, y1=100, fillcolor='rgba(255,75,75,0.04)', line_width=0)
            fig_rsi.add_hrect(y0=0,  y1=30,  fillcolor='rgba(0,212,170,0.04)', line_width=0)
            fig_rsi.update_layout(**_PLOT, height=220, showlegend=False,
                yaxis=dict(range=[0, 100], **_GRID), xaxis=_GRID, margin=dict(t=10, b=10))
            st.plotly_chart(fig_rsi, use_container_width=True)

        # ─ 5-day hourly chart ─
        st.markdown("**Last 5 Days — Hourly** (intraday price movement)")
        try:
            df_h = load_hourly_history(selected)
            if not df_h.empty:
                fig_h = make_subplots(rows=2, cols=1, row_heights=[0.75, 0.25],
                                      shared_xaxes=True, vertical_spacing=0.04)
                fig_h.add_trace(go.Scatter(x=df_h.index, y=df_h['Close'], name='Price',
                    line=dict(color='#4a90e2', width=1.5),
                    hovertemplate='%{x|%b %d %I%p}<br><b>$%{y:.2f}</b><extra></extra>'), row=1, col=1)
                h_colors = ['#00d4aa' if c >= o else '#ff4b4b'
                            for c, o in zip(df_h['Close'], df_h['Open'])]
                fig_h.add_trace(go.Bar(x=df_h.index, y=df_h['Volume'],
                    marker_color=h_colors, showlegend=False,
                    hovertemplate='Vol: %{y:,.0f}<extra></extra>'), row=2, col=1)
                fig_h.update_layout(**_PLOT, height=300, hovermode='x unified',
                    yaxis_title='Price ($)', yaxis2_title='Volume',
                    showlegend=False, margin=dict(t=10, b=10))
                fig_h.update_xaxes(**_GRID, tickformat='%b %d\n%I%p')
                fig_h.update_yaxes(**_GRID)
                st.plotly_chart(fig_h, use_container_width=True)
        except Exception as e:
            st.caption(f"Hourly data unavailable: {e}")

        # ─ Recent OHLCV table ─
        if not df_d.empty:
            with st.expander("📋 Recent daily OHLCV — last 10 trading days"):
                recent = df_d.tail(11)[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
                recent['Daily Chg'] = recent['Close'].pct_change().apply(
                    lambda x: f"{x*100:+.2f}%" if pd.notna(x) else '—')
                recent['Volume'] = recent['Volume'].apply(lambda x: f"{x:,.0f}")
                for col in ['Open', 'High', 'Low', 'Close']:
                    recent[col] = recent[col].apply(lambda x: f"${x:.2f}")
                recent.index = recent.index.strftime('%a %b %d')
                recent = recent.iloc[::-1].iloc[:10]
                st.dataframe(recent, use_container_width=True)

        # ─ Holdings & news context ─
        holdings = ETF_TOP_HOLDINGS.get(selected, [])
        news_items = load_etf_news(selected)

        if holdings or news_items:
            with st.expander(f"📰 Context: {selected} holdings & recent news"):
                if holdings:
                    h_str = " · ".join(f"**{h[0]}** ({h[1]:.0f}%)" for h in holdings[:5])
                    st.markdown(f"**Top holdings:** {h_str}")
                if news_items:
                    st.markdown("**Recent news:**")
                    for a in news_items:
                        if a.get('title'):
                            st.markdown(f"- {a['title']} — *{a.get('publisher', '')}*")

    except Exception as e:
        st.error(f"Deep dive error: {e}")


# ── TAB 4: Positions & Portfolio ─────────────────────────────────────────────
with tab4:
    st.markdown('<div class="section-title">💼 Open Positions</div>', unsafe_allow_html=True)

    if positions:
        open_trades_db = get_open_trades(module='etf')

        # Group by symbol — one symbol can have multiple bracket orders (separate sessions)
        trade_groups: dict[str, list] = defaultdict(list)
        for t in open_trades_db:
            trade_groups[t.symbol].append(t)

        # Worst-case simultaneous stop-out: sum (entry - stop) × qty across all open orders
        worst_case_loss = sum(
            (t.entry_price - t.stop_loss) * t.qty
            for t in open_trades_db
            if t.entry_price and t.stop_loss
        )
        portfolio_value = acct['portfolio_value'] if acct else 1.0
        worst_case_pct  = worst_case_loss / portfolio_value * 100

        wc1, wc2, wc3 = st.columns(3)
        wc1.metric(
            "Worst-Case Stop-Out",
            f"-${worst_case_loss:,.0f}",
            delta=f"-{worst_case_pct:.1f}% of portfolio",
            delta_color="inverse",
        )
        wc2.metric("Open Orders", len(open_trades_db))
        wc3.metric("Open Positions", len(positions))

        rows = []
        for p in positions:
            pnl     = float(p.unrealized_pl)
            pnl_pct = float(p.unrealized_plpc) * 100
            current = float(p.current_price)

            # Multiple orders per symbol: show tightest stop (highest price) and nearest TP (lowest price)
            trades = trade_groups.get(p.symbol, [])
            stop_vals = [t.stop_loss   for t in trades if t.stop_loss]
            tp_vals   = [t.take_profit for t in trades if t.take_profit]
            stop = max(stop_vals) if stop_vals else None
            tp   = min(tp_vals)   if tp_vals   else None

            rows.append({
                'Symbol':    p.symbol,
                'Name':      ETF_NAMES.get(p.symbol, p.symbol),
                'Shares':    int(float(p.qty)),
                'Entry':     f"${float(p.avg_entry_price):.2f}",
                'Current':   f"${current:.2f}",
                'Value':     f"${float(p.market_value):,.2f}",
                'P&L $':     f"{'+'if pnl>=0 else ''}${pnl:.2f}",
                'P&L %':     f"{'+'if pnl_pct>=0 else ''}{pnl_pct:.2f}%",
                '→ Stop':    f"{(current-stop)/current*100:+.1f}%" if stop else '—',
                '→ Target':  f"{(tp-current)/current*100:+.1f}%"  if tp  else '—',
            })

        def _color_pnl(v):
            if isinstance(v, str) and v.startswith('+'):  return 'color:#00d4aa; font-weight:bold'
            if isinstance(v, str) and '-' in v and not v.startswith('→'): return 'color:#ff6b6b; font-weight:bold'
            return ''

        st.dataframe(pd.DataFrame(rows).style.map(_color_pnl, subset=['P&L $', 'P&L %']),
                     use_container_width=True, hide_index=True)
    else:
        st.caption("No open positions. Run `python run.py etf-execute` in the terminal to enter positions.")

    st.divider()

    snapshots = get_snapshots()
    if len(snapshots) > 1:
        st.markdown('<div class="section-title">📉 Portfolio Value Over Time</div>', unsafe_allow_html=True)
        snap_df = pd.DataFrame([{'Time': s.timestamp, 'Value': s.portfolio_value} for s in snapshots])
        fig_p = go.Figure()
        fig_p.add_trace(go.Scatter(
            x=snap_df['Time'], y=snap_df['Value'],
            fill='tozeroy', fillcolor='rgba(74,144,226,0.12)',
            line=dict(color='#4a90e2', width=2),
            name='Portfolio',
            hovertemplate='%{x|%b %d %I:%M %p}<br><b>$%{y:,.2f}</b><extra></extra>',
        ))

        # SPY benchmark: normalize SPY to same starting value as portfolio
        try:
            spy_hist = get_price_history('SPY', period='3mo')
            start_ts = snap_df['Time'].iloc[0]
            start_value = snap_df['Value'].iloc[0]
            # Align SPY from the first snapshot date onward
            spy_hist.index = spy_hist.index.tz_localize(None) if spy_hist.index.tz is not None else spy_hist.index
            spy_since = spy_hist[spy_hist.index >= pd.Timestamp(start_ts).tz_localize(None).normalize()]
            if not spy_since.empty:
                spy_start_price = spy_since['Close'].iloc[0]
                spy_normalized  = spy_since['Close'] / spy_start_price * start_value
                fig_p.add_trace(go.Scatter(
                    x=spy_since.index, y=spy_normalized,
                    line=dict(color='rgba(180,180,180,0.5)', dash='dot', width=1.5),
                    name='SPY (benchmark)',
                    hovertemplate='%{x|%b %d}<br>SPY benchmark: <b>$%{y:,.2f}</b><extra></extra>',
                ))
        except Exception:
            pass  # silently skip if SPY data unavailable

        fig_p.update_layout(**_PLOT, height=300, xaxis_title='Date', yaxis_title='Value ($)',
                            showlegend=True, legend=dict(x=0.01, y=0.99, bgcolor='rgba(0,0,0,0)'),
                            margin=dict(t=10))
        fig_p.update_xaxes(**_GRID)
        fig_p.update_yaxes(**_GRID)
        st.plotly_chart(fig_p, use_container_width=True)
    else:
        st.caption("Portfolio chart will appear after at least 2 snapshots are recorded.")


# ── TAB 5: Trade History ─────────────────────────────────────────────────────
with tab5:
    st.markdown('<div class="section-title">📜 Trade History</div>', unsafe_allow_html=True)
    trades = get_all_trades()

    if trades:
        closed = [t for t in trades if t.exit_price and t.pnl is not None]
        wins   = [t for t in closed if t.pnl > 0]
        losses = [t for t in closed if t.pnl <= 0]
        total_pnl = sum(t.pnl for t in closed)

        s1, s2, s3, s4, s5 = st.columns(5)
        s1.metric("Total Trades",  len(trades))
        s2.metric("Closed",        len(closed))
        s3.metric("Wins",          len(wins))
        s4.metric("Losses",        len(losses))
        s5.metric("Realized P&L",  f"{'+'if total_pnl>=0 else ''}${total_pnl:.2f}")

        if closed:
            win_rate = len(wins) / len(closed) * 100
            avg_win  = sum(t.pnl for t in wins)   / len(wins)   if wins   else 0
            avg_loss = sum(t.pnl for t in losses)  / len(losses) if losses else 0
            exp      = total_pnl / len(closed)
            st.caption(
                f"Win rate: **{win_rate:.0f}%** · Avg win: +${avg_win:.2f} · "
                f"Avg loss: ${avg_loss:.2f} · Expectancy: {'+'if exp>=0 else ''}${exp:.2f}/trade"
            )

        st.divider()
        rows = []
        for t in trades:
            rows.append({
                'Date':   t.entry_time.strftime('%b %d %H:%M') if t.entry_time else '—',
                'Symbol': t.symbol,
                'Module': t.module,
                'Qty':    int(t.qty),
                'Entry':  f"${t.entry_price:.2f}"  if t.entry_price  else '—',
                'Exit':   f"${t.exit_price:.2f}"   if t.exit_price   else 'Open',
                'Stop':   f"${t.stop_loss:.2f}"    if t.stop_loss    else '—',
                'Target': f"${t.take_profit:.2f}"  if t.take_profit  else '—',
                'P&L':    f"{'+'if (t.pnl or 0)>=0 else ''}${t.pnl:.2f}" if t.pnl else ('Open' if not t.exit_price else '—'),
                'Status': t.status,
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    else:
        st.caption("No trades in the database yet.")


# ── TAB 6: Stock Screener ────────────────────────────────────────────────────
with tab6:
    st.markdown('<div class="section-title">📋 Stock Screener — S&P MidCap 400 + SmallCap 600</div>', unsafe_allow_html=True)

    candidates, cache_time = load_screener_results()

    if not candidates:
        st.warning(
            "No screener results for today. "
            "Run `python run.py stocks` in the terminal (3–5 min on first run, instant after). "
            "Use `python run.py stocks-refresh` to force a fresh download."
        )
    else:
        if cache_time:
            st.caption(f"Last screener run: **{cache_time}** today  ·  Run `python run.py stocks-refresh` in the terminal to update.")

        # ── Opening briefing ──────────────────────────────────────────────────
        st.markdown('<div class="section-title">🌅 Morning Overview</div>', unsafe_allow_html=True)
        try:
            intro = generate_stock_intro_narrative(candidates, regime_data)
            st.markdown(f'<div class="narrative-box">{intro}</div>', unsafe_allow_html=True)
        except Exception as e:
            st.caption(f"Intro narrative error: {e}")

        # ── Summary metrics ───────────────────────────────────────────────────
        high_cs = [c for c in candidates if c.get('tier_label') == 'HIGH']
        mid_cs  = [c for c in candidates if c.get('tier_label') == 'MID']
        low_cs  = [c for c in candidates if c.get('tier_label') == 'LOW']

        sm1, sm2, sm3, sm4 = st.columns(4)
        sm1.metric("🟢 HIGH Tier", len(high_cs))
        sm2.metric("🟡 MID Tier",  len(mid_cs))
        sm3.metric("🔴 LOW Tier",  len(low_cs))
        sm4.metric("Total Candidates", len(candidates))

        st.divider()

        # ── Tier sections ─────────────────────────────────────────────────────
        tier_config = [
            ('HIGH', high_cs, '#00d4aa', '🟢', 'Score 8–10 — highest signal confluence, strongest setups'),
            ('MID',  mid_cs,  '#ffd700', '🟡', 'Score 5–7 — solid setups, fewer confirming signals'),
            ('LOW',  low_cs,  '#ff6b6b', '🔴', 'Score 0–4 — passes all hard gates, requires strong qualitative case'),
        ]

        for tier_label, tier_cands, tier_color, tier_emoji, tier_desc in tier_config:
            if not tier_cands:
                continue

            st.markdown(
                f"<div style='font-size:1.15rem;font-weight:700;color:{tier_color};"
                f"padding:10px 0 4px;border-bottom:2px solid {tier_color}33;margin:16px 0 8px'>"
                f"{tier_emoji} {tier_label} TIER — {tier_desc}</div>",
                unsafe_allow_html=True
            )

            for c in tier_cands:
                sym     = c.get('symbol', '?')
                name    = (c.get('name') or sym)[:40]
                rank    = c.get('rank', '?')
                score   = c.get('score', 0)
                setup   = c.get('setup_name', 'Momentum Setup')
                s_tier  = c.get('setup_tier', 'B')
                emoji   = _STOCK_SETUP_EMOJI.get(setup, '📊')
                sector  = c.get('sector', 'Unknown')
                cap     = c.get('market_cap') or 0
                price   = c.get('price')
                rsi     = c.get('rsi') or 0
                rvol    = c.get('rvol') or 1.0
                prox    = c.get('prox_52w') or 0
                rs_spy  = c.get('rs_spy')
                atr_pct = c.get('atr_pct') or 0

                ret_1d = c.get('ret_1d') or 0
                ret_5d = c.get('ret_5d') or 0
                ret_1m = c.get('ret_1m') or 0
                ret_3m = c.get('ret_3m') or 0
                ret_6m = c.get('ret_6m') or 0

                earnings_flag = c.get('earnings_soon', False)
                short_flag    = c.get('short_flag', False)
                short_pct     = (c.get('short_float') or 0) * 100

                cap_label = f"${cap/1e9:.1f}B" if cap >= 1e9 else f"${cap/1e6:.0f}M" if cap > 0 else "n/a"

                # Warning flags inline
                flags = ""
                if earnings_flag: flags += "  ⚠️ EARNINGS SOON"
                if short_flag:    flags += f"  ⚠️ SHORT {short_pct:.0f}%"

                # Candidate header
                st.markdown(
                    f"**#{rank}  {sym}** — {name}  &nbsp;·&nbsp;  "
                    f"Score **{score}/10**  &nbsp;·&nbsp;  "
                    f"{emoji} {setup} (Setup {s_tier})  &nbsp;·&nbsp;  "
                    f"{sector} · {cap_label}"
                    + (f"  &nbsp;<span style='color:#ffaa00'>{flags}</span>" if flags else ""),
                    unsafe_allow_html=True
                )

                # Metrics row
                mc1, mc2, mc3, mc4, mc5, mc6, mc7 = st.columns(7)
                mc1.metric("Price",    f"${price:.2f}"      if price else "n/a")
                mc2.metric("RSI",      f"{rsi:.0f}",        help=">70 overbought, <35 oversold")
                mc3.metric("RVOL",     f"{rvol:.1f}×",      help="Volume vs. 20-day average")
                mc4.metric("Today",    f"{ret_1d*100:+.2f}%")
                mc5.metric("1 Month",  f"{ret_1m*100:+.1f}%")
                mc6.metric("3 Month",  f"{ret_3m*100:+.1f}%")
                mc7.metric("6 Month",  f"{ret_6m*100:+.1f}%")

                # RS vs SPY and proximity badges
                badge_parts = [f"52W High: **{prox*100:.1f}%**  ·  ATR: {atr_pct*100:.1f}%/day"]
                if _nv(rs_spy):
                    color = '#00d4aa' if rs_spy > 0 else '#ff6b6b'
                    badge_parts.append(
                        f"<span style='color:{color}'>RS vs SPY (3m): {rs_spy*100:+.1f}%</span>"
                    )
                st.markdown(
                    f"<span style='font-size:0.82rem;color:#aaa'>{' &nbsp;·&nbsp; '.join(badge_parts)}</span>",
                    unsafe_allow_html=True
                )

                # Deep-dive expander
                with st.expander(f"📊 Full analysis — {sym}: chart, signals, news & research guide"):
                    # Price chart
                    try:
                        df_st = load_daily_history(sym, period='3mo')
                        if not df_st.empty and 'Close' in df_st.columns:
                            close_s = df_st['Close']
                            ma20_s  = close_s.rolling(20).mean()
                            ma50_s  = close_s.rolling(50).mean()
                            rsi_ser = _rsi_series(close_s)

                            fig_st = make_subplots(rows=2, cols=1, row_heights=[0.72, 0.28],
                                                   shared_xaxes=True, vertical_spacing=0.03)
                            fig_st.add_trace(go.Scatter(
                                x=df_st.index, y=close_s, name='Price',
                                line=dict(color='#4a90e2', width=2),
                                hovertemplate='%{x|%b %d}<br><b>$%{y:.2f}</b><extra></extra>',
                            ), row=1, col=1)
                            fig_st.add_trace(go.Scatter(
                                x=df_st.index, y=ma20_s, name='20d EMA',
                                line=dict(color='#ffd700', width=1.5, dash='dash'), opacity=0.85,
                            ), row=1, col=1)
                            fig_st.add_trace(go.Scatter(
                                x=df_st.index, y=ma50_s, name='50d EMA',
                                line=dict(color='#ff9500', width=1.5, dash='dot'), opacity=0.85,
                            ), row=1, col=1)

                            vcol = ['#00d4aa' if c2 >= o else '#ff4b4b'
                                    for c2, o in zip(df_st['Close'], df_st['Open'])]
                            fig_st.add_trace(go.Bar(
                                x=df_st.index, y=df_st['Volume'],
                                marker_color=vcol, showlegend=False,
                                hovertemplate='Vol: %{y:,.0f}<extra></extra>',
                            ), row=2, col=1)

                            fig_st.update_layout(
                                **_PLOT, height=370, hovermode='x unified',
                                yaxis_title='Price ($)', yaxis2_title='Volume',
                                legend=dict(orientation='h', yanchor='bottom', y=1.02, x=0),
                                margin=dict(t=10, b=8),
                            )
                            fig_st.update_xaxes(**_GRID)
                            fig_st.update_yaxes(**_GRID)
                            st.plotly_chart(fig_st, use_container_width=True)

                            # RSI chart
                            fig_r2 = go.Figure()
                            fig_r2.add_trace(go.Scatter(
                                x=df_st.index, y=rsi_ser, name='RSI',
                                line=dict(color='#a855f7', width=1.5),
                                hovertemplate='%{x|%b %d}  RSI: %{y:.1f}<extra></extra>',
                            ))
                            fig_r2.add_hline(y=70, line_color='rgba(255,75,75,0.55)', line_dash='dash',
                                             annotation_text='70', annotation_position='right')
                            fig_r2.add_hline(y=30, line_color='rgba(0,212,170,0.55)', line_dash='dash',
                                             annotation_text='30', annotation_position='right')
                            fig_r2.add_hrect(y0=70, y1=100, fillcolor='rgba(255,75,75,0.05)', line_width=0)
                            fig_r2.add_hrect(y0=0,  y1=30,  fillcolor='rgba(0,212,170,0.05)', line_width=0)
                            fig_r2.update_layout(**_PLOT, height=150, showlegend=False,
                                yaxis=dict(range=[0, 100], title='RSI', **_GRID),
                                xaxis=_GRID, margin=dict(t=4, b=4))
                            st.plotly_chart(fig_r2, use_container_width=True)

                    except Exception:
                        st.caption("Chart unavailable for this symbol.")

                    # Narrative + news + research
                    try:
                        news_items = load_stock_news(sym)
                    except Exception:
                        news_items = []

                    try:
                        narrative = generate_stock_candidate_narrative(c, news_items)
                        st.markdown(narrative)
                    except Exception as e:
                        st.caption(f"Narrative error: {e}")

                st.markdown("<div style='margin-bottom:6px'></div>", unsafe_allow_html=True)

        st.divider()

        # ── Scoring key ───────────────────────────────────────────────────────
        with st.expander("📖 Scoring methodology — how the 0–10 score is calculated"):
            st.markdown("""
| Dimension | Signal | Max Points | How scored |
|-----------|--------|-----------|------------|
| **RS vs SPY (3m)** | Stock return minus SPY return over 3 months | 2 pts | Top 80% of universe = 2; top 50% = 1 |
| **12-1M Momentum** | Return from 12 months ago to 1 month ago (skips last month to avoid microstructure reversal) | 2 pts | Top 80% of universe = 2; top 50% = 1 |
| **52W Proximity** | Current price ÷ 52-week high | 2 pts | ≥95% of high = 2; ≥85% = 1 |
| **Relative Volume** | 3-day avg volume ÷ 20-day baseline | 2 pts | ≥2.0× = 2; ≥1.5× = 1 |
| **Setup Tier** | Quality of technical structure | 2 pts | Tier A = 2; Tier B = 1; Tier C = 0 |

**Tier thresholds:** HIGH ≥ 8 · MID ≥ 5 · LOW < 5

Scoring is **relative** within today's survivor universe — a score reflects how this stock ranks against the others that passed all gates, not against an absolute threshold.
""")

        with st.expander("📖 Hard gates — what gets eliminated before scoring"):
            st.markdown("""
Every stock must pass ALL of these before it's even considered:

| Gate | Threshold | Why |
|------|-----------|-----|
| **Minimum price** | ≥ $5 | Eliminates penny stocks with wide spreads and manipulation risk |
| **Dollar volume** | ≥ $5M/day (20-day avg) | Ensures you can enter and exit without moving the market |
| **ATR range** | 1.5% – 8% daily | Too quiet = no return potential; too wild = unmanageable risk |
| **MA stack** | Price > 20 EMA > 50 EMA > 200 SMA | All four levels aligned = confirmed uptrend at every timeframe |
| **RSI** | 35 – 82 | Avoid falling knives (<35) and severely extended moves (>82) |
| **3-month return** | > 0% | Must show positive recent momentum — not chasing declining assets |
| **Relative volume** | ≥ 1.0× baseline | Must have at least average participation — no dead, ignored names |
""")
