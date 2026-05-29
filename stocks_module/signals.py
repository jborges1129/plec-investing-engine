"""
Signal computation library for individual stock screening.

Every function takes a pre-fetched OHLCV DataFrame and returns a single value.
No network calls here — pure math on price data.

Design principles:
  - Hard gates eliminate clearly unqualified stocks before scoring starts
  - Relative signals (RS vs SPY, 12-1 momentum) rank within the survivor universe
  - Setup types tell the human reviewer WHAT KIND of trade this is before they look at a chart
"""

import pandas as pd
import numpy as np

# ── Screening constants ───────────────────────────────────────────────────────

MIN_PRICE          = 5.0        # no penny stocks
MIN_DOLLAR_VOLUME  = 5_000_000  # $5M avg daily dollar volume — liquidity floor
MIN_ATR_PCT        = 0.015      # 1.5% — too quiet to generate meaningful returns
MAX_ATR_PCT        = 0.080      # 8%  — too wild to manage risk reliably
RSI_MIN            = 35         # below = too weak, avoid catching falling knives
RSI_MAX            = 82         # above = too extended for new entries
MIN_RVOL           = 1.0        # must have at least average volume (no dead names)
MIN_3M_RETURN      = 0.0        # positive 3-month momentum required (same as ETF module)

# Scoring thresholds
SCORE_HIGH_TIER = 8   # 8-10 → HIGH
SCORE_MID_TIER  = 5   # 5-7  → MID
                      # 0-4  → LOW

# Setup detection thresholds
BREAKOUT_PROX_MIN    = 0.97   # within 3% of 52-week high for breakout
BREAKOUT_RVOL_MIN    = 1.5    # volume confirmation for breakout
ATR_CONTRACT_RATIO   = 0.75   # recent ATR < 75% of 3m ATR = coiling/base
FIRST_PB_MIN_MOVE    = 0.12   # prior 1m move > 12% to qualify as "first pullback"
EARNINGS_SOON_DAYS   = 7      # flag earnings within 7 days as catalyst setup
POST_EARNINGS_GAP    = 0.04   # 4% gap-up = proxy for earnings beat

# Maps yfinance sector names to the corresponding sector ETF
SECTOR_ETF = {
    'Technology':             'XLK',
    'Financial Services':     'XLF',
    'Financials':             'XLF',
    'Energy':                 'XLE',
    'Consumer Cyclical':      'XLY',
    'Consumer Discretionary': 'XLY',
    'Basic Materials':        'XLB',
    'Materials':              'XLB',
    'Industrials':            'XLI',
    'Communication Services': 'XLC',
    'Consumer Defensive':     'XLP',
    'Consumer Staples':       'XLP',
    'Healthcare':             'XLV',
    'Health Care':            'XLV',
    'Utilities':              'XLU',
    'Real Estate':            'XLRE',
}


# ── Core signal functions ─────────────────────────────────────────────────────

def compute_signals(df: pd.DataFrame, spy_close: pd.Series = None) -> dict | None:
    """
    Compute all technical screening signals for one stock.
    Returns a dict of signals, or None if the stock fails any hard gate.

    Hard gates (must ALL pass):
      - Price >= $5
      - Average daily dollar volume >= $5M
      - ATR% between 1.5% and 8%
      - Full MA stack: price > 20EMA > 50EMA > 200SMA
      - RSI between 35 and 82
      - 3-month return > 0
      - Relative volume >= 1.0x

    df: Daily OHLCV DataFrame with columns Open/High/Low/Close/Volume
    spy_close: SPY closing price series for relative strength calculation (optional)
    """
    try:
        close  = df['Close'].dropna()
        high   = df['High'].dropna()
        low    = df['Low'].dropna()
        volume = df['Volume'].dropna()

        if len(close) < 200:
            return None

        price = float(close.iloc[-1])
        if price < MIN_PRICE:
            return None

        # ── Dollar volume ────────────────────────────────────────────────────
        # Use dollar volume (shares × price) not raw share volume — filters
        # illiquid small caps that pass share-volume screens but can't be traded
        dollar_vol = float((close.iloc[-20:] * volume.iloc[-20:]).mean())
        if dollar_vol < MIN_DOLLAR_VOLUME:
            return None

        # ── True Range and ATR ───────────────────────────────────────────────
        # True Range captures overnight gaps that H-L alone would miss
        tr = pd.concat([
            high - low,
            (high - close.shift(1)).abs(),
            (low  - close.shift(1)).abs(),
        ], axis=1).max(axis=1)
        atr14 = float(tr.rolling(14).mean().iloc[-1])
        if pd.isna(atr14) or atr14 == 0:
            return None
        atr_pct = atr14 / price
        if atr_pct < MIN_ATR_PCT or atr_pct > MAX_ATR_PCT:
            return None

        # ── MA stack ─────────────────────────────────────────────────────────
        # All four levels must be aligned: price > 20EMA > 50EMA > 200SMA.
        # A broken stack = stock is in a downtrend at some timeframe.
        # EMA (exponential) for short/mid to respond faster; SMA for long-term.
        ema20  = float(close.ewm(span=20,  adjust=False).mean().iloc[-1])
        ema50  = float(close.ewm(span=50,  adjust=False).mean().iloc[-1])
        sma200 = float(close.rolling(200).mean().iloc[-1])
        if not (price > ema20 > ema50 > sma200):
            return None

        # ── RSI (14-period) ──────────────────────────────────────────────────
        delta = close.diff()
        gain  = delta.clip(lower=0).rolling(14).mean()
        loss  = (-delta.clip(upper=0)).rolling(14).mean().replace(0, float('nan'))
        rsi   = float((100 - (100 / (1 + gain / loss))).fillna(100).iloc[-1])
        if rsi < RSI_MIN or rsi > RSI_MAX:
            return None

        # ── Price returns ────────────────────────────────────────────────────
        def _ret(n: int) -> float:
            if len(close) <= n:
                return float('nan')
            s = float(close.iloc[-n - 1])
            return (price - s) / s if s > 0 else float('nan')

        ret_1d = _ret(1)
        ret_5d = _ret(5)
        ret_1m = _ret(21)
        ret_3m = _ret(63)
        ret_6m = _ret(126)

        if pd.isna(ret_3m) or ret_3m <= MIN_3M_RETURN:
            return None

        # ── 12-1 month momentum ──────────────────────────────────────────────
        # The academic version: return from 12 months ago to 1 month ago.
        # Skipping the most recent month avoids the short-term reversal
        # documented by Jegadeesh & Titman — last month's big winners often
        # mean-revert in the very near term.
        mom_12_1 = float('nan')
        if len(close) >= 252:
            s12m = float(close.iloc[-252])
            s1m  = float(close.iloc[-21])
            if s12m > 0:
                mom_12_1 = (s1m - s12m) / s12m

        # ── Relative strength vs SPY ─────────────────────────────────────────
        # How much has this stock outperformed the S&P 500 over 3 months?
        # A stock in the top 20% of RS vs SPY in a rising market is the
        # single strongest predictor of near-term continuation.
        rs_spy = float('nan')
        if spy_close is not None and len(spy_close) >= 63:
            spy_ret_3m = (float(spy_close.iloc[-1]) - float(spy_close.iloc[-63])) / float(spy_close.iloc[-63])
            if not pd.isna(ret_3m):
                rs_spy = ret_3m - spy_ret_3m

        # ── 52-week proximity ────────────────────────────────────────────────
        # Price / 52-week high. 1.0 = at the high. 0.90 = 10% off the high.
        # George & Hwang (2004): anchoring to prior highs creates a momentum
        # anomaly — stocks near highs continue to outperform. The market is
        # slow to adjust past the reference point of a well-known prior high.
        n_52w = min(252, len(close))
        high_52w = float(close.iloc[-n_52w:].max())
        prox_52w = price / high_52w if high_52w > 0 else float('nan')

        # ── Relative volume ──────────────────────────────────────────────────
        # 3-day average vs 20-day baseline. Rising RVOL on up days signals
        # institutional accumulation — the kind of buying that sustains a move.
        vol_baseline = float(volume.iloc[-23:-3].mean()) if len(volume) >= 23 else float(volume.mean())
        vol_recent   = float(volume.iloc[-3:].mean())
        rvol = vol_recent / vol_baseline if vol_baseline > 0 else 1.0
        if rvol < MIN_RVOL:
            return None

        # Volume trend label (consistent with ETF module)
        if rvol > 1.5:
            vol_trend = 'rising'
        elif rvol < 0.85:
            vol_trend = 'falling'
        else:
            vol_trend = 'neutral'

        # ── ATR contraction (base/coiling detection) ─────────────────────────
        # When recent volatility is contracting vs the 3-month average, the
        # stock is "coiling" — building energy. A breakout from this state
        # on high volume is the highest-conviction technical setup.
        atr_10d    = float(tr.iloc[-10:].mean())
        atr_3m_avg = float(tr.iloc[-63:].mean()) if len(tr) >= 63 else atr14
        atr_contracting = (atr_10d < atr_3m_avg * ATR_CONTRACT_RATIO) if atr_3m_avg > 0 else False

        # ── Post-earnings gap proxy ──────────────────────────────────────────
        # A 4%+ gap-up on 1.5x+ volume in the past 10 days is a strong proxy
        # for an earnings beat. PEAD research: these stocks continue higher
        # for 2-8 weeks as the market slowly reprices the new earnings level.
        post_earnings_proxy = False
        open_prices = df['Open'].dropna()
        for i in range(-10, -1):
            try:
                if abs(i) > len(df) - 1:
                    break
                prev_c  = float(close.iloc[i - 1])
                curr_o  = float(open_prices.iloc[i])
                v_today = float(volume.iloc[i])
                v_avg   = float(volume.iloc[-20:].mean())
                if prev_c > 0 and v_avg > 0:
                    gap = (curr_o - prev_c) / prev_c
                    if gap > POST_EARNINGS_GAP and v_today > v_avg * 1.5:
                        post_earnings_proxy = True
                        break
            except Exception:
                pass

        return {
            'price':               price,
            'atr':                 atr14,
            'atr_pct':             atr_pct,
            'rsi':                 rsi,
            'rvol':                rvol,
            'vol_trend':           vol_trend,
            'dollar_vol':          dollar_vol,
            'ret_1d':              ret_1d,
            'ret_5d':              ret_5d,
            'ret_1m':              ret_1m,
            'ret_3m':              ret_3m,
            'ret_6m':              ret_6m,
            'mom_12_1':            mom_12_1,
            'rs_spy':              rs_spy,
            'prox_52w':            prox_52w,
            'high_52w':            high_52w,
            'ema20':               ema20,
            'ema50':               ema50,
            'sma200':              sma200,
            'atr_contracting':     atr_contracting,
            'post_earnings_proxy': post_earnings_proxy,
        }
    except Exception:
        return None


def detect_setup(signals: dict) -> tuple[str, str]:
    """
    Classify the technical setup into a named type and conviction tier.

    Returns (setup_name, tier) where tier is:
      'A' — highest conviction (institutional-grade structure)
      'B' — solid momentum setup (actionable with normal diligence)
      'C' — lower conviction (requires strong qualitative justification)

    Setup types are mutually exclusive — the first matching rule wins,
    ordered by conviction level.
    """
    rsi             = signals.get('rsi', 50)
    rvol            = signals.get('rvol', 1.0)
    prox            = signals.get('prox_52w', 0) or 0
    ret_1m          = signals.get('ret_1m', 0) or 0
    atr_contracting = signals.get('atr_contracting', False)
    post_earnings   = signals.get('post_earnings_proxy', False)
    earnings_soon   = signals.get('earnings_soon', False)

    # ── Tier A: institutional-grade setups ───────────────────────────────────
    # Post-earnings drift: price gapped up on volume after a beat,
    # now holding the gap. PEAD research supports 2-8 week continuation.
    if post_earnings:
        return ('Earnings Gap / Post-Beat Drift', 'A')

    # Breakout from base: ATR contracting (coiling), near highs,
    # volume surging on the breakout attempt. Classic VCP structure.
    if atr_contracting and prox >= BREAKOUT_PROX_MIN and rvol >= BREAKOUT_RVOL_MIN:
        return ('Breakout from Base', 'A')

    # ── Tier B: solid momentum setups ────────────────────────────────────────
    # Momentum continuation: near highs, RSI in sweet spot (not overbought),
    # trend-following entry on any mild pullback.
    if prox >= 0.90 and 55 <= rsi <= 75:
        return ('Momentum Continuation', 'B')

    # First pullback: made a strong prior move, now pulling back to 20 EMA
    # on declining volume. Classic "buy the first dip in an uptrend."
    if ret_1m >= FIRST_PB_MIN_MOVE and 40 <= rsi <= 60:
        return ('First Pullback in Uptrend', 'B')

    # ── Tier C: lower conviction, require extra qualitative justification ────
    # Pre-earnings: technically strong but binary event risk within a week.
    if earnings_soon and rsi >= 50:
        return ('Pre-Earnings Setup', 'C')

    # Deep pullback: well off highs but potentially setting up at support.
    # High failure rate — only take with a very strong qualitative thesis.
    if prox < 0.78:
        return ('Deep Pullback to Support', 'C')

    # Default: momentum but doesn't fit a tighter category
    return ('Momentum Setup', 'B')


def score_candidate(row: pd.Series, universe_df: pd.DataFrame) -> int:
    """
    Score 0–10 across five independent signal dimensions (2 pts each).
    Scoring is relative — each signal is ranked within the survivor universe,
    so the score reflects how this stock compares to other qualified candidates,
    not against an absolute threshold.

    Dimensions:
      1. Relative strength vs SPY (3m) — are institutions choosing this over the index?
      2. 12-1 month momentum — academically validated predictor of near-term continuation
      3. 52-week proximity — anchoring anomaly; near-high stocks continue outperforming
      4. Relative volume — confirms institutional participation in the move
      5. Setup tier — quality of the technical structure (A=2, B=1, C=0)
    """
    score = 0

    # 1. RS vs SPY
    rs_spy_val = row.get('rs_spy')
    if not pd.isna(rs_spy_val):
        valid = universe_df['rs_spy'].dropna()
        if len(valid) > 1:
            pct = (valid < rs_spy_val).mean()
            if pct >= 0.80:   score += 2
            elif pct >= 0.50: score += 1

    # 2. 12-1 month momentum
    mom_val = row.get('mom_12_1')
    if not pd.isna(mom_val):
        valid = universe_df['mom_12_1'].dropna()
        if len(valid) > 1:
            pct = (valid < mom_val).mean()
            if pct >= 0.80:   score += 2
            elif pct >= 0.50: score += 1

    # 3. 52-week proximity
    prox = row.get('prox_52w', 0) or 0
    if prox >= 0.95:   score += 2
    elif prox >= 0.85: score += 1

    # 4. Relative volume
    rvol = row.get('rvol', 1.0)
    if rvol >= 2.0:   score += 2
    elif rvol >= 1.5: score += 1

    # 5. Setup tier
    tier = row.get('setup_tier', 'C')
    if tier == 'A':   score += 2
    elif tier == 'B': score += 1

    return int(score)
