import math
import pandas as pd
from shared.data import (
    rank_etfs_by_momentum, get_market_regime, get_news_sentiment,
    ETF_UNIVERSE, BULL_ETFS, BEAR_ETFS
)

TRAIL_PCT = 0.30  # trailing stop: 30% below the position's running high

# Kept for live Alpaca bracket order construction in execute.py
ATR_MULTIPLIER    = 4.5
REWARD_RISK_RATIO = 2.5
MAX_STOP_PCT      = 0.12

MAX_POSITIONS = 3  # top 3 ETFs by 12m momentum

# Antonacci absolute momentum filter: 0.0 allows AGG (bonds) to qualify in flight-to-safety
# environments where bonds have flat/slightly positive returns.
QUALITY_FLOOR_12M = 0.01  # require positive 12m momentum (Antonacci absolute filter)

MIN_ALLOCATION_USD = 200.0

# Trailing stops handle "let winners ride" — monthly rotation should be forced.
# Set threshold very high to effectively disable winner protection.
WINNER_PROTECTION_THRESHOLD = 999.0

# 95% deployment in bull/neutral — maximize participation in the trend.
# Bear regime still pulls back to 50% to limit exposure during confirmed downtrends.
REGIME_ALLOCATION = {
    'bull':    0.95,
    'neutral': 0.80,
    'bear':    0.50,
}

# ── 12-Factor Composite Scoring Model ────────────────────────────────────────
#
# Each ETF is scored 0–100 across 12 independent factors, then combined into a
# single composite score that drives the hold/rank decision.
#
# Factor groups and rationale:
#   Momentum (50%): cross-sectional and absolute momentum signal quality
#   Risk-adjusted quality (26%): reward returns achieved with less volatility/drawdown
#   Technical confirmation (18%): trend alignment signals that reduce false positives
#   Regime fit (6%): bonus for ETFs aligned with the current macro regime
#
# All z-score factors are normalized cross-sectionally within the candidate universe
# each run, then clipped ±2.5σ and scaled to [0, 100]. Absolute-scored factors
# (RSI, MACD, regime fit) use fixed thresholds instead, to avoid circular ranking.

FACTOR_WEIGHTS = {
    'return_12m':       0.30,  # Antonacci primary signal — 12m cross-sectional momentum (academic consensus)
    'return_3m':        0.12,  # medium-term continuation — 3m used as secondary confirmation
    'momentum_accel':   0.08,  # 3m pace vs 12m pace — is momentum building or fading?
    'alpha_vs_spy':     0.14,  # 12m excess return vs SPY — rewards genuine alpha over beta
    'sharpe_12m':       0.10,  # risk-adjusted 12m return — same return at lower vol ranks higher
    'drawdown_score':   0.08,  # max 12m drawdown (inverted) — penalizes choppy/unstable trends
    'rsi_score':        0.04,  # RSI positioning — sweet spot 45–65, penalize extremes
    'macd_score':       0.06,  # MACD vs signal line — trend direction confirmation
    'ma50_score':       0.04,  # % above 50-day MA — near-term trend alignment
    'ma200_score':      0.02,  # % above 200-day MA — long-term structural trend
    'vol_score':        0.02,  # relative volume (5d / 20d) — institutional conviction
    'regime_fit':       0.00,  # disabled — regime filtering handled at candidate selection
}
# sum(FACTOR_WEIGHTS.values()) == 1.0


def _zscore_to_score(series: pd.Series) -> pd.Series:
    """Cross-sectional z-score, clipped ±2.5σ, scaled to [0, 100]."""
    std = series.std()
    if std == 0 or pd.isna(std):
        return pd.Series(50.0, index=series.index)
    z = (series - series.mean()) / std
    return (z.clip(-2.5, 2.5) + 2.5) / 5.0 * 100.0


def _rsi_to_score(rsi: float) -> float:
    """
    Absolute RSI scoring. The 45–65 zone is the momentum sweet spot:
    strong trend but not yet overbought. Extremes signal mean-reversion risk.
    """
    if 45 <= rsi <= 65:  return 100.0
    if 35 <= rsi <  45:  return  70.0
    if 65 <  rsi <= 75:  return  60.0
    if 25 <= rsi <  35:  return  35.0
    if 75 <  rsi <= 85:  return  25.0
    return 10.0  # RSI < 25 or > 85


def _macd_to_score(macd: float, signal: float) -> float:
    """
    MACD trend-direction scoring.
    MACD > signal AND positive: full bullish momentum (100).
    MACD > signal but negative: recovering, upward crossover (65).
    MACD < signal but positive: weakening, still above zero (35).
    MACD < signal AND negative: bearish (0).
    """
    above    = macd > signal
    positive = macd > 0
    if above and positive:      return 100.0
    if above and not positive:  return  65.0
    if not above and positive:  return  35.0
    return 0.0


def _regime_fit_to_score(symbol: str, regime: str) -> float:
    """Bonus for ETFs structurally aligned with the current macro regime."""
    if regime == 'neutral':
        return 50.0
    in_bull = symbol in BULL_ETFS
    in_bear = symbol in BEAR_ETFS
    if regime == 'bull':
        return 100.0 if in_bull else (0.0 if in_bear else 50.0)
    return 100.0 if in_bear else (0.0 if in_bull else 50.0)


def _compute_composite_scores(df: pd.DataFrame, regime: str) -> pd.DataFrame:
    """
    Add per-factor scores (f_*) and composite_score to df.
    All z-score factors normalize cross-sectionally within the passed universe.
    """
    df = df.copy()

    # ── Derived factors ───────────────────────────────────────────────────────
    # Momentum acceleration: 3m annualized pace vs 12m pace (positive = building)
    df['momentum_accel'] = df['return_3m'] * 4 - df['return_12m']

    # 12m excess return over SPY — rewards genuine alpha over broad market beta
    df['alpha_vs_spy'] = df['return_12m'] - df.get('spy_return_12m', pd.Series(0.0, index=df.index))

    # Risk-adjusted 12m return (Sharpe proxy: annual return / annual vol)
    safe_vol = df['realized_vol_12m'].replace(0, float('nan'))
    df['sharpe_12m'] = df['return_12m'] / safe_vol

    # ── Z-score factors → [0, 100] ────────────────────────────────────────────
    df['f_return_12m']     = _zscore_to_score(df['return_12m'])
    df['f_return_3m']      = _zscore_to_score(df['return_3m'])
    df['f_momentum_accel'] = _zscore_to_score(df['momentum_accel'])
    df['f_alpha_vs_spy']   = _zscore_to_score(df['alpha_vs_spy'])
    df['f_sharpe_12m']     = _zscore_to_score(df['sharpe_12m'].fillna(0))
    df['f_drawdown_score'] = _zscore_to_score(-df['max_drawdown_12m'])  # invert: less drawdown = better
    df['f_ma50_score']     = _zscore_to_score(df['pct_above_ma50'])
    df['f_ma200_score']    = _zscore_to_score(df['pct_above_ma200'])
    df['f_vol_score']      = _zscore_to_score(df['volume_ratio'])

    # ── Absolute-scored factors → [0, 100] ───────────────────────────────────
    df['f_rsi_score']  = df['rsi'].apply(_rsi_to_score)
    df['f_macd_score'] = df.apply(lambda r: _macd_to_score(r['macd'], r['macd_signal']), axis=1)
    df['f_regime_fit'] = df['symbol'].apply(lambda s: _regime_fit_to_score(s, regime))

    # ── Weighted composite ────────────────────────────────────────────────────
    w = FACTOR_WEIGHTS
    df['composite_score'] = (
        df['f_return_12m']     * w['return_12m']     +
        df['f_return_3m']      * w['return_3m']      +
        df['f_momentum_accel'] * w['momentum_accel'] +
        df['f_alpha_vs_spy']   * w['alpha_vs_spy']   +
        df['f_sharpe_12m']     * w['sharpe_12m']     +
        df['f_drawdown_score'] * w['drawdown_score'] +
        df['f_rsi_score']      * w['rsi_score']      +
        df['f_macd_score']     * w['macd_score']     +
        df['f_ma50_score']     * w['ma50_score']     +
        df['f_ma200_score']    * w['ma200_score']    +
        df['f_vol_score']      * w['vol_score']      +
        df['f_regime_fit']     * w['regime_fit']
    )

    return df


def get_etf_candidates(regime: str) -> list[str]:
    # Pure momentum ranking — consider all ETFs regardless of regime.
    # The 12m return signal naturally rotates toward defensive names (XLP, AGG)
    # when equity momentum fades, with no hard regime-based exclusions.
    return list(ETF_UNIVERSE)


def build_signal_dashboard(regime_data: dict) -> pd.DataFrame:
    regime     = regime_data['regime']
    candidates = get_etf_candidates(regime)
    df         = rank_etfs_by_momentum(candidates)
    if df.empty:
        return df

    df['regime_fit_label'] = df['symbol'].apply(
        lambda s: 'yes' if (
            (regime == 'bull' and s in BULL_ETFS) or
            (regime == 'bear' and s not in BULL_ETFS) or
            regime == 'neutral'
        ) else 'no'
    )
    df['rsi_signal'] = df['rsi'].apply(
        lambda r: 'overbought' if r > 70 else ('oversold' if r < 30 else 'neutral')
    )

    # ATR-based stop, capped at hard max
    df['stop_loss_pct']   = df.apply(
        lambda row: min((ATR_MULTIPLIER * row['atr']) / row['price'], MAX_STOP_PCT), axis=1
    )
    df['take_profit_pct'] = df['stop_loss_pct'] * REWARD_RISK_RATIO

    # Composite scoring — ranks all candidates by 12-factor model
    df = _compute_composite_scores(df, regime)

    # Sort by composite score; QUALITY_FLOOR_12M is enforced at sizing time
    df = df.sort_values('composite_score', ascending=False).reset_index(drop=True)
    df['rank'] = df.index + 1
    return df


def get_position_sizing(
    df: pd.DataFrame,
    portfolio_value: float,
    regime: str,
    top_n: int = 3,
    deployed_usd: float = 0.0,
) -> list[dict]:
    # Hard filter: absolute momentum must be positive before composite score can qualify
    qualified = df[df['return_12m'] > QUALITY_FLOOR_12M].head(top_n).copy()
    top = qualified[qualified['atr'] > 0]
    if top.empty:
        return []

    total_etf_budget = portfolio_value * REGIME_ALLOCATION[regime]
    available_budget = max(0.0, total_etf_budget - deployed_usd)

    if available_budget < MIN_ALLOCATION_USD:
        return []

    top['atr_pct']       = top['atr'] / top['price']
    top['inv_vol']       = 1.0 / top['atr_pct']
    top['weight']        = top['inv_vol'] / top['inv_vol'].sum()
    top['allocation_usd']= top['weight'] * available_budget
    top['qty']           = top.apply(lambda row: math.floor(row['allocation_usd'] / row['price']), axis=1)

    top['stop_price']       = top.apply(lambda row: round(row['price'] * (1 - row['stop_loss_pct']), 2), axis=1)
    top['take_profit_price']= top.apply(lambda row: round(row['price'] * (1 + row['take_profit_pct']), 2), axis=1)

    top = top[top['qty'] > 0]
    if top.empty:
        return []

    return top.to_dict('records')


def get_news_for_candidates(symbols: list[str]) -> dict[str, list]:
    return get_news_sentiment(symbols, limit=3)


def _factor_bar(score: float, width: int = 12) -> str:
    """ASCII bar showing a factor's 0-100 score."""
    filled = round(score / 100 * width)
    return f"[{'█' * filled}{'░' * (width - filled)}] {score:5.1f}"


def print_morning_briefing(regime_data: dict, dashboard: pd.DataFrame, sizing: list[dict], news: dict):
    regime = regime_data['regime']
    W = 72
    print(f"\n{'='*W}")
    print(f"  MORNING BRIEFING")
    print(f"{'='*W}")
    print(f"\nMarket Regime: {regime.upper()}")
    print(f"  VIX: {regime_data['vix']:.1f}  |  SPY: ${regime_data['spy_price']:.2f}  |  200MA: ${regime_data['ma200']:.2f}")

    print(f"\n{'─'*W}")
    print(f"  ETF SIGNAL DASHBOARD  (ranked by 12-factor composite score)")
    print(f"{'─'*W}")
    print(f"{'Rank':<5} {'ETF':<6} {'Score':>6} {'1M':>7} {'3M':>7} {'6M':>7} {'RSI':>6} {'MACD':>7} {'Vol':>6} {'Stop%':>7} {'TP%':>7}")
    print(f"{'─'*W}")
    for _, row in dashboard.iterrows():
        macd_flag = '▲' if row['macd'] > row['macd_signal'] else '▼'
        print(
            f"{int(row['rank']):<5} {row['symbol']:<6} "
            f"{row['composite_score']:>6.1f} "
            f"{row['return_1m']*100:>6.1f}% "
            f"{row['return_3m']*100:>6.1f}% "
            f"{row['return_6m']*100:>6.1f}% "
            f"{row['rsi']:>6.1f} "
            f"  {macd_flag}{'above' if macd_flag=='▲' else 'below':>5} "
            f"{row['volume_ratio']:>5.2f}x "
            f"{row['stop_loss_pct']*100:>6.1f}% "
            f"{row['take_profit_pct']*100:>6.1f}%"
        )

    if sizing:
        print(f"\n{'─'*W}")
        print(f"  RECOMMENDED POSITIONS (top {len(sizing)} by composite score, positive 3m return)")
        print(f"{'─'*W}")
        factor_labels = [
            ('f_return_3m',      '3m momentum'),
            ('f_return_1m',      '1m momentum'),
            ('f_momentum_accel', 'accel'),
            ('f_alpha_vs_spy',   'alpha vs SPY'),
            ('f_sharpe_3m',      'Sharpe'),
            ('f_drawdown_score', 'drawdown'),
            ('f_rsi_score',      'RSI'),
            ('f_macd_score',     'MACD'),
            ('f_ma50_score',     '50d MA'),
            ('f_ma200_score',    '200d MA'),
            ('f_vol_score',      'volume'),
            ('f_regime_fit',     'regime fit'),
        ]
        for pos in sizing:
            print(f"\n  {pos['symbol']}  (composite score: {pos['composite_score']:.1f}/100)")
            print(f"    Qty: {int(pos['qty'])} shares @ ~${pos['price']:.2f}")
            print(f"    Allocation: ${pos['allocation_usd']:,.0f} ({pos['weight']*100:.0f}% of ETF budget)")
            print(f"    Stop-loss:   ${pos['stop_price']:.2f}  ({pos['stop_loss_pct']*100:.1f}% below entry)")
            print(f"    Take-profit: ${pos['take_profit_price']:.2f}  ({pos['take_profit_pct']*100:.1f}% above entry)")
            print(f"    ATR (14d):   ${pos['atr']:.2f}")
            print(f"    Factor breakdown:")
            for col, label in factor_labels:
                if col in pos:
                    print(f"      {label:<18} {_factor_bar(pos[col])}")
    else:
        print("\n  No viable candidates for current regime.")

    if news:
        print(f"\n{'─'*W}")
        print(f"  NEWS")
        print(f"{'─'*W}")
        for symbol, articles in news.items():
            for a in articles:
                print(f"\n  [{symbol}] {a['sentiment_label'].upper()} ({a['sentiment_score']:+.2f})")
                print(f"  {a['title']}")
                print(f"  {a['summary'][:200]}...")

    print(f"\n{'='*W}")
    print(f"  Review above and confirm which positions to enter.")
    print(f"{'='*W}\n")


def print_intraday_briefing(
    regime_data: dict,
    available: pd.DataFrame,
    sizing: list[dict],
    current_positions,
    deployed_usd: float,
    portfolio_value: float,
    news: dict,
):
    regime = regime_data['regime']
    total_etf_budget = portfolio_value * REGIME_ALLOCATION[regime]
    remaining_budget = total_etf_budget - deployed_usd

    W = 72
    print(f"\n{'='*W}")
    print(f"  INTRADAY SCAN")
    print(f"{'='*W}")
    print(f"\nMarket Regime: {regime.upper()}")
    print(f"  VIX: {regime_data['vix']:.1f}  |  SPY: ${regime_data['spy_price']:.2f}  |  200MA: ${regime_data['ma200']:.2f}")
    print(f"\n  ETF Budget: ${total_etf_budget:,.0f} total | ${deployed_usd:,.0f} deployed | ${remaining_budget:,.0f} available")

    if current_positions:
        print(f"\n{'─'*W}")
        print(f"  CURRENT POSITIONS ({len(current_positions)} / {MAX_POSITIONS} max)")
        print(f"{'─'*W}")
        for p in current_positions:
            pnl  = float(p.unrealized_pl)
            sign = '+' if pnl >= 0 else ''
            print(f"  {p.symbol:<6}  {float(p.qty):.0f} shares  "
                  f"avg ${float(p.avg_entry_price):.2f}  "
                  f"now ${float(p.current_price):.2f}  "
                  f"P&L {sign}${pnl:.2f}")

    if not available.empty:
        print(f"\n{'─'*W}")
        print(f"  AVAILABLE CANDIDATES (ranked by composite score)")
        print(f"{'─'*W}")
        print(f"{'Rank':<5} {'ETF':<6} {'Score':>6} {'1M':>7} {'3M':>7} {'6M':>7} {'RSI':>6} {'MACD':>7} {'Qualifies'}")
        print(f"{'─'*W}")
        for _, row in available.head(10).iterrows():
            qualifies = 'YES' if row['return_12m'] > QUALITY_FLOOR_12M else 'no (neg 12m)'
            macd_flag = '▲' if row['macd'] > row['macd_signal'] else '▼'
            print(
                f"{int(row['rank']):<5} {row['symbol']:<6} "
                f"{row['composite_score']:>6.1f} "
                f"{row['return_1m']*100:>6.1f}% "
                f"{row['return_3m']*100:>6.1f}% "
                f"{row['return_6m']*100:>6.1f}% "
                f"{row['rsi']:>6.1f} "
                f"  {macd_flag}{'above' if macd_flag=='▲' else 'below':>5} "
                f"{qualifies}"
            )

    if sizing:
        print(f"\n{'─'*W}")
        print(f"  PROPOSED ADDITIONS ({len(sizing)})")
        print(f"{'─'*W}")
        for pos in sizing:
            print(f"\n  {pos['symbol']}  (composite score: {pos['composite_score']:.1f}/100)")
            print(f"    Qty: {int(pos['qty'])} shares @ ~${pos['price']:.2f}")
            print(f"    Allocation: ${pos['allocation_usd']:,.0f} ({pos['weight']*100:.0f}% of available budget)")
            print(f"    Stop-loss:   ${pos['stop_price']:.2f}  ({pos['stop_loss_pct']*100:.1f}% below entry)")
            print(f"    Take-profit: ${pos['take_profit_price']:.2f}  ({pos['take_profit_pct']*100:.1f}% above entry)")
    else:
        print("\n  No new candidates qualify (need positive 3m return and sufficient remaining budget).")

    if news:
        print(f"\n{'─'*W}")
        print(f"  NEWS")
        print(f"{'─'*W}")
        for symbol, articles in news.items():
            for a in articles:
                print(f"\n  [{symbol}] {a['sentiment_label'].upper()} ({a['sentiment_score']:+.2f})")
                print(f"  {a['title']}")
                print(f"  {a['summary'][:200]}...")

    print(f"\n{'='*W}\n")
