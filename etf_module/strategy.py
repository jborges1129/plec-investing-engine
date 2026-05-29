import math
import pandas as pd
from shared.data import (
    rank_etfs_by_momentum, get_market_regime, get_news_sentiment,
    ETF_UNIVERSE, BULL_ETFS, BEAR_ETFS
)

# ATR × 3 stop: requires a 3-sigma-equivalent intraday move to trigger.
# ATR₁₄ ≈ typical daily range. At 3×ATR, stops survive normal 1-2σ intraday noise
# while catching genuine trend reversals. Engineering analogy: a 3× factor of safety.
ATR_MULTIPLIER = 3.0

# 2.5:1 reward-to-risk: requires only a 29% win rate for positive expected value.
# Antonacci Dual Momentum (2014) and CXO SACEMS data show ETF momentum win rates
# of 55-65%. At 60%: E[per trade] = 0.60×2.5 − 0.40×1.0 = 1.10 per unit of stop risk.
REWARD_RISK_RATIO = 2.5

# 7% hard cap: bounds single-trade loss to $7 per $100 invested regardless of ATR.
# With regime allocation ≤50% and inverse-vol sizing across ≤5 positions,
# simultaneous stop-out of all positions is bounded at roughly 3–4% of total portfolio.
MAX_STOP_PCT = 0.07

MAX_POSITIONS = 5  # max concurrent ETF positions; limits concentration risk

# Absolute momentum filter (Antonacci Dual Momentum): only hold ETFs above their own
# price 3 months ago. Set to 1% (not 0%) to exclude near-zero-momentum ETFs — a
# return of <1% over 3 months has no actionable signal and falls within noise.
QUALITY_FLOOR_3M = 0.01

MIN_ALLOCATION_USD = 200.0  # skip any position too small to matter

# Capital deployed scales inversely with market risk. Never fully deployed:
# even in bull mode, 50% stays in cash/alternatives for dry-powder.
# Thresholds: bull = VIX<20 AND SPY>200MA; bear = VIX>25 AND SPY<200MA.
REGIME_ALLOCATION = {
    'bull': 0.50,
    'neutral': 0.40,
    'bear': 0.30,
}


def get_etf_candidates(regime: str) -> list[str]:
    if regime == 'bull':
        return [s for s in ETF_UNIVERSE if s in BULL_ETFS]
    elif regime == 'bear':
        return [s for s in ETF_UNIVERSE if s in BEAR_ETFS]
    else:
        return ETF_UNIVERSE


def build_signal_dashboard(regime_data: dict) -> pd.DataFrame:
    regime = regime_data['regime']
    candidates = get_etf_candidates(regime)
    df = rank_etfs_by_momentum(candidates)
    if df.empty:
        return df

    df['regime_fit'] = df['symbol'].apply(
        lambda s: 'yes' if (
            (regime == 'bull' and s in BULL_ETFS) or
            (regime == 'bear' and s not in BULL_ETFS) or
            regime == 'neutral'
        ) else 'no'
    )
    df['rsi_signal'] = df['rsi'].apply(
        lambda r: 'overbought' if r > 70 else ('oversold' if r < 30 else 'neutral')
    )

    # Cap stop at MAX_STOP_PCT to match risk targets
    df['stop_loss_pct'] = df.apply(
        lambda row: min((ATR_MULTIPLIER * row['atr']) / row['price'], MAX_STOP_PCT), axis=1
    )
    df['take_profit_pct'] = df['stop_loss_pct'] * REWARD_RISK_RATIO

    df = df.sort_values('return_3m', ascending=False).reset_index(drop=True)
    df['rank'] = df.index + 1
    return df


def get_position_sizing(
    df: pd.DataFrame,
    portfolio_value: float,
    regime: str,
    top_n: int = 3,
    deployed_usd: float = 0.0,
) -> list[dict]:
    qualified = df[df['return_3m'] > QUALITY_FLOOR_3M].head(top_n).copy()
    top = qualified[qualified['atr'] > 0]
    if top.empty:
        return []

    total_etf_budget = portfolio_value * REGIME_ALLOCATION[regime]
    available_budget = max(0.0, total_etf_budget - deployed_usd)

    if available_budget < MIN_ALLOCATION_USD:
        return []

    top['atr_pct'] = top['atr'] / top['price']
    top['inv_vol'] = 1.0 / top['atr_pct']
    top['weight'] = top['inv_vol'] / top['inv_vol'].sum()
    top['allocation_usd'] = top['weight'] * available_budget
    top['qty'] = top.apply(lambda row: math.floor(row['allocation_usd'] / row['price']), axis=1)

    top['stop_price'] = top.apply(
        lambda row: round(row['price'] * (1 - row['stop_loss_pct']), 2), axis=1
    )
    top['take_profit_price'] = top.apply(
        lambda row: round(row['price'] * (1 + row['take_profit_pct']), 2), axis=1
    )

    top = top[top['qty'] > 0]
    if top.empty:
        return []

    return top.to_dict('records')


def get_news_for_candidates(symbols: list[str]) -> dict[str, list]:
    return get_news_sentiment(symbols, limit=3)


def print_morning_briefing(regime_data: dict, dashboard: pd.DataFrame, sizing: list[dict], news: dict):
    regime = regime_data['regime']
    print(f"\n{'='*60}")
    print(f"  MORNING BRIEFING")
    print(f"{'='*60}")
    print(f"\nMarket Regime: {regime.upper()}")
    print(f"  VIX: {regime_data['vix']:.1f}  |  SPY: ${regime_data['spy_price']:.2f}  |  200MA: ${regime_data['ma200']:.2f}")

    print(f"\n{'─'*60}")
    print(f"  ETF SIGNAL DASHBOARD")
    print(f"{'─'*60}")
    print(f"{'Rank':<5} {'ETF':<6} {'1M':>7} {'3M':>7} {'6M':>7} {'RSI':>6} {'Volume':<10} {'Stop%':>7} {'TP%':>7}")
    print(f"{'─'*60}")
    for _, row in dashboard.iterrows():
        print(
            f"{int(row['rank']):<5} {row['symbol']:<6} "
            f"{row['return_1m']*100:>6.1f}% "
            f"{row['return_3m']*100:>6.1f}% "
            f"{row['return_6m']*100:>6.1f}% "
            f"{row['rsi']:>6.1f} "
            f"{row['volume_trend']:<10} "
            f"{row['stop_loss_pct']*100:>6.1f}% "
            f"{row['take_profit_pct']*100:>6.1f}%"
        )

    if sizing:
        print(f"\n{'─'*60}")
        print(f"  RECOMMENDED POSITIONS (top {len(sizing)} by momentum)")
        print(f"{'─'*60}")
        for pos in sizing:
            print(f"\n  {pos['symbol']}")
            print(f"    Qty: {int(pos['qty'])} shares @ ~${pos['price']:.2f}")
            print(f"    Allocation: ${pos['allocation_usd']:,.0f} ({pos['weight']*100:.0f}% of ETF budget)")
            print(f"    Stop-loss:  ${pos['stop_price']:.2f}  ({pos['stop_loss_pct']*100:.1f}% below entry)")
            print(f"    Take-profit: ${pos['take_profit_price']:.2f}  ({pos['take_profit_pct']*100:.1f}% above entry)")
            print(f"    ATR (14d):  ${pos['atr']:.2f}")
    else:
        print("\n  No viable candidates for current regime.")

    if news:
        print(f"\n{'─'*60}")
        print(f"  NEWS")
        print(f"{'─'*60}")
        for symbol, articles in news.items():
            for a in articles:
                print(f"\n  [{symbol}] {a['sentiment_label'].upper()} ({a['sentiment_score']:+.2f})")
                print(f"  {a['title']}")
                print(f"  {a['summary'][:200]}...")

    print(f"\n{'='*60}")
    print(f"  Review above and confirm which positions to enter.")
    print(f"{'='*60}\n")


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

    print(f"\n{'='*60}")
    print(f"  INTRADAY SCAN")
    print(f"{'='*60}")
    print(f"\nMarket Regime: {regime.upper()}")
    print(f"  VIX: {regime_data['vix']:.1f}  |  SPY: ${regime_data['spy_price']:.2f}  |  200MA: ${regime_data['ma200']:.2f}")
    print(f"\n  ETF Budget: ${total_etf_budget:,.0f} total | ${deployed_usd:,.0f} deployed | ${remaining_budget:,.0f} available")

    if current_positions:
        print(f"\n{'─'*60}")
        print(f"  CURRENT POSITIONS ({len(current_positions)} / {MAX_POSITIONS} max)")
        print(f"{'─'*60}")
        for p in current_positions:
            pnl = float(p.unrealized_pl)
            sign = '+' if pnl >= 0 else ''
            print(f"  {p.symbol:<6}  {float(p.qty):.0f} shares  "
                  f"avg ${float(p.avg_entry_price):.2f}  "
                  f"now ${float(p.current_price):.2f}  "
                  f"P&L {sign}${pnl:.2f}")

    if not available.empty:
        print(f"\n{'─'*60}")
        print(f"  AVAILABLE CANDIDATES (not currently held, ranked by 3m momentum)")
        print(f"{'─'*60}")
        print(f"{'Rank':<5} {'ETF':<6} {'1M':>7} {'3M':>7} {'6M':>7} {'RSI':>6} {'Volume':<10} {'Qualifies'}")
        print(f"{'─'*60}")
        for _, row in available.head(10).iterrows():
            qualifies = 'YES' if row['return_3m'] > QUALITY_FLOOR_3M else 'no (neg 3m)'
            print(
                f"{int(row['rank']):<5} {row['symbol']:<6} "
                f"{row['return_1m']*100:>6.1f}% "
                f"{row['return_3m']*100:>6.1f}% "
                f"{row['return_6m']*100:>6.1f}% "
                f"{row['rsi']:>6.1f} "
                f"{row['volume_trend']:<10} "
                f"{qualifies}"
            )

    if sizing:
        print(f"\n{'─'*60}")
        print(f"  PROPOSED ADDITIONS ({len(sizing)})")
        print(f"{'─'*60}")
        for pos in sizing:
            print(f"\n  {pos['symbol']}")
            print(f"    Qty: {int(pos['qty'])} shares @ ~${pos['price']:.2f}")
            print(f"    Allocation: ${pos['allocation_usd']:,.0f} ({pos['weight']*100:.0f}% of available budget)")
            print(f"    Stop-loss:  ${pos['stop_price']:.2f}  ({pos['stop_loss_pct']*100:.1f}% below entry)")
            print(f"    Take-profit: ${pos['take_profit_price']:.2f}  ({pos['take_profit_pct']*100:.1f}% above entry)")
    else:
        print("\n  No new candidates qualify (need positive 3m return and sufficient remaining budget).")

    if news:
        print(f"\n{'─'*60}")
        print(f"  NEWS")
        print(f"{'─'*60}")
        for symbol, articles in news.items():
            for a in articles:
                print(f"\n  [{symbol}] {a['sentiment_label'].upper()} ({a['sentiment_score']:+.2f})")
                print(f"  {a['title']}")
                print(f"  {a['summary'][:200]}...")

    print(f"\n{'='*60}\n")
