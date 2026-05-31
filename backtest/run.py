"""
ETF Momentum Strategy — Backtester Entry Point

Usage:
    python -m backtest.run
    python -m backtest.run --capital 10000 --start 2010-01-01 --end 2026-05-31 --sims 10000

Output files (written to --out directory, default: backtest/results/):
    backtest_results.png  — four-panel chart (equity, drawdown, Monte Carlo, annual returns)
    portfolio.csv         — daily portfolio value
    trades.csv            — full trade log with P&L, stop, take-profit, exit reason
    metrics.json          — all performance statistics

Look-ahead bias prevention summary (verifiable in engine.py):
    1. Signals at date T use data[symbol].loc[:T] only — zero future bars
    2. Entries fill at next trading day's open, not signal day's close
    3. Stops fill at open (if gap) or stop_price — never magically at exact level
    4. Factors normalized cross-sectionally within each rebalance date, not full history
    5. Regime detection uses only SPY/VIX data available as of the rebalance date
"""

import argparse
import json
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shared.data import ETF_UNIVERSE
from backtest.data_loader import (
    fetch_all_history,
    get_trading_dates,
    get_rebalance_dates,
)
from backtest.engine import run_backtest
from backtest.metrics import compute_metrics, print_metrics, trades_to_dataframe
from backtest.monte_carlo import run_monte_carlo, plot_results, print_monte_carlo_summary


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description='ETF Momentum Strategy Backtester',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument('--capital',  type=float, default=10_000.0,
                   help='Starting capital in USD')
    p.add_argument('--start',    type=str,   default='2010-01-01',
                   help='Backtest start date (YYYY-MM-DD)')
    p.add_argument('--end',      type=str,   default='2026-05-31',
                   help='Backtest end date (YYYY-MM-DD)')
    p.add_argument('--sims',     type=int,   default=10_000,
                   help='Number of Monte Carlo simulations')
    p.add_argument('--slippage', type=float, default=7.5,
                   help='Slippage in basis points per trade leg')
    p.add_argument('--out',      type=str,   default='backtest/results',
                   help='Output directory for results and charts')
    return p.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(args.out, exist_ok=True)

    # ── 1. Download historical data ───────────────────────────────────────────
    # Fetch 1 extra year before the start date so that 200-day MAs and other
    # indicators have enough warmup data on the first signal date.
    fetch_start = str(pd.Timestamp(args.start) - pd.DateOffset(years=1))[:10]
    data = fetch_all_history(ETF_UNIVERSE, start=fetch_start, end=args.end)

    if not data:
        print("ERROR: No data downloaded. Check internet connection.")
        return

    all_trading_dates = get_trading_dates(data)

    # Backtest loop runs from args.start onwards; warmup data is used for slices
    loop_dates = all_trading_dates[all_trading_dates >= pd.Timestamp(args.start)]
    if len(loop_dates) == 0:
        print(f"ERROR: No trading dates found in [{args.start}, {args.end}].")
        return

    rebalance_dates = get_rebalance_dates(loop_dates, args.start, args.end)
    print(f"\n  Backtest period : {args.start} → {args.end}")
    print(f"  Trading days    : {len(loop_dates):,}")
    print(f"  Rebalance dates : {len(rebalance_dates)}")
    print(f"  Starting capital: ${args.capital:,.2f}")
    print(f"  Slippage        : {args.slippage} bps per leg\n")

    # ── 2. Run event-driven backtest ──────────────────────────────────────────
    print("Running backtest...")
    trades, portfolio = run_backtest(
        data              = data,
        all_trading_dates = all_trading_dates,
        loop_dates        = loop_dates,
        rebalance_dates   = rebalance_dates,
        initial_capital   = args.capital,
        slippage_bps      = args.slippage,
    )
    print(f"  Completed: {len(trades)} trades")

    # ── 3. Performance metrics ────────────────────────────────────────────────
    spy_close = data['SPY']['Close'] if 'SPY' in data else None
    metrics   = compute_metrics(trades, portfolio, args.capital, spy_data=spy_close)
    print_metrics(metrics, args.capital)

    # ── 4. Monte Carlo simulation ─────────────────────────────────────────────
    print(f"Running Monte Carlo ({args.sims:,} simulations)...")
    mc = run_monte_carlo(trades, args.capital, n_sims=args.sims)
    print_monte_carlo_summary(mc, args.capital)

    # ── 5. Save outputs ───────────────────────────────────────────────────────
    portfolio_path = os.path.join(args.out, 'portfolio.csv')
    portfolio.to_csv(portfolio_path, header=['portfolio_value'])
    print(f"  Saved: {portfolio_path}")

    trades_path = os.path.join(args.out, 'trades.csv')
    trades_to_dataframe(trades).to_csv(trades_path, index=False)
    print(f"  Saved: {trades_path}")

    # Serialize metrics (handle non-JSON types)
    def _jsonify(v):
        if isinstance(v, float) and (v != v):  # NaN
            return None
        if isinstance(v, (float, int)):
            return v
        return str(v)

    metrics_out  = {k: _jsonify(v) if not isinstance(v, dict) else v
                    for k, v in metrics.items()}
    metrics_path = os.path.join(args.out, 'metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics_out, f, indent=2, default=str)
    print(f"  Saved: {metrics_path}")

    # ── 6. Charts ─────────────────────────────────────────────────────────────
    try:
        plot_results(
            portfolio       = portfolio,
            metrics         = metrics,
            mc_results      = mc,
            initial_capital = args.capital,
            spy_close       = spy_close,
            output_dir      = args.out,
        )
    except ImportError as e:
        print(f"  Skipping charts — matplotlib unavailable: {e}")
        print("  Install with: pip install matplotlib")

    print(f"\nBacktest complete. Results in: {args.out}/")


if __name__ == '__main__':
    main()
