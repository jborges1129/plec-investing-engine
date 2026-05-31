"""
Monte Carlo resampling of trade outcomes.

Approach: bootstrap-resample the empirical trade P&L distribution N times,
compound the returns sequentially, compute ending capital and max drawdown
for each simulation, then plot the percentile fan and 5th-pct worst-case path.

Assumption of independence between trades is a simplification (concurrent
ETF positions have correlated returns), but is appropriate for a pitch-level
analysis. A more rigorous approach would resample monthly portfolio returns.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')   # headless — no display required
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

from backtest.engine import Trade


def run_monte_carlo(
    trades:          list[Trade],
    initial_capital: float,
    n_sims:          int = 10_000,
    seed:            int = 42,
) -> dict:
    """
    Bootstrap-resample trade P&L percentages to produce a distribution of outcomes.

    Returns:
        paths          — (n_sims, n_trades+1) array of portfolio values per simulation
        final_values   — (n_sims,) array of terminal portfolio values
        max_drawdowns  — (n_sims,) array of worst peak-to-trough drawdown per sim
        percentiles    — {metric: {5: ..., 25: ..., 50: ..., 75: ..., 95: ...}}
    """
    closed = [t for t in trades if t.pnl_pct is not None]
    if not closed:
        print("  No completed trades — skipping Monte Carlo.")
        return {}

    rng      = np.random.default_rng(seed)
    pnl_pcts = np.array([t.pnl_pct for t in closed])
    n_trades = len(pnl_pcts)

    # (n_sims, n_trades): each row is one resampled sequence of trade returns
    samples    = rng.choice(pnl_pcts, size=(n_sims, n_trades), replace=True)
    cumulative = np.cumprod(1 + samples, axis=1) * initial_capital

    # Prepend starting capital so paths begin at initial_capital
    starting = np.full((n_sims, 1), initial_capital)
    paths    = np.hstack([starting, cumulative])

    final_values = paths[:, -1]

    # Max drawdown per simulation
    roll_max     = np.maximum.accumulate(paths, axis=1)
    dd_matrix    = (paths - roll_max) / roll_max
    max_drawdowns = dd_matrix.min(axis=1)

    pcts = [5, 25, 50, 75, 95]
    return {
        'paths':         paths,
        'final_values':  final_values,
        'max_drawdowns': max_drawdowns,
        'n_trades':      n_trades,
        'n_sims':        n_sims,
        'percentiles': {
            'final_value':  {p: np.percentile(final_values,  p) for p in pcts},
            'max_drawdown': {p: np.percentile(max_drawdowns, p) for p in pcts},
        },
    }


def plot_results(
    portfolio:       pd.Series,
    metrics:         dict,
    mc_results:      dict,
    initial_capital: float,
    spy_close:       pd.Series = None,
    output_dir:      str       = 'backtest',
) -> str:
    """
    Save four-panel chart to output_dir/backtest_results.png.
    Returns the saved file path.
    """
    import os
    os.makedirs(output_dir, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle(
        'ETF Momentum Strategy — Backtest Results',
        fontsize=15, fontweight='bold', y=1.01,
    )

    usd_fmt = mticker.FuncFormatter(lambda x, _: f'${x:,.0f}')

    # ── Panel 1: Equity curve ─────────────────────────────────────────────────
    ax = axes[0, 0]
    ax.plot(portfolio.index, portfolio.values,
            color='#2196F3', linewidth=1.5, label='Strategy', zorder=3)
    if spy_close is not None:
        spy_range   = spy_close.loc[portfolio.index[0]:portfolio.index[-1]].dropna()
        spy_scaled  = spy_range / spy_range.iloc[0] * initial_capital
        ax.plot(spy_scaled.index, spy_scaled.values,
                color='#9E9E9E', linewidth=1.0, linestyle='--', label='SPY (scaled)', zorder=2)
    ax.axhline(initial_capital, color='#BDBDBD', linewidth=0.8, linestyle=':')
    ax.set_title('Portfolio Equity Curve')
    ax.set_ylabel('Portfolio Value')
    ax.yaxis.set_major_formatter(usd_fmt)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.25)

    # ── Panel 2: Drawdown ─────────────────────────────────────────────────────
    ax = axes[0, 1]
    roll_max  = portfolio.cummax()
    dd_series = (portfolio - roll_max) / roll_max * 100
    ax.fill_between(dd_series.index, dd_series.values, 0, color='#F44336', alpha=0.45)
    ax.plot(dd_series.index, dd_series.values, color='#B71C1C', linewidth=0.8)
    ax.set_title(f'Drawdown  (max: {metrics["max_drawdown"]*100:.1f}%)')
    ax.set_ylabel('Drawdown (%)')
    ax.grid(True, alpha=0.25)

    # ── Panel 3: Monte Carlo fan ──────────────────────────────────────────────
    ax = axes[1, 0]
    if mc_results:
        paths    = mc_results['paths']
        n_trades = mc_results['n_trades']
        x        = np.arange(paths.shape[1])

        p5  = np.percentile(paths,  5, axis=0)
        p25 = np.percentile(paths, 25, axis=0)
        p50 = np.percentile(paths, 50, axis=0)
        p75 = np.percentile(paths, 75, axis=0)
        p95 = np.percentile(paths, 95, axis=0)

        ax.fill_between(x, p5,  p95,  color='#90CAF9', alpha=0.25, label='5–95th pct')
        ax.fill_between(x, p25, p75,  color='#42A5F5', alpha=0.35, label='25–75th pct')
        ax.plot(x, p50, color='#1565C0', linewidth=1.6, label='Median')
        ax.plot(x, p5,  color='#B71C1C', linewidth=1.2, linestyle='--',
                label='5th pct — worst-case path')
        ax.axhline(initial_capital, color='#BDBDBD', linewidth=0.8, linestyle=':')
        ax.set_title(
            f'Monte Carlo  ({mc_results["n_sims"]:,} sims, {n_trades} trades resampled)'
        )
        ax.set_xlabel('Trade number')
        ax.set_ylabel('Portfolio Value')
        ax.yaxis.set_major_formatter(usd_fmt)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.25)
    else:
        ax.text(0.5, 0.5, 'Insufficient trades for Monte Carlo',
                ha='center', va='center', transform=ax.transAxes)

    # ── Panel 4: Annual returns bar ───────────────────────────────────────────
    ax = axes[1, 1]
    annual = portfolio.resample('YE').last().pct_change().dropna() * 100
    colors = ['#4CAF50' if v >= 0 else '#F44336' for v in annual.values]
    ax.bar(annual.index.year, annual.values, color=colors, width=0.7, zorder=2)
    ax.axhline(0, color='black', linewidth=0.8)
    ax.set_title('Annual Returns')
    ax.set_ylabel('Return (%)')
    ax.set_xlabel('Year')
    ax.grid(True, alpha=0.25, axis='y')

    plt.tight_layout()
    out_path = os.path.join(output_dir, 'backtest_results.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Chart saved: {out_path}")
    return out_path


def print_monte_carlo_summary(mc_results: dict, initial_capital: float) -> None:
    if not mc_results:
        return
    p = mc_results['percentiles']
    W = 62
    print(f"\n{'='*W}")
    print(f"  MONTE CARLO  ({mc_results['n_sims']:,} simulations, "
          f"{mc_results['n_trades']} trades resampled)")
    print(f"{'─'*W}")
    print(f"  Final portfolio value distribution:")
    for pct in [5, 25, 50, 75, 95]:
        v    = p['final_value'][pct]
        mult = v / initial_capital
        print(f"    {pct:2d}th pct:  ${v:>10,.0f}  ({mult:.2f}x starting capital)")
    print(f"{'─'*W}")
    print(f"  Max drawdown distribution:")
    for pct in [5, 25, 50, 75, 95]:
        dd = p['max_drawdown'][pct] * 100
        print(f"    {pct:2d}th pct:  {dd:>7.1f}%")
    worst_case = p['final_value'][5]
    print(f"{'─'*W}")
    print(f"  5th-pct scenario: ${worst_case:,.0f} terminal — "
          f"{(worst_case/initial_capital - 1)*100:.1f}% total return")
    print(f"{'='*W}\n")
