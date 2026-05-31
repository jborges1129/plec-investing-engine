"""Performance metrics for the backtest."""

import numpy as np
import pandas as pd
from backtest.engine import Trade


def compute_metrics(
    trades:          list[Trade],
    portfolio:       pd.Series,
    initial_capital: float,
    spy_data:        pd.Series = None,
) -> dict:
    """
    Compute CAGR, annualized vol, Sharpe (4% risk-free), max drawdown,
    win rate, avg win/loss, quarter-Kelly fraction, and SPY benchmark.
    """
    years        = (portfolio.index[-1] - portfolio.index[0]).days / 365.25
    final_value  = float(portfolio.iloc[-1])
    cagr         = (final_value / initial_capital) ** (1 / years) - 1 if years > 0 else 0.0

    daily_returns = portfolio.pct_change().dropna()
    ann_vol       = float(daily_returns.std() * np.sqrt(252))

    risk_free_daily  = 0.04 / 252
    excess           = daily_returns - risk_free_daily
    sharpe           = float(excess.mean() / excess.std() * np.sqrt(252)) if excess.std() > 0 else 0.0

    roll_max    = portfolio.cummax()
    drawdowns   = (portfolio - roll_max) / roll_max
    max_drawdown = float(drawdowns.min())

    closed   = [t for t in trades if t.pnl_pct is not None]
    pnl_pcts = [t.pnl_pct for t in closed]
    winners  = [p for p in pnl_pcts if p > 0]
    losers   = [p for p in pnl_pcts if p <= 0]

    win_rate       = len(winners) / len(closed) if closed else 0.0
    avg_win        = float(np.mean(winners)) if winners else 0.0
    avg_loss       = float(np.mean(losers))  if losers  else 0.0
    win_loss_ratio = abs(avg_win / avg_loss)  if avg_loss != 0 else float('inf')

    # Kelly criterion: f* = p - q/b  where p=win_rate, q=1-p, b=win/loss ratio
    kelly_f       = win_rate - (1 - win_rate) / win_loss_ratio if win_loss_ratio > 0 and win_loss_ratio != float('inf') else 0.0
    quarter_kelly = kelly_f / 4

    exit_counts: dict[str, int] = {}
    for t in closed:
        exit_counts[t.exit_reason] = exit_counts.get(t.exit_reason, 0) + 1

    # SPY buy-and-hold benchmark over same period
    spy_metrics: dict = {}
    if spy_data is not None and len(spy_data) > 0:
        spy_range = spy_data.loc[portfolio.index[0]:portfolio.index[-1]].dropna()
        if len(spy_range) > 1:
            spy_cagr    = (spy_range.iloc[-1] / spy_range.iloc[0]) ** (1 / years) - 1
            spy_dd      = ((spy_range - spy_range.cummax()) / spy_range.cummax()).min()
            spy_ret     = spy_range.pct_change().dropna()
            spy_vol     = float(spy_ret.std() * np.sqrt(252))
            spy_sharpe  = float((spy_ret.mean() - risk_free_daily) / spy_ret.std() * np.sqrt(252))
            spy_metrics = {
                'spy_cagr':        spy_cagr,
                'spy_max_drawdown': float(spy_dd),
                'spy_ann_vol':     spy_vol,
                'spy_sharpe':      spy_sharpe,
            }

    return {
        'cagr':           cagr,
        'ann_vol':        ann_vol,
        'sharpe':         sharpe,
        'max_drawdown':   max_drawdown,
        'win_rate':       win_rate,
        'avg_win_pct':    avg_win,
        'avg_loss_pct':   avg_loss,
        'win_loss_ratio': win_loss_ratio,
        'kelly_f':        kelly_f,
        'quarter_kelly':  quarter_kelly,
        'total_trades':   len(closed),
        'years':          years,
        'final_value':    final_value,
        'exit_reasons':   exit_counts,
        **spy_metrics,
    }


def print_metrics(metrics: dict, initial_capital: float) -> None:
    W = 62
    print(f"\n{'='*W}")
    print(f"  BACKTEST RESULTS  ({metrics['years']:.1f} years)")
    print(f"{'='*W}")
    print(f"  Initial capital:   ${initial_capital:>12,.2f}")
    print(f"  Final value:       ${metrics['final_value']:>12,.2f}")
    print(f"  CAGR:              {metrics['cagr']*100:>10.2f}%")
    print(f"  Annualized vol:    {metrics['ann_vol']*100:>10.2f}%")
    print(f"  Sharpe (4% RF):    {metrics['sharpe']:>10.2f}")
    print(f"  Max drawdown:      {metrics['max_drawdown']*100:>10.2f}%")

    if 'spy_cagr' in metrics:
        print(f"{'─'*W}")
        print(f"  SPY buy-and-hold (same period):")
        print(f"    CAGR:            {metrics['spy_cagr']*100:>10.2f}%")
        print(f"    Max drawdown:    {metrics['spy_max_drawdown']*100:>10.2f}%")
        print(f"    Sharpe:          {metrics['spy_sharpe']:>10.2f}")
        excess_cagr = metrics['cagr'] - metrics['spy_cagr']
        print(f"  Excess CAGR:       {excess_cagr*100:>+10.2f}%")

    print(f"{'─'*W}")
    print(f"  Total trades:      {metrics['total_trades']:>10}")
    print(f"  Win rate:          {metrics['win_rate']*100:>10.2f}%")
    print(f"  Avg win:           {metrics['avg_win_pct']*100:>10.2f}%")
    print(f"  Avg loss:          {metrics['avg_loss_pct']*100:>10.2f}%")
    print(f"  Win/loss ratio:    {metrics['win_loss_ratio']:>10.2f}x")
    print(f"{'─'*W}")
    print(f"  Kelly fraction:    {metrics['kelly_f']*100:>9.2f}%")
    print(f"  Quarter-Kelly:     {metrics['quarter_kelly']*100:>9.2f}%  ← suggested sizing")

    if metrics.get('exit_reasons'):
        print(f"{'─'*W}")
        print(f"  Exit reasons:")
        for reason, count in sorted(metrics['exit_reasons'].items(), key=lambda x: -x[1]):
            print(f"    {reason:<15} {count:>5}")
    print(f"{'='*W}\n")


def trades_to_dataframe(trades: list[Trade]) -> pd.DataFrame:
    rows = []
    for t in trades:
        rows.append({
            'symbol':            t.symbol,
            'entry_date':        t.entry_date,
            'entry_price':       t.entry_price,
            'exit_date':         t.exit_date,
            'exit_price':        t.exit_price,
            'exit_reason':       t.exit_reason,
            'qty':               t.qty,
            'pnl_pct':           t.pnl_pct,
            'pnl_usd':           t.pnl_usd,
            'stop_price':        t.stop_price,
            'take_profit_price': t.take_profit_price,
            'stop_loss_pct':     t.stop_loss_pct,
            'composite_score':   t.composite_score,
        })
    return pd.DataFrame(rows)
