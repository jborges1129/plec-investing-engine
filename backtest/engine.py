"""
Event-driven backtest engine for the ETF momentum strategy.

Design principles that prevent look-ahead bias:
  - Signals at date T use only data[symbol].loc[:T] (no future bars).
  - Entries fill at next trading day's open, not the signal day's close.
  - Stops fill at stop_price or open (whichever is worse) — gap-aware.
  - Portfolio value for sizing is computed from current cash + MTM positions
    using only data available at that moment.
"""

import math
import pandas as pd
from dataclasses import dataclass
from typing import Optional

from shared.data import (
    _compute_symbol_row,
    get_momentum,
    ETF_UNIVERSE,
)
from etf_module.strategy import (
    get_etf_candidates,
    QUALITY_FLOOR_12M,
    REGIME_ALLOCATION,
    MAX_POSITIONS,
    TRAIL_PCT,
    MIN_ALLOCATION_USD,
    WINNER_PROTECTION_THRESHOLD,
)
from backtest.data_loader import get_symbol_slice, get_next_trading_day

SLIPPAGE_BPS = 7.5  # per trade leg; ~15 bps round-trip


@dataclass
class Trade:
    symbol:             str
    entry_date:         pd.Timestamp
    entry_price:        float
    qty:                float
    stop_price:         float          # current trailing stop level (updated daily)
    position_high:      float          # highest price seen since entry (for trailing)
    trail_pct:          float          # e.g. 0.30 means stop trails 30% below high
    take_profit_price:  float          # set to large value — no fixed TP with trailing stops
    stop_loss_pct:      float          # initial stop distance at entry (= trail_pct)
    composite_score:    float          # 12m return used as rank score
    exit_date:          Optional[pd.Timestamp] = None
    exit_price:         Optional[float]        = None
    exit_reason:        Optional[str]          = None  # 'stop' | 'rebalance' | 'end'

    @property
    def pnl_pct(self) -> Optional[float]:
        if self.exit_price is None:
            return None
        return (self.exit_price - self.entry_price) / self.entry_price

    @property
    def pnl_usd(self) -> Optional[float]:
        if self.exit_price is None:
            return None
        return (self.exit_price - self.entry_price) * self.qty


# ── Internal helpers ──────────────────────────────────────────────────────────

def _compute_regime(
    data: dict,
    as_of: pd.Timestamp,
    all_trading_dates: pd.DatetimeIndex,
) -> dict:
    """Regime detection using only data available as of `as_of`."""
    spy_slice = get_symbol_slice(data, 'SPY', as_of)
    vix_slice = get_symbol_slice(data, '^VIX', as_of)

    if spy_slice.empty:
        return {'regime': 'neutral', 'vix': 20.0, 'spy_price': 0.0, 'ma200': 0.0}

    spy_close = spy_slice['Close']
    current   = float(spy_close.iloc[-1])
    ma200     = (
        float(spy_close.rolling(200).mean().iloc[-1])
        if len(spy_close) >= 200
        else current
    )

    vix_level = 20.0
    if not vix_slice.empty:
        vix_level = float(vix_slice['Close'].iloc[-1])

    if vix_level < 20 and current > ma200:
        regime = 'bull'
    elif vix_level > 25 and current < ma200:
        regime = 'bear'
    else:
        regime = 'neutral'

    return {'regime': regime, 'vix': vix_level, 'spy_price': current, 'ma200': ma200}


def _build_factor_df(
    data: dict,
    candidates: list[str],
    as_of: pd.Timestamp,
) -> pd.DataFrame:
    """
    Compute all 12 factor inputs for each candidate using only trailing data.
    Delegates to shared.data._compute_symbol_row — same logic as the live system.
    """
    rows: list[dict] = []

    spy_slice      = get_symbol_slice(data, 'SPY', as_of)
    spy_return_3m  = 0.0
    spy_return_12m = 0.0
    if len(spy_slice) >= 252:
        try:
            spy_mom        = get_momentum(spy_slice)
            spy_return_3m  = spy_mom['return_3m']
            spy_return_12m = spy_mom.get('return_12m', 0.0) or 0.0
        except Exception:
            pass

    for sym in candidates:
        df_slice = get_symbol_slice(data, sym, as_of)
        if len(df_slice) < 252:   # need ≥12m of data for the primary signal
            continue
        try:
            row = _compute_symbol_row(sym, df_slice)
            rows.append(row)
        except Exception:
            pass

    if not rows:
        return pd.DataFrame()

    out = pd.DataFrame(rows)
    out['spy_return_3m']  = spy_return_3m
    out['spy_return_12m'] = spy_return_12m
    return out


def _get_sizing_rows(
    scored_df: pd.DataFrame,
    portfolio_value: float,
    regime: str,
) -> list[dict]:
    """
    Inverse-vol sizing using 12m realized vol; trailing stop replaces ATR bracket.
    Each row carries trail_pct so the engine can initialize position_high at fill.
    """
    qualified = scored_df[scored_df['return_12m'] > QUALITY_FLOOR_12M].head(MAX_POSITIONS).copy()
    if qualified.empty:
        return []

    total_etf_budget = portfolio_value * REGIME_ALLOCATION[regime]
    if total_etf_budget < MIN_ALLOCATION_USD:
        return []

    safe_vol = qualified['realized_vol_12m'].replace(0, float('nan'))
    qualified['inv_vol'] = 1.0 / safe_vol
    qualified = qualified.dropna(subset=['inv_vol'])
    if qualified.empty:
        return []

    qualified['weight']         = qualified['inv_vol'] / qualified['inv_vol'].sum()
    qualified['allocation_usd'] = qualified['weight'] * total_etf_budget
    qualified['trail_pct']      = TRAIL_PCT

    return qualified.to_dict('records')


# ── Main backtest loop ────────────────────────────────────────────────────────

def run_backtest(
    data:              dict[str, pd.DataFrame],
    all_trading_dates: pd.DatetimeIndex,
    loop_dates:        pd.DatetimeIndex,
    rebalance_dates:   list[pd.Timestamp],
    initial_capital:   float = 10_000.0,
    slippage_bps:      float = SLIPPAGE_BPS,
) -> tuple[list[Trade], pd.Series]:
    """
    Walk forward through `loop_dates` day by day.
    Returns (completed_trades, portfolio_value_series).

    Ordering within each date:
      1. Fill pending entries at today's open (scheduled from prior rebalance)
      2. Check every open position for stop / take-profit triggers
      3. On rebalance days: score candidates, close removed positions, schedule new entries
      4. Mark-to-market portfolio value at close
    """
    cash               = initial_capital
    open_positions:    dict[str, Trade] = {}
    completed_trades:  list[Trade]      = []
    portfolio_series:  dict             = {}
    pending_entries:   list[dict]       = []   # entries to fill on their entry_date

    slippage_mult = slippage_bps / 10_000.0
    rebalance_set = set(rebalance_dates)

    for date in loop_dates:

        # ── 1. Fill pending entries at today's open ───────────────────────────
        remaining_pending: list[dict] = []
        for entry in pending_entries:
            if entry['entry_date'] != date:
                remaining_pending.append(entry)
                continue
            sym = entry['symbol']
            if sym not in data or date not in data[sym].index:
                continue
            entry_price = float(data[sym].loc[date, 'Open']) * (1 + slippage_mult)
            qty = math.floor(entry['allocation_usd'] / entry_price)
            if qty <= 0:
                continue
            cost = entry_price * qty
            if cost > cash:
                qty  = math.floor(cash / entry_price)
                cost = entry_price * qty
            if qty <= 0:
                continue
            trail = entry['trail_pct']
            trade = Trade(
                symbol            = sym,
                entry_date        = date,
                entry_price       = entry_price,
                qty               = qty,
                stop_price        = round(entry_price * (1 - trail), 4),
                position_high     = entry_price,
                trail_pct         = trail,
                take_profit_price = entry_price * 100.0,  # effectively disabled
                stop_loss_pct     = trail,
                composite_score   = entry['composite_score'],
            )
            cash -= cost
            open_positions[sym] = trade
        pending_entries = remaining_pending

        # ── 2. Check trailing stops ───────────────────────────────────────────
        to_close: list[str] = []
        for sym, pos in open_positions.items():
            if sym not in data or date not in data[sym].index:
                continue
            row      = data[sym].loc[date]
            day_open = float(row['Open'])
            day_low  = float(row['Low'])
            day_high = float(row['High'])

            # Trailing stop from yesterday's position high
            trail_stop = pos.position_high * (1 - pos.trail_pct)

            fill_price  = None
            exit_reason = None

            # Gap down through trailing stop?
            if day_open <= trail_stop:
                fill_price  = day_open * (1 - slippage_mult)
                exit_reason = 'stop'
            else:
                # Ratchet position high up on today's action, then check intraday
                new_high = max(pos.position_high, day_high)
                pos.position_high = new_high
                trail_stop_updated = new_high * (1 - pos.trail_pct)
                pos.stop_price = trail_stop_updated  # keep current for reference

                if day_low <= trail_stop_updated:
                    fill_price  = trail_stop_updated * (1 - slippage_mult)
                    exit_reason = 'stop'

            if fill_price is not None:
                pos.exit_date   = date
                pos.exit_price  = fill_price
                pos.exit_reason = exit_reason
                cash += fill_price * pos.qty
                completed_trades.append(pos)
                to_close.append(sym)

        for sym in to_close:
            del open_positions[sym]

        # ── 3. Rebalance ──────────────────────────────────────────────────────
        if date in rebalance_set:
            regime_data = _compute_regime(data, date, all_trading_dates)
            regime      = regime_data['regime']
            candidates  = get_etf_candidates(regime)
            factor_df   = _build_factor_df(data, candidates, date)

            if not factor_df.empty:
                # Pure 12m momentum rank — no composite scoring model
                scored_df = factor_df.copy()
                scored_df['composite_score'] = scored_df['return_12m']
                scored_df = scored_df.sort_values(
                    'return_12m', ascending=False
                ).reset_index(drop=True)

                # Portfolio value for sizing (cash + current open positions at close)
                portfolio_value = cash
                for sym, pos in open_positions.items():
                    if sym in data and date in data[sym].index:
                        portfolio_value += float(data[sym].loc[date, 'Close']) * pos.qty

                sizing_rows   = _get_sizing_rows(scored_df, portfolio_value, regime)
                target_symbols = {r['symbol'] for r in sizing_rows}

                # Close positions no longer in the target, but protect winning ones.
                # If a position is up > WINNER_PROTECTION_THRESHOLD, skip rotating it out —
                # let it ride to its natural stop or take-profit rather than cutting early.
                to_rebalance_out = [s for s in list(open_positions) if s not in target_symbols]
                for sym in to_rebalance_out:
                    pos = open_positions[sym]
                    current_close = (
                        float(data[sym].loc[date, 'Close'])
                        if sym in data and date in data[sym].index
                        else pos.entry_price
                    )
                    unrealized_pct = (current_close - pos.entry_price) / pos.entry_price
                    if unrealized_pct > WINNER_PROTECTION_THRESHOLD:
                        continue  # let winner ride — stop/TP will handle exit
                    fill = current_close * (1 - slippage_mult)
                    pos.exit_date   = date
                    pos.exit_price  = fill
                    pos.exit_reason = 'rebalance'
                    cash += fill * pos.qty
                    completed_trades.append(pos)
                    del open_positions[sym]

                # Capital already locked in staying positions
                staying_value = sum(
                    float(data[sym].loc[date, 'Close']) * open_positions[sym].qty
                    for sym in open_positions
                    if sym in data and date in data[sym].index
                )

                total_etf_budget = portfolio_value * REGIME_ALLOCATION[regime]
                available        = max(0.0, total_etf_budget - staying_value)

                # Schedule entries only for symbols not already held
                new_entries    = [r for r in sizing_rows if r['symbol'] not in open_positions]
                total_new_wt   = sum(r['weight'] for r in new_entries)
                next_day       = get_next_trading_day(all_trading_dates, date)

                if new_entries and available >= MIN_ALLOCATION_USD and next_day is not None:
                    for row in new_entries:
                        sym = row['symbol']
                        if sym not in data:
                            continue
                        rel_wt = row['weight'] / total_new_wt if total_new_wt > 0 else 1.0 / len(new_entries)
                        alloc  = rel_wt * available
                        if alloc < MIN_ALLOCATION_USD:
                            continue
                        pending_entries.append({
                            'symbol':          sym,
                            'entry_date':      next_day,
                            'allocation_usd':  alloc,
                            'trail_pct':       row['trail_pct'],
                            'composite_score': row['composite_score'],
                        })

        # ── 4. Mark-to-market at close ────────────────────────────────────────
        mtm = cash
        for sym, pos in open_positions.items():
            if sym in data and date in data[sym].index:
                mtm += float(data[sym].loc[date, 'Close']) * pos.qty
        portfolio_series[date] = mtm

    # End of backtest: close all remaining positions at last available close
    if len(loop_dates) > 0:
        last_date = loop_dates[-1]
        for sym, pos in list(open_positions.items()):
            fill = (
                float(data[sym].loc[last_date, 'Close']) * (1 - slippage_mult)
                if sym in data and last_date in data[sym].index
                else pos.entry_price
            )
            pos.exit_date   = last_date
            pos.exit_price  = fill
            pos.exit_reason = 'end'
            completed_trades.append(pos)

    return completed_trades, pd.Series(portfolio_series)
