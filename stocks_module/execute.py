"""
Human-in-the-loop execution workflow for individual stock positions.

Flow:
  1. Human runs `python run.py stocks` → reviews today's screener output
  2. Picks a symbol → `python run.py stocks-buy SYMBOL`
  3. This module shows the proposed order (entry, stop, target, notional, risk $)
  4. Human confirms → bracket order placed on Alpaca, logged to database

Position sizing:
  - Risk-based: risk at most STOCK_RISK_PCT of portfolio value per trade
  - Notional cap: no single stock position exceeds STOCK_MAX_POSITION_PCT
  - Stop: STOCK_ATR_MULTIPLIER × ATR below entry
  - Take-profit: STOCK_REWARD_RISK × stop distance above entry
"""

import json
import os
from collections import defaultdict
from datetime import datetime

import yfinance as yf

from shared.alpaca import get_account, get_positions, place_bracket_order, get_client
from shared.database import init_db, log_trade, get_open_trades, close_trade
from shared.notify import send_alert, format_position_alert
from shared.data import get_atr

STOCK_RISK_PCT         = 0.01   # risk at most 1% of portfolio per trade
STOCK_MAX_POSITION_PCT = 0.08   # max 8% of portfolio notional per position
STOCK_MAX_POSITIONS    = 8      # max concurrent stock positions
STOCK_ATR_MULTIPLIER   = 2.0    # stop = 2× ATR below entry
STOCK_REWARD_RISK      = 3.0    # take-profit = 3:1 reward-to-risk ratio


def compute_sizing(
    symbol: str,
    price: float,
    atr: float,
    portfolio_value: float,
    available_cash: float,
) -> dict | None:
    """
    Risk-based position sizing. Returns a sizing dict or None if the position
    would be too small after applying both the risk cap and notional cap.
    """
    if price <= 0 or atr <= 0:
        return None

    stop_distance = atr * STOCK_ATR_MULTIPLIER
    tp_distance   = stop_distance * STOCK_REWARD_RISK

    # Qty limited by 1% portfolio risk
    max_risk_usd = portfolio_value * STOCK_RISK_PCT
    risk_qty     = int(max_risk_usd / stop_distance)

    # Qty limited by 8% notional cap
    max_notional   = portfolio_value * STOCK_MAX_POSITION_PCT
    notional_qty   = int(max_notional / price)

    qty = min(risk_qty, notional_qty)
    if qty <= 0:
        return None

    # Cash guard
    cost = qty * price
    if cost > available_cash:
        qty  = int(available_cash / price)
        cost = qty * price
    if qty <= 0:
        return None

    stop_loss_pct    = stop_distance / price
    take_profit_pct  = tp_distance   / price

    return {
        'symbol':            symbol,
        'qty':               qty,
        'price':             price,
        'atr':               atr,
        'allocation_usd':    cost,
        'stop_price':        round(price * (1 - stop_loss_pct),  2),
        'take_profit_price': round(price * (1 + take_profit_pct), 2),
        'stop_loss_pct':     stop_loss_pct,
        'take_profit_pct':   take_profit_pct,
        'risk_usd':          qty * stop_distance,
        'risk_pct':          (qty * stop_distance) / portfolio_value,
    }


def _load_candidate(symbol: str) -> dict | None:
    """Load screener data for SYMBOL from today's cache, if available."""
    today_str = datetime.now().strftime('%Y-%m-%d')
    base_dir  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cache     = os.path.join(base_dir, 'data', f'screener_cache_{today_str}.json')
    if not os.path.exists(cache):
        return None
    with open(cache) as f:
        candidates = json.load(f)
    return next((c for c in candidates if c.get('symbol') == symbol), None)


def _fetch_live_quote(symbol: str) -> tuple[float, float]:
    """Fetch current price and 14-day ATR via yfinance. Returns (price, atr)."""
    df = yf.Ticker(symbol).history(period='3mo', interval='1d')
    if df.empty:
        raise ValueError(f"No price data for {symbol}")
    price = float(df['Close'].iloc[-1])
    atr   = get_atr(df)
    return price, atr


def buy_stock(symbol: str, force_refresh: bool = False) -> None:
    """
    Interactive single-stock entry flow. Shows the proposed bracket order,
    asks for confirmation, then places it on Alpaca and logs to the database.
    """
    init_db()

    # Guard: already at max positions
    open_stock_trades = get_open_trades(module='stocks')
    if len(open_stock_trades) >= STOCK_MAX_POSITIONS:
        print(f"  Already at max {STOCK_MAX_POSITIONS} stock positions — close one first.")
        return

    # Guard: already holding this symbol in stocks module
    open_stock_symbols = {t.symbol for t in open_stock_trades}
    if symbol in open_stock_symbols:
        print(f"  {symbol} is already an open stock position.")
        return

    # Account snapshot
    account         = get_account()
    portfolio_value = float(account.portfolio_value)
    cash            = float(account.cash)

    # Try to pull candidate from today's screener cache
    candidate = _load_candidate(symbol)
    if candidate is None and not force_refresh:
        print(f"\n  ⚠  {symbol} is not in today's screener cache.")
        print(f"  Run 'python run.py stocks' first to generate today's candidates.")
        proceed = input("  Proceed anyway using a live price quote? (yes/no): ").strip().lower()
        if proceed != 'yes':
            return
        candidate = {}

    # Get price + ATR from screener data or live quote
    try:
        if candidate.get('price') and candidate.get('atr'):
            price = float(candidate['price'])
            atr   = float(candidate['atr'])
        else:
            print(f"  Fetching live quote for {symbol}...")
            price, atr = _fetch_live_quote(symbol)
            if candidate:
                candidate['price'] = price
                candidate['atr']   = atr
    except Exception as e:
        print(f"  Could not fetch price data: {e}")
        return

    # Compute sizing
    sizing = compute_sizing(symbol, price, atr, portfolio_value, cash)
    if sizing is None:
        print(f"  Position too small — not enough cash or allocation for {symbol}.")
        return

    # Display proposed order
    W = 62
    print(f"\n{'─'*W}")
    print(f"  PROPOSED STOCK ORDER: {symbol}")
    print(f"{'─'*W}")
    print(f"  Setup:       {candidate.get('setup_name', 'Manual')} (Tier {candidate.get('setup_tier', '?')})")
    print(f"  Score:       {candidate.get('score', '—')}/10  |  Sector: {candidate.get('sector', 'Unknown')}")
    print(f"  3M return:   {candidate.get('ret_3m', 0)*100:+.1f}%  |  RS vs SPY: {(candidate.get('rs_spy') or 0)*100:+.1f}%  |  RSI: {candidate.get('rsi', 0):.0f}")
    print(f"{'─'*W}")
    print(f"  Entry (mkt): ~${price:.2f}")
    print(f"  Shares:      {sizing['qty']}")
    print(f"  Notional:    ${sizing['allocation_usd']:,.2f}  ({sizing['allocation_usd']/portfolio_value*100:.1f}% of ${portfolio_value:,.0f})")
    print(f"{'─'*W}")
    print(f"  Stop-loss:   ${sizing['stop_price']:.2f}  ({sizing['stop_loss_pct']*100:.1f}% below, {STOCK_ATR_MULTIPLIER}× ATR)")
    print(f"  Take-profit: ${sizing['take_profit_price']:.2f}  ({sizing['take_profit_pct']*100:.1f}% above, {STOCK_REWARD_RISK}:1 R/R)")
    print(f"  Max risk:    ${sizing['risk_usd']:.2f}  ({sizing['risk_pct']*100:.2f}% of portfolio)")
    print(f"{'─'*W}")
    print(f"  Cash avail:  ${cash:,.2f}")
    print(f"  Open stocks: {len(open_stock_trades)}/{STOCK_MAX_POSITIONS} positions")
    print(f"{'─'*W}\n")

    confirm = input("  Execute this order? (yes/no): ").strip().lower()
    if confirm != 'yes':
        print("  Cancelled.")
        return

    # Guard: pending orders on this symbol
    client = get_client()
    try:
        open_orders    = client.get_orders()
        pending_syms   = {o.symbol for o in open_orders}
        if symbol in pending_syms:
            print(f"  {symbol} already has a pending order — skipping.")
            return
    except Exception:
        pass

    # Guard: already held on Alpaca (outside of this module)
    current_positions = get_positions()
    if any(p.symbol == symbol for p in current_positions):
        print(f"  {symbol} is already an open Alpaca position.")
        return

    # Place bracket order
    try:
        result, stop, tp = place_bracket_order(
            symbol          = symbol,
            qty             = sizing['qty'],
            entry_price     = price,
            stop_loss_pct   = sizing['stop_loss_pct'],
            take_profit_pct = sizing['take_profit_pct'],
        )
        fill_price = float(result.filled_avg_price) if result.filled_avg_price else price

        log_trade(
            order_id    = str(result.id),
            symbol      = symbol,
            module      = 'stocks',
            side        = 'buy',
            qty         = sizing['qty'],
            entry_price = fill_price,
            stop_loss   = stop,
            take_profit = tp,
        )

        print(f"\n  Order submitted: {result.id}")
        print(f"    {sizing['qty']} shares of {symbol}")
        print(f"    Stop:   ${stop:.2f}")
        print(f"    Target: ${tp:.2f}")
        print(f"    Max loss if stopped: -${sizing['risk_usd']:.2f} ({sizing['risk_pct']*100:.2f}%)")

    except Exception as e:
        print(f"\n  Error placing order: {e}")


def check_closed_positions() -> None:
    """
    Detect any stock positions that closed on Alpaca since the last check,
    update the database, and send an alert.
    """
    open_trades = get_open_trades(module='stocks')
    if not open_trades:
        print("  No open stock trades in database.")
        return

    client            = get_client()
    current_positions = {p.symbol: p for p in get_positions()}

    symbol_trades: dict[str, list] = defaultdict(list)
    for t in open_trades:
        symbol_trades[t.symbol].append(t)
    for trades in symbol_trades.values():
        trades.sort(key=lambda t: t.entry_time)

    for symbol, trades in symbol_trades.items():
        pos       = current_positions.get(symbol)
        alpaca_qty = float(pos.qty) if pos else 0.0
        db_total   = sum(t.qty for t in trades)

        if alpaca_qty >= db_total - 0.5:
            continue  # still fully open

        cumulative = 0.0
        for trade in trades:
            cumulative += trade.qty
            if alpaca_qty >= cumulative - 0.5:
                continue

            exit_price = _lookup_exit_price(client, trade)
            if exit_price is None:
                print(f"  {symbol}: order {trade.order_id[:8]} closed — exit price unknown.")
                exit_price = 0.0

            close_trade(trade.order_id, exit_price)
            pnl     = (exit_price - trade.entry_price) * trade.qty if trade.entry_price else 0.0
            pnl_pct = (exit_price - trade.entry_price) / trade.entry_price if trade.entry_price else 0.0
            subj, body = format_position_alert(symbol, 'closed', exit_price, pnl, pnl_pct)
            send_alert(subj, body)
            print(f"  {symbol} closed — P&L ${pnl:+.2f} ({pnl_pct*100:+.1f}%)")


def _lookup_exit_price(client, trade) -> float | None:
    """Return the fill price for a closed bracket order, or None if not found."""
    try:
        from alpaca.trading.requests import GetOrdersRequest
        from alpaca.trading.enums import QueryOrderStatus
        req    = GetOrdersRequest(
            status=QueryOrderStatus.CLOSED,
            symbols=[trade.symbol],
            limit=20,
        )
        orders = client.get_orders(filter=req)
        for o in orders:
            matched = str(o.id) == trade.order_id or (
                hasattr(o, 'legs') and o.legs and
                any(str(leg.id) == trade.order_id for leg in o.legs)
            )
            if matched and o.filled_avg_price:
                return float(o.filled_avg_price)
    except Exception as e:
        print(f"  Could not fetch exit for {trade.symbol}: {e}")
    return None
