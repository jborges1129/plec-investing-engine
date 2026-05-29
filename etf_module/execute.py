from shared.alpaca import get_account, get_positions, place_bracket_order, close_position, get_client
from shared.database import init_db, log_trade, snapshot_portfolio, get_open_trades, close_trade
from shared.notify import send_alert, format_position_alert, format_rotation_alert
from shared.data import get_market_regime, ETF_UNIVERSE
from etf_module.strategy import (
    build_signal_dashboard, get_position_sizing, get_news_for_candidates,
    print_morning_briefing, print_intraday_briefing, MAX_POSITIONS,
)


def morning_briefing(execute: bool = False):
    init_db()

    account = get_account()
    portfolio_value = float(account.portfolio_value)
    cash = float(account.cash)
    equity = float(account.equity)

    snapshot_portfolio(portfolio_value, cash, equity)
    print(f"Portfolio: ${portfolio_value:,.2f} | Cash: ${cash:,.2f}")

    regime_data = get_market_regime()
    dashboard = build_signal_dashboard(regime_data)

    if dashboard.empty:
        print("No ETF data available.")
        return [], regime_data  # always return a tuple

    top_symbols = list(dashboard.head(3)['symbol'])
    sizing = get_position_sizing(dashboard, portfolio_value, regime_data['regime'])
    news = get_news_for_candidates(top_symbols)

    print_morning_briefing(regime_data, dashboard, sizing, news)

    if not execute:
        print("Run etf-execute after reviewing to place orders.")

    return sizing, regime_data


def execute_positions(sizing: list[dict]):
    """Execute the recommended positions after human review."""
    if not sizing:
        print("No positions to execute.")
        return

    client = get_client()

    # Guard: check for pending orders to avoid duplicates
    try:
        open_orders = client.get_orders()
        pending_symbols = {o.symbol for o in open_orders}
    except Exception:
        pending_symbols = set()

    current_positions = get_positions()
    current_symbols = {p.symbol for p in current_positions}

    if len(current_symbols) >= MAX_POSITIONS:
        print(f"Already at max {MAX_POSITIONS} positions — skipping all orders.")
        return

    for pos in sizing:
        if len(current_symbols) >= MAX_POSITIONS:
            print(f"  Reached max {MAX_POSITIONS} positions — stopping.")
            break

        symbol = pos['symbol']
        qty = int(pos['qty'])  # Alpaca requires whole shares for bracket orders

        if qty == 0:
            print(f"  {symbol}: qty=0 (position too small for allocation), skipping.")
            continue
        if symbol in current_symbols:
            print(f"  {symbol} already held — skipping.")
            continue
        if symbol in pending_symbols:
            print(f"  {symbol} already has a pending order — skipping.")
            continue

        current_est = pos['price']
        print(f"  Buying {qty} shares of {symbol} @ ~${current_est:.2f}...")
        try:
            result, stop, tp = place_bracket_order(
                symbol=symbol,
                qty=qty,
                entry_price=current_est,
                stop_loss_pct=pos['stop_loss_pct'],
                take_profit_pct=pos['take_profit_pct'],
            )

            fill_price = float(result.filled_avg_price) if result.filled_avg_price else current_est

            log_trade(
                order_id=str(result.id),
                symbol=symbol,
                module='etf',
                side='buy',
                qty=qty,
                entry_price=fill_price,
                stop_loss=stop,
                take_profit=tp,
            )
            current_symbols.add(symbol)
            print(f"  Submitted: {result.id} | stop ${stop:.2f} | target ${tp:.2f}")
        except Exception as e:
            print(f"  Error buying {symbol}: {e}")


def intraday_briefing(execute: bool = False):
    """Scan for new ETF opportunities mid-day using remaining budget and open position slots."""
    init_db()

    account = get_account()
    portfolio_value = float(account.portfolio_value)

    current_positions = get_positions()
    current_symbols = {p.symbol for p in current_positions}
    etf_set = set(ETF_UNIVERSE)

    open_etf_count = sum(1 for p in current_positions if p.symbol in etf_set)

    if open_etf_count >= MAX_POSITIONS:
        print(f"Already at max {MAX_POSITIONS} ETF positions — no new buys available.")
        return [], get_market_regime()

    deployed_usd = sum(
        float(p.qty) * float(p.current_price)
        for p in current_positions if p.symbol in etf_set
    )

    regime_data = get_market_regime()
    full_dashboard = build_signal_dashboard(regime_data)

    if full_dashboard.empty:
        print("No ETF data available.")
        return [], regime_data

    available = full_dashboard[~full_dashboard['symbol'].isin(current_symbols)].copy().reset_index(drop=True)
    available['rank'] = available.index + 1

    remaining_slots = MAX_POSITIONS - open_etf_count
    sizing = get_position_sizing(
        available, portfolio_value, regime_data['regime'],
        top_n=remaining_slots, deployed_usd=deployed_usd,
    )

    news = get_news_for_candidates([p['symbol'] for p in sizing[:2]]) if sizing else {}

    print_intraday_briefing(
        regime_data, available, sizing, current_positions,
        deployed_usd, portfolio_value, news,
    )

    if not execute:
        print("Run python run.py scan to execute new positions after reviewing.")

    return sizing, regime_data


_ROTATION_RANK_THRESHOLD = 4    # held ETF must have slipped to this rank or worse
_ROTATION_MOMENTUM_GAP  = 0.05  # candidate must beat held by at least 5% 3m return


def check_rotation() -> bool:
    """
    Compare held ETF ranks against current momentum dashboard.
    Sends an email alert if a held ETF slipped to rank 4+ while a better ETF is rank 1-2.
    Returns True if a rotation alert was sent.
    """
    init_db()

    regime_data = get_market_regime()
    dashboard   = build_signal_dashboard(regime_data)

    if dashboard.empty:
        return False

    current_positions = get_positions()
    etf_set = set(ETF_UNIVERSE)
    held_symbols = {p.symbol for p in current_positions if p.symbol in etf_set}

    if not held_symbols:
        return False

    # Enrich dashboard rows
    rank_map = {row['symbol']: row for row in dashboard.to_dict('records')}

    held_lagging = [
        {'symbol': sym, 'rank': rank_map[sym]['rank'], 'return_3m': rank_map[sym]['return_3m']}
        for sym in held_symbols
        if sym in rank_map and rank_map[sym]['rank'] >= _ROTATION_RANK_THRESHOLD
    ]

    if not held_lagging:
        return False

    best_held_3m = max(rank_map[h['symbol']]['return_3m'] for h in held_lagging if h['symbol'] in rank_map)

    top_candidates = [
        {
            'symbol':          row['symbol'],
            'rank':            row['rank'],
            'return_3m':       row['return_3m'],
            'price':           row['price'],
            'stop_loss_pct':   row['stop_loss_pct'],
            'take_profit_pct': row['take_profit_pct'],
        }
        for row in dashboard.to_dict('records')
        if row['symbol'] not in held_symbols
        and row['rank'] <= 2
        and row['return_3m'] > best_held_3m + _ROTATION_MOMENTUM_GAP
    ]

    if not top_candidates:
        return False

    held_str = ', '.join(f"{h['symbol']}(#{h['rank']})" for h in held_lagging)
    cand_str = ', '.join(c['symbol'] for c in top_candidates)
    print(f"  Rotation signal: {held_str} lagging — {cand_str} ranked higher")

    subject, body = format_rotation_alert(held_lagging, top_candidates)
    send_alert(subject, body)
    return True


def _lookup_exit(client, trade) -> tuple[float | None, str]:
    """Return (exit_price, event_type) for a single closed bracket order."""
    try:
        from alpaca.trading.requests import GetOrdersRequest
        from alpaca.trading.enums import QueryOrderStatus
        req = GetOrdersRequest(
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
                event = str(getattr(o, 'order_type', 'closed'))
                return float(o.filled_avg_price), event
    except Exception as e:
        print(f"  Could not fetch exit order for {trade.symbol}: {e}")
    return None, 'closed'


def check_closed_positions():
    """
    Compare open DB trades against current Alpaca positions.

    Handles multiple bracket orders per symbol (e.g., XLK bought in two batches):
    compares total DB qty vs Alpaca position qty to detect partial closures, then
    identifies which specific order(s) filled by working oldest-first.
    """
    open_trades = get_open_trades(module='etf')
    if not open_trades:
        return

    client = get_client()
    current_positions = {p.symbol: p for p in get_positions()}

    # Group open DB trades by symbol; sort each group oldest-first
    from collections import defaultdict
    symbol_trades: dict[str, list] = defaultdict(list)
    for t in open_trades:
        symbol_trades[t.symbol].append(t)
    for trades in symbol_trades.values():
        trades.sort(key=lambda t: t.entry_time)

    for symbol, trades in symbol_trades.items():
        pos = current_positions.get(symbol)
        alpaca_qty = float(pos.qty) if pos else 0.0
        db_total   = sum(t.qty for t in trades)

        # All shares still present — nothing to do
        if alpaca_qty >= db_total - 0.5:
            continue

        # One or more orders have closed. Walk oldest-first:
        # if cumulative DB qty exceeds what Alpaca still holds, that order closed.
        cumulative = 0.0
        for trade in trades:
            cumulative += trade.qty
            still_open = alpaca_qty >= cumulative - 0.5
            if still_open:
                continue

            exit_price, exit_event = _lookup_exit(client, trade)
            if exit_price is None:
                print(f"  {symbol} order {trade.order_id[:8]} closed — exit price unknown, logging $0.")
                exit_price = 0.0

            close_trade(trade.order_id, exit_price, status=exit_event)
            pnl     = (exit_price - trade.entry_price) * trade.qty if trade.entry_price else 0
            pnl_pct = (exit_price - trade.entry_price) / trade.entry_price if trade.entry_price else 0
            subj, body = format_position_alert(symbol, exit_event, exit_price, pnl, pnl_pct)
            send_alert(subj, body)
            print(f"  {symbol} order {trade.order_id[:8]} closed — P&L ${pnl:+.2f} ({pnl_pct*100:+.1f}%)")
