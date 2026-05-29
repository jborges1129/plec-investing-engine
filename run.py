import sys
from shared.alpaca import get_account, get_positions
from shared.database import init_db


def get_market_regime_cached():
    try:
        from shared.data import get_market_regime
        return get_market_regime()
    except Exception:
        return {'regime': 'unknown'}


def status():
    account = get_account()
    print(f"\n=== Account Status ===")
    print(f"Portfolio Value: ${float(account.portfolio_value):,.2f}")
    print(f"Cash:            ${float(account.cash):,.2f}")
    print(f"Equity:          ${float(account.equity):,.2f}")
    print(f"Buying Power:    ${float(account.buying_power):,.2f}")

    positions = get_positions()
    if positions:
        print(f"\nOpen Positions ({len(positions)}):")
        for p in positions:
            pnl = float(p.unrealized_pl)
            sign = '+' if pnl >= 0 else ''
            print(f"  {p.symbol:<6}  {float(p.qty):.0f} shares  "
                  f"avg ${float(p.avg_entry_price):.2f}  "
                  f"current ${float(p.current_price):.2f}  "
                  f"P&L {sign}${pnl:.2f}")
    else:
        print("\nNo open positions.")


def main():
    command = sys.argv[1] if len(sys.argv) > 1 else 'status'

    if command == 'status':
        status()

    elif command == 'briefing':
        # Morning briefing — analysis only, no trades
        from etf_module.execute import morning_briefing
        morning_briefing(execute=False)

    elif command == 'etf-execute':
        # Run briefing, then immediately execute top 2
        # Use only after reviewing the briefing output
        from etf_module.execute import morning_briefing, execute_positions
        sizing, _ = morning_briefing(execute=False)
        confirm = input("\nExecute these positions? (yes/no): ").strip().lower()
        if confirm == 'yes':
            execute_positions(sizing)
        else:
            print("Cancelled.")

    elif command == 'scan':
        # Intraday scan — check for new ETF opportunities based on remaining budget
        from etf_module.execute import intraday_briefing, execute_positions
        sizing, _ = intraday_briefing(execute=False)
        if sizing:
            confirm = input("\nAdd these positions? (yes/no): ").strip().lower()
            if confirm == 'yes':
                execute_positions(sizing)
            else:
                print("Cancelled.")

    elif command == 'check':
        # Check if any positions closed since last run
        from etf_module.execute import check_closed_positions
        check_closed_positions()

    elif command == 'rotate-check':
        # One-shot rotation check — compare held ETF ranks, email if rotation warranted
        from etf_module.execute import check_rotation
        fired = check_rotation()
        if not fired:
            print("No rotation signal — held positions still rank well.")

    elif command == 'scheduler':
        # Start the intraday scheduler (runs until Ctrl-C)
        # Runs check_closed_positions + check_rotation every 30 min during market hours
        from scheduler import run_scheduler
        run_scheduler()

    elif command == 'stocks':
        # Run the stock screener — shows top 20 ranked candidates
        from stocks_module.screener import run_screener
        from stocks_module.alerts import print_stock_briefing
        from etf_module.execute import morning_briefing
        candidates = run_screener(top_n=20)
        regime_data = get_market_regime_cached()
        print_stock_briefing(candidates, regime=regime_data.get('regime', 'unknown'))

    elif command == 'stocks-refresh':
        # Force a fresh screener run (ignores today's cache)
        from stocks_module.screener import run_screener
        from stocks_module.alerts import print_stock_briefing
        candidates = run_screener(top_n=20, force_refresh=True)
        regime_data = get_market_regime_cached()
        print_stock_briefing(candidates, regime=regime_data.get('regime', 'unknown'))

    else:
        print(f"Unknown command: {command}")
        print("Usage: python run.py [status|briefing|etf-execute|scan|check|rotate-check|scheduler|stocks|stocks-refresh]")


if __name__ == '__main__':
    init_db()
    main()
