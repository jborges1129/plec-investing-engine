# Investing System — PLEC Projects

## What this is

A two-module algorithmic trading system built on Alpaca's brokerage API. The goal is to generate consistent, defensible returns that justify scaling up capital from an initial ~$2-3k pilot to a larger allocation from a $25k company budget.

This is one of two revenue-generation projects. The other is a sports betting / Kalshi prediction market system (separate project). This project handles investing.

## Boss context

- Company has ~$25k available, potentially $75k
- Boss is skeptical — needs to see proof before committing more capital
- May 31, 2026 pitch: show paper trading dashboard with 2-3 days of live data
- Goal for pitch: demonstrate the system works as designed, risk is bounded, returns are positive
- Success metric for month 1: consistent positive P&L, no catastrophic drawdowns, behavior matches the architecture

## Architecture

Two modules in one project, sharing the same database and data infrastructure:

```
investing-system/
├── shared/          # data fetching, database, Alpaca connection
├── etf_module/      # fully automated momentum strategy
├── stocks_module/   # screener + signals, human approves trades
├── dashboard/       # Streamlit dashboard (what the boss sees)
├── data/            # SQLite database, CSVs
└── logs/            # trade logs, error logs
```

### ETF Module (fully automated)
- Strategy: momentum ranking — rank a basket of ETFs by 3-month return, hold top 1-3, rebalance monthly
- Runs on a schedule with no human input required
- Alpaca bracket orders: entry + stop-loss + take-profit in one API call
- Capital allocation: ~40-50% of deployed capital
- Risk tolerance: lower — this is the steady passive layer

### Stocks Module (human-in-the-loop)
- Universe: small-to-mid cap stocks ($100M–$2B market cap)
- Screener runs daily pre-market, surfaces 3-10 candidates
- Signal engine scores and ranks candidates
- Output: alert/notification with top 2-3 setups
- Human (Josiah) reviews, approves or skips — 15 min per morning
- Execution: Alpaca bracket orders after human approval
- Risk tolerance: medium — willing to test riskier setups here, will validate with paper trading first

## Risk management (critical — this is the pitch)

Every position uses Alpaca bracket orders:
- **Entry**: buy at signal price
- **Stop-loss**: auto-sell if price drops X% (typically 5-7%)
- **Take-profit**: auto-sell if price rises Y% (typically 12-15%)
- **Trailing stops**: preferred over fixed stops — stop moves up as price rises, locking in gains

Position sizing:
- Max 5-10% of portfolio per position
- Max daily loss mathematically bounded: even if all positions stop out simultaneously, loss is capped at ~5% of portfolio
- This is the core defensibility argument for the boss

## Tech stack

- **Python** — everything is Python scripts
- **yfinance** — free historical price data, fundamentals (P/E, EPS, market cap, earnings dates)
- **alpaca-py** — official Alpaca Python SDK for order execution and market data
- **pandas** — data manipulation
- **SQLite** — local database for trade logs and performance tracking
- **Streamlit** — dashboard (shows boss live P&L, open positions, closed trade history, metrics)
- **schedule** or **APScheduler** — runs the ETF module on a cron-like schedule

## Alpaca setup

- Using **paper trading** mode initially (real market prices, no real money)
- Paper trading API base URL: `https://paper-api.alpaca.markets`
- API keys go in `.env` file: `ALPACA_API_KEY` and `ALPACA_SECRET_KEY`
- Switch to live trading by changing the base URL and using live keys

## Key decisions already made

1. One project for both modules — they share database and data pipeline, splitting would be redundant
2. ETF module = passive/automated, not where we're hunting for big returns
3. Stocks module = where Josiah spends active time, small cap focus for larger edges
4. Paper trade for at least 1 week before any real money (pitch on May 31 will show paper data)
5. No fully autonomous stock trading — human approves individual stock entries
6. ETF trades can be fully autonomous once validated
7. Dashboard (Streamlit) is the artifact shown to the boss — must look clean and professional
8. Start simple: prove the system works before adding complexity (news sentiment, ML, etc.)

## Build order

1. `shared/data.py` — yfinance data fetcher
2. `shared/alpaca.py` — Alpaca connection, bracket order helper
3. `shared/database.py` — SQLite setup, trade logging
4. `etf_module/strategy.py` — momentum ranking logic
5. `etf_module/execute.py` — automated ETF rebalancing
6. `stocks_module/screener.py` — daily candidate filter
7. `stocks_module/signals.py` — entry signal scoring
8. `stocks_module/alerts.py` — output top candidates for human review
9. `dashboard/app.py` — Streamlit dashboard
10. `run.py` — main entry point tying everything together

## What "done" looks like for the May 31 pitch

- Paper trading running for 2-3 days
- Dashboard showing: open positions, closed trades with P&L, win rate, portfolio value over time
- At least one stop-loss that fired correctly (demonstrates risk management works)
- Clean, readable UI that a non-technical boss can understand in 60 seconds
