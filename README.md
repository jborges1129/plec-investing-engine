# PLEC Investing Engine

**Algorithmic paper trading system — ETF momentum strategy with human-supervised stock selection.**

> **Live dashboard:** run `streamlit run dashboard/app.py` from the project directory.  
> All trades are paper (simulated at real market prices via Alpaca). No real capital at risk during the pilot phase.

---

## What It Does

Two parallel strategies sharing one database and risk framework:

| Module | Strategy | Automation |
|--------|----------|------------|
| **ETF** | Momentum-ranks 20 sector ETFs, holds the top 3 by 3-month return. Regime-aware: reduces exposure when VIX > 25 or SPY < 200-day MA. | Fully automated. Orders placed and managed without intervention. |
| **Stocks** | Daily screen of ~1,000 S&P 400/600 stocks. Scores setups 0–10 across 7 technical criteria. Surfaces top 20 candidates each morning. | Human reviews and approves every entry. |

---

## Why the Money Is Safe

Every position uses Alpaca **bracket orders** — a single API call that submits entry, stop-loss, and take-profit simultaneously. The stop fires automatically; no manual action needed to limit losses.

**Risk parameters (ETF module):**

| Parameter | Value | Reasoning |
|-----------|-------|-----------|
| Stop-loss | ATR × 3, capped at 7% | 3× the average daily range — survives normal volatility, catches genuine reversals |
| Take-profit | 2.5× the stop distance | Requires only a **29% win rate** for positive expected value. Momentum strategies historically achieve 55–65%. |
| Capital deployed | 30–50% of portfolio | Scales with regime: 50% in bull markets, 30% in bear markets. Remainder stays in cash. |
| Max concurrent positions | 5 ETFs | Limits concentration; each position sized by inverse volatility |

**Worst-case scenario:** If all open positions hit their stop-loss simultaneously — a historically unprecedented event — the bounded loss is visible in real time on the dashboard. With current parameters and 5 positions, simultaneous stop-out is capped at approximately 3–4% of total portfolio.

---

## Performance Tracking

Paper trading began **May 26, 2026**. The dashboard shows:

- Live open positions with entry price, current P&L, distance to stop, and distance to target
- Closed trades with realized P&L, win/loss breakdown, and expectancy per trade
- Portfolio value chart vs. SPY benchmark (same starting value)
- Market regime indicator (Bull / Neutral / Bear) driving current allocation

*A 90-day paper trading period is the standard evaluation window. The strategy parameters were locked before the first trade was placed.*

---

## Strategy Parameters — Academic Basis

- **ATR × 3 stop:** factor-of-safety framing — 3 standard deviations of daily range to trigger (Wilder, 1978)
- **2.5:1 reward-risk:** expected value = 0.60 × 2.5 − 0.40 × 1.0 = **+1.10 per unit of risk** at a 60% win rate
- **3-month momentum filter:** Jegadeesh & Titman (1993); Antonacci *Dual Momentum Investing* (2014) — ETFs must show >1% gain over the prior 3 months to qualify
- **Regime detection:** VIX threshold + 200-day MA crossover — reduces exposure in bear markets, consistent with risk-parity frameworks

---

## Project Roadmap

| Phase | Module | Status |
|-------|--------|--------|
| **Phase I** | ETF momentum engine — automated, bounded downside, regime-aware | ✅ Running |
| **Phase II** | Individual stock screener — human-supervised entries, 7-factor scoring | ✅ Built, validating |
| **Phase III** | Prediction markets / Kalshi — structural arbitrage, different risk hypothesis | Gated by Phase I & II results |

Phase III does not begin until Phase I and II demonstrate consistent signal over at least one full market cycle.

---

## Running the System

```bash
# Install dependencies
pip install -r requirements.txt

# Copy and fill in API keys
cp .env.example .env

# View dashboard
streamlit run dashboard/app.py

# Morning briefing (analysis only, no trades)
python run.py briefing

# Execute ETF positions after reviewing briefing
python run.py etf-execute

# Start intraday scheduler (position monitoring + rotation alerts, every 30 min)
python run.py scheduler

# Stock screener
python run.py stocks
```

---

## Repository Structure

```
investing-system/
├── shared/            # Alpaca API, database, data fetching
├── etf_module/        # Momentum strategy and automated execution
├── stocks_module/     # Daily screener, signal scoring, candidate alerts
├── dashboard/         # Streamlit dashboard
├── scheduler.py       # APScheduler — runs during market hours
├── run.py             # CLI entry point
└── docs/              # Strategy notes and ideation
```

---

*Built with Python · Alpaca paper trading API · yfinance · Streamlit · SQLite*
