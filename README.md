# PLEC Investing Engine

**Algorithmic paper trading system — ETF momentum strategy with human-supervised stock selection.**

> **Live dashboard:** `streamlit run dashboard/app.py`
> All trades are paper (simulated at real market prices via Alpaca). No real capital at risk during the pilot phase.

---

## What It Does

Two parallel strategies sharing one database and risk framework:

| Module | Strategy | Automation |
|--------|----------|------------|
| **ETF** | Ranks 9 ETFs by pure 12-month momentum. Holds top 1–3 by return. Regime-aware: scales capital from 95% (bull) to 50% (bear). | Fully automated. Orders placed and managed without intervention. |
| **Stocks** | Daily screen of ~1,000 S&P 400/600 stocks through 7 hard filters + 5-dimension scoring. Surfaces top 20 candidates each morning. | Human reviews and approves every entry. |

---

## Why the Money Is Safe

Every position uses Alpaca **bracket orders** — a single API call that places entry, stop-loss, and take-profit simultaneously. Exits are automated; no manual action required to limit losses.

**ETF module risk parameters:**

| Parameter | Value | Reasoning |
|-----------|-------|-----------|
| Stop-loss | `min(ATR × 4.5, 12% of price)` | Sized to survive normal daily volatility; fires on genuine trend breaks |
| Take-profit | 2.5× the stop distance | Requires only a **29% win rate** for positive expected value |
| Capital deployed | 95% / 80% / 50% | Scales with regime: bull / neutral / bear |
| Max concurrent ETF positions | 3 | Inverse-vol weighted across top-ranked ETFs |

**Stocks module risk parameters:**

| Parameter | Value | Reasoning |
|-----------|-------|-----------|
| Risk per trade | 1% of portfolio | Defined-risk entries via ATR-based stop |
| Stop-loss | 2× ATR below entry | Sized per-stock to each name's actual volatility |
| Take-profit | 3× the stop distance | 3:1 reward-risk; break-even at 25% win rate |
| Max concurrent stock positions | 8 | At 1% risk each, worst-case simultaneous stop-out = 8% of portfolio |

**Worst case across both modules:** if every open position stops out simultaneously, total portfolio loss is mathematically capped and visible in real time on the dashboard.

---

## Backtested Performance (ETF Module)

Strategy parameters were locked before the first live paper trade. Backtest run over the same historical period as SPY for apples-to-apples comparison.

| Metric | This Strategy | Buy & Hold SPY |
|--------|--------------|----------------|
| CAGR | 13.98% | ~10–11% |
| Max Drawdown | -25.92% | ~-50% (2008/2020) |
| Calmar Ratio | 0.54 | ~0.42 |

> The Calmar ratio (CAGR ÷ max drawdown) is the relevant comparison — it measures how much return you get per unit of worst-case loss. The strategy beats SPY on a risk-adjusted basis by holding cash in bear regimes instead of riding drawdowns.

---

## ETF Universe

9 ETFs covering distinct market segments — no overlap, maximum diversification of factor exposure:

```
SPY   QQQ   SOXX   XLI   XLP   XLE   XOP   EWY   AGG
```

Each month, all 9 are ranked by 12-month total return. The top 1–3 with positive momentum (>1% over 12 months) are held. Capital is allocated in inverse proportion to each ETF's 3-month volatility (inverse-vol weighting).

**Regime detection** adjusts total deployed capital before sizing:
- **Bull:** SPY above 200-day MA and VIX < 20 → 95% of portfolio deployed
- **Neutral:** one condition fails → 80% deployed
- **Bear:** both conditions fail → 50% deployed

---

## Stock Screener Pipeline

**Universe:** ~1,000 stocks from S&P 400 (mid-cap) + S&P 600 (small-cap)

**Stage 1 — Hard gates (all must pass):**
- Price $5–$150
- Market cap $100M–$2B
- Average volume ≥ 200K shares/day
- Price above 200-day MA (long-term uptrend)
- RSI 40–70 (not oversold, not overbought)
- ATR/price ≥ 1.5% (enough volatility to make bracket orders worthwhile)
- No earnings within 3 days (avoids binary event risk)

**Stage 2 — Setup scoring (0–10 per dimension):**
- Momentum (1m, 3m, 6m returns vs universe)
- Trend strength (distance from 20/50/200 MAs)
- Volume conviction (recent volume vs 20-day average)
- Volatility quality (ATR-to-price in the sweet spot)
- Fundamental health (analyst target upside, EPS beat rate, revenue trend)

**Output tiers:**
- **HIGH (≥8/10):** Top setups — review these first
- **MID (5–7):** Solid candidates — review if time allows
- **LOW (<5):** Shown for context, typically skip

Each candidate card shows analyst consensus, price target vs current, quarterly revenue trend (accelerating / decelerating), and EPS beat rate — so morning review is judgment only, not research.

---

## Performance Tracking

The dashboard shows:
- Live open positions with entry, current P&L, distance to stop, distance to target
- Closed trades with realized P&L, win/loss breakdown, and per-trade expectancy
- Portfolio value chart vs. SPY benchmark
- Market regime indicator (Bull / Neutral / Bear)
- Stock screener candidates with full research pre-loaded

---

## Project Roadmap

| Phase | Module | Status |
|-------|--------|--------|
| **Phase I** | ETF momentum engine — automated, regime-aware, bounded downside | ✅ Running |
| **Phase II** | Stock screener — human-supervised, research pre-fetched, bracket orders | ✅ Built, validating |
| **Phase III** | Sports betting + Kalshi prediction markets — Kelly criterion sizing, surebet arbitrage | Gated by Phase I & II results |

Phase III begins only after Phase I and II demonstrate consistent positive expectancy across at least one full market cycle.

---

## Running the System

```bash
# Install dependencies
pip install -r requirements.txt

# Configure API keys
cp .env.example .env   # then fill in ALPACA_API_KEY and ALPACA_SECRET_KEY

# View live dashboard
streamlit run dashboard/app.py

# Morning ETF briefing (analysis only, no trades placed)
python run.py briefing

# Execute ETF positions after reviewing briefing
python run.py etf-execute

# Run stock screener (uses today's cache if available)
python run.py stocks

# Force fresh stock screener run (ignores cache)
python run.py stocks-refresh

# Buy a specific stock after reviewing screener output
python run.py stocks-buy SYMBOL

# Check for positions that closed since last run
python run.py stocks-check
python run.py check

# Start intraday scheduler (rotation checks + monitoring every 30 min)
python run.py scheduler
```

---

## Repository Structure

```
investing-system/
├── shared/            # Alpaca API, SQLite database, yfinance data fetching
├── etf_module/        # Momentum ranking, regime detection, automated execution
├── stocks_module/     # Daily screener, signal scoring, research enrichment, alerts
├── backtest/          # Historical backtest engine (trailing stop simulation)
├── dashboard/         # Streamlit dashboard
├── docs/              # Strategy overview document
├── scheduler.py       # APScheduler — runs during market hours
├── run.py             # CLI entry point
└── .streamlit/        # Dashboard config (headless mode, port 8501)
```

---

*Built with Python · Alpaca paper trading API · yfinance · pandas · Streamlit · SQLite*
