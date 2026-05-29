# ETF Strategy — Requirements

**Date:** 2026-05-26
**Status:** Approved for planning

---

## What we're building

A decision-support trading system for ETFs. The system does the analysis; Josiah makes every trade decision. Nothing executes without approval.

The core principle: **every number is derived from data, nothing is arbitrary.** Stop-losses come from each ETF's own volatility. Position sizes come from inverse volatility weighting. Regime detection comes from measurable market indicators. Each signal is displayed independently — no composite scores that hide their own assumptions.

---

## The four pillars

### 1. Market regime detection

Every morning, the system classifies the current market state:

- **Bull**: VIX below 20, SPY above its 200-day moving average → favor momentum ETFs (QQQ, XLK, XLE, IWM)
- **Bear**: VIX above 25, SPY below 200-day MA → rotate toward defensive ETFs (TLT, GLD, XLV)
- **Neutral**: everything in between → balanced consideration of both groups

This determines which ETFs are even worth evaluating that day. You don't look at QQQ in a bear regime.

### 2. Individual signal dashboard (no composite scores)

Each ETF gets scored across independent dimensions — displayed as separate numbers, never collapsed into one. Josiah looks at the full row and makes a judgment call.

| Signal | What it measures | Why it matters |
|---|---|---|
| Momentum (1m / 3m / 6m) | Price return at each timeframe | Trend strength and consistency |
| RSI (14-day) | Overbought/oversold pressure | Avoid buying at the top of a move |
| Volume trend | Whether volume is confirming price | High volume = conviction behind the move |
| Regime alignment | Does this ETF fit the current regime | Bull regime + defensive ETF = low conviction |
| News sentiment | Recent coverage tone for this ticker | Macro tailwinds or headwinds |

No weightings, no addition. The signals are informational inputs to a human decision.

### 3. Per-ETF dynamic risk parameters

Stop-loss and take-profit are calculated from each ETF's own Average True Range (ATR) — its average daily price movement over 14 days. This means:

- A volatile ETF like XLE gets a wider stop (more room to move)
- A stable ETF like GLD gets a tighter stop
- The math is fully transparent and re-derivable at any time

**Formula:**
- Stop-loss = entry price − (3 × ATR₁₄)
- Take-profit = entry price + (2.5 × stop distance) → 2.5:1 reward-to-risk ratio

**Position sizing** uses inverse volatility weighting: calmer ETF gets proportionally larger allocation. Total ETF allocation shifts by regime — more deployed in bull markets, more cash held in bear markets.

### 4. News intelligence and alerts

**Inbound**: the system subscribes to a financial news feed (Alpha Vantage News Sentiment API or equivalent). It reads full article content, not just headlines. Articles are filtered for relevance to current or candidate ETF positions.

**Triage logic**: not every article is worth surfacing. The system evaluates whether the news materially affects the investment thesis — Fed policy, major earnings, sector-level macro shifts, significant CEO statements. Noise is filtered out.

**Routing**:
- **Slack alert**: any article the system judges as actionable or position-threatening gets pushed immediately, even if Josiah isn't in a Claude session
- **Morning briefing**: each session opens with a summary of overnight developments, current positions, regime status, and any signals that changed since last session

**Historical context**: when a relevant news event arrives, the system pulls historical data to answer: "what happened to this ETF the last several times this type of event occurred?" That context accompanies the alert, not just the headline.

---

## Trading cadence

This is a medium-term momentum strategy, not a day trading strategy.

- **Daily**: monitor regime, review news alerts, check if any stop/take-profit orders fired
- **Act when signals change**: regime flip, take-profit fires and re-entry decision needed, major news materially changes thesis
- **Full rebalance**: roughly monthly — re-rank ETF universe, rotate if top picks have changed

Rebalancing on a fixed daily schedule defeats the purpose of signal-based ranking. The strategy acts when something changes, not when a day passes.

---

## ETF universe

10 ETFs across asset classes, selected to give momentum ranking meaningful variation:

| ETF | Exposure | Regime |
|---|---|---|
| SPY | S&P 500 broad market | Bull |
| QQQ | Nasdaq 100 / tech-heavy | Bull |
| XLK | Technology sector | Bull |
| XLE | Energy sector | Bull |
| IWM | Small-cap Russell 2000 | Bull |
| XLF | Financials | Bull |
| TLT | Long-term treasuries | Bear/defensive |
| GLD | Gold | Bear/defensive |
| XLV | Healthcare | Bear/defensive |
| EFA | International developed markets | Neutral |

---

## Alerts and communication

- **Slack**: push alerts for actionable news, regime changes, stop-loss fires, take-profit fires
- **Morning briefing** (in Claude session): regime status, open positions with current P&L, ETF dashboard with individual signals, overnight news summary
- **Article depth**: full article content analyzed, not headlines — the important information is always in the body

---

## Human approval

Every trade requires Josiah's explicit go-ahead. The system surfaces:
1. Recommended action (buy X, close Y, hold)
2. Individual signal dashboard for the candidate ETF
3. Suggested stop-loss and take-profit (with ATR calculation shown)
4. Suggested position size (with inverse volatility math shown)
5. Relevant news context

Josiah reviews, asks questions if needed, and says go.

---

## What this is not

- Not a day trading system — not designed to capture intraday moves
- Not a composite score system — signals are never collapsed into one number
- Not fully autonomous — human approval is required for every execution
- Not a backtesting platform (yet) — historical context is pulled for advisory purposes, not for systematic optimization

---

## Open questions / future additions

- Congressional trading data (Quiver Quantitative API) — strong signal, add after core system is stable
- Backtesting framework — important for validating ATR multiplier and reward-risk ratio choices
- Stocks module integration — regime detection from this module should inform stocks module as well
- Slack workspace setup — need webhook URL from Josiah's Slack
