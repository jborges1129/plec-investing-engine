# Kalshi Strategy & Data Roadmap

**Date:** 2026-06-02
**Status:** Pre-build — learning and manual validation phase
**Goal:** Establish which prediction market strategies to pursue, what data feeds them, what to learn, and how to validate edge before writing code or risking real capital.

---

## The Three Strategies

### 1. Weather Uncertainty Shorting
**Frequency:** Daily — multiple tradeable contracts every day across many U.S. cities
**Edge type:** Structural overpricing. Kalshi temperature markets systematically price in more uncertainty than actually materializes. Market-implied volatility is 1.27× higher than realized — meaning the "uncertain" contracts are systematically too expensive.

**How it works:**
1. Kalshi lists binary temperature threshold contracts: "Will NYC max temp exceed 90°F on July 5?"
2. The price ($0.65 = 65% implied probability) reflects what the market thinks
3. NOAA's GFS ensemble forecast gives you an *actual* probability distribution over temperature, updated every 6 hours for free
4. If GFS says 50% chance and Kalshi says 65%, Kalshi is overpriced by 15¢ → buy "No" at $0.35
5. The systematic overpricing means you don't need to be directionally right most of the time — a 35% win rate can still be profitable when you size correctly on high-confidence signals

**Data needed:** NOAA GFS ensemble forecasts (free, `api.weather.gov`), ECMWF Open Data (free tier). No API key required for NOAA GFS.

**What to learn first:** How to read probability distributions from weather ensembles. An ensemble forecast gives you 50+ model runs — the fraction that exceed the threshold IS the probability. This is simple math once you have the data.

---

### 2. Top-Trader Consensus Aggregation
**Frequency:** Active — as frequent as Polymarket top traders take positions (daily)
**Edge type:** Informed consensus. Polymarket's top traders by ROI have skin in the game and track records. When 20+ of the top 50 wallets hold the same side of a market, that's a strong signal that the current price is wrong.

**How it works:**
1. Polymarket publishes a public leaderboard — top traders by profit/ROI
2. Each wallet's open positions are visible via Polymarket's public API (no auth required)
3. Build a tracker (eventually): for each open Polymarket market, count how many top-50 wallets hold Yes vs No
4. When 20+ of the top 50 agree on the same side → strong consensus signal
5. Cross-check: if the same market exists on Kalshi at a different price, even better

**The math (why 20/50 matters):** If 40% of top traders are already in a position, and those traders are better-calibrated than the market average, the market price is likely lagging. You're not predicting the outcome — you're betting that the price will drift toward where the smart money already is.

**Data needed:** Polymarket public API (`gamma-api.polymarket.com`) — no authentication needed. Returns open markets, positions per market, and can be cross-referenced against the leaderboard.

**What to learn first:** How to read the Polymarket API. It returns raw market data in JSON. You don't need to write a bot — start by checking manually which markets have high top-trader consensus.

---

### 3. CME–Kalshi Economic Nowcasting
**Frequency:** 8-12 events/year (FOMC meetings, monthly CPI releases) — use as a supplement
**Edge type:** Information arbitrage. Kalshi's retail participants don't fully incorporate CME futures prices, which represent billions in institutional hedging capital. The gap has been 3–8¢ and is not latency-dependent.

**How it works:**
1. Before each FOMC meeting, CME FedWatch shows the institutional probability of each outcome (e.g., 72% chance Fed holds rates)
2. Kalshi has a matching contract: "Will Fed hold rates at July meeting?" — currently priced at $0.64
3. Gap = 72% − 64% = 8¢ → buy Yes at $0.64 when institutional money says it should be $0.72
4. Enter the position days before the meeting. Exit before resolution or hold to settlement.

**Why the gap exists:** CME futures are priced by institutional traders hedging real rate exposure. Kalshi is mostly retail. The two markets don't talk to each other, so the prices diverge.

**Data needed:** CME FedWatch Tool (free website, `cmegroup.com/markets/interest-rates/cme-fedwatch-tool`), FRED API (free, `fred.stlouisfed.org/docs/api/fred/`), Cleveland Fed Nowcast (free daily CSV, `clevelandfed.org/indicators-and-data/inflation-nowcasting`).

**What to learn first:** How to read CME FedWatch probabilities and how they map to specific Kalshi contract resolution conditions.

---

## The Four Things to Learn (in order)

### 1. Kalshi Order Mechanics
*Do this first — before any strategy analysis.*

- Create your Kalshi account and deposit $50-100
- Place a small trade ($5) manually on any market with decent liquidity
- Understand: limit orders vs market orders, how the order book works, how fees are charged (7¢ × C × (1-C) per contract), how positions show P&L
- Try buying Yes on something you think will happen, and No on something you think won't
- Watch the position resolve and understand the settlement

**Time required:** 1-2 hours of hands-on exploration.

### 2. Probability Math: Edge, Calibration, and Closing Line Value
*Understand this before sizing any real money.*

**The core concept:** Every Kalshi price IS a probability. $0.65 = 65% chance. If you think the true probability is 75%, your edge is 10¢ per contract. If you're right on average, you make money.

**Edge = your probability − market price**

The key question is: how do you know if your probability estimate is any good? The answer is **Closing Line Value (CLV)** — did the market drift toward your position after you entered? If you bought at $0.65 and the contract settled with a final pre-resolution price of $0.72, you beat the closing line by 7¢. Positive CLV over 30+ bets means you have genuine edge, not luck.

**What to track from your first trade onward:**
- Entry price
- Final price before resolution (the "closing line")
- Resolution outcome (Yes/No)
- CLV = closing line − your entry price

A **Brier score** measures calibration. If you say 70% and the event happens 50% of the time, your model is wrong. You want: when you say 70%, the event should happen ~70% of the time.

**Start a simple spreadsheet with these columns.** Every trade you ever make goes in here. After 30 resolved trades, your CLV tells you if you have real edge.

### 3. Data Sources Reference

| Source | What it gives you | Cost | How to access |
|---|---|---|---|
| NOAA GFS Ensemble | Temperature probability distributions, weather forecasts | Free | `api.weather.gov` — no API key |
| ECMWF Open Data | European weather model ensemble | Free (limited) | `data.ecmwf.int/open-data` |
| Polymarket API | Open markets, positions, leaderboard | Free | `gamma-api.polymarket.com` |
| Kalshi API | Your positions, open markets, order placement | Free with account | `trading-api.kalshi.com` |
| CME FedWatch | Fed funds futures implied probabilities | Free (website) | `cmegroup.com/markets/interest-rates/cme-fedwatch-tool` |
| FRED API | All US economic data, Fed funds rate | Free | `api.stlouisfed.org/fred` — get a free API key |
| Cleveland Fed Nowcast | Daily CPI/PCE probability distributions | Free | `clevelandfed.org` — CSV download |

**Start here:** Open `api.weather.gov` and `gamma-api.polymarket.com` in your browser. Both return readable JSON. You don't need to write a single line of code to start exploring what data is available.

### 4. Position Sizing: Kelly Criterion
*Learn this before putting more than $20 in any single trade.*

**The Kelly formula for a binary bet:**
```
f* = (p - c) / (1 - c)
```
Where:
- `p` = your estimated probability of winning (e.g., 0.72)
- `c` = the contract price / cost (e.g., 0.65)
- `f*` = the fraction of your bankroll to bet

**Example:** You estimate 72% chance, market price is 65¢.
```
f* = (0.72 - 0.65) / (1 - 0.65) = 0.07 / 0.35 = 20%
```
20% of your bankroll on this one trade. That sounds aggressive — and it is. **Always start with half-Kelly or less:**

- **Half-Kelly** = `f* / 2` = 10% in this example
- **Quarter-Kelly** = `f* / 4` = 5% — safer when you're not sure how good your model is

With $100 in your account, half-Kelly on this trade = $10. That's the right starting size.

**The key rule:** Never bet more than 10% of your bankroll on a single trade until you have 50+ resolved trades and positive CLV. Your model is probably not as good as you think it is until you've validated it.

---

## Capital Staging

| Stage | Capital | Condition to advance |
|---|---|---|
| Stage 1: Learn | $50–100 personal | After 10 manually placed trades, you understand the mechanics |
| Stage 2: Validate | $200–500 personal | After 30 resolved trades with tracked CLV — is it positive? |
| Stage 3: Scale | $1k personal | CLV is positive, Brier score is below 0.20 in at least one category |
| Stage 4: Float deployment | $20k float | Edge is demonstrated in Stage 3; use JIT float scheduler (contracts resolving within float window only) |

Do not skip stages. The float is idle capital you owe back to the business — it's not your risk capital. Validate edge with personal money first.

---

## What to Build (Eventually — Not Yet)

Once manual validation confirms edge in at least one strategy, the build order is:

1. **CLV tracking spreadsheet** → Python script that auto-logs from Kalshi API (Week 1–2)
2. **NOAA GFS probability extractor** → pulls ensemble data for target cities, outputs probability per threshold (Month 1)
3. **Polymarket top-trader scanner** → lists open markets with top-50 consensus count (Month 1–2)
4. **CME-Kalshi gap checker** → pulls CME FedWatch + Kalshi prices, flags gaps > 5¢ (Month 2)
5. **JIT float deployment scheduler** → filters all opportunities to contracts resolving within the float window (Month 2–3)

None of this needs to be automated initially. A Python script that prints "here are today's signals" and you manually enter the trades is perfectly valid until Stage 3.

---

## Success Criteria (How to Know It's Working)

- **30 resolved trades logged:** CLV is positive (your entries beat the closing line on average)
- **Weather model:** Your GFS-vs-Kalshi comparison is profitable in at least 3 of 5 cities you track
- **Top-trader:** Markets where 20+ top traders agree are drifting toward the consensus position within 48 hours
- **Economic:** CME-Kalshi gaps you entered are closing (market drifting toward CME price) before resolution

If any of these fail after 30 bets, that strategy gets deprioritized and capital allocation shrinks. The CLV spreadsheet makes this automatic — numbers tell you which strategies work.

---

## Deferred (Not V1)

- LLM semantic non-fungibility arb — needs NLP pipeline, higher engineering complexity
- Logical arbitrage graph — same
- Kalshi API automation / autonomous trade execution
- Bayesian model portfolio (build after you have 3+ validated strategies)
- Float deployment (after Stage 3 validation)
