---
date: 2026-05-29
topic: pitch-evaluation
focus: Evaluate the full investing system + sports betting against the May 31 boss pitch goal. What's needed to make it defensible?
mode: repo-grounded
---

# Ideation: Pre-Pitch Evaluation — May 31 Boss Pitch

## Grounding Context

**System state as of May 29, 2026:**
- ETF module (complete): momentum-ranks ~20 ETFs by 3m return, holds top 3, ATR×3 stop + 2.5:1 R:R take-profit, regime detection (Bull/Neutral/Bear), rotation alerts via email, APScheduler 30-min cadence
- Stocks module (complete): daily screener of S&P 400/600 (~1,000 stocks), 7 setup types, 0-10 scoring, top 20 candidates surfaced; human approves via CLI
- Dashboard: Streamlit (1,630 lines) — regime narrative, ETF ranking, positions with P&L, closed trades, portfolio value chart, Stock Screener tab
- Database: SQLite, logs all trades + portfolio snapshots
- Sports betting: NOT BUILT — only mentioned in CLAUDE.md as a separate future project
- **CRITICAL**: No paper trades have been logged yet. Trading DB empty. Pitch is May 31 (2 days away)
- ETF momentum alpha has degraded from ~10% (1990s) to ~2% today due to institutional crowding (academically documented)
- Institutional investors require 3-6 month live track record; May 31 pitch is necessarily proof-of-concept

**External research signals:**
- Paper trading pitch credibility: pre-registered parameters + walk-forward validation is the institutional standard (arXiv 2602.10785, Feb 2026)
- ETF momentum: academically backed via Antonacci Dual Momentum, Jegadeesh-Titman 1993 — but crowded
- Prediction market edge: primarily structural/arbitrage, bot-dominated; Kalshi $22B, CFTC-regulated (May 2026)

**Key institutional learning:**
- Lead the pitch with the RISK GUARANTEE (bounded max daily loss), not returns
- Dashboard must be readable in 60 seconds by a non-technical boss
- One stop-loss that fired correctly is more persuasive than any return chart
- ETF module running cleanly with stops visible IS the pitch

## Topic Axes

1. Pitch readiness — operational actions needed in the next 2 days
2. Strategy defensibility — academic/empirical backing for the parameters chosen
3. Dashboard narrative — what the boss actually sees; does it tell the right story in 60 seconds
4. Sports betting framing — how to position the unbuilt module
5. Risk framework presentation — making the mathematical loss guarantee visceral and credible

## Ranked Ideas

### 1. "Worst-Case Stop-Out" Dollar Number Pinned to Dashboard
**Description:** Compute `sum(qty × stop_pct × entry_price)` across all open ETF positions and display it as a single prominent metric: "Worst-case simultaneous stop-out: −$X (−Y% of portfolio)." This converts the bracketed-order risk architecture from an implicit promise into a verifiable number the boss can see and point to.
**Axis:** Risk framework presentation
**Basis:** `direct:` `etf_module/execute.py` computes `stop_loss_pct` per position (ATR×3, capped at 7%). All inputs are available in the DB. CLAUDE.md line 57: "This is the core defensibility argument for the boss." Nothing currently surfaces this aggregate number in the dashboard.
**Rationale:** "You cannot lose more than $X this week, mathematically" is the single most persuasive sentence in the pitch for a non-technical capital allocator. The dashboard already has all the data. One metric box converts an architecture promise into a verifiable number.
**Downsides:** Number fluctuates as positions change; if execution differs from paper, boss may anchor on it incorrectly
**Confidence:** 95%
**Complexity:** Low — one computation + display widget in `dashboard/app.py`
**Status:** Unexplored

---

### 2. Pre-Registration Frame: "Empty DB = Parameters Locked Before Clock Started"
**Description:** Write a 1-2 page "pre-registration document" today — strategy parameters, benchmark (SPY), evaluation period (90 days), primary metric (Sharpe vs SPY), and stopping conditions — timestamped before the first forward trade is logged. This mirrors clinical trial pre-registration and converts the "no trade history" weakness into the strongest opening: "We registered the parameters before any trades were observed."
**Axis:** Pitch readiness
**Basis:** `external:` Clinical trial pre-registration (FDA / ClinicalTrials.gov) is the institutional standard for distinguishing prospective evidence from post-hoc data fitting. Web research explicitly identified this as the most credible framing for a paper trading pitch to a skeptical non-technical executive. The grounding summary confirms: "pre-registered parameters" framing works because the timestamp is verifiable.
**Rationale:** Requires zero code. One document written today does more credibility work than two days of paper results. The empty DB stops being a liability and becomes evidence of rigor — "we haven't started because the controlled trial just began."
**Downsides:** Requires actually writing the document; works best if timestamped via email or git commit before the first trade
**Confidence:** 90%
**Complexity:** Low — 1-2 hours, no code, pure pitch preparation
**Status:** Unexplored

---

### 3. Operational Pitch-Readiness: DB Integrity + Equity Curve Update
**Description:** Two specific operational fixes before tomorrow's open: (a) verify and clean any duplicate trade rows in SQLite (suspected: two XLK + two QQQ entries from May 26/27 at different prices — run `sqlite3 data/trading.db "SELECT symbol, COUNT(*) FROM trades GROUP BY symbol"` to confirm); (b) add `snapshot_portfolio()` call to the scheduler so the equity curve updates every 30 minutes during market hours rather than only when `morning_briefing` is run manually.
**Axis:** Pitch readiness
**Basis:** `direct:` Codebase scan identified both defects — duplicate rows will cause the dashboard to double-count unrealized P&L; `snapshot_portfolio()` is only called in `etf_module/execute.py:19` inside `morning_briefing()`, never in `scheduler.py`.
**Rationale:** The dashboard is the primary pitch artifact. A doubled P&L table or a two-day-flat equity curve are the first things visible, before the boss reads any narrative. Both fixable in under an hour.
**Downsides:** Requires verifying DB state is actually corrupted (the scan may have inferred from logic rather than reading the database directly)
**Confidence:** 85% (verify DB state first; snapshot gap is confirmed)
**Complexity:** Low — SQL verification + 3-line scheduler addition
**Status:** Unexplored

---

### 4. Benchmark Line: Overlay SPY on the Equity Curve
**Description:** Pull SPY daily returns from the same start date as the paper trading run and normalize it to the same starting portfolio value ($100,000). Overlay this line on the "Portfolio Value Over Time" chart. The visual exists in `dashboard/app.py` already pulling SPY data — it just isn't rendered alongside portfolio value.
**Axis:** Dashboard narrative / Strategy defensibility
**Basis:** `external:` Academic walk-forward validation literature (arXiv 2602.10785, CXO SACEMS tracking) treats benchmark comparison as a non-negotiable credibility signal. `direct:` `dashboard/app.py` already calls `get_price_history('SPY')` — the data is being fetched but not rendered alongside portfolio value.
**Rationale:** The "just buy VOO" objection is the hardest objection in any active management pitch. It can only be answered with a chart showing both lines. Even with 2-3 days of paper data, the visual anchors the right question: "Is this system generating alpha?" The honest answer with 2 days is "can't tell yet" — but that lands far better with a visible benchmark than without one.
**Downsides:** With 2-3 days of data, the comparison is statistically meaningless; must be framed as "this is what we're measuring against, not what we've proven yet"
**Confidence:** 90%
**Complexity:** Low — normalize SPY price to $100k baseline, add overlay to existing chart
**Status:** Unexplored

---

### 5. "We Stop at 15% Drawdown" — Governance Commitment, Not a Forecast
**Description:** In the pitch (not necessarily in the dashboard), make an explicit stopping-rule commitment: "If paper portfolio drops more than 15% at any monthly review, I halt the system, diagnose it, and require your sign-off before resuming." This is a governance promise, not a risk statistic. It can optionally be reflected on the dashboard as a "current drawdown from peak" metric with a red line at 15%.
**Axis:** Risk framework presentation
**Basis:** `reasoned:` Institutional risk governance (ISDA frameworks, hedge fund side-pocket agreements) is built on trigger-based stopping rules, not VaR forecasts. A forecast can be wrong with no accountability; a stopping rule is a promise with teeth. The external research notes the 2026 shift toward drawdown containment as the primary pitch credibility signal for capital allocators.
**Rationale:** The most credible risk statement you can make is one that works against your own interests if violated. "We stop at 15%" is adversarial to yourself — the boss doesn't need to trust your models, only your willingness to be held accountable to a stated threshold.
**Downsides:** Commits you to the threshold; requires a difficult conversation if violated legitimately (e.g., flash crash)
**Confidence:** 88%
**Complexity:** Zero (pure pitch positioning); Low if adding drawdown metric to dashboard
**Status:** Unexplored

---

### 6. Phase I / Phase II / Phase III: The Sports Betting Sequencing Argument
**Description:** Frame the full project arc using drug development phase logic: ETF module = Phase I (safety established: bounded downside, regime detection working), stocks module = Phase II (signal validation under controlled human-oversight conditions), prediction markets = Phase III (different efficacy hypothesis, gated by Phase I/II completion). If sports betting comes up, the answer is: "We're in Phase I. We don't advance to Phase III until Phase I and II confirm signal. That's the protocol."
**Axis:** Sports betting framing
**Basis:** `reasoned:` Drug development uses legally mandated phases specifically because advancement should be gated by evidence from prior phases. The investing system maps directly: each module is a different risk/signal hypothesis with increasing complexity. The external research confirms prediction market edge is structural arbitrage, not forecasting — framing it as a later-stage module prevents overpromising what it can deliver.
**Rationale:** Without a prepared framing, "sports betting isn't built yet" reads as disorganization. With Phase I/II/III framing, the same fact reads as disciplined sequencing. It also sets the correct expectation: sports betting is not being promised as a near-term return driver.
**Downsides:** Requires rehearsing the framing so it sounds natural; doesn't answer "when will Phase III be ready?"
**Confidence:** 92%
**Complexity:** Zero — pure pitch preparation
**Status:** Unexplored

---

### 7. QUALITY_FLOOR_3M = 0.0 — Fix or Justify Before the Pitch
**Description:** `etf_module/strategy.py` line 8 sets `QUALITY_FLOOR_3M = 0.0`, meaning any ETF with any positive 3-month return qualifies. Raise to 2% (matching the strategy's documented expected alpha degradation) or add a one-line comment explicitly justifying why 0% is correct. Either is defensible; the current state is not.
**Axis:** Strategy defensibility
**Basis:** `direct:` `etf_module/strategy.py:8` — `QUALITY_FLOOR_3M = 0.0`; line 67 — `qualified = df[df['return_3m'] > QUALITY_FLOOR_3M]`. External research: ETF momentum alpha has degraded to ~2% annually; at a 0% floor, the strategy holds ETFs in the degraded-alpha cohort with no meaningful signal requirement.
**Rationale:** If a technically literate reviewer asks "what's the minimum momentum threshold for an ETF to enter?" and the answer is "any positive return," that sounds like no filter exists. The fix is a 5-minute change or a one-sentence comment; the current state invites a question that undermines the "defensible parameters" claim.
**Downsides:** Raising the floor to 2% could reduce the candidate pool in bear/neutral regimes when most ETFs are barely positive
**Confidence:** 95%
**Complexity:** Low — 5-minute change + one-line comment
**Status:** Unexplored

---

### 8. Regime History Timeline Strip Below the Equity Curve
**Description:** Render the historical regime state (bull/neutral/bear) as a horizontal colored strip below the portfolio equity curve — each bar colored by regime (green/yellow/red). This requires storing regime state in the DB alongside portfolio snapshots (currently computed live, not stored historically).
**Axis:** Dashboard narrative
**Basis:** `direct:` Regime detection runs on every briefing via `get_market_regime()`. The regime allocation module allocates 50%/40%/30% of portfolio by regime. This information is never rendered historically in `dashboard/app.py` — flat or down periods look like failures rather than intentional risk reduction.
**Rationale:** Without the regime strip, any flat period on the equity curve looks like the system failed. With it, the boss sees "the system reduced exposure here because VIX spiked — that's the risk management working." Every period of reduced returns is reframed as intentional defensive posture. This is also the only visual proof that the regime detection module does anything, which is otherwise invisible to a non-technical viewer.
**Downsides:** Requires adding a regime column to `portfolio_snapshots` table and backfilling from existing briefing logs (medium complexity); flat equity curve with no regime variation makes the strip less compelling
**Confidence:** 85%
**Complexity:** Medium — DB schema addition + scheduler update + Streamlit rendering
**Status:** Unexplored

---

**Cross-cutting pattern — The Governance Trifecta (ideas 1, 2, 5):**
Present ideas 1, 2, and 5 as a single narrative arc in the pitch: "We locked the parameters before any trades were observed [pre-registration doc] → here is the worst-case daily loss visible in real time [dashboard number] → if the portfolio drops 15% at any monthly review, we stop and you sign off before we resume [stopping rule]." This sequence answers every risk skepticism question before it is raised. Each element is individually credible; together they form an unfalsifiable governance argument.

---

## Rejection Summary

| # | Idea | Reason Rejected |
|---|------|-----------------|
| — | Walk-forward DB seeding | Ethics: presenting simulated trades as paper trading history undermines the pre-registration credibility argument |
| — | Stocks execution path (CLI) | Scope mismatch: CLAUDE.md says ETF module running cleanly is the May 31 pitch; stocks not in scope |
| — | One-page pitch packet export | Nice-to-have; lower leverage than fixing operational gaps in 48 hours |
| — | "How This Goes Live" paragraph | Subsumed by Capital Ask / Stopping Rule (more specific) |
| — | Defense-in-depth fault tree (nuclear) | Overlaps with Factor-of-Safety + Max Loss Number survivors |
| — | Live $10 Kalshi account | High execution risk, scope creep, could undermine investing pitch if it goes badly |
| — | Risk guarantee as signed operating agreement | Too adversarial for an internal company pitch |
| — | Weather forecaster calibration frame | Overlaps with pre-registration framing which is stronger and more specific |
| — | Degrees-of-freedom disclosure | Too technical for non-quant boss; pre-registration achieves same credibility signal |
| — | Two-crew CRM verification (aviation) | Stocks not the demo focus; lower leverage |
| — | All non-Phase I/II/III sports betting variants | Subsumed by Phase I/II/III which is stronger |
| — | 6-month walk-forward backtest | Same ethical concern as DB seeding; pre-registration framing is the correct alternative |
| — | Pre-market checklist header | Below ambition floor relative to other survivors |
| — | 60-second boss view (one number/chart/signal) | Subsumed by Regime Timeline + Benchmark Line |
| — | $0 budget / 30-day free trial framing | Too tactical, doesn't advance the system |
| — | Give boss words to convince himself | Too vague, not actionable |
