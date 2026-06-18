# Weather Model — Calibration Findings & Changes (2026-06-18)

**Context:** Acting on the "deepen the weather model" next-steps. Goal was the
principled overconfidence fix (ensembles → real σ), plus discipline, NWS-anchoring,
and per-city bias. The data redirected the work — documented honestly below.

## What the data said

### 1. Ensemble / multi-model σ does NOT fix overconfidence (measured, not shipped)
`calibrate_ensemble.py` pulls 4 deterministic models (GFS/ECMWF/ICON/GEM) over
2021–2025 (the live ensemble API only retains members ~2 days back, so the full
historical backtest uses cross-model spread as the disagreement proxy). Train
2021–2024 / test 2025:

- `corr(model-spread, |forecast error|) = +0.074` — disagreement predicts error, but
  weakly. Error std rises only 1.54 → 1.73°F from the lowest to highest spread quartile.
- Spread-aware σ vs hand-fit σ: **identical OOS Brier (0.023), both calibration "ok."**

Conclusion: σ is already well-calibrated; ensemble spread adds negligible signal. A
live 30-member ensemble fetch would add an API dependency + latency for no measured
gain. **Not shipped.** Kept `calibrate_ensemble.py` as the measurement artifact.

### 2. The overconfidence is on confident-Yes, and no center/σ tweak fixes it
`validate_live.py` on **1,468 real Kalshi settlements** (2026-04-15..06-17, model @1pm):

| Model P(Yes) | Actual settle | n | verdict |
|---|---|---|---|
| 0.01–0.44 (No/low side) | 0.02–0.50 | 1,333 | **well-calibrated** |
| 0.54–0.98 (confident-Yes) | 0.24–0.78 | 135 | **badly overconfident** |

Swept the obvious fixes — all RAISED Brier (0.105 → worse):
- `PEAK_BIAS` −0.5 / −0.82 / −1.2 → Brier 0.116 / 0.125 / 0.136
- `PEAK_DISCOUNT` 0.7 / 0.5 / 0.3 → Brier 0.118 / 0.137 / 0.158

A global center shift corrupts the (correct) low bins faster than it helps the
confident-Yes bins, which stay broken regardless. The confident-Yes failures are
forecast busts on rise-dependent cases, not a tunable location error.

### 3. Betting sim OVERTURNED the calibration-only discipline (478 bets, real prices)
`validate_live.py --edge` simulates betting the model's side at real Kalshi prices and
reports CLV (did the book drift toward our entry — the robust edge signal). This is what
the data actually said, which is NOT what the calibration table alone implied:

| Cut | bets | win | ROI/bet | CLV |
|---|---|---|---|---|
| No side | 236 | 53% | +2.2% | **+3.6¢** |
| Yes side | 242 | 24% | +2.6% | **+3.9¢** |
| edge 0–10¢ | 87 | 44% | −0.9% | +0.1¢ |
| edge 10–30¢ | 227 | — | +1.8% | +3.2¢ |
| **edge 30¢+** | 164 | 30% | **+5.0%** | **+6.5¢** |
| entry <20¢ (deep longshot) | 175 | 8% | +0.4% | +1.7¢ |
| **entry 60–100¢ (favorite)** | 96 | 86% | **+9.9%** | **+10.4¢** |

So: both sides profitable; edge is monotonic (bigger = better, 30¢+ is best — NOT bugs);
favorites strongest; deep longshots high-variance/marginal. The calibration-only rules I
first shipped ("No-side only, cap 30¢, reject forecast-Yes") would have discarded the
most profitable bets — they were removed.

## What shipped

1. **Discipline filter** (`weather_signal.py`, on by default; `--no-discipline` to opt out)
   — rewritten to match the betting evidence, not the calibration guess:
   - Default `min_edge` raised **0.05 → 0.10** (the 0–10¢ band is breakeven).
   - Skip **deep longshots** (entry < 15¢, ~8% win — too high-variance for the pilot).
   - **No max-edge cap** (big edges are the best bucket); a 50¢ guard remains *only* to
     catch data errors (bad threshold parse / dead book), with `min_oi` screening staleness.
   - **No side restriction** — both No and forecast-driven Yes are profitable.
   - Live smoke test (2026-06-18): surfaced both sides incl. a +34¢ No-favorite that the
     old 30¢ cap wrongly rejected; dropped 6 deep-longshots.

2. **NWS daily-high anchor** (`intraday_max_cdf(..., daily_high=)`): the raw NWS hourly
   grid runs ~1.6°F warm vs the Climatological Report Kalshi resolves on. The intraday
   path never corrected this (the "Chicago bug"). Now the remaining-hours peak is capped
   at the official NWS daily high — hourly grid for *timing*, daily high for *level*.

3. **Per-city bias measurement** in `--grade`: reports mean(forecast − actual) per city
   from graded trades. Reported, **not auto-applied** — finding #2 shows a naive center
   shift hurts; any PEAK_BIAS change must first be shown to lower Brier in validate_live.

4. **Bug fix**: `validate_live.candle_prices` returned 2 values on the empty path vs 3
   otherwise, crashing the `--edge` sim. Added research knobs (`--peak-bias`,
   `--peak-discount`, `--std-at-peak`) and a betting report split by side.

### 4. How real is the edge? (city-day-clustered bootstrap)
Bets within a city-day share one weather outcome, so the honest sample is **200 distinct
city-days**, not 402 bets. Resampling city-days (min-edge 0.10 window):

| Metric | Point | 95% CI | Read |
|---|---|---|---|
| ROI/bet | +2.8% | **[−1.9%, +7.7%]** | straddles 0 — not yet significant |
| CLV (closing) | +4.2¢ | [−0.4¢, +9.0¢] | degenerate (settles to 0/1) |
| **3h line-drift** | **+7.3¢** | **[+3.2¢, +11.6¢]** | **clearly positive — real edge** |

The same-day P&L is too high-variance to call significant on two months, but the market
**systematically drifts toward our side in the ~3h after entry** — the methodologically
clean edge signal, and it's significant. The model finds mispricings the book later
corrects; we just can't yet prove that converts to significant realized P&L on this
sample.

## Bottom line
The edge is **real but thin** — confirmed by a significant post-entry line drift
(+7.3¢, CI clear of 0), not yet by significant realized ROI (CI straddles 0). It is NOT
where calibration pointed: the win comes from requiring a genuine edge (≥10¢, bigger is
better) on liquid markets, leaning toward favorites, on both sides, avoiding deep
longshots. σ refinement (ensembles) was measured and does not help. **Paper/small stakes
until the roadmap gate (30+ resolved/category, positive P&L AND drift) is met**; the
`--grade` loop now captures this automatically.

## How to reproduce
```
venv/bin/python kalshi_module/calibrate_ensemble.py                 # σ vs spread test
venv/bin/python kalshi_module/validate_live.py --start 2026-04-15 \
    --end 2026-06-17 --edge --min-edge 0.05                         # calibration + betting
venv/bin/python kalshi_module/validate_live.py ... --peak-bias -0.8 # bias/discount sweeps
```
