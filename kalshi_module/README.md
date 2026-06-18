# Kalshi Weather Module

Trades Kalshi daily high-temperature markets (NYC, Chicago, Miami, Denver) against an
intraday model: live station observations set a hard floor on the day's max, and a
bias-checked forecast fills in the remaining hours. The edge is structural — Kalshi
temperature markets lag the observations as the afternoon unfolds.

**Status:** validated but thin. Paper / tiny stakes only until the gate is met (below).

## Daily use (one command)

```bash
# Run around 12–1pm LOCAL for the city you're trading (see "When" below).
venv/bin/python kalshi_module/weather_signal.py --paper
```

`--paper` does the whole loop hands-off:
1. settles & grades yesterday's logged trades and auto-captures their closing line (CLV),
2. scans today's markets, applies the discipline filter, and
3. auto-logs today's new picks to `data/kalshi_trades.csv` (deduped — safe to re-run).

Then place the trustworthy picks manually at kalshi.com. To look without logging, drop
`--paper`. To just re-grade: `--grade`.

Add `--extra-cities` to also scan **Philadelphia, Dallas, Atlanta, Phoenix**. Their
resolution station + NWS grid are derived automatically from Kalshi's own settlement
rules (verified to reproduce the 4 core cities exactly — no wrong-station risk). These 4
were kept from a 10-city candidate set because the model is **well-calibrated** there on a
2025 backtest; LA, Houston, Boston, DC (overconfident) and Seattle (underconfident) were
dropped. Their *edge* isn't price-validated yet, so they forward-validate via `--paper`.

## Reading the output

Each pick carries a **trust tier** — read this first:

| Tier | Meaning | Action |
|---|---|---|
| **LOCKED** | Observed high already decides it | Safest; act if liquid |
| **SOLID** | Peak imminent and temp nearly there | Trust |
| **RISKY** | Needs an unobserved further rise; forecast-dependent | Small or skip |
| **NEXT-DAY** | Pure forecast, no live obs yet | Most speculative |

The header line "▶ WHAT TO DO" summarises how many are trustworthy and which to start with.
`Suggested bet` is ¼-Kelly, hard-capped (`--max-bet`). Bet **No** freely; only bet **Yes**
when LOCKED.

## When to trade (validated)

The edge is a **midday phenomenon**. Across 1,468 settlements the post-entry line drift
toward our side is significant at **12pm (+5.9¢) and 1pm (+7.3¢)**, marginal at 10am, and
**gone by 2pm** — by late afternoon the market has priced the same observations you see.
So run and trade **~12–1pm local** for the target city.

## What's validated (and what isn't)

- **Discipline filter** (default on): min edge 10¢, skip deep longshots (<15¢ entry), no
  max-edge cap, no side restriction — derived from a 478-bet real-price sim, not a guess.
- **Honest edge:** raw same-day ROI +2.8% but its 95% CI straddles 0; the **3h line-drift
  is significantly positive** (the clean signal). Real but thin; high same-day variance.
- **Not validated live:** the live path uses NWS forecasts; the backtest used Open-Meteo
  (no NWS forecast archive exists). The `--paper` loop closes this gap forward.
- σ from ensembles was measured and does **not** help (kept the calibrated hand-fit + a
  rise-based floor). See `../docs/brainstorms/weather-model-findings-2026-06-18.md`.

## Gate before real capital

**30+ resolved trades per category with positive P&L AND positive line-drift** (roadmap
`../docs/brainstorms/kalshi-strategy-roadmap.md`). `--grade` prints the running record,
distinct city-day count (the honest effective sample), and per-city forecast bias.

## The tools

| File | What it does |
|---|---|
| `weather_signal.py` | Live scanner + `--paper`/`--log`/`--grade`. The thing you run. |
| `validate_live.py` | Backtest vs **real** Kalshi settlements + prices (`--edge` for the betting sim, `--decision-hour` to sweep). |
| `backtest_weather.py` | Calibration/skill backtest on ERA5 history (no prices). |
| `calibrate_weather.py` | Fits the intraday σ/center params (train 2021–24 / test 2025). |
| `calibrate_ensemble.py` | Tests whether ensemble spread improves σ (it doesn't). |
| `test_weather_math.py` | `python kalshi_module/test_weather_math.py` — core-math invariants. |

## Setup

```bash
python3 -m venv venv && venv/bin/pip install -r requirements.txt   # needs scipy, numpy, requests
```
No API key needed for the data (NWS, Open-Meteo, Kalshi public endpoints, Iowa ASOS).
