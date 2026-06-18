#!/usr/bin/env python3
"""
Historical backtest for the Kalshi weather intraday model.

It replays `weather_signal`'s REAL model functions against past days: for each
day it feeds the model the hourly forecast that was actually issued (Open-Meteo
historical-forecast archive) plus the observations as they would have been known
at a given decision hour (Open-Meteo ERA5 actuals), then compares the model's
probability for every "greater than X°F" threshold to what actually happened.

This measures CALIBRATION and SKILL (is the model's probability accurate, and
does it sharpen through the day) — not market edge, since we have no historical
Kalshi prices. Open-Meteo temps are model-based and differ slightly from the ASOS
reading Kalshi resolves on, so treat results as methodology validation.

Usage:
    python kalshi_module/backtest_weather.py
    python kalshi_module/backtest_weather.py --start 2025-06-01 --end 2025-08-31
    python kalshi_module/backtest_weather.py --cities NYC,CHI --hours 10,13,16
"""

import argparse
from collections import defaultdict
from datetime import datetime
from zoneinfo import ZoneInfo

import requests

import weather_signal
from weather_signal import (
    intraday_max_cdf, prob_yes_from_cdf, model_prob_yes, forecast_std,
)

ARCHIVE = "https://archive-api.open-meteo.com/v1/archive"
HIST_FC = "https://historical-forecast-api.open-meteo.com/v1/forecast"

# Resolution-station coordinates + local tz (same stations the live model uses).
CITIES = {
    "NYC": (40.779, -73.969, "America/New_York"),
    "CHI": (41.786, -87.752, "America/Chicago"),
    "MIA": (25.790, -80.316, "America/New_York"),
    "DEN": (39.847, -104.656, "America/Denver"),
}


def fetch_hourly(url: str, lat: float, lon: float, start: str, end: str, tz: str) -> dict[str, list]:
    r = requests.get(url, params={
        "latitude": lat, "longitude": lon, "start_date": start, "end_date": end,
        "hourly": "temperature_2m", "temperature_unit": "fahrenheit", "timezone": tz,
    }, timeout=60)
    r.raise_for_status()
    return r.json().get("hourly", {})


def by_day(hourly: dict, tz: ZoneInfo) -> dict[str, list[tuple[datetime, float]]]:
    """Group an Open-Meteo hourly block into {date: [(aware_dt, tempF)]}."""
    out: dict[str, list[tuple[datetime, float]]] = defaultdict(list)
    for t, v in zip(hourly.get("time", []), hourly.get("temperature_2m", [])):
        if v is None:
            continue
        dt = datetime.fromisoformat(t).replace(tzinfo=tz)
        out[t[:10]].append((dt, float(v)))
    for d in out:
        out[d].sort(key=lambda x: x[0])
    return out


def run_city(name: str, start: str, end: str, decision_hours: list[int]) -> None:
    lat, lon, tzname = CITIES[name]
    tz = ZoneInfo(tzname)
    actual = by_day(fetch_hourly(ARCHIVE, lat, lon, start, end, tzname), tz)
    forecast = by_day(fetch_hourly(HIST_FC, lat, lon, start, end, tzname), tz)

    days = sorted(set(actual) & set(forecast))
    # Fixed threshold grid spanning the realized range, so the test isn't centred
    # on the answer. Evaluate every "greater than F" threshold each day.
    realized_maxes = [max(t for _, t in actual[d]) for d in days if actual[d]]
    lo, hi = int(min(realized_maxes)) - 2, int(max(realized_maxes)) + 2
    thresholds = list(range(lo, hi + 1))

    print(f"\n{'='*78}\n{name}: {len(days)} days {start}..{end}  | thresholds {lo}-{hi}°F")
    print(f"{'='*78}")
    print("'uncertain zone' = predictions with 0.15<p<0.85 (the only place edge exists):")
    print("  n, mean predicted prob, actual frequency — if pred>>actual the model is overconfident.")
    print(f"\n{'decision':>9} | {'Brier':>6} | {'pt-MAE':>6} | {'uncertain-zone':>22} | calib")

    # Baseline: forecast-only model (the OLD approach) — issued daily max + fixed std,
    # using the *full-day* forecast known the morning of (no intraday obs).
    base_sq = base_n = 0
    bu_pred = bu_out = bu_n = 0.0
    for d in days:
        fmax = max(t for _, t in forecast[d])
        rmax = max(t for _, t in actual[d])
        for F in thresholds:
            p = model_prob_yes(fmax, "greater", float(F), None, 1.0)
            o = 1.0 if round(rmax) > F else 0.0
            base_sq += (p - o) ** 2; base_n += 1
            if 0.15 < p < 0.85:
                bu_pred += p; bu_out += o; bu_n += 1
    uz = f"n={int(bu_n)} {bu_pred/bu_n:.2f}→{bu_out/bu_n:.2f}" if bu_n else "n=0"
    calib = "over" if bu_n and bu_pred/bu_n > bu_out/bu_n + 0.05 else ("under" if bu_n and bu_pred/bu_n < bu_out/bu_n - 0.05 else "ok")
    print(f"{'forecast':>9} | {base_sq/base_n:>6.3f} | {'   — ':>6} | {uz:>22} | {calib}  (day-ahead baseline)")

    for H in decision_hours:
        sq = n = 0
        abs_err = pt_n = 0.0
        u_pred = u_out = u_n = 0.0  # uncertain zone aggregates
        for d in days:
            day_actual = actual[d]
            day_fc = forecast[d]
            if not day_actual or not day_fc:
                continue
            now = datetime.fromisoformat(f"{d}T{H:02d}:00").replace(tzinfo=tz)
            obs_so_far = [(dt, t) for dt, t in day_actual if dt <= now]
            if not obs_so_far:
                continue
            cdf, dbg = intraday_max_cdf(obs_so_far, day_fc, now)
            if cdf is None:
                continue
            rmax = max(t for _, t in day_actual)
            pt = max(dbg["observed_max"], dbg.get("remaining_peak") or dbg["observed_max"])
            abs_err += abs(pt - rmax); pt_n += 1
            for F in thresholds:
                p = min(1.0, max(0.0, prob_yes_from_cdf(cdf, "greater", float(F), None)))
                o = 1.0 if round(rmax) > F else 0.0
                sq += (p - o) ** 2; n += 1
                if 0.15 < p < 0.85:
                    u_pred += p; u_out += o; u_n += 1
        if n == 0:
            continue
        uz = f"n={int(u_n)} {u_pred/u_n:.2f}→{u_out/u_n:.2f}" if u_n else "n=0 (all locked)"
        if u_n:
            gap = u_pred/u_n - u_out/u_n
            calib = "OVERCONF" if gap > 0.05 else ("underconf" if gap < -0.05 else "ok")
        else:
            calib = "—"
        print(f"{H:02d}:00 loc | {sq/n:>6.3f} | {abs_err/pt_n:>5.1f}° | {uz:>22} | {calib}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Backtest the intraday weather model on history")
    ap.add_argument("--start", default="2025-06-01")
    ap.add_argument("--end", default="2025-08-31")
    ap.add_argument("--cities", default="NYC,CHI,MIA,DEN")
    ap.add_argument("--hours", default="10,13,16,18")
    ap.add_argument("--resid-damp", type=float, default=None,
                    help="Override RESIDUAL_DAMP (e.g. 0 to disable morning bias carry)")
    ap.add_argument("--std-add", type=float, default=0.0,
                    help="Add this many °F to STD_AT_PEAK (widen uncertainty)")
    ap.add_argument("--peak-discount", type=float, default=None,
                    help="Override PEAK_DISCOUNT (fraction of projected rise above observed to trust)")
    args = ap.parse_args()
    if args.resid_damp is not None:
        weather_signal.RESIDUAL_DAMP = args.resid_damp
    if args.peak_discount is not None:
        weather_signal.PEAK_DISCOUNT = args.peak_discount
    weather_signal.STD_AT_PEAK += args.std_add
    print(f"params: RESIDUAL_DAMP={weather_signal.RESIDUAL_DAMP} "
          f"PEAK_DISCOUNT={weather_signal.PEAK_DISCOUNT} "
          f"STD_AT_PEAK={weather_signal.STD_AT_PEAK}")
    hours = [int(h) for h in args.hours.split(",")]
    print("Brier: lower is better (0=perfect, 0.25=coin flip). pt-MAE: avg error of the")
    print("model's implied daily-max vs actual. reliability: predicted→actual frequency.")
    for c in args.cities.split(","):
        run_city(c.strip(), args.start, args.end, hours)


if __name__ == "__main__":
    main()
