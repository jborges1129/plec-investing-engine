#!/usr/bin/env python3
"""
Calibrate the intraday weather model on history, with a real train/test split.

Train: summers 2021-2024 (4 cities).  Test (out-of-sample): summer 2025.

Model structure (must match weather_signal.intraday_max_cdf):
    center = observed_max + d * (remaining_peak_forecast - observed_max) + b
    final_max ~ Normal(center, σ),  σ = STD_AT_PEAK + STD_PER_HOUR_TO_PEAK * hours_to_peak
    final_max is floored at observed_max.

We FIT, by regression on the training years (not by eye):
    d, b               from  (realized_max - observed_max)  ~  (remaining_peak - observed_max)
    STD_AT_PEAK, STD_PER_HOUR_TO_PEAK   from residual std as a function of hours_to_peak

Then we report calibration (Brier + uncertain-zone reliability) on the held-out
2025 test set, for the OLD params vs the FITTED params.

Caveat that cannot be engineered away: forecasts come from Open-Meteo, not NWS, and
its forecast runs slightly warm vs its own reanalysis. The spread (σ) transfers
reasonably; the bias term `b` is partly source-specific and must be refined on live
NWS prediction-vs-outcome pairs via `weather_signal.py --grade`.

Usage:
    python kalshi_module/calibrate_weather.py
"""

import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import requests
from scipy.stats import norm

ARCHIVE = "https://archive-api.open-meteo.com/v1/archive"
HIST_FC = "https://historical-forecast-api.open-meteo.com/v1/forecast"
CACHE = Path(__file__).parent.parent / "data" / "calib_cache.json"

CITIES = {
    "NYC": (40.779, -73.969, "America/New_York"),
    "CHI": (41.786, -87.752, "America/Chicago"),
    "MIA": (25.790, -80.316, "America/New_York"),
    "DEN": (39.847, -104.656, "America/Denver"),
}
TRAIN_YEARS = [2021, 2022, 2023, 2024]
TEST_YEARS = [2025]
DECISION_HOURS = list(range(9, 17))  # 9am..4pm local, to span hours-to-peak

# Old (pre-calibration) params, for comparison.
OLD = dict(d=1.0, b=0.0, s0=0.7, s1=0.45)


def _load_cache() -> dict:
    if CACHE.exists():
        return json.loads(CACHE.read_text())
    return {}


def _save_cache(c: dict) -> None:
    CACHE.write_text(json.dumps(c))


def fetch(url: str, lat: float, lon: float, start: str, end: str, tz: str, cache: dict) -> dict:
    key = f"{url}|{lat}|{lon}|{start}|{end}"
    if key in cache:
        return cache[key]
    r = requests.get(url, params={
        "latitude": lat, "longitude": lon, "start_date": start, "end_date": end,
        "hourly": "temperature_2m", "temperature_unit": "fahrenheit", "timezone": tz,
    }, timeout=90)
    r.raise_for_status()
    cache[key] = r.json().get("hourly", {})
    return cache[key]


def instances(years: list[int], cache: dict) -> list[dict]:
    """Build prediction instances across cities/days/decision-hours."""
    rows = []
    for city, (lat, lon, tzname) in CITIES.items():
        tz = ZoneInfo(tzname)
        for yr in years:
            start, end = f"{yr}-06-01", f"{yr}-08-31"
            act = fetch(ARCHIVE, lat, lon, start, end, tzname, cache)
            fc = fetch(HIST_FC, lat, lon, start, end, tzname, cache)
            a_by, f_by = defaultdict(list), defaultdict(list)
            for t, v in zip(act.get("time", []), act.get("temperature_2m", [])):
                if v is not None:
                    a_by[t[:10]].append((datetime.fromisoformat(t).replace(tzinfo=tz), float(v)))
            for t, v in zip(fc.get("time", []), fc.get("temperature_2m", [])):
                if v is not None:
                    f_by[t[:10]].append((datetime.fromisoformat(t).replace(tzinfo=tz), float(v)))
            for d in sorted(set(a_by) & set(f_by)):
                day_a, day_f = sorted(a_by[d]), sorted(f_by[d])
                if not day_a or not day_f:
                    continue
                realized = max(t for _, t in day_a)
                for H in DECISION_HOURS:
                    now = datetime.fromisoformat(f"{d}T{H:02d}:00").replace(tzinfo=tz)
                    obs = [t for dt, t in day_a if dt <= now]
                    rem = [(dt, t) for dt, t in day_f if dt > now]
                    if not obs or not rem:
                        continue
                    omax = max(obs)
                    rpeak = max(t for _, t in rem)
                    peak_dt = max(rem, key=lambda p: p[1])[0]
                    h2p = max(0.0, (peak_dt - now).total_seconds() / 3600.0)
                    rows.append(dict(city=city, omax=omax, rpeak=rpeak, h2p=h2p, realized=realized))
    return rows


def fit_center(rows: list[dict]) -> tuple[float, float]:
    """Fit (realized-omax) = d*(rpeak-omax) + b on the rise cases."""
    x = np.array([r["rpeak"] - r["omax"] for r in rows if r["rpeak"] > r["omax"]])
    y = np.array([r["realized"] - r["omax"] for r in rows if r["rpeak"] > r["omax"]])
    A = np.vstack([x, np.ones_like(x)]).T
    (d, b), *_ = np.linalg.lstsq(A, y, rcond=None)
    return float(d), float(b)


def center(r: dict, d: float, b: float) -> float:
    if r["rpeak"] > r["omax"]:
        return r["omax"] + d * (r["rpeak"] - r["omax"]) + b
    return r["rpeak"]


def fit_sigma(rows: list[dict], d: float, b: float) -> tuple[float, float]:
    """Residual std as a function of hours-to-peak → (intercept, slope).
    Only the genuinely-uncertain 'rise' cases (peak not yet reached). Peak-passed
    cases are pinned by the observed-floor, not the center, so they'd pollute σ."""
    rise = [r for r in rows if r["rpeak"] > r["omax"]]
    resid = np.array([r["realized"] - center(r, d, b) for r in rise])
    h2p = np.array([r["h2p"] for r in rise])
    mids, stds = [], []
    for lo in range(0, 9):
        m = (h2p >= lo) & (h2p < lo + 1)
        if m.sum() >= 30:
            mids.append(lo + 0.5)
            stds.append(resid[m].std())
    mids, stds = np.array(mids), np.array(stds)
    A = np.vstack([mids, np.ones_like(mids)]).T
    (slope, intercept), *_ = np.linalg.lstsq(A, stds, rcond=None)
    print("  σ by hours-to-peak (rise cases): " +
          "  ".join(f"{m:.0f}h:{s:.1f}" for m, s in zip(mids, stds)))
    return max(0.3, float(intercept)), max(0.0, float(slope))


def evaluate(rows: list[dict], d: float, b: float, s0: float, s1: float) -> dict:
    """Brier + uncertain-zone calibration over a per-city threshold grid."""
    grids = {}
    for c in CITIES:
        rs = [r["realized"] for r in rows if r["city"] == c]
        if rs:
            grids[c] = range(int(min(rs)) - 2, int(max(rs)) + 3)
    sq = n = 0
    up = uo = un = 0.0
    for r in rows:
        ctr = center(r, d, b)
        sig = max(0.3, s0 + s1 * r["h2p"])
        rmax = r["realized"]
        for F in grids[r["city"]]:
            x = F + 0.5
            p = 1.0 if x < r["omax"] else float(1.0 - norm.cdf(x, ctr, sig))
            p = min(1.0, max(0.0, p))
            o = 1.0 if round(rmax) > F else 0.0
            sq += (p - o) ** 2; n += 1
            if 0.15 < p < 0.85:
                up += p; uo += o; un += 1
    return dict(brier=sq / n, uz_n=int(un),
                uz_pred=(up / un if un else 0), uz_act=(uo / un if un else 0))


def main() -> None:
    cache = _load_cache()
    print("Fetching (cached after first run)…")
    train = instances(TRAIN_YEARS, cache)
    test = instances(TEST_YEARS, cache)
    _save_cache(cache)
    print(f"Train instances: {len(train)}  ({TRAIN_YEARS})")
    print(f"Test  instances: {len(test)}   ({TEST_YEARS})\n")

    d, b = fit_center(train)
    s0, s1 = fit_sigma(train, d, b)
    print("FITTED PARAMETERS (from training years):")
    print(f"  PEAK_DISCOUNT (d)        = {d:.3f}   (trust this much of the forecast's projected rise)")
    print(f"  forecast bias (b)        = {b:+.2f}°F (global; partly Open-Meteo-specific)")
    print(f"  STD_AT_PEAK (s0)         = {s0:.2f}°F")
    print(f"  STD_PER_HOUR_TO_PEAK (s1)= {s1:.2f}°F/hr\n")

    print(f"{'params':<26} | {'set':<5} | {'Brier':>6} | uncertain-zone (n, pred→actual)")
    for label, p in [("OLD (d=1, σ=0.7+0.45h)", OLD),
                     ("FITTED", dict(d=d, b=b, s0=s0, s1=s1))]:
        for setname, rows in [("train", train), ("test", test)]:
            e = evaluate(rows, p["d"], p.get("b", 0.0), p["s0"], p["s1"])
            tag = "OVERCONF" if e["uz_pred"] > e["uz_act"] + 0.05 else (
                  "underconf" if e["uz_pred"] < e["uz_act"] - 0.05 else "ok")
            print(f"{label:<26} | {setname:<5} | {e['brier']:>6.3f} | "
                  f"n={e['uz_n']:<5} {e['uz_pred']:.2f}→{e['uz_act']:.2f}  {tag}")

    print("\nUpdate weather_signal.py with the FITTED params above if test calibration is 'ok'.")


if __name__ == "__main__":
    main()
