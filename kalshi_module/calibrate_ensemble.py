#!/usr/bin/env python3
"""
Does forecast DISAGREEMENT predict forecast ERROR?  (condition-aware σ test)

The current intraday model sets uncertainty as a fixed function of hours-to-peak:
    σ = STD_AT_PEAK + STD_PER_HOUR_TO_PEAK * hours_to_peak
That treats a calm, models-agree day the same as a convective, models-disagree
day. The principled fix is to let σ track the actual forecast uncertainty. We
can't pull historical ENSEMBLE member spread from Open-Meteo (members are only
retained ~2 days back), but we CAN pull multiple deterministic MODELS
(GFS / ECMWF / ICON / GEM) over the full 2021-2025 history. Their disagreement
at a given hour is a condition-aware proxy for forecast uncertainty — it widens
on convective/uncertain days exactly when we want σ to widen.

This script MEASURES, with a real train/test split (train 2021-2024, test 2025):
  1. Is model-spread correlated with |forecast error|?  (the core hypothesis)
  2. Does a spread-aware σ beat the hand-fit σ on out-of-sample calibration
     (Brier + uncertain-zone reliability)?

If (1) is weak or (2) doesn't improve OOS, condition-aware σ is NOT shipped.
Live, the model uses the richer 30-member GFS ensemble spread (strictly better
than this 4-model proxy); this script validates the *mechanism*.

Usage:
    python kalshi_module/calibrate_ensemble.py
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
DATA = Path(__file__).parent.parent / "data"
ENS_CACHE = DATA / "ensemble_calib_cache.json"
OLD_CACHE = DATA / "calib_cache.json"  # reuse already-fetched ERA5 actuals

CITIES = {
    "NYC": (40.779, -73.969, "America/New_York"),
    "CHI": (41.786, -87.752, "America/Chicago"),
    "MIA": (25.790, -80.316, "America/New_York"),
    "DEN": (39.847, -104.656, "America/Denver"),
}
# Models with good historical coverage on Open-Meteo's historical-forecast API.
MODELS = ["gfs_seamless", "ecmwf_ifs04", "icon_seamless", "gem_seamless"]
TRAIN_YEARS = [2021, 2022, 2023, 2024]
TEST_YEARS = [2025]
DECISION_HOURS = list(range(9, 17))  # 9am..4pm local


def _load(p):
    return json.loads(p.read_text()) if p.exists() else {}


def fetch_actuals(lat, lon, start, end, tz, cache, old):
    """ERA5 hourly actuals. Reuse the existing calib_cache key if present."""
    key = f"{ARCHIVE}|{lat}|{lon}|{start}|{end}"
    if key in old:
        return old[key]
    if key in cache:
        return cache[key]
    r = requests.get(ARCHIVE, params={
        "latitude": lat, "longitude": lon, "start_date": start, "end_date": end,
        "hourly": "temperature_2m", "temperature_unit": "fahrenheit", "timezone": tz,
    }, timeout=90)
    r.raise_for_status()
    cache[key] = r.json().get("hourly", {})
    return cache[key]


def fetch_models(lat, lon, start, end, tz, cache):
    """Multi-model hourly forecast block: {time:[...], temperature_2m_<model>:[...]}."""
    key = f"multi|{lat}|{lon}|{start}|{end}|{','.join(MODELS)}"
    if key in cache:
        return cache[key]
    r = requests.get(HIST_FC, params={
        "latitude": lat, "longitude": lon, "start_date": start, "end_date": end,
        "hourly": "temperature_2m", "temperature_unit": "fahrenheit",
        "models": ",".join(MODELS), "timezone": tz,
    }, timeout=120)
    r.raise_for_status()
    cache[key] = r.json().get("hourly", {})
    return cache[key]


def _by_day_single(block, tz):
    out = defaultdict(list)
    for t, v in zip(block.get("time", []), block.get("temperature_2m", [])):
        if v is not None:
            out[t[:10]].append((datetime.fromisoformat(t).replace(tzinfo=tz), float(v)))
    for d in out:
        out[d].sort(key=lambda x: x[0])
    return out


def _models_by_day(block, tz):
    """{date: [(dt, {model: tempF})]} keeping per-model values for spread."""
    times = block.get("time", [])
    series = {m: block.get(f"temperature_2m_{m}", []) for m in MODELS}
    out = defaultdict(list)
    for i, t in enumerate(times):
        dt = datetime.fromisoformat(t).replace(tzinfo=tz)
        vals = {m: (float(series[m][i]) if i < len(series[m]) and series[m][i] is not None else None)
                for m in MODELS}
        out[t[:10]].append((dt, vals))
    for d in out:
        out[d].sort(key=lambda x: x[0])
    return out


def build_rows(years, ens_cache, old_cache):
    """Prediction instances with per-model remaining peaks → spread."""
    rows = []
    for city, (lat, lon, tzname) in CITIES.items():
        tz = ZoneInfo(tzname)
        for yr in years:
            start, end = f"{yr}-06-01", f"{yr}-08-31"
            act = _by_day_single(fetch_actuals(lat, lon, start, end, tzname, ens_cache, old_cache), tz)
            mdl = _models_by_day(fetch_models(lat, lon, start, end, tzname, ens_cache), tz)
            for d in sorted(set(act) & set(mdl)):
                day_a = act[d]
                if not day_a:
                    continue
                realized = max(t for _, t in day_a)
                for H in DECISION_HOURS:
                    now = datetime.fromisoformat(f"{d}T{H:02d}:00").replace(tzinfo=tz)
                    obs = [t for dt, t in day_a if dt <= now]
                    rem = [(dt, vals) for dt, vals in mdl[d] if dt > now]
                    if not obs or not rem:
                        continue
                    omax = max(obs)
                    # Per-model remaining-hours peak (only models present through the rest of day)
                    peaks = {}
                    for m in MODELS:
                        mvals = [vals[m] for _, vals in rem if vals.get(m) is not None]
                        if mvals:
                            peaks[m] = max(mvals)
                    if len(peaks) < 2:
                        continue
                    ens_mean_peak = float(np.mean(list(peaks.values())))
                    model_spread = float(np.std(list(peaks.values())))  # disagreement
                    # hours to the ensemble-mean peak (approx via gfs timing, else any model)
                    ref_model = "gfs_seamless" if "gfs_seamless" in peaks else next(iter(peaks))
                    ref_series = [(dt, vals[ref_model]) for dt, vals in rem if vals.get(ref_model) is not None]
                    peak_dt = max(ref_series, key=lambda p: p[1])[0]
                    h2p = max(0.0, (peak_dt - now).total_seconds() / 3600.0)
                    rows.append(dict(city=city, omax=omax, ens_peak=ens_mean_peak,
                                     spread=model_spread, h2p=h2p, realized=realized))
    return rows


def fit_center(rows):
    """(realized-omax) = d*(ens_peak-omax) + b on rise cases."""
    rise = [r for r in rows if r["ens_peak"] > r["omax"]]
    x = np.array([r["ens_peak"] - r["omax"] for r in rise])
    y = np.array([r["realized"] - r["omax"] for r in rise])
    A = np.vstack([x, np.ones_like(x)]).T
    (d, b), *_ = np.linalg.lstsq(A, y, rcond=None)
    return float(d), float(b)


def center(r, d, b):
    if r["ens_peak"] > r["omax"]:
        return r["omax"] + d * (r["ens_peak"] - r["omax"]) + b
    return r["ens_peak"]


def correlation_test(rows, d, b):
    """Core hypothesis: does model-spread track |forecast error|?"""
    rise = [r for r in rows if r["ens_peak"] > r["omax"]]
    resid = np.array([r["realized"] - center(r, d, b) for r in rise])
    spread = np.array([r["spread"] for r in rise])
    abs_err = np.abs(resid)
    corr = float(np.corrcoef(spread, abs_err)[0, 1])
    print(f"\nCORE HYPOTHESIS — model-spread vs |forecast error|  (n={len(rise)} rise cases)")
    print(f"  corr(spread, |error|) = {corr:+.3f}   (>0 means disagreement predicts error)")
    print("  error std within model-spread quartiles:")
    qs = np.quantile(spread, [0, .25, .5, .75, 1.0])
    for i in range(4):
        m = (spread >= qs[i]) & (spread <= qs[i + 1])
        if m.sum():
            print(f"    spread {qs[i]:.2f}-{qs[i+1]:.2f}°F : |err| mean {abs_err[m].mean():.2f}  "
                  f"err std {resid[m].std():.2f}  n={int(m.sum())}")
    return corr


def fit_sigma_handfit(rows, d, b):
    """σ = s0 + s1*h2p (current model structure)."""
    rise = [r for r in rows if r["ens_peak"] > r["omax"]]
    resid = np.array([r["realized"] - center(r, d, b) for r in rise])
    h2p = np.array([r["h2p"] for r in rise])
    mids, stds = [], []
    for lo in range(0, 9):
        m = (h2p >= lo) & (h2p < lo + 1)
        if m.sum() >= 30:
            mids.append(lo + 0.5); stds.append(resid[m].std())
    A = np.vstack([np.array(mids), np.ones(len(mids))]).T
    (s1, s0), *_ = np.linalg.lstsq(A, np.array(stds), rcond=None)
    return max(0.3, float(s0)), max(0.0, float(s1))


def fit_sigma_spread(rows, d, b):
    """σ² modeled as variance ~ a + b1*h2p + b2*spread (regress squared resid)."""
    rise = [r for r in rows if r["ens_peak"] > r["omax"]]
    resid = np.array([r["realized"] - center(r, d, b) for r in rise])
    h2p = np.array([r["h2p"] for r in rise])
    spread = np.array([r["spread"] for r in rise])
    # Regress |resid|*sqrt(pi/2) (unbiased σ est per point) on features.
    target = np.abs(resid) * np.sqrt(np.pi / 2)
    A = np.vstack([np.ones_like(h2p), h2p, spread]).T
    coef, *_ = np.linalg.lstsq(A, target, rcond=None)
    return [float(c) for c in coef]  # [a, b_h2p, b_spread]


def sigma_handfit(r, s0, s1):
    return max(0.3, s0 + s1 * r["h2p"])


def sigma_spread(r, coef):
    a, b1, b2 = coef
    return max(0.3, a + b1 * r["h2p"] + b2 * r["spread"])


def evaluate(rows, d, b, sigfn):
    """Brier + uncertain-zone reliability over a per-city threshold grid."""
    grids = {}
    for c in CITIES:
        rs = [r["realized"] for r in rows if r["city"] == c]
        if rs:
            grids[c] = range(int(min(rs)) - 2, int(max(rs)) + 3)
    sq = n = 0
    up = uo = un = 0.0
    for r in rows:
        ctr = center(r, d, b)
        sig = sigfn(r)
        for F in grids[r["city"]]:
            x = F + 0.5
            p = 1.0 if x < r["omax"] else float(1.0 - norm.cdf(x, ctr, sig))
            p = min(1.0, max(0.0, p))
            o = 1.0 if round(r["realized"]) > F else 0.0
            sq += (p - o) ** 2; n += 1
            if 0.15 < p < 0.85:
                up += p; uo += o; un += 1
    return dict(brier=sq / n, uz_n=int(un),
                uz_pred=(up / un if un else 0), uz_act=(uo / un if un else 0))


def _tag(e):
    g = e["uz_pred"] - e["uz_act"]
    return "OVERCONF" if g > 0.05 else ("underconf" if g < -0.05 else "ok")


def main():
    ens_cache, old_cache = _load(ENS_CACHE), _load(OLD_CACHE)
    print("Fetching multi-model history (cached after first run; first run is slow)…")
    train = build_rows(TRAIN_YEARS, ens_cache, old_cache)
    test = build_rows(TEST_YEARS, ens_cache, old_cache)
    ENS_CACHE.write_text(json.dumps(ens_cache))
    print(f"Train instances: {len(train)} ({TRAIN_YEARS})  Test: {len(test)} ({TEST_YEARS})")

    d, b = fit_center(train)
    print(f"\nCenter fit (ensemble-mean):  d(discount)={d:.3f}  b(bias)={b:+.2f}°F")

    correlation_test(train, d, b)

    s0, s1 = fit_sigma_handfit(train, d, b)
    coef = fit_sigma_spread(train, d, b)
    print(f"\nσ models fit on train:")
    print(f"  hand-fit:    σ = {s0:.2f} + {s1:.3f}·h2p")
    print(f"  spread-aware: σ = {coef[0]:.2f} + {coef[1]:.3f}·h2p + {coef[2]:.3f}·spread")

    print(f"\n{'σ model':<14} | {'set':<5} | {'Brier':>6} | uncertain-zone (n, pred→actual)  flag")
    for label, fn in [("hand-fit", lambda r: sigma_handfit(r, s0, s1)),
                      ("spread-aware", lambda r: sigma_spread(r, coef))]:
        for setname, rows in [("train", train), ("test", test)]:
            e = evaluate(rows, d, b, fn)
            print(f"{label:<14} | {setname:<5} | {e['brier']:>6.3f} | "
                  f"n={e['uz_n']:<6} {e['uz_pred']:.2f}→{e['uz_act']:.2f}  {_tag(e)}")

    print("\nDecision: ship spread-aware σ only if it lowers OOS (test) Brier AND")
    print("does not worsen uncertain-zone calibration vs hand-fit.")


if __name__ == "__main__":
    main()
