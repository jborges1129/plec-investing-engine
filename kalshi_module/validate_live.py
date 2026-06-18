#!/usr/bin/env python3
"""
Validate the calibrated intraday model against REAL Kalshi settlements.

For every settled Kalshi high-temp market in a window, this:
  - takes the ACTUAL outcome (`result`) and ACTUAL settled high (`expiration_value`),
  - reconstructs what the calibrated model would have predicted at a decision hour
    (live observations-so-far + issued forecast, from Open-Meteo, run through the
    real weather_signal model functions),
  - with --edge, pulls Kalshi hourly candlesticks to get the real market price at the
    decision hour and at close, and simulates betting the model's side.

So the OUTCOMES and PRICES are 100% real Kalshi; only the weather inputs are modeled.
Everything is cached to disk so re-runs are free and rate-limit-friendly.

Usage:
    python kalshi_module/validate_live.py --start 2025-06-01 --end 2025-08-31
    python kalshi_module/validate_live.py --start 2026-05-20 --end 2026-06-17 --edge --min-edge 0.07
"""

import argparse
import json
import time
from collections import defaultdict
from datetime import date, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import requests

import weather_signal as ws
from weather_signal import intraday_max_cdf, prob_yes_from_cdf, kalshi_taker_fee

KAPI = "https://external-api.kalshi.com/trade-api/v2"
KH = {"accept": "application/json"}
ARCHIVE = "https://archive-api.open-meteo.com/v1/archive"
HIST_FC = "https://historical-forecast-api.open-meteo.com/v1/forecast"
DATA = Path(__file__).parent.parent / "data"
KCACHE = DATA / "kalshi_api_cache.json"
OCACHE = DATA / "calib_cache.json"

CITIES = {  # series: (name, lat, lon, tz, IEM ASOS station id)
    "KXHIGHNY": ("NYC", 40.779, -73.969, "America/New_York", "NYC"),
    "KXHIGHCHI": ("CHI", 41.786, -87.752, "America/Chicago", "MDW"),
    "KXHIGHMIA": ("MIA", 25.790, -80.316, "America/New_York", "MIA"),
    "KXHIGHDEN": ("DEN", 39.847, -104.656, "America/Denver", "DEN"),
}
IEM = "https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py"
DECISION_HOUR = 13  # 1pm local
DRIFT_HOURS = 3     # measure line drift this many hours after entry (pre-resolution)


def _load(p):
    return json.loads(p.read_text()) if p.exists() else {}


def kget(url, params, cache, tag):
    key = tag + "|" + "&".join(f"{k}={v}" for k, v in sorted(params.items()))
    if key in cache:
        return cache[key]
    for attempt in range(6):
        r = requests.get(url, headers=KH, params=params, timeout=25)
        if r.status_code == 429:
            time.sleep(3 * (attempt + 1))
            continue
        r.raise_for_status()
        cache[key] = r.json()
        time.sleep(1.2)
        return cache[key]
    raise RuntimeError("rate limited repeatedly")


def settled_markets(series, start, end, cache):
    """All settled markets whose occurrence date is in [start, end]."""
    out, cursor, pages = [], None, 0
    while pages < 40:
        p = {"series_ticker": series, "status": "settled", "limit": 200}
        if cursor:
            p["cursor"] = cursor
        j = kget(f"{KAPI}/markets", p, cache, f"settled:{series}:{pages}")
        ms = j.get("markets", [])
        out += ms
        cursor = j.get("cursor")
        pages += 1
        if not cursor or not ms:
            break
        # stop early once we've paged before the window
        if all(m.get("occurrence_datetime", "9")[:10] < start for m in ms):
            break
    return [m for m in out if start <= m.get("occurrence_datetime", "")[:10] <= end]


def ometeo(url, lat, lon, start, end, tz, cache):
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


def hourly_by_day(block, tz):
    out = defaultdict(list)
    for t, v in zip(block.get("time", []), block.get("temperature_2m", [])):
        if v is not None:
            out[t[:10]].append((datetime.fromisoformat(t).replace(tzinfo=tz), float(v)))
    for d in out:
        out[d].sort(key=lambda x: x[0])
    return out


def iem_obs_by_day(iem_station, start, end, tz, cache):
    """REAL station METAR temps from the Iowa ASOS archive (what Kalshi resolves on),
    grouped by LOCAL date: {date: [(aware_dt_utc, tempF)]}."""
    key = f"iem|{iem_station}|{start}|{end}"
    if key in cache:
        rows = cache[key]
    else:
        utc = ZoneInfo("UTC")
        s, e = date.fromisoformat(start), date.fromisoformat(end)
        for attempt in range(6):
            r = requests.get(IEM, params={
                "station": iem_station, "data": "tmpf",
                "year1": s.year, "month1": s.month, "day1": s.day,
                "year2": e.year, "month2": e.month, "day2": e.day,
                "tz": "Etc/UTC", "format": "onlycomma", "latlon": "no",
                "missing": "M", "trace": "T",
            }, timeout=90)
            if r.status_code == 429:
                time.sleep(5 * (attempt + 1))
                continue
            break
        r.raise_for_status()
        rows = []
        for ln in r.text.strip().split("\n")[1:]:
            parts = ln.split(",")
            if len(parts) < 3:
                continue
            try:
                rows.append((parts[1], float(parts[2])))
            except ValueError:
                continue
        cache[key] = rows
        _ = utc  # noqa
    tzu = ZoneInfo("UTC")
    out = defaultdict(list)
    for ts, v in rows:
        dt = datetime.fromisoformat(ts).replace(tzinfo=tzu)
        out[dt.astimezone(tz).date().isoformat()].append((dt, v))
    for d in out:
        out[d].sort(key=lambda x: x[0])
    return out


def model_prob_for(market, act_day, fc_day, tz):
    """(P(Yes), yes_observation_locked) for `market` at DECISION_HOUR, or (None, False)."""
    d = market["occurrence_datetime"][:10]
    now = datetime.fromisoformat(f"{d}T{DECISION_HOUR:02d}:00").replace(tzinfo=tz)
    obs = [(dt, t) for dt, t in act_day if dt <= now]
    if not obs or not fc_day:
        return None, False
    cdf, dbg = intraday_max_cdf(obs, fc_day, now)
    if cdf is None:
        return None, False
    st = market.get("strike_type")
    fl = market.get("floor_strike")
    cp = market.get("cap_strike")
    fl = float(fl) if fl is not None else None
    cp = float(cp) if cp is not None else None
    if st == "greater" and fl is None:
        return None, False
    if st == "less" and cp is None:
        return None, False
    if st == "between" and (fl is None or cp is None):
        return None, False
    locked = ws.yes_observation_locked(st, fl, cp, dbg.get("observed_max"))
    return min(1.0, max(0.0, prob_yes_from_cdf(cdf, st, fl, cp))), locked


def candle_prices(series, ticker, day, tz, kcache):
    """
    Returns (yes_ask, yes_bid, close_px, mid_dec, mid_drift) from Kalshi candlesticks:
      - yes_ask/yes_bid at the decision hour (our entry costs)
      - close_px: final-candle traded price (settles to ~0/1 — degenerate CLV)
      - mid_dec:  traded price at the decision hour
      - mid_drift: traded price ~DRIFT_HOURS later but still pre-resolution — lets us
        measure genuine line drift toward our side BEFORE the outcome is obvious, the
        methodologically clean edge signal (vs the degenerate same-day closing CLV).
    """
    o = int(datetime.fromisoformat(f"{day}T00:00").replace(tzinfo=tz).timestamp())
    c = int(datetime.fromisoformat(f"{day}T23:59").replace(tzinfo=tz).timestamp())
    j = kget(f"{KAPI}/series/{series}/markets/{ticker}/candlesticks",
             {"start_ts": o, "end_ts": c, "period_interval": 60}, kcache, f"cs:{ticker}")
    cs = j.get("candlesticks", [])
    if not cs:
        return None, None, None, None, None
    target = int(datetime.fromisoformat(f"{day}T{DECISION_HOUR:02d}:00").replace(tzinfo=tz).timestamp())
    dec = min(cs, key=lambda k: abs(k["end_period_ts"] - target))
    drift_target = target + DRIFT_HOURS * 3600
    drift = min(cs, key=lambda k: abs(k["end_period_ts"] - drift_target))

    def f(k, side, field):
        v = (k.get(side) or {}).get(field)
        return float(v) if v not in (None, "") else None
    yes_ask = f(dec, "yes_ask", "close_dollars")
    yes_bid = f(dec, "yes_bid", "close_dollars")
    close_px = (cs[-1].get("price") or {}).get("close_dollars")
    mid_dec = (dec.get("price") or {}).get("close_dollars")
    mid_drift = (drift.get("price") or {}).get("close_dollars")
    g = lambda v: float(v) if v not in (None, "") else None
    return yes_ask, yes_bid, g(close_px), g(mid_dec), g(mid_drift)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2025-06-01")
    ap.add_argument("--end", default="2025-08-31")
    ap.add_argument("--cities", default="KXHIGHNY,KXHIGHCHI,KXHIGHMIA,KXHIGHDEN")
    ap.add_argument("--edge", action="store_true", help="also pull prices and simulate betting")
    ap.add_argument("--min-edge", type=float, default=0.07)
    ap.add_argument("--peak-bias", type=float, default=None,
                    help="Override PEAK_BIAS (°F) to test forecast de-biasing")
    ap.add_argument("--peak-discount", type=float, default=None,
                    help="Override PEAK_DISCOUNT (fraction of projected rise to trust)")
    ap.add_argument("--std-at-peak", type=float, default=None,
                    help="Override STD_AT_PEAK (°F)")
    ap.add_argument("--decision-hour", type=int, default=None,
                    help="Local hour to evaluate at (default 13). Sweep to find when edge is best.")
    args = ap.parse_args()

    if args.peak_bias is not None:
        ws.PEAK_BIAS = args.peak_bias
    if args.peak_discount is not None:
        ws.PEAK_DISCOUNT = args.peak_discount
    if args.std_at_peak is not None:
        ws.STD_AT_PEAK = args.std_at_peak
    if args.decision_hour is not None:
        global DECISION_HOUR
        DECISION_HOUR = args.decision_hour

    kcache, ocache = _load(KCACHE), _load(OCACHE)
    print(f"Validating {args.start}..{args.end} at {DECISION_HOUR}:00 local  "
          f"(model: PEAK_DISCOUNT={ws.PEAK_DISCOUNT} PEAK_BIAS={ws.PEAK_BIAS} "
          f"STD_AT_PEAK={ws.STD_AT_PEAK})\n")

    overall = defaultdict(int)
    bins = defaultdict(lambda: [0.0, 0.0, 0])
    bet_rows = []

    for series in args.cities.split(","):
        name, lat, lon, tzname, iem_id = CITIES[series]
        tz = ZoneInfo(tzname)
        try:
            mkts = settled_markets(series, args.start, args.end, kcache)
        except Exception as e:
            print(f"{name}: kalshi error {e}")
            continue
        # observed-max FLOOR from the real resolution station; forecast from Open-Meteo.
        act = iem_obs_by_day(iem_id, args.start, args.end, tz, ocache)
        fc = hourly_by_day(ometeo(HIST_FC, lat, lon, args.start, args.end, tzname, ocache), tz)

        n = wins = sq = 0
        for m in mkts:
            if m.get("result") not in ("yes", "no"):
                continue
            d = m["occurrence_datetime"][:10]
            p, locked = model_prob_for(m, act.get(d, []), fc.get(d, []), tz)
            if p is None:
                continue
            outcome = 1.0 if m["result"] == "yes" else 0.0
            n += 1
            sq += (p - outcome) ** 2
            bins[min(9, int(p * 10))][0] += p
            bins[min(9, int(p * 10))][1] += outcome
            bins[min(9, int(p * 10))][2] += 1
            if (p > 0.5) == (outcome == 1.0):
                wins += 1

            if args.edge:
                yes_ask, yes_bid, close_px, mid_dec, mid_drift = candle_prices(
                    series, m["ticker"], d, tz, kcache)
                if yes_ask is None or yes_bid is None or close_px is None:
                    continue
                # Real entry costs: buy Yes at yes_ask; buy No at (1 - yes_bid).
                no_ask = round(1.0 - yes_bid, 4)
                yes_edge = p - yes_ask - kalshi_taker_fee(yes_ask)
                no_edge = (1 - p) - no_ask - kalshi_taker_fee(no_ask)
                side, edge, entry = ("Yes", yes_edge, yes_ask) if yes_edge >= no_edge else ("No", no_edge, no_ask)
                if edge < args.min_edge or entry <= 0 or entry >= 1:
                    continue
                won = (side == "Yes" and outcome == 1) or (side == "No" and outcome == 0)
                pnl = (1 - entry if won else -entry) - kalshi_taker_fee(entry)
                clv = (close_px - entry) if side == "Yes" else ((1 - close_px) - entry)
                # Pre-resolution line drift toward our side (clean edge signal, not degenerate).
                drift = None
                if mid_dec is not None and mid_drift is not None:
                    drift = (mid_drift - mid_dec) if side == "Yes" else (mid_dec - mid_drift)
                bet_rows.append(dict(city=name, city_day=f"{name}:{d}", side=side, won=won,
                                     pnl=pnl, clv=clv, drift=drift, edge=edge, entry=entry,
                                     locked=locked))

        overall["n"] += n
        overall["wins"] += wins
        overall["sq"] += sq
        brier = sq / n if n else 0
        acc = wins / n if n else 0
        print(f"{name}: {n:>4} settled markets | dir-acc {acc*100:4.0f}% | Brier {brier:.3f}")
        KCACHE.write_text(json.dumps(kcache))   # persist progress after each city
        OCACHE.write_text(json.dumps(ocache))

    KCACHE.write_text(json.dumps(kcache))
    OCACHE.write_text(json.dumps(ocache))

    N = overall["n"]
    print(f"\nTOTAL: {N} markets | direction acc {overall['wins']/N*100:.0f}% | "
          f"Brier {overall['sq']/N:.3f}  (0.25=coin flip)")
    print("\nCalibration (model prob → actual Kalshi settle rate):")
    for b in range(10):
        sp, so, c = bins[b]
        if c:
            print(f"  [{b/10:.1f}-{b/10+0.1:.1f})  n={c:<5} pred {sp/c:.2f} → actual {so/c:.2f}")

    if args.edge and bet_rows:
        def report(label, rows):
            if not rows:
                print(f"  {label:<26} (no bets)")
                return
            n = len(rows)
            w = sum(1 for r in rows if r["won"])
            pnl = sum(r["pnl"] for r in rows)
            clv = sum(r["clv"] for r in rows) / n
            print(f"  {label:<26} bets {n:>4} | win {w/n*100:3.0f}% | "
                  f"P&L {pnl/n*100:+5.1f}% ROI/bet | avg CLV {clv*100:+5.1f}¢")

        def F(**kw):  # filter helper
            return [r for r in bet_rows if all(r[k] == v for k, v in kw.items())]

        print(f"\nSIMULATED BETTING (edge≥{args.min_edge}, real Kalshi prices)")
        print(" by side:")
        report("ALL", bet_rows)
        report("No side", F(side="No"))
        report("Yes side", F(side="Yes"))
        print(" by edge size:")
        for lo, hi in [(0.0, 0.10), (0.10, 0.20), (0.20, 0.30), (0.30, 1.01)]:
            report(f"edge {int(lo*100)}-{int(hi*100)}¢",
                   [r for r in bet_rows if lo <= r["edge"] < hi])
        print(" by entry price (longshot vs near-money):")
        for lo, hi in [(0.0, 0.20), (0.20, 0.40), (0.40, 0.60), (0.60, 1.01)]:
            report(f"entry {int(lo*100)}-{int(hi*100)}¢",
                   [r for r in bet_rows if lo <= r["entry"] < hi])
        print(" Yes side, observation-locked vs forecast-driven:")
        report("Yes, obs-LOCKED", F(side="Yes", locked=True))
        report("Yes, forecast-driven", F(side="Yes", locked=False))

        # ── Honest uncertainty: bets within a city-day are correlated (one weather
        #    outcome drives every threshold), so the effective sample is the number of
        #    distinct city-days, not bets. Bootstrap by RESAMPLING city-days. ──
        import random
        days = sorted({r["city_day"] for r in bet_rows})
        by_day = {d: [r for r in bet_rows if r["city_day"] == d] for d in days}

        def stat(rows, key):
            xs = [r[key] for r in rows if r.get(key) is not None]
            return sum(xs) / len(xs) if xs else float("nan")

        def boot_ci(key, iters=2000):
            means = []
            for _ in range(iters):
                samp = [r for _ in days for r in by_day[random.choice(days)]]
                means.append(stat(samp, key) * 100)
            means.sort()
            return means[int(0.025 * iters)], means[int(0.975 * iters)]

        roi_lo, roi_hi = boot_ci("pnl")
        clv_lo, clv_hi = boot_ci("clv")
        drift_rows = [r for r in bet_rows if r.get("drift") is not None]
        print(f"\n EFFECTIVE SAMPLE: {len(bet_rows)} bets but only {len(days)} independent "
              f"city-days (bets within a day share one outcome).")
        print(f"  ROI/bet  {stat(bet_rows,'pnl')*100:+.1f}%  | 95% CI (city-day bootstrap) "
              f"[{roi_lo:+.1f}%, {roi_hi:+.1f}%]")
        print(f"  CLV/bet  {stat(bet_rows,'clv')*100:+.1f}¢  | 95% CI "
              f"[{clv_lo:+.1f}¢, {clv_hi:+.1f}¢]  (degenerate — settles to 0/1)")
        if drift_rows:
            d_lo, d_hi = boot_ci("drift")
            print(f"  {DRIFT_HOURS}h line-drift toward our side {stat(drift_rows,'drift')*100:+.1f}¢ "
                  f"| 95% CI [{d_lo:+.1f}¢, {d_hi:+.1f}¢]  ← clean pre-resolution edge signal")
        print("  Edge is real only if these CIs sit clearly above zero. Treat anything whose")
        print("  CI straddles 0 as unproven on this window.")


if __name__ == "__main__":
    main()
