#!/usr/bin/env python3
"""
Kalshi Weather Signal Scanner

Fetches active Kalshi temperature markets (greater, less, and between types),
pulls NOAA NWS hourly forecasts, and outputs ranked trading signals where the
model's edge exceeds the fee threshold.

Key modeling note: NWS Climatological Reports use integer degrees, so
resolution thresholds are effectively at X.5°F (e.g., "greater than 83"
resolves Yes for any observation that rounds to ≥84, i.e., actual ≥83.5°F).

Usage:
    python kalshi_module/weather_signal.py
    python kalshi_module/weather_signal.py --bankroll 200 --min-edge 0.07
    python kalshi_module/weather_signal.py --log
"""

import argparse
import csv
from datetime import date, datetime, timezone
from pathlib import Path

import requests
from scipy.stats import norm

# ── API configuration ─────────────────────────────────────────────────────────

KALSHI_API = "https://external-api.kalshi.com/trade-api/v2"
KALSHI_HEADERS = {"accept": "application/json"}
NWS_HEADERS = {
    "User-Agent": "kalshi-weather-signal/1.0 (josiah)",
    "Accept": "application/json",
}

# ── City configuration ────────────────────────────────────────────────────────
# Each city's grid + observation station was verified against Kalshi's published
# settlement rules (the NWS Climatological Report station the contract resolves on)
# on 2026-06-18:
#   NY  → Central Park        CLI NYC  · station KNYC · grid OKX/34,45   (verified)
#   CHI → Chicago Midway      CLI MDW  · station KMDW · grid LOT/72,69   (verified)
#   MIA → Miami Intl Airport  CLI MIA  · station KMIA · grid MFL/105,51  (was 106,51 — fixed)
#   DEN → Denver Intl         CLI DEN  · station KDEN · grid BOU/75,66   (was 63,62 — fixed, ~3°F warm)
#
# `station` is the METAR site whose live observations Kalshi resolves against.
# Grids come from api.weather.gov/points/{station_lat},{station_lon}.
CITIES = {
    "KXHIGHNY": {
        "name": "NYC (Central Park)",
        "station": "KNYC",
        "forecast_url": "https://api.weather.gov/gridpoints/OKX/34,45/forecast/hourly",
        "daily_url":    "https://api.weather.gov/gridpoints/OKX/34,45/forecast",
    },
    "KXHIGHCHI": {
        "name": "Chicago (Midway)",
        "station": "KMDW",
        "forecast_url": "https://api.weather.gov/gridpoints/LOT/72,69/forecast/hourly",
        "daily_url":    "https://api.weather.gov/gridpoints/LOT/72,69/forecast",
    },
    "KXHIGHMIA": {
        "name": "Miami (Intl Airport)",
        "station": "KMIA",
        "forecast_url": "https://api.weather.gov/gridpoints/MFL/105,51/forecast/hourly",
        "daily_url":    "https://api.weather.gov/gridpoints/MFL/105,51/forecast",
    },
    "KXHIGHDEN": {
        "name": "Denver (Intl)",
        "station": "KDEN",
        "forecast_url": "https://api.weather.gov/gridpoints/BOU/75,66/forecast/hourly",
        "daily_url":    "https://api.weather.gov/gridpoints/BOU/75,66/forecast",
    },
}

# ── Signal parameters ─────────────────────────────────────────────────────────

DEFAULT_MIN_EDGE = 0.10  # 0-10¢ "edges" were breakeven in the 478-bet sim (fees/noise)
DEFAULT_BANKROLL = 100.0
KELLY_SCALE = 0.25  # quarter-Kelly: the roadmap's rule until 30+ resolved trades with +CLV
DEFAULT_MIN_OI = 200  # skip markets with phantom/stale prices
DEFAULT_MAX_BET = 10.0  # hard cap per bet until model is validated

# ── Trading discipline (validated: 1,468 real Kalshi settlements + 478-bet sim) ──
# The calibration table alone suggested "only bet the No/low-prob side." The BETTING
# sim against real Kalshi prices (validate_live.py --edge) overturned that — what
# actually has edge is different and was measured, not assumed:
#   • Edge size is monotonic: 0-10¢ ≈ breakeven (+0.1¢ CLV); 10-30¢ ≈ +3¢ CLV;
#     30¢+ is the BEST bucket (+6.5¢ CLV, +5% ROI). Big edges are signal, not bugs.
#   • BOTH sides are profitable (No +3.6¢ CLV, Yes +3.9¢ CLV). Forecast-driven Yes —
#     which the calibration table maligned — pays as cheap longshots the book underprices.
#   • Favorites (entry 60-100¢) are the single strongest bucket (+10.4¢ CLV, 86% win);
#     deep longshots (entry <20¢) win only ~8% — high-variance, only marginal CLV.
# So discipline = require a REAL edge, skip the high-variance deep longshots, keep
# liquidity, and rely on min_oi (not an edge cap) to screen stale books. The earlier
# "No-only / cap edge at 30¢ / reject forecast-Yes" rules were REMOVED — they would
# have thrown away the most profitable bets. Sample is ~2 months and correlated by
# city-day, so treat as provisional and keep validating forward via --grade.
MIN_ENTRY_PRICE = 0.15   # below this, win rate ≈8% — too high-variance for the pilot
ABSURD_EDGE = 0.50       # pure data-error guard (bad threshold parse / dead book), not a signal filter

CSV_PATH = Path(__file__).parent.parent / "data" / "kalshi_trades.csv"
CSV_FIELDNAMES = [
    "Date", "Market", "Side", "My_Model_Pct", "Kalshi_Price",
    "Edge", "Entry_Price", "Position_Size_USD", "Kelly_Fraction",
    "Closing_Price", "CLV", "Outcome", "PnL_USD", "Notes",
]


# ── Core math ─────────────────────────────────────────────────────────────────

def kalshi_taker_fee(price: float) -> float:
    """Taker fee per $1 notional: 7¢ × C × (1 - C)."""
    return 0.07 * price * (1.0 - price)


def forecast_std(days_out: float) -> float:
    """
    Max-temperature forecast uncertainty (1-sigma, °F) for the DAY-AHEAD fallback
    used only on markets resolving in the future (no live observations yet). The
    same-day intraday model — the calibrated, validated path — does not use this.

    Anchor: the same-day-known full-day forecast error measured over 1,840
    city-days of Open-Meteo data is σ≈2.2°F. A genuine day-1-ahead forecast is
    wider than that, and skill degrades further out, so these widen with horizon.
    Deliberately conservative (wider → fewer false future-day signals); the
    future-day path is not yet validated against settled outcomes.
    """
    if days_out <= 1:
        return 3.0
    elif days_out <= 2:
        return 3.5
    elif days_out <= 3:
        return 4.0
    elif days_out <= 4:
        return 5.0
    else:
        return 6.5


def model_prob_yes(
    forecast_max: float,
    strike_type: str,
    floor: float | None,
    cap: float | None,
    days_out: float,
) -> float:
    """
    P(Yes resolves) for each Kalshi strike type.

    NWS reports integer °F, so resolution thresholds shift by ±0.5°F:
      - greater (>floor): Yes if reported ≥ floor+1  →  actual ≥ floor+0.5
      - less    (≤cap):   Yes if reported ≤ cap       →  actual < cap+0.5
      - between [floor,cap]: Yes if reported ∈ {floor…cap} →  actual ∈ [floor-0.5, cap+0.5)
    """
    std = forecast_std(days_out)
    if strike_type == "greater":
        return float(1.0 - norm.cdf(floor + 0.5, loc=forecast_max, scale=std))
    elif strike_type == "less":
        return float(norm.cdf(cap + 0.5, loc=forecast_max, scale=std))
    else:  # between
        return float(
            norm.cdf(cap + 0.5, loc=forecast_max, scale=std)
            - norm.cdf(floor - 0.5, loc=forecast_max, scale=std)
        )


def yes_observation_locked(
    strike_type: str, floor: float | None, cap: float | None, observed_max: float | None
) -> bool:
    """
    True when today's observed max ALREADY guarantees a Yes resolution, so the bet
    no longer depends on the forecast. Only "greater than floor" can lock upward
    intraday (the temp can't un-happen); "less"/"between" can't lock Yes from above
    midday because a later rise could still break them.
    """
    if observed_max is None:
        return False
    if strike_type == "greater" and floor is not None:
        return observed_max >= floor + 0.5  # rounds to ≥ floor+1 → reported > floor
    return False


def kelly_fraction(model_prob: float, entry_price: float) -> float:
    """Kelly criterion for a binary bet: f* = (p - c) / (1 - c)."""
    if entry_price >= 1.0:
        return 0.0
    return max(0.0, (model_prob - entry_price) / (1.0 - entry_price))


# ── Intraday model ─────────────────────────────────────────────────────────────
# The day's final max temperature obeys a hard identity:
#       final_max = max(observed_max_so_far, max over remaining hours)
# `observed_max_so_far` is a floor the temperature can never go below, so once it
# clears a threshold the outcome is locked. As the afternoon peak approaches there
# are fewer unknown hours, so the distribution of the remaining max narrows toward
# zero. This is the information a market anchored on the morning forecast lags.
#
# ── Calibrated parameters ───────────────────────────────────────────────────────
# Fitted by regression in calibrate_weather.py on 4 training summers (2021-2024, all
# 4 cities, ~11.8k samples) and validated OUT-OF-SAMPLE on summer 2025. The original
# hand-set values were badly overconfident (predicted ~48% on thresholds that hit
# ~38%); the fitted set is well-calibrated to mildly conservative (0.47→0.52 OOS).
#   σ(hours_to_peak) = STD_AT_PEAK + STD_PER_HOUR_TO_PEAK * hours_to_peak  (capped)
STD_AT_PEAK = 1.4          # °F uncertainty when the peak is imminent (was 0.7 — too tight)
STD_PER_HOUR_TO_PEAK = 0.10  # °F added per forecast-hour until the peak (was 0.45)
STD_CAP = 3.2              # °F ceiling on σ

# σ floor for the UNDER-OBSERVED case. Normally hours-to-peak captures uncertainty, but
# when the forecast front-loads a big jump the labeled peak can be "imminent" (h2p≈0)
# while observations are still many degrees below it — e.g. obs 81°F at 2pm, forecast
# 91°F at 3pm. The h2p formula would call that near-certain (σ≈1.4); it is not. The
# training data shows a large unrealized rise carries irreducible error std ≈1.8-1.9°F
# regardless of h2p, so we floor σ at STD_AT_PEAK + STD_PER_DEGREE_RISE·(projected rise).
STD_PER_DEGREE_RISE = 0.10  # °F of σ per °F of still-unobserved projected rise

# Carrying the morning residual forward worsened calibration in the backtest — off.
RESIDUAL_DAMP = 0.0        # fraction of the running residual carried to the peak
RESID_CAP = 2.0            # °F: max magnitude of the carried residual

# center = observed_max + PEAK_DISCOUNT*(forecast_remaining_peak - observed_max) + PEAK_BIAS
# Fit showed the forecast's projected rise above observed is ~92% trustworthy. The fit
# also wanted PEAK_BIAS=-0.7°F, but that reflects Open-Meteo's warm forecast bias and
# does NOT transfer to live NWS data (there it becomes a spurious cool bias that invents
# "No" edges). So live default is 0; the true per-source bias must be learned from
# settled NWS outcomes via `--grade`. The σ widening below is the transferable fix.
PEAK_DISCOUNT = 0.92
PEAK_BIAS = 0.0            # °F (calibrate_weather.py found -0.7 on Open-Meteo; see note)

# Hours-to-peak below which an intraday signal is considered high-confidence
# (the unknown window is small). Above it, the signal still leans on the forecast.
PEAK_SOON_HOURS = 2.0


def intraday_max_cdf(
    obs_today: list[tuple[datetime, float]],
    hourly_today: list[tuple[datetime, float]],
    now: datetime,
    daily_high: float | None = None,
):
    """
    Build CDF F(x) = P(day's final max ≤ x) given live observations + the hourly
    forecast for the remaining hours.

    `daily_high`, when supplied, is the official NWS *daily* forecast high. The raw
    NWS hourly grid runs ~1.6°F warm vs the Climatological Report Kalshi resolves on
    (and vs the official daily high), which inflates the projected afternoon peak and
    biases Yes probabilities upward — the live "Chicago bug". We anchor by capping the
    raw remaining-hours peak at the official daily high: trust the hourly grid for peak
    *timing*, the daily high for peak *level*.

    Returns (cdf, debug) where debug carries the pieces used so signals can be
    explained and audited. Returns (None, reason) when there isn't enough data.
    """
    observed = [t for _, t in obs_today]
    if not observed:
        return None, "no observations yet today"
    observed_max = max(observed)

    # How is today actually tracking vs the forecast for the hours already elapsed?
    # A morning anomaly mean-reverts, so damp it hard and cap it before carrying it
    # forward to the afternoon peak.
    raw_resid = _intraday_residual(obs_today, hourly_today)
    resid = max(-RESID_CAP, min(RESID_CAP, RESIDUAL_DAMP * raw_resid))

    # Bias-corrected forecast for hours strictly after `now`.
    remaining = [(dt, t + resid) for dt, t in hourly_today if dt > now]

    if not remaining:
        # No daylight/forecast hours left: the max is effectively locked in.
        def cdf(x: float) -> float:
            return 1.0 if x >= observed_max else 0.0
        debug = {
            "observed_max": observed_max, "raw_residual": raw_resid, "residual": resid,
            "remaining_peak": None, "std": 0.0, "hours_to_peak": 0.0, "confidence": "locked",
        }
        return cdf, debug

    raw_remaining_peak = max(t for _, t in remaining)
    peak_dt = max(remaining, key=lambda p: p[1])[0]
    # Anchor the (warm) hourly grid to the official NWS daily high: never project the
    # remaining peak above it, as long as the day isn't already past that high.
    anchored = False
    if daily_high is not None and observed_max <= daily_high < raw_remaining_peak:
        raw_remaining_peak = daily_high
        anchored = True
    # Discount the projected rise above what's already observed, plus a small global
    # bias (see PEAK_DISCOUNT / PEAK_BIAS — both calibrated in calibrate_weather.py).
    if raw_remaining_peak > observed_max:
        remaining_peak = observed_max + PEAK_DISCOUNT * (raw_remaining_peak - observed_max) + PEAK_BIAS
    else:
        remaining_peak = raw_remaining_peak
    hours_to_peak = max(0.0, (peak_dt - now).total_seconds() / 3600.0)
    projected_rise = max(0.0, remaining_peak - observed_max)  # still-unobserved climb
    std = min(STD_CAP, max(
        STD_AT_PEAK + STD_PER_HOUR_TO_PEAK * hours_to_peak,
        STD_AT_PEAK + STD_PER_DEGREE_RISE * projected_rise,
    ))
    # Confidence reflects what we actually KNOW, not just the clock: a peak that is
    # "imminent" but still 3°F+ above the observed max is forecast-dependent, not locked.
    if observed_max >= remaining_peak:
        confidence = "locked"        # obs already at/above the projected peak
    elif hours_to_peak <= PEAK_SOON_HOURS and projected_rise <= 3.0:
        confidence = "high"          # peak soon AND obs nearly there
    else:
        confidence = "speculative"   # outcome still hinges on an unobserved rise

    def cdf(x: float) -> float:
        # final = max(observed_max, R), R ~ Normal(remaining_peak, std)
        if x < observed_max:
            return 0.0
        return float(norm.cdf(x, loc=remaining_peak, scale=std))

    debug = {
        "observed_max": observed_max, "raw_residual": raw_resid, "residual": resid,
        "remaining_peak": remaining_peak, "std": std, "hours_to_peak": hours_to_peak,
        "projected_rise": projected_rise, "confidence": confidence,
        "anchored_to_daily_high": anchored,
    }
    return cdf, debug


def _intraday_residual(
    obs_today: list[tuple[datetime, float]],
    hourly_today: list[tuple[datetime, float]],
) -> float:
    """Mean (observed − forecast) over elapsed hours: today's running warm/cool bias."""
    fc_by_hour = {dt.replace(minute=0, second=0, microsecond=0): t for dt, t in hourly_today}
    diffs = []
    for dt, t in obs_today:
        key = dt.replace(minute=0, second=0, microsecond=0)
        if key in fc_by_hour:
            diffs.append(t - fc_by_hour[key])
    if not diffs:
        return 0.0
    # Weight recent hours more: simple mean of the last 6 matched hours.
    recent = diffs[-6:]
    return sum(recent) / len(recent)


def prob_yes_from_cdf(cdf, strike_type: str, floor: float | None, cap: float | None) -> float:
    """P(Yes) for each strike type, using the integer-rounding thresholds (±0.5°F)."""
    if strike_type == "greater":
        return 1.0 - cdf(floor + 0.5)
    elif strike_type == "less":
        return cdf(cap + 0.5)
    else:  # between
        return cdf(cap + 0.5) - cdf(floor - 0.5)


# ── Data fetching ─────────────────────────────────────────────────────────────

def fetch_kalshi_markets(series_ticker: str) -> list[dict]:
    """
    Return all open temperature markets for a series.
    Includes greater, less, and between types.
    Requires at least one of floor_strike or cap_strike to be present.
    """
    r = requests.get(
        f"{KALSHI_API}/markets",
        headers=KALSHI_HEADERS,
        params={"series_ticker": series_ticker, "status": "open", "limit": 100},
        timeout=10,
    )
    r.raise_for_status()
    return [
        m for m in r.json().get("markets", [])
        if m.get("strike_type") in ("greater", "less", "between")
        and (m.get("floor_strike") is not None or m.get("cap_strike") is not None)
    ]


def fetch_nws_forecasts(city_cfg: dict) -> dict[str, float]:
    """
    Combine NWS daily and hourly forecasts to return {YYYY-MM-DD: max_temp_F}.
    Uses the daily forecast (daytime periods) as the primary source since it
    gives the official human-forecasted high — more accurate than raw hourly
    model output and matches what's displayed on weather apps.
    Falls back to hourly max-per-day when daily is unavailable for a date.
    """
    daily: dict[str, float] = {}

    # Daily forecast: daytime periods have the official high
    try:
        r = requests.get(city_cfg["daily_url"], headers=NWS_HEADERS, timeout=15)
        r.raise_for_status()
        for p in r.json().get("properties", {}).get("periods", []):
            if p.get("isDaytime") and p.get("temperatureUnit") == "F":
                # startTime like "2026-06-03T06:00:00-04:00" → date "2026-06-03"
                day = p.get("startTime", "")[:10]
                if day:
                    daily[day] = float(p["temperature"])
    except Exception:
        pass

    # Hourly fallback for any dates not covered by daily.
    # NWS raw hourly grid runs +1.63°F warm vs the NWS Climatological Report
    # (empirical from 34 settled Kalshi NYC markets, Apr–Jun 2026).
    # Apply bias correction when using this source.
    HOURLY_BIAS = -1.63
    try:
        r = requests.get(city_cfg["forecast_url"], headers=NWS_HEADERS, timeout=15)
        r.raise_for_status()
        hourly_by_day: dict[str, list[float]] = {}
        for p in r.json().get("properties", {}).get("periods", []):
            if p.get("temperatureUnit") != "F":
                continue
            day = p.get("startTime", "")[:10]
            if day:
                hourly_by_day.setdefault(day, []).append(float(p["temperature"]))
        for day, temps in hourly_by_day.items():
            if day not in daily:
                daily[day] = max(temps) + HOURLY_BIAS
    except Exception:
        pass

    return daily


def _c_to_f(c: float) -> float:
    return c * 9.0 / 5.0 + 32.0


def fetch_station_observations(station: str, limit: int = 60) -> list[tuple[datetime, float]]:
    """
    Recent METAR observations for `station` as [(aware_datetime, temp_F)], oldest
    first. This is the data Kalshi resolves against. Caller filters to the local day.
    """
    r = requests.get(
        f"https://api.weather.gov/stations/{station}/observations",
        headers=NWS_HEADERS, params={"limit": limit}, timeout=15,
    )
    r.raise_for_status()
    out: list[tuple[datetime, float]] = []
    for f in r.json().get("features", []):
        p = f.get("properties", {})
        tval = (p.get("temperature") or {}).get("value")
        ts = p.get("timestamp")
        if tval is None or not ts:
            continue
        out.append((datetime.fromisoformat(ts), _c_to_f(tval)))
    out.sort(key=lambda x: x[0])
    return out


def observations_for_local_day(
    obs: list[tuple[datetime, float]], tz, target_date: str
) -> list[tuple[datetime, float]]:
    """Keep observations whose LOCAL (station-tz) calendar date equals target_date."""
    return [(dt, t) for dt, t in obs if dt.astimezone(tz).date().isoformat() == target_date]


def fetch_hourly_forecast(city_cfg: dict, target_date: str | None = None) -> list[tuple[datetime, float]]:
    """Timestamped hourly forecast as [(aware_datetime, temp_F)]; all hours unless
    `target_date` is given. The tz of each datetime is the station's local tz."""
    r = requests.get(city_cfg["forecast_url"], headers=NWS_HEADERS, timeout=15)
    r.raise_for_status()
    out: list[tuple[datetime, float]] = []
    for p in r.json().get("properties", {}).get("periods", []):
        if p.get("temperatureUnit") != "F":
            continue
        ts = p.get("startTime", "")
        if target_date is not None and ts[:10] != target_date:
            continue
        out.append((datetime.fromisoformat(ts), float(p["temperature"])))
    out.sort(key=lambda x: x[0])
    return out


# ── Signal computation ────────────────────────────────────────────────────────

def compute_signals(
    today: date,
    bankroll: float = DEFAULT_BANKROLL,
    min_edge: float = DEFAULT_MIN_EDGE,
    min_days: int = 0,
    min_oi: float = DEFAULT_MIN_OI,
    max_bet: float = DEFAULT_MAX_BET,
    discipline: bool = True,
) -> list[dict]:
    signals: list[dict] = []
    disc_skips = {"buggy_edge": 0, "deep_longshot": 0}

    for series_ticker, city_cfg in CITIES.items():
        print(f"  {city_cfg['name']:<28}", end=" ", flush=True)

        try:
            markets = fetch_kalshi_markets(series_ticker)
        except Exception as e:
            print(f"[kalshi error: {e}]")
            continue

        if not markets:
            print("[no open markets]")
            continue

        try:
            daily_maxes = fetch_nws_forecasts(city_cfg)
        except Exception as e:
            print(f"[nws error: {e}]")
            continue

        # ── Intraday state for the CURRENT local day (same-day markets) ──
        now = datetime.now(timezone.utc)
        intraday_cdf = None
        intraday_dbg: dict = {}
        today_local = today.isoformat()
        try:
            hourly_all = fetch_hourly_forecast(city_cfg)
            tz = hourly_all[0][0].tzinfo if hourly_all else timezone.utc
            today_local = now.astimezone(tz).date().isoformat()
            hourly_today = [
                (dt, t) for dt, t in hourly_all
                if dt.astimezone(tz).date().isoformat() == today_local
            ]
            obs_today = observations_for_local_day(
                fetch_station_observations(city_cfg["station"]), tz, today_local
            )
            intraday_cdf, intraday_dbg = intraday_max_cdf(
                obs_today, hourly_today, now, daily_high=daily_maxes.get(today_local)
            )
        except Exception as e:
            intraday_dbg = {"error": str(e)}

        city_signals = 0
        for market in markets:
            occurrence = market.get("occurrence_datetime", "")
            if not occurrence:
                continue
            target_date = occurrence[:10]

            days_out = (date.fromisoformat(target_date) - today).days
            if days_out < min_days:
                continue

            strike_type = market.get("strike_type", "")
            floor = market.get("floor_strike")
            cap = market.get("cap_strike")

            # Ensure we have the threshold we need for this strike type
            if strike_type == "greater" and floor is None:
                continue
            if strike_type == "less" and cap is None:
                continue
            if strike_type == "between" and (floor is None or cap is None):
                continue

            floor = float(floor) if floor is not None else None
            cap = float(cap) if cap is not None else None

            # ── Choose model: intraday for today, forecast-Gaussian for future days ──
            is_today = (target_date == today_local) and callable(intraday_cdf)
            if is_today:
                p_yes = prob_yes_from_cdf(intraday_cdf, strike_type, floor, cap)
                model_kind = "intraday"
                forecast_max = intraday_dbg.get("remaining_peak") or intraday_dbg.get("observed_max")
            else:
                forecast_max = daily_maxes.get(target_date)
                if forecast_max is None:
                    continue
                p_yes = model_prob_yes(forecast_max, strike_type, floor, cap, max(days_out, 0.5))
                model_kind = "forecast"
            p_yes = min(1.0, max(0.0, p_yes))
            p_no = 1.0 - p_yes

            yes_ask = float(market.get("yes_ask_dollars") or 1.0)
            no_ask = float(market.get("no_ask_dollars") or 1.0)

            yes_net_edge = p_yes - yes_ask - kalshi_taker_fee(yes_ask)
            no_net_edge = p_no - no_ask - kalshi_taker_fee(no_ask)

            best_edge = max(yes_net_edge, no_net_edge)
            if best_edge < min_edge:
                continue

            if yes_net_edge >= no_net_edge:
                side, entry_price, model_pct, edge = "Yes", yes_ask, p_yes, yes_net_edge
            else:
                side, entry_price, model_pct, edge = "No", no_ask, p_no, no_net_edge

            # ── Discipline: trade only where the betting sim showed real edge ──
            if discipline:
                if edge > ABSURD_EDGE:
                    disc_skips["buggy_edge"] += 1
                    continue  # data-error guard only (min_oi handles staleness)
                if entry_price < MIN_ENTRY_PRICE:
                    disc_skips["deep_longshot"] += 1
                    continue  # ~8% win rate — too high-variance for the pilot

            oi = float(market.get("open_interest_fp") or 0)
            if oi < min_oi:
                continue  # Stale/phantom price — no real liquidity

            kf = kelly_fraction(model_pct, entry_price)
            bet_size = min(bankroll * kf * KELLY_SCALE, max_bet)

            # Human-readable threshold description
            if strike_type == "greater":
                threshold_str = f">{int(floor)}°F"
            elif strike_type == "less":
                threshold_str = f"≤{int(cap)}°F"
            else:
                threshold_str = f"{int(floor)}-{int(cap)}°F"

            signals.append({
                "city": city_cfg["name"],
                "ticker": market["ticker"],
                "title": market.get("title", ""),
                "target_date": target_date,
                "days_out": days_out,
                "strike_type": strike_type,
                "threshold_str": threshold_str,
                "model_kind": model_kind,
                "confidence": (intraday_dbg.get("confidence") if model_kind == "intraday" else "forecast"),
                "forecast_max": round(forecast_max, 1) if forecast_max is not None else None,
                "observed_max": (round(intraday_dbg["observed_max"], 1)
                                 if model_kind == "intraday" and intraday_dbg.get("observed_max") is not None else None),
                "remaining_peak": (round(intraday_dbg["remaining_peak"], 1)
                                   if model_kind == "intraday" and intraday_dbg.get("remaining_peak") is not None else None),
                "residual": (round(intraday_dbg.get("residual", 0.0), 1) if model_kind == "intraday" else None),
                "hours_to_peak": (round(intraday_dbg.get("hours_to_peak", 0.0), 1) if model_kind == "intraday" else None),
                "projected_rise": (round(intraday_dbg.get("projected_rise", 0.0), 1) if model_kind == "intraday" else None),
                "std": (round(intraday_dbg.get("std", 0.0), 2) if model_kind == "intraday" else None),
                "model_pct": round(p_yes, 4),
                "side": side,
                "entry_price": round(entry_price, 4),
                "edge": round(edge, 4),
                "kelly_fraction": round(kf, 4),
                "bet_size_usd": round(bet_size, 2),
                "open_interest": oi,
                "volume_24h": float(market.get("volume_24h_fp") or 0),
            })
            city_signals += 1

        total = len(markets)
        if callable(intraday_cdf):
            om = intraday_dbg.get("observed_max")
            rp = intraday_dbg.get("remaining_peak")
            state = f"obs_max={om:.0f}F" + (f" peak~{rp:.0f}F" if rp is not None else " (peaked)")
        else:
            state = intraday_dbg.get("error", "no intraday")
        print(f"[{total} markets, {city_signals} signal{'s' if city_signals != 1 else ''}, {state}]")

    if discipline and (disc_skips["buggy_edge"] or disc_skips["deep_longshot"]):
        print(f"  discipline filtered: {disc_skips['deep_longshot']} deep-longshot "
              f"(entry <{MIN_ENTRY_PRICE*100:.0f}¢), {disc_skips['buggy_edge']} absurd-edge "
              f"(>{ABSURD_EDGE*100:.0f}¢ — data error)")

    signals.sort(key=lambda x: x["edge"], reverse=True)
    return signals


# ── Output ────────────────────────────────────────────────────────────────────

# Plain-language trust tiers. Ordering = how much to trust the pick (best first).
TRUST_RANK = {"locked": 0, "high": 1, "speculative": 2, "forecast": 3}


def _risk_note(s: dict) -> str:
    """One-line, plain-English read on what a pick depends on."""
    c = s["confidence"]
    if c == "locked":
        return "LOCKED — today's observed high already decides this; lowest risk"
    if c == "high":
        return "SOLID — peak is imminent and the temp is nearly there"
    if c == "speculative":
        rise = s.get("projected_rise")
        extra = f" a further +{rise:.0f}°F" if rise else " more warming"
        return f"RISKY — needs{extra} that hasn't happened yet; forecast-dependent"
    return "NEXT-DAY — pure forecast, no live observations yet; most speculative"


def print_signals(signals: list[dict], bankroll: float) -> None:
    print(f"\n{'━' * 68}")
    print(f"  KALSHI WEATHER SIGNALS  ·  {date.today().isoformat()}")
    print(f"{'━' * 68}")

    if not signals:
        print("\n  No signals above edge threshold today.\n")
        return

    # Sort by trust tier first, then edge — the safest, strongest picks rise to the top.
    signals = sorted(signals, key=lambda s: (TRUST_RANK.get(s["confidence"], 9), -s["edge"]))

    trustworthy = [s for s in signals if s["confidence"] in ("locked", "high")]
    print(f"\n  ▶ WHAT TO DO: {len(trustworthy)} trustworthy pick(s) "
          f"(locked/solid); {len(signals) - len(trustworthy)} speculative (act small or skip).")
    if trustworthy:
        names = ", ".join(f"{s['city'].split(' (')[0]} {s['threshold_str']} {s['side']}"
                          for s in trustworthy[:3])
        print(f"    Start with: {names}.")
    print(f"    ⏰ Best traded ~12–1pm LOCAL per city — validation shows the edge fades to")
    print(f"       zero after ~2pm (the market prices the same observations you see).")
    print(f"    All bets are paper/small until 30+ resolved trades show positive edge.")

    for i, s in enumerate(signals, 1):
        print(f"\n  #{i}  {s['ticker']}  [{s['city']}]  ({s['model_kind']}·{s['confidence']})")
        print(f"       {s['title']}")
        print(f"       → {_risk_note(s)}")
        print(f"       Resolves: {s['target_date']} ({s['days_out']}d out)  |  Type: {s['strike_type']}  |  Threshold: {s['threshold_str']}")
        if s["model_kind"] == "intraday":
            print(f"       Observed max so far: {s['observed_max']}°F  |  "
                  f"Projected peak: {s['remaining_peak']}°F  (needs +{s['projected_rise']:.0f}°F more)")
            print(f"       Hours to peak: {s['hours_to_peak']}  |  Uncertainty σ: {s['std']}°F  |  Confidence: {s['confidence']}")
        else:
            print(f"       NWS forecast max: {s['forecast_max']}°F  (day-ahead Gaussian)")
        print(f"       Model prob (Yes): {s['model_pct']*100:.1f}%  |  Kalshi ask: {s['entry_price']*100:.0f}¢")
        print(f"       Side: {s['side']}  |  Net edge: +{s['edge']*100:.1f}¢  |  Kelly: {s['kelly_fraction']*100:.1f}%")
        print(f"       Suggested bet: ${s['bet_size_usd']:.2f}  (¼-Kelly on ${bankroll:.0f} bankroll)")
        print(f"       OI: {s['open_interest']:.0f}  |  Vol 24h: {s['volume_24h']:.0f}")

    print(f"\n{'━' * 68}")
    print(f"  {len(signals)} signal(s).  Enter trades manually at kalshi.com")
    print(f"  Trust tiers: LOCKED (decided) > SOLID (peak imminent, obs there) > "
          f"RISKY (needs unobserved rise) > NEXT-DAY (forecast only).")
    print(f"  Log with --log, then grade the next day with --grade to build a CLV record.")
    print(f"{'━' * 68}\n")


def log_signals_to_csv(signals: list[dict], auto_log: bool, dedupe: bool = False) -> None:
    if not signals:
        return

    if dedupe and CSV_PATH.exists():
        with open(CSV_PATH, newline="") as f:
            already = {r.get("Market", "") for r in csv.DictReader(f)}
        signals = [s for s in signals if s["ticker"] not in already]
        if not signals:
            print("✓ Paper log: all of today's picks already logged (nothing new).")
            return

    if not auto_log:
        print(f"Log to {CSV_PATH.name}?  [a]ll / [1] top only / [n]o  → ", end="")
        choice = input().strip().lower()
        if choice == "n" or choice == "":
            return
        to_log = signals if choice == "a" else signals[:1]
    else:
        to_log = signals

    today_str = date.today().isoformat()
    rows = [
        {
            "Date": today_str,
            "Market": s["ticker"],
            "Side": s["side"],
            "My_Model_Pct": s["model_pct"],
            "Kalshi_Price": s["entry_price"],
            "Edge": s["edge"],
            "Entry_Price": s["entry_price"],
            "Position_Size_USD": s["bet_size_usd"],
            "Kelly_Fraction": s["kelly_fraction"],
            "Closing_Price": "",
            "CLV": "",
            "Outcome": "",
            "PnL_USD": "",
            "Notes": (
                f"NWS_max={s['forecast_max']}F "
                f"threshold={s['threshold_str']} "
                f"{s['days_out']}d_out "
                f"{s['strike_type']}"
            ),
        }
        for s in to_log
    ]

    file_exists = CSV_PATH.exists()
    with open(CSV_PATH, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDNAMES)
        if not file_exists:
            writer.writeheader()
        writer.writerows(rows)

    print(f"✓ Logged {len(rows)} signal(s) to {CSV_PATH}")


# ── Grading / validation ──────────────────────────────────────────────────────
# The roadmap's gate for real capital is "30 resolved trades with positive CLV."
# This grades logged trades against the actual high temperature Kalshi resolved on.

def station_max_for_date(station: str, day: str) -> float | None:
    """Actual observed max temp (°F) at `station` for local calendar day `day`.
    Uses the NWS observations archive (reliable ~7 days back, longer for many sites)."""
    try:
        r = requests.get(
            f"https://api.weather.gov/stations/{station}/observations",
            headers=NWS_HEADERS,
            params={"start": f"{day}T00:00:00+00:00", "end": f"{day}T23:59:59+00:00", "limit": 200},
            timeout=20,
        )
        r.raise_for_status()
        temps = [
            (f["properties"]["temperature"]["value"])
            for f in r.json().get("features", [])
            if (f["properties"].get("temperature") or {}).get("value") is not None
        ]
        return max(_c_to_f(t) for t in temps) if temps else None
    except Exception:
        return None


def yes_closing_price(series: str, ticker: str, res_date: str) -> float | None:
    """
    Last traded Yes price (0-1) before the market resolved — the "closing line".
    CLV = (our side's closing price − our entry) is the roadmap's robust edge metric,
    so grading must capture it. Pulls Kalshi hourly candlesticks over a generous UTC
    window around the resolution day and returns the final candle with a valid close.
    """
    try:
        start = int(datetime.fromisoformat(f"{res_date}T00:00:00+00:00").timestamp())
        end = int(datetime.fromisoformat(f"{res_date}T23:59:59+00:00").timestamp()) + 86400
        r = requests.get(
            f"{KALSHI_API}/series/{series}/markets/{ticker}/candlesticks",
            headers=KALSHI_HEADERS,
            params={"start_ts": start, "end_ts": end, "period_interval": 60},
            timeout=15,
        )
        r.raise_for_status()
        for k in reversed(r.json().get("candlesticks", [])):
            px = (k.get("price") or {}).get("close_dollars")
            if px not in (None, ""):
                return float(px)
    except Exception:
        return None
    return None


def _parse_threshold_from_notes(notes: str) -> tuple[str, float | None, float | None] | None:
    """Recover (strike_type, floor, cap) from the Notes string we log."""
    import re
    m = re.search(r"threshold=([^ ]+)", notes or "")
    if not m:
        return None
    th = m.group(1)
    if th.startswith(">"):
        return "greater", float(re.sub(r"[^\d.]", "", th)), None
    if th.startswith("≤") or th.startswith("<"):
        return "less", None, float(re.sub(r"[^\d.]", "", th))
    if "-" in th:
        lo, hi = re.findall(r"[\d.]+", th)[:2]
        return "between", float(lo), float(hi)
    return None


def _yes_resolved(strike_type: str, floor, cap, actual_max: float) -> bool:
    reported = round(actual_max)  # CLI reports integer °F
    if strike_type == "greater":
        return reported > floor
    if strike_type == "less":
        return reported <= cap
    return floor <= reported <= cap


def grade_trades() -> None:
    if not CSV_PATH.exists():
        print("No trades file to grade.")
        return
    with open(CSV_PATH, newline="") as f:
        rows = list(csv.DictReader(f))

    series_to_station = {sk: cfg["station"] for sk, cfg in CITIES.items()}
    today = date.today()
    graded = 0
    for row in rows:
        if row.get("Outcome"):
            continue  # already graded
        if row.get("Market", "").startswith("EXAMPLE"):
            continue
        # Resolution date is encoded in the ticker, e.g. KXHIGHDEN-26JUN03-T88
        parts = row.get("Market", "").split("-")
        if len(parts) < 2:
            continue
        try:
            res_date = datetime.strptime(parts[1], "%y%b%d").date()
        except Exception:
            continue
        if res_date >= today:
            continue  # not resolved yet
        day = res_date.isoformat()
        series = parts[0]
        station = series_to_station.get(series)
        parsed = _parse_threshold_from_notes(row.get("Notes", ""))
        if not station or not parsed:
            continue
        actual = station_max_for_date(station, day)
        if actual is None:
            row["Outcome"] = ""  # leave for retry; obs archive may not reach this far
            continue
        strike_type, floor, cap = parsed
        yes = _yes_resolved(strike_type, floor, cap, actual)
        won = (yes and row["Side"] == "Yes") or ((not yes) and row["Side"] == "No")
        entry = float(row["Entry_Price"])
        size = float(row["Position_Size_USD"])
        contracts = size / entry if entry > 0 else 0.0
        pnl = contracts * ((1.0 if won else 0.0) - entry) - contracts * kalshi_taker_fee(entry)
        row["Outcome"] = "Win" if won else "Loss"
        row["PnL_USD"] = f"{pnl:.2f}"
        row["Notes"] = (row.get("Notes", "") + f" | actual_max={actual:.0f}F").strip()

        # Closing line value: store OUR SIDE's closing price so CLV = close − entry
        # works uniformly for Yes and No (the summary below relies on that).
        if not row.get("Closing_Price"):
            yes_close = yes_closing_price(series, row["Market"], day)
            if yes_close is not None:
                our_close = yes_close if row["Side"] == "Yes" else (1.0 - yes_close)
                row["Closing_Price"] = f"{our_close:.4f}"
                row["CLV"] = f"{our_close - entry:.4f}"
        graded += 1

    with open(CSV_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)

    # Summary
    resolved = [r for r in rows if r.get("Outcome") in ("Win", "Loss")]
    wins = sum(1 for r in resolved if r["Outcome"] == "Win")
    pnl = sum(float(r["PnL_USD"]) for r in resolved if r.get("PnL_USD"))
    clvs = [float(r["CLV"]) for r in resolved if r.get("CLV") not in (None, "")]
    # Distinct city-days = the honest effective sample (a day's bets share one outcome).
    city_days = {(r.get("Market", "").split("-")[0], r.get("Market", "").split("-")[1])
                 for r in resolved if len(r.get("Market", "").split("-")) >= 2}
    print(f"\nGraded {graded} new trade(s). Resolved total: {len(resolved)} "
          f"across {len(city_days)} independent city-days.")
    if resolved:
        print(f"  Record: {wins}-{len(resolved)-wins}  ({wins/len(resolved)*100:.0f}% hit rate)")
        print(f"  Total P&L: ${pnl:+.2f}  (avg {pnl/len(resolved):+.2f}/trade)")
        if clvs:
            print(f"  Avg CLV: {sum(clvs)/len(clvs)*100:+.1f}¢ over {len(clvs)} trades (closing line auto-captured)")
            print("  NOTE: same-day weather settles to ~0/1, so CLV ≈ P&L here (not an")
            print("        independent signal). Robust edge = the multi-month betting sim + ")
            print("        forward consistency. Treat ROI and CLV agreement as one confirmation.")
        print("  → Gate for real capital: 30+ resolved trades, positive P&L AND CLV.")

        _report_per_city_bias(resolved)


def _report_per_city_bias(resolved: list[dict]) -> None:
    """
    Measure the LIVE forecast bias per city: mean(forecast_max − actual_max) from
    graded trades. PEAK_BIAS is 0 because the Open-Meteo fit's −0.7°F did not transfer
    to NWS; this is how we learn the real NWS-vs-station bias. We REPORT it, we do not
    auto-apply it: validation showed a global center shift raises Brier (it corrupts the
    well-calibrated No side), so any bias correction must be confirmed to improve
    calibration before it's wired into PEAK_BIAS — and ideally applied per city.
    """
    import re
    by_city: dict[str, list[float]] = {}
    for r in resolved:
        notes = r.get("Notes", "")
        fm = re.search(r"NWS_max=([\d.]+)F", notes)
        am = re.search(r"actual_max=([\d.]+)F", notes)
        if not fm or not am:
            continue
        series = r.get("Market", "").split("-")[0]
        city = CITIES.get(series, {}).get("name", series)
        by_city.setdefault(city, []).append(float(fm.group(1)) - float(am.group(1)))
    if not by_city:
        return
    print("\n  Measured forecast bias (forecast − actual, °F) — for calibration, not auto-applied:")
    for city, diffs in sorted(by_city.items()):
        mean = sum(diffs) / len(diffs)
        warm = "warm→inflates Yes" if mean > 0.5 else ("cool→inflates No" if mean < -0.5 else "≈unbiased")
        print(f"    {city:<24} {mean:+.1f}°F over {len(diffs):>2} trades   ({warm})")
    print("  Only consider a PEAK_BIAS change once a city has 20+ trades AND a re-run of")
    print("  validate_live.py with that bias LOWERS Brier vs the current 0.")


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Kalshi weather signal scanner — all market types, corrected edge model"
    )
    parser.add_argument("--bankroll", type=float, default=DEFAULT_BANKROLL)
    parser.add_argument("--min-edge", type=float, default=DEFAULT_MIN_EDGE,
                        help="Minimum net edge in dollars (default: 0.05 = 5¢)")
    parser.add_argument("--min-days", type=int, default=0,
                        help="Skip markets resolving in fewer than N days (default: 0 = include "
                             "today; the intraday model is built for same-day)")
    parser.add_argument("--min-oi", type=float, default=DEFAULT_MIN_OI,
                        help=f"Minimum open interest to include (default: {DEFAULT_MIN_OI})")
    parser.add_argument("--max-bet", type=float, default=DEFAULT_MAX_BET,
                        help=f"Hard cap per bet in USD (default: ${DEFAULT_MAX_BET:.0f})")
    parser.add_argument("--log", action="store_true",
                        help="Auto-log all signals to CSV without prompting")
    parser.add_argument("--grade", action="store_true",
                        help="Grade past logged trades against actual resolved highs and exit")
    parser.add_argument("--no-discipline", action="store_true",
                        help="Disable discipline filters (deep-longshot + data-error guards); "
                             "research only")
    parser.add_argument("--paper", action="store_true",
                        help="Hands-off daily loop: settle/grade prior trades (with CLV), then "
                             "auto-log today's new disciplined picks (deduped). Run once a day.")
    args = parser.parse_args()

    if args.grade:
        grade_trades()
        return

    today = date.today()
    if args.paper:
        print("Paper loop — step 1/2: settling & grading prior trades…")
        grade_trades()
        print("\nPaper loop — step 2/2: scanning today's picks…")

    print(f"\nKalshi Weather Signal Scanner  ·  {today.isoformat()}")
    print(f"Bankroll: ${args.bankroll:.0f}  |  Min edge: {args.min_edge*100:.0f}¢  |  ¼-Kelly\n")
    print("Fetching markets + NWS forecasts:")

    signals = compute_signals(
        today,
        bankroll=args.bankroll,
        min_edge=args.min_edge,
        min_days=args.min_days,
        min_oi=args.min_oi,
        max_bet=args.max_bet,
        discipline=not args.no_discipline,
    )
    print_signals(signals, bankroll=args.bankroll)

    if signals:
        # --paper auto-logs deduped (idempotent daily run); --log auto-logs all; else prompt.
        log_signals_to_csv(signals, auto_log=args.log or args.paper, dedupe=args.paper)


if __name__ == "__main__":
    main()
