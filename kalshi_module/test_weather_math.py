#!/usr/bin/env python3
"""
Fast, dependency-light tests for the weather model's core math and discipline rules.
Run directly (no pytest needed):  python kalshi_module/test_weather_math.py
These lock the invariants that must hold before any real-money use.
"""

from datetime import datetime, timedelta, timezone

import weather_signal as ws


def test_taker_fee_symmetric_and_peaks_midbook():
    assert abs(ws.kalshi_taker_fee(0.5) - 0.0175) < 1e-9
    assert abs(ws.kalshi_taker_fee(0.2) - ws.kalshi_taker_fee(0.8)) < 1e-9  # C(1-C) symmetry
    assert ws.kalshi_taker_fee(0.5) > ws.kalshi_taker_fee(0.1)             # max at the middle


def test_kelly_is_zero_when_no_edge():
    assert ws.kelly_fraction(0.3, 0.5) == 0.0      # model below price → don't bet
    assert abs(ws.kelly_fraction(0.7, 0.5) - 0.4) < 1e-9
    assert ws.kelly_fraction(0.99, 1.0) == 0.0     # degenerate price guarded


def test_greater_probability_rises_with_forecast():
    ps = [ws.model_prob_yes(f, "greater", 85, None, 1.0) for f in (80, 85, 90)]
    assert ps[0] < ps[1] < ps[2]
    assert 0.0 <= ps[0] and ps[2] <= 1.0


def test_observation_lock_only_for_cleared_greater():
    assert ws.yes_observation_locked("greater", 83, None, 83.5) is True   # rounds to ≥84
    assert ws.yes_observation_locked("greater", 83, None, 83.0) is False
    assert ws.yes_observation_locked("less", None, 90, 99) is False        # can't lock from above
    assert ws.yes_observation_locked("between", 80, 85, 99) is False
    assert ws.yes_observation_locked("greater", 83, None, None) is False   # no obs


def _series(start_temp, peak_temp, now_hour=13):
    """Build (obs, hourly_forecast, now) for a synthetic day climbing to peak_temp."""
    base = datetime(2026, 7, 1, tzinfo=timezone.utc)
    obs = [(base + timedelta(hours=h), start_temp) for h in range(6, now_hour + 1)]
    hourly = [(base + timedelta(hours=h), peak_temp if h in (15, 16) else start_temp)
              for h in range(now_hour, 21)]
    now = base + timedelta(hours=now_hour, minutes=30)
    return obs, hourly, now


def test_sigma_widens_when_big_rise_still_unobserved():
    # obs stuck at 81, forecast peak 91 → big unobserved rise must NOT be near-certain.
    _, _, now = _series(81, 91)
    obs, hourly, now = _series(81, 91)
    cdf, dbg = ws.intraday_max_cdf(obs, hourly, now)
    assert dbg["projected_rise"] >= 5
    assert dbg["std"] > ws.STD_AT_PEAK + 0.5          # floored up by the rise
    assert dbg["confidence"] == "speculative"          # not falsely "high"


def test_locked_confidence_when_obs_already_at_peak():
    obs, hourly, now = _series(90, 90)                 # obs already at the projected peak
    cdf, dbg = ws.intraday_max_cdf(obs, hourly, now)
    assert dbg["confidence"] in ("locked", "high")
    assert dbg["projected_rise"] <= 1


def test_daily_high_anchor_caps_warm_hourly_grid():
    obs, hourly, now = _series(81, 95)                 # hourly grid says 95
    _, dbg_raw = ws.intraday_max_cdf(obs, hourly, now)
    _, dbg_anc = ws.intraday_max_cdf(obs, hourly, now, daily_high=90.0)
    assert dbg_anc["remaining_peak"] < dbg_raw["remaining_peak"]
    assert dbg_anc["anchored_to_daily_high"] is True


def test_between_probability_in_unit_interval():
    for fc in (70, 85, 100):
        p = ws.model_prob_yes(fc, "between", 84, 86, 1.0)
        assert 0.0 <= p <= 1.0


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"  ✓ {t.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"  ✗ {t.__name__}  — {e or 'assertion failed'}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    raise SystemExit(1 if failed else 0)
