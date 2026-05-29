"""
Daily stock screener — S&P 400 + S&P 600 universe.

Pipeline:
  1. Load universe (~1,000 tickers from S&P MidCap 400 + SmallCap 600)
  2. Batch-download 1 year of daily OHLCV for all tickers
  3. Apply hard technical gates (fast, pure math — eliminates ~80% of universe)
  4. Fetch fundamentals only for survivors (~100-200 stocks, threaded)
  5. Apply fundamental disqualifiers
  6. Detect setup type for each survivor
  7. Score 0-10 based on signal confluence (relative within survivor universe)
  8. Return top 20 ranked candidates, tiered HIGH/MID/LOW

Performance note:
  First run of the day: ~3-5 minutes (downloading 1,000 stocks).
  Results are cached daily — subsequent calls return instantly.
"""

import io
import os
import json
import time
import warnings
import requests
import pandas as pd
import yfinance as yf
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

from stocks_module.signals import (
    compute_signals, detect_setup, score_candidate,
    SCORE_HIGH_TIER, SCORE_MID_TIER, EARNINGS_SOON_DAYS,
)

warnings.filterwarnings('ignore')

BASE_DIR       = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
UNIVERSE_CACHE = os.path.join(BASE_DIR, 'data', 'sp_universe.csv')
RESULTS_CACHE  = os.path.join(BASE_DIR, 'data', 'screener_cache_{date}.json')
UNIVERSE_TTL_DAYS = 7   # re-fetch universe from Wikipedia every 7 days
BATCH_SIZE        = 50   # tickers per yf.download call — smaller batches avoid DNS thread exhaustion

# All sector ETFs fetched alongside SPY for relative strength calculations
BENCHMARK_TICKERS = ['SPY', 'XLK', 'XLF', 'XLE', 'XLY', 'XLB',
                     'XLI', 'XLC', 'XLP', 'XLV', 'XLU', 'XLRE']


# ── Universe ──────────────────────────────────────────────────────────────────

def get_universe() -> list[str]:
    """
    Return the full screening universe: S&P MidCap 400 + S&P SmallCap 600.
    Fetches from Wikipedia on first call, then caches for UNIVERSE_TTL_DAYS days.
    """
    if os.path.exists(UNIVERSE_CACHE):
        age_days = (datetime.now() - datetime.fromtimestamp(os.path.getmtime(UNIVERSE_CACHE))).days
        if age_days < UNIVERSE_TTL_DAYS:
            df = pd.read_csv(UNIVERSE_CACHE)
            return df['symbol'].dropna().tolist()

    print("Fetching S&P 400 + S&P 600 universe from Wikipedia...")
    tickers = set()
    headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'}

    for url, label in [
        ('https://en.wikipedia.org/wiki/List_of_S%26P_400_companies', 'S&P 400'),
        ('https://en.wikipedia.org/wiki/List_of_S%26P_600_companies', 'S&P 600'),
    ]:
        try:
            resp = requests.get(url, headers=headers, timeout=15)
            resp.raise_for_status()
            tables = pd.read_html(io.StringIO(resp.text))
            for tbl in tables:
                for col in ['Ticker symbol', 'Symbol', 'Ticker']:
                    if col in tbl.columns:
                        raw = tbl[col].dropna().astype(str).str.strip()
                        # yfinance uses '-' for class shares (e.g. BRK-B not BRK.B)
                        symbols = raw.str.replace(r'\.', '-', regex=True).tolist()
                        tickers.update(symbols)
                        print(f"  {label}: {len(symbols)} symbols loaded")
                        break
        except Exception as e:
            print(f"  Warning: could not fetch {label}: {e}")

    if not tickers:
        raise RuntimeError(
            "Failed to fetch universe from Wikipedia. Check internet connection.\n"
            "If the problem persists, manually populate data/sp_universe.csv with a 'symbol' column."
        )

    tickers = sorted(t for t in tickers if t and len(t) <= 6 and t.isalpha() or '-' in t)
    pd.DataFrame({'symbol': tickers}).to_csv(UNIVERSE_CACHE, index=False)
    print(f"Universe cached: {len(tickers)} symbols total\n")
    return tickers


# ── Price data ────────────────────────────────────────────────────────────────

def _download_batch(tickers: list[str], period: str = '1y') -> dict[str, pd.DataFrame]:
    """Download OHLCV for a batch of tickers. Returns {symbol: DataFrame}."""
    result = {}
    if not tickers:
        return result
    try:
        batch_str = ' '.join(tickers)
        raw = yf.download(
            batch_str,
            period=period,
            interval='1d',
            group_by='ticker',
            auto_adjust=True,
            progress=False,
            threads=False,
        )
        if raw.empty:
            return result

        for sym in tickers:
            try:
                # Single-ticker download returns flat columns; multi returns MultiIndex
                if len(tickers) == 1:
                    df = raw.copy()
                else:
                    if sym not in raw.columns.get_level_values(0):
                        continue
                    df = raw[sym].copy()

                df = df.dropna(how='all')
                if df.empty or 'Close' not in df.columns:
                    continue
                if df.index.tz is not None:
                    df.index = df.index.tz_localize(None)
                result[sym] = df
            except Exception:
                pass
    except Exception as e:
        print(f"  Batch download error: {e}")
    return result


def download_universe_data(tickers: list[str]) -> dict[str, pd.DataFrame]:
    """Download all tickers in batches. Prints progress."""
    all_data: dict[str, pd.DataFrame] = {}
    n_batches = (len(tickers) + BATCH_SIZE - 1) // BATCH_SIZE
    for i in range(0, len(tickers), BATCH_SIZE):
        batch = tickers[i:i + BATCH_SIZE]
        batch_num = i // BATCH_SIZE + 1
        print(f"  Downloading batch {batch_num}/{n_batches} ({len(batch)} symbols)...", end='\r')
        result = _download_batch(batch)
        all_data.update(result)
        time.sleep(1.0)  # give DNS resolver breathing room between batches
    print(f"\n  Downloaded {len(all_data)} symbols with sufficient data.")
    return all_data


def download_benchmarks() -> dict[str, pd.Series]:
    """Download SPY + sector ETF close prices. Returns {symbol: close_series}."""
    result = {}
    try:
        raw = yf.download(
            ' '.join(BENCHMARK_TICKERS),
            period='1y',
            group_by='ticker',
            auto_adjust=True,
            progress=False,
        )
        for sym in BENCHMARK_TICKERS:
            try:
                if len(BENCHMARK_TICKERS) == 1:
                    close = raw['Close'].dropna()
                else:
                    close = raw[sym]['Close'].dropna()
                if close.index.tz is not None:
                    close.index = close.index.tz_localize(None)
                result[sym] = close
            except Exception:
                pass
    except Exception as e:
        print(f"  Warning: benchmark download failed: {e}")
    return result


# ── Fundamentals ──────────────────────────────────────────────────────────────

def _fetch_one_fundamental(symbol: str) -> dict:
    """Fetch fundamental data for a single symbol via yfinance."""
    default = {
        'earnings_soon': False,
        'eps_positive':  True,
        'revenue_ok':    True,
        'short_float':   0.0,
        'short_flag':    False,
        'short_disqualify': False,
        'sector':        'Unknown',
        'name':          symbol,
        'market_cap':    0,
    }
    try:
        t    = yf.Ticker(symbol)
        info = t.info or {}

        # Earnings date — is there one within EARNINGS_SOON_DAYS?
        earnings_soon = False
        try:
            cal = t.calendar
            if cal is not None:
                dates = None
                if isinstance(cal, pd.DataFrame) and not cal.empty:
                    if 'Earnings Date' in cal.index:
                        dates = cal.loc['Earnings Date']
                elif isinstance(cal, dict):
                    dates = cal.get('Earnings Date')
                if dates is not None:
                    vals = [dates] if not hasattr(dates, '__iter__') else dates
                    for d in vals:
                        try:
                            delta = (pd.Timestamp(d) - pd.Timestamp.now()).days
                            if 0 <= delta <= EARNINGS_SOON_DAYS:
                                earnings_soon = True
                                break
                        except Exception:
                            pass
        except Exception:
            pass

        trailing_eps  = info.get('trailingEps')
        rev_growth    = info.get('revenueGrowth')
        short_float   = float(info.get('shortPercentOfFloat') or 0)
        sector        = info.get('sector', 'Unknown') or 'Unknown'
        name          = info.get('longName') or info.get('shortName', symbol) or symbol
        market_cap    = int(info.get('marketCap') or 0)

        return {
            'earnings_soon':     earnings_soon,
            'eps_positive':      trailing_eps is None or trailing_eps > -1.0,
            'revenue_ok':        rev_growth is None or rev_growth > -0.15,
            'short_float':       short_float,
            'short_flag':        short_float > 0.15,
            'short_disqualify':  short_float > 0.25,
            'sector':            sector,
            'name':              name,
            'market_cap':        market_cap,
        }
    except Exception:
        return default


def fetch_fundamentals(symbols: list[str], max_workers: int = 15) -> dict[str, dict]:
    """
    Fetch fundamentals for all survivors in parallel (threaded).
    Only runs on symbols that passed technical gates — typically 100-200 stocks.
    """
    results = {}
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_fetch_one_fundamental, sym): sym for sym in symbols}
        done = 0
        for future in as_completed(futures):
            sym = futures[future]
            try:
                results[sym] = future.result()
            except Exception:
                results[sym] = _fetch_one_fundamental.__wrapped__(sym) if hasattr(_fetch_one_fundamental, '__wrapped__') else {}
            done += 1
            print(f"  Fetching fundamentals: {done}/{len(symbols)}...", end='\r')
    print(f"\n  Fundamentals fetched for {len(results)} symbols.")
    return results


# ── Main screener ─────────────────────────────────────────────────────────────

def run_screener(top_n: int = 20, force_refresh: bool = False) -> list[dict]:
    """
    Run the full screening pipeline. Returns top_n ranked candidates.

    Results are cached for the calendar day — call repeatedly without re-downloading.
    Use force_refresh=True to force a fresh run even if today's cache exists.

    Each returned dict contains:
      symbol, name, sector, price, score, tier_label, setup_name, setup_tier,
      ret_1d, ret_5d, ret_1m, ret_3m, ret_6m, rsi, rvol, vol_trend,
      prox_52w, high_52w, atr, atr_pct, rs_spy, mom_12_1,
      earnings_soon, short_float, short_flag, market_cap
    """
    today_str   = datetime.now().strftime('%Y-%m-%d')
    cache_path  = RESULTS_CACHE.format(date=today_str)

    # Return cached results if available
    if not force_refresh and os.path.exists(cache_path):
        with open(cache_path) as f:
            cached = json.load(f)
        print(f"Loaded today's screener results from cache ({len(cached)} candidates).")
        return cached[:top_n]

    t_start = time.time()

    # ── Step 1: Universe ──────────────────────────────────────────────────────
    tickers = get_universe()

    # ── Step 2: Download price data ───────────────────────────────────────────
    print(f"\nDownloading benchmarks (SPY + sector ETFs)...")
    benchmarks = download_benchmarks()
    spy_close   = benchmarks.get('SPY')

    print(f"\nDownloading universe price data (~{len(tickers)} stocks)...")
    print("This takes 3-5 minutes on first run. Results are cached for the rest of the day.")
    price_data = download_universe_data(tickers)

    # ── Step 3: Technical gates + signal computation ──────────────────────────
    print("\nComputing technical signals...")
    tech_survivors = []
    for sym, df in price_data.items():
        sig = compute_signals(df, spy_close=spy_close)
        if sig is not None:
            sig['symbol'] = sym
            tech_survivors.append(sig)

    print(f"  {len(tech_survivors)} symbols passed all technical gates "
          f"({len(price_data) - len(tech_survivors)} eliminated).")

    if not tech_survivors:
        print("No candidates passed technical gates.")
        return []

    # ── Step 4: Fundamentals (survivors only) ─────────────────────────────────
    print(f"\nFetching fundamental data for {len(tech_survivors)} survivors...")
    survivor_symbols = [s['symbol'] for s in tech_survivors]
    fundamentals     = fetch_fundamentals(survivor_symbols)

    # ── Step 5: Fundamental gates ─────────────────────────────────────────────
    qualified = []
    for sig in tech_survivors:
        sym  = sig['symbol']
        fund = fundamentals.get(sym, {})
        # Disqualify structural losers and heavily-shorted names
        if not fund.get('eps_positive', True):
            continue
        if not fund.get('revenue_ok', True):
            continue
        if fund.get('short_disqualify', False):
            continue
        sig.update({
            'earnings_soon': fund.get('earnings_soon', False),
            'short_float':   fund.get('short_float', 0.0),
            'short_flag':    fund.get('short_flag', False),
            'sector':        fund.get('sector', 'Unknown'),
            'name':          fund.get('name', sym),
            'market_cap':    fund.get('market_cap', 0),
        })
        qualified.append(sig)

    print(f"  {len(qualified)} candidates survived fundamental gates.")

    if not qualified:
        print("No candidates passed fundamental gates.")
        return []

    # ── Step 6: Setup type classification ─────────────────────────────────────
    for sig in qualified:
        setup_name, tier = detect_setup(sig)
        sig['setup_name'] = setup_name
        sig['setup_tier'] = tier

    # ── Step 7: Score (relative within survivor universe) ─────────────────────
    df_qual = pd.DataFrame(qualified)
    df_qual['score'] = df_qual.apply(
        lambda row: score_candidate(row, df_qual), axis=1
    )

    # ── Step 8: Tier labels and final ranking ──────────────────────────────────
    def _tier_label(s: int) -> str:
        if s >= SCORE_HIGH_TIER: return 'HIGH'
        if s >= SCORE_MID_TIER:  return 'MID'
        return 'LOW'

    df_qual['tier_label']  = df_qual['score'].apply(_tier_label)
    df_qual['tier_order']  = df_qual['tier_label'].map({'HIGH': 0, 'MID': 1, 'LOW': 2})
    df_qual = df_qual.sort_values(
        ['tier_order', 'score', 'rs_spy'],
        ascending=[True, False, False]
    ).drop(columns=['tier_order']).head(top_n).reset_index(drop=True)
    df_qual['rank'] = df_qual.index + 1

    elapsed = time.time() - t_start
    print(f"\nScreener complete in {elapsed/60:.1f} minutes. "
          f"{len(df_qual)} candidates ranked.")

    # Cache results
    # Convert NaN to None for JSON serialization
    records = json.loads(df_qual.to_json(orient='records'))
    with open(cache_path, 'w') as f:
        json.dump(records, f, indent=2, default=str)

    return records
