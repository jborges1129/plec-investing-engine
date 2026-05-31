"""
Historical data fetching and slicing for the backtester.
All functions are pure with respect to time — no live data fetches.
"""

import pandas as pd
import yfinance as yf
from typing import Optional


def fetch_all_history(symbols: list[str], start: str, end: str) -> dict[str, pd.DataFrame]:
    """
    Download full adjusted OHLCV history for all symbols plus VIX via yfinance.
    Returns {symbol: DataFrame} with columns [Open, High, Low, Close, Volume].
    auto_adjust=True bakes in splits and dividends — use Close for return math.
    """
    all_syms = list(dict.fromkeys(symbols + ['^VIX', 'SPY']))
    print(f"Downloading {len(all_syms)} symbols from {start} to {end} ...")
    raw = yf.download(all_syms, start=start, end=end, auto_adjust=True, progress=True)

    result: dict[str, pd.DataFrame] = {}

    if isinstance(raw.columns, pd.MultiIndex):
        # Determine which MultiIndex level holds the ticker symbols vs price types
        price_types = {'Open', 'High', 'Low', 'Close', 'Volume', 'Adj Close'}
        lv0_sample  = set(raw.columns.get_level_values(0).unique()[:6])
        ticker_level = 1 if lv0_sample <= price_types or lv0_sample & price_types else 0

        tickers_present = raw.columns.get_level_values(ticker_level).unique()
        for sym in tickers_present:
            try:
                df = raw.xs(sym, axis=1, level=ticker_level).dropna(how='all')
                if df.empty:
                    continue
                if df.index.tz is not None:
                    df.index = df.index.tz_localize(None)
                result[sym] = df
            except KeyError:
                pass
    else:
        # Single ticker — shouldn't happen with our list, but handle it
        sym = all_syms[0]
        df  = raw.dropna(how='all')
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)
        result[sym] = df

    missing = [s for s in all_syms if s not in result]
    if missing:
        print(f"  Warning: no data for {missing}")
    print(f"  Downloaded {len(result)} symbols, "
          f"{len(result.get('SPY', pd.DataFrame()))} SPY bars.")
    return result


def get_trading_dates(data: dict[str, pd.DataFrame]) -> pd.DatetimeIndex:
    """All trading dates from the SPY series (proxy for market-open days)."""
    ref = data.get('SPY') or next(iter(data.values()))
    return ref.index.sort_values()


def get_rebalance_dates(
    trading_dates: pd.DatetimeIndex,
    start_date: str,
    end_date: str,
) -> list[pd.Timestamp]:
    """
    First trading day of each calendar month within [start_date, end_date].
    Signal is computed on this date; entry fills at next trading day's open.
    """
    start = pd.Timestamp(start_date)
    end   = pd.Timestamp(end_date)
    in_range = trading_dates[(trading_dates >= start) & (trading_dates <= end)]
    if len(in_range) == 0:
        return []
    frame = pd.DataFrame({'date': in_range})
    frame['ym'] = frame['date'].dt.to_period('M')
    return frame.groupby('ym')['date'].first().tolist()


def get_symbol_slice(
    data: dict[str, pd.DataFrame],
    symbol: str,
    as_of: pd.Timestamp,
) -> pd.DataFrame:
    """Data for symbol up to and including as_of — no future peeking."""
    df = data.get(symbol)
    if df is None or df.empty:
        return pd.DataFrame()
    return df[df.index <= as_of]


def get_next_trading_day(
    trading_dates: pd.DatetimeIndex,
    after: pd.Timestamp,
) -> Optional[pd.Timestamp]:
    """First trading date strictly after `after`."""
    candidates = trading_dates[trading_dates > after]
    return candidates[0] if len(candidates) > 0 else None
