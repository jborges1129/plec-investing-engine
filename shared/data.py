import os
import time
import requests
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
from dotenv import load_dotenv

load_dotenv()

ETF_UNIVERSE = [
    # Broad market
    'SPY', 'QQQ', 'IWM', 'DIA',
    # All 11 S&P 500 sectors
    'XLK',   # Technology
    'XLF',   # Financials
    'XLE',   # Energy
    'XLY',   # Consumer Discretionary
    'XLB',   # Materials
    'XLI',   # Industrials
    'XLC',   # Communication Services
    'XLP',   # Consumer Staples
    'XLV',   # Healthcare
    'XLU',   # Utilities
    'XLRE',  # Real Estate
    # Defensive / alternatives
    'TLT',   # Long-term treasuries
    'GLD',   # Gold
    'SLV',   # Silver
    # International
    'EFA',   # Developed markets
    'EEM',   # Emerging markets
]

ETF_TOP_HOLDINGS = {
    'SPY':  [('Apple', 7.2), ('Microsoft', 6.4), ('NVIDIA', 5.5), ('Amazon', 3.7), ('Meta', 2.5)],
    'QQQ':  [('Microsoft', 9.0), ('Apple', 8.5), ('NVIDIA', 8.3), ('Amazon', 5.0), ('Broadcom', 4.8)],
    'IWM':  [],  # ~2,000 small-cap holdings; no single position above 0.5%
    'DIA':  [('UnitedHealth', 9.8), ('Goldman Sachs', 8.2), ('Microsoft', 5.3), ('Home Depot', 4.9)],
    'XLK':  [('Apple', 20.1), ('Microsoft', 18.2), ('NVIDIA', 9.4), ('Broadcom', 4.9), ('AMD', 3.2)],
    'XLF':  [('Berkshire Hathaway', 13.1), ('JPMorgan Chase', 10.2), ('Visa', 4.9), ('Mastercard', 4.5)],
    'XLE':  [('ExxonMobil', 22.3), ('Chevron', 17.1), ('ConocoPhillips', 8.2), ('EOG Resources', 4.5)],
    'XLY':  [('Amazon', 24.1), ('Tesla', 13.8), ('Home Depot', 5.4), ("McDonald's", 4.1)],
    'XLB':  [('Linde', 17.2), ('Sherwin-Williams', 8.1), ('Air Products', 6.3), ('Freeport-McMoRan', 5.9)],
    'XLI':  [('GE Aerospace', 5.2), ('Caterpillar', 4.9), ('Honeywell', 4.3), ('Uber', 4.1)],
    'XLC':  [('Meta Platforms', 23.1), ('Alphabet A', 14.3), ('Alphabet C', 11.2), ('Netflix', 5.1)],
    'XLP':  [('Procter & Gamble', 15.1), ('Costco', 12.8), ('Coca-Cola', 9.7), ('PepsiCo', 8.5)],
    'XLV':  [('Eli Lilly', 12.1), ('Johnson & Johnson', 8.9), ('UnitedHealth', 8.5), ('AbbVie', 5.2)],
    'XLU':  [('NextEra Energy', 14.1), ('Southern Company', 5.3), ('Duke Energy', 4.9)],
    'XLRE': [('Prologis', 10.2), ('American Tower', 9.1), ('Equinix', 7.3), ('Welltower', 5.8)],
    'TLT':  [],  # US Treasury bonds, 20+ year maturity; moves inversely with interest rates
    'GLD':  [],  # Physical gold; ~0.095 troy oz per share
    'SLV':  [],  # Physical silver
    'EFA':  [('Nestlé', 2.1), ('ASML', 1.9), ('LVMH', 1.8), ('Novo Nordisk', 1.7)],
    'EEM':  [('Samsung Electronics', 3.8), ('Taiwan Semi', 3.5), ('Tencent', 3.3), ('Alibaba', 2.1)],
}

ETF_NAMES = {
    'SPY': 'S&P 500', 'QQQ': 'Nasdaq 100', 'IWM': 'Russell 2000 Small Cap', 'DIA': 'Dow Jones',
    'XLK': 'Technology', 'XLF': 'Financials', 'XLE': 'Energy',
    'XLY': 'Consumer Discretionary', 'XLB': 'Materials', 'XLI': 'Industrials',
    'XLC': 'Communication Services', 'XLP': 'Consumer Staples',
    'XLV': 'Healthcare', 'XLU': 'Utilities', 'XLRE': 'Real Estate',
    'TLT': 'Long-Term Treasuries', 'GLD': 'Gold', 'SLV': 'Silver',
    'EFA': 'Intl Developed Markets', 'EEM': 'Emerging Markets',
}

BULL_ETFS = {'SPY', 'QQQ', 'IWM', 'DIA', 'XLK', 'XLF', 'XLE', 'XLY', 'XLB', 'XLI', 'XLC'}
BEAR_ETFS = {'TLT', 'GLD', 'SLV', 'XLV', 'XLP', 'XLU', 'XLRE'}
NEUTRAL_ETFS = {'EFA', 'EEM'}

# Fetch extra buffer so 6-month lookback (126 days) is always covered
_MOMENTUM_PERIOD = '8mo'
_MIN_BARS_FOR_6M = 126


def _yf_fetch(symbol: str, period: str, interval: str = '1d') -> pd.DataFrame:
    """
    yfinance fetch with exponential-backoff retry (3 attempts: 0s, 2s, 4s).
    Used for VIX and any path that cannot use the Alpaca data API.
    """
    last_exc: Exception = RuntimeError("unreachable")
    for attempt in range(3):
        try:
            df = yf.Ticker(symbol).history(period=period, interval=interval)
            if not df.empty:
                return df
            raise ValueError(f"Empty response for {symbol}")
        except Exception as exc:
            last_exc = exc
            if attempt < 2:
                time.sleep(2 ** attempt)   # 0 s, 2 s, 4 s
    raise last_exc


def _fetch_alpaca_batch(symbols: list[str]) -> dict[str, pd.DataFrame]:
    """
    Fetch ~1 year of daily OHLCV for all symbols in a single Alpaca API call.
    Returns {symbol: DataFrame} with capitalized column names (Open/High/Low/Close/Volume)
    to match the yfinance format expected by all downstream computations.
    """
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame
    from shared.alpaca import get_data_client

    client  = get_data_client()
    request = StockBarsRequest(
        symbol_or_symbols=symbols,
        timeframe=TimeFrame.Day,
        start=datetime.now() - timedelta(days=370),
        adjustment='all',   # split- and dividend-adjusted, matches yfinance auto_adjust
    )
    bars     = client.get_stock_bars(request)
    full_df  = bars.df   # MultiIndex: (symbol, timestamp), columns lowercase

    result: dict[str, pd.DataFrame] = {}
    for sym in symbols:
        try:
            df = full_df.loc[sym].copy()
            df = df.rename(columns={
                'open': 'Open', 'high': 'High', 'low': 'Low',
                'close': 'Close', 'volume': 'Volume',
            })
            if df.index.tz is not None:
                df.index = df.index.tz_convert('UTC').tz_localize(None)
            result[sym] = df
        except KeyError:
            pass   # symbol absent from response — caller falls back to yfinance
    return result


def get_price_history(symbol: str, period: str = '6mo') -> pd.DataFrame:
    """yfinance fetch with retry. Used for dashboard charts and VIX in regime detection."""
    return _yf_fetch(symbol, period=period)


def get_hourly_history(symbol: str, period: str = '5d') -> pd.DataFrame:
    df = _yf_fetch(symbol, period=period, interval='1h')
    if df.empty:
        raise ValueError(f"No hourly data for {symbol}")
    return df


def _fetch_full_history(symbol: str) -> pd.DataFrame:
    """yfinance fallback for a single symbol — used when Alpaca batch misses a symbol."""
    return _yf_fetch(symbol, period=_MOMENTUM_PERIOD)


def get_current_price(symbol: str) -> float:
    df = get_price_history(symbol, period='5d')
    return float(df['Close'].iloc[-1])


def get_momentum(df: pd.DataFrame) -> dict:
    """Compute 1m/3m/6m momentum from a pre-fetched history DataFrame."""
    close = df['Close']
    if len(close) < _MIN_BARS_FOR_6M:
        raise ValueError(f"Insufficient history: {len(close)} bars (need {_MIN_BARS_FOR_6M})")
    current = float(close.iloc[-1])

    def _ret(days):
        if len(close) < days:
            raise ValueError(f"Need {days} bars for momentum, have {len(close)}")
        start = float(close.iloc[-days])
        return (current - start) / start

    return {
        'return_1d': _ret(2),   # yesterday close → today
        'return_5d': _ret(6),   # 5 trading days ago → today (~1 week)
        'return_1m': _ret(21),
        'return_3m': _ret(63),
        'return_6m': _ret(126),
    }


def get_rsi(df: pd.DataFrame, period: int = 14) -> float:
    """Compute RSI from a pre-fetched history DataFrame. Returns 100 on all-gain periods."""
    close = df['Close']
    if len(close) < period * 2:
        raise ValueError(f"Insufficient data for RSI")
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    # Avoid division by zero: zero loss = RSI of 100
    loss_safe = loss.replace(0, float('nan'))
    rs = gain / loss_safe
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.fillna(100)
    result = float(rsi.iloc[-1])
    if pd.isna(result):
        raise ValueError("RSI calculation produced NaN")
    return result


def get_atr(df: pd.DataFrame, period: int = 14) -> float:
    """Compute ATR from a pre-fetched history DataFrame."""
    if len(df) < period * 2:
        raise ValueError(f"Insufficient data for ATR")
    high = df['High']
    low = df['Low']
    close = df['Close']
    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low - close.shift(1)).abs(),
    ], axis=1).max(axis=1)
    result = float(tr.rolling(period).mean().iloc[-1])
    if pd.isna(result) or result == 0:
        raise ValueError("ATR is zero or NaN — cannot size position")
    return result


def get_volume_trend(df: pd.DataFrame) -> str:
    """Compute volume trend label from a pre-fetched history DataFrame."""
    vol = df['Volume']
    if len(vol) < 20:
        return 'neutral'
    recent = vol.iloc[-5:].mean()
    baseline = vol.iloc[-20:-5].mean()
    if baseline == 0:
        return 'neutral'
    ratio = recent / baseline
    if ratio > 1.15:
        return 'rising'
    elif ratio < 0.85:
        return 'falling'
    return 'neutral'


def get_volume_ratio(df: pd.DataFrame) -> float:
    """5-day average volume divided by 20-day baseline average."""
    vol = df['Volume']
    if len(vol) < 20:
        return 1.0
    recent   = float(vol.iloc[-5:].mean())
    baseline = float(vol.iloc[-20:-5].mean())
    return recent / baseline if baseline > 0 else 1.0


def get_macd(df: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9) -> dict:
    """MACD line, signal line, and histogram (standard 12/26/9 parameters)."""
    close = df['Close']
    ema_fast   = close.ewm(span=fast,   adjust=False).mean()
    ema_slow   = close.ewm(span=slow,   adjust=False).mean()
    macd_line  = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    return {
        'macd':        float(macd_line.iloc[-1]),
        'macd_signal': float(signal_line.iloc[-1]),
        'macd_hist':   float((macd_line - signal_line).iloc[-1]),
    }


def get_ma_distances(df: pd.DataFrame) -> dict:
    """Price position relative to 50-day and 200-day moving averages."""
    close = df['Close']
    price = float(close.iloc[-1])
    n     = len(close)
    ma50  = float(close.rolling(50).mean().iloc[-1])  if n >= 50  else price
    ma200 = float(close.rolling(200).mean().iloc[-1]) if n >= 200 else price
    return {
        'ma50':            ma50,
        'ma200':           ma200,
        'pct_above_ma50':  (price - ma50)  / ma50  if ma50  > 0 else 0.0,
        'pct_above_ma200': (price - ma200) / ma200 if ma200 > 0 else 0.0,
    }


def get_realized_vol(df: pd.DataFrame, period: int = 63) -> float:
    """Annualized realized volatility over the trailing `period` trading days (~3 months)."""
    close = df['Close']
    rets  = close.pct_change().dropna()
    tail  = rets.iloc[-min(period, len(rets)):]
    return float(tail.std() * (252 ** 0.5))


def get_max_drawdown(df: pd.DataFrame, period: int = 63) -> float:
    """Max drawdown (negative fraction) over the trailing `period` bars."""
    close    = df['Close'].iloc[-period:]
    roll_max = close.cummax()
    dd       = (close - roll_max) / roll_max
    return float(dd.min())


def get_market_regime() -> dict:
    # SPY via Alpaca (official API, single call, no rate-limit risk)
    spy_data = _fetch_alpaca_batch(['SPY'])
    if spy_data:
        spy = spy_data['SPY']
    else:
        # Alpaca failed — fall back to yfinance with retry
        spy = _yf_fetch('SPY', period='1y')

    close   = spy['Close']
    current = float(close.iloc[-1])
    ma200   = float(close.rolling(200).mean().iloc[-1])
    ma50    = float(close.rolling(50).mean().iloc[-1])

    # VIX: yfinance only (Alpaca does not carry index data).
    # If the fetch fails after retries, default to 20 (neutral) so the scheduler
    # continues running rather than crashing the whole cycle.
    vix_level = 20.0
    try:
        vix_hist  = _yf_fetch('^VIX', period='5d')
        vix_level = float(vix_hist['Close'].iloc[-1])
    except Exception as e:
        print(f"  VIX fetch failed — defaulting to 20.0 (neutral): {e}")

    # Both conditions must be true simultaneously to avoid false regime flips
    if vix_level < 20 and current > ma200:
        regime = 'bull'
    elif vix_level > 25 and current < ma200:
        regime = 'bear'
    else:
        regime = 'neutral'

    return {
        'regime': regime,
        'vix': vix_level,
        'spy_price': current,
        'ma50': ma50,
        'ma200': ma200,
        'spy_above_ma200': current > ma200,
    }


# News sentiment is cached to avoid exceeding Alpha Vantage free tier (25 calls/day)
_news_cache: dict = {}


def get_news_sentiment(symbols: list[str], limit: int = 3) -> dict[str, list]:
    """
    Fetch news for multiple symbols in one API call (batch request).
    Cached for the session to avoid hitting the 25 calls/day free tier limit.
    """
    api_key = os.getenv('ALPHA_VANTAGE_API_KEY')
    if not api_key:
        return {}

    cache_key = ','.join(sorted(symbols))
    if cache_key in _news_cache:
        return _news_cache[cache_key]

    tickers_param = ','.join(symbols)
    url = (
        f"https://www.alphavantage.co/query"
        f"?function=NEWS_SENTIMENT"
        f"&tickers={tickers_param}"
        f"&limit={limit * len(symbols)}"
        f"&apikey={api_key}"
    )
    try:
        resp = requests.get(url, timeout=10)
        data = resp.json()
        articles = data.get('feed', [])
        results: dict[str, list] = {s: [] for s in symbols}
        for a in articles:
            for ts in a.get('ticker_sentiment', []):
                sym = ts.get('ticker', '')
                if sym in results:
                    score = float(ts.get('ticker_sentiment_score', 0))
                    if abs(score) > 0.15:
                        results[sym].append({
                            'title': a.get('title', ''),
                            'summary': a.get('summary', ''),
                            'url': a.get('url', ''),
                            'sentiment_score': score,
                            'sentiment_label': ts.get('ticker_sentiment_label', 'neutral'),
                            'source': a.get('source', ''),
                            'published': a.get('time_published', ''),
                        })
        _news_cache[cache_key] = results
        return results
    except Exception as e:
        print(f"News fetch failed: {e}")
        return {}


def get_etf_news_yf(symbol: str, limit: int = 4) -> list[dict]:
    """Fetch recent news headlines via yfinance (no API key required)."""
    try:
        raw = yf.Ticker(symbol).news or []
        results = []
        for item in raw:
            if not isinstance(item, dict):
                continue
            # Handle both old and new yfinance news formats
            content = item.get('content', item)
            title = content.get('title', '')
            publisher = (
                content.get('provider', {}).get('displayName', '')
                or item.get('publisher', '')
            )
            if title:
                results.append({'title': title, 'publisher': publisher})
            if len(results) >= limit:
                break
        return results
    except Exception:
        return []


def get_prior_day_summary() -> dict:
    """Returns the last two trading days for the briefing narrative."""
    results = {}
    for symbol in ['SPY', 'QQQ', 'TLT', 'GLD', '^VIX']:
        try:
            df = _yf_fetch(symbol, period='5d')
            if len(df) >= 2:
                prev_close = float(df['Close'].iloc[-2])
                last_close = float(df['Close'].iloc[-1])
                last_date  = df.index[-1].date()
                results[symbol] = {
                    'prev_close': prev_close,
                    'last_close': last_close,
                    'change_pct': (last_close - prev_close) / prev_close,
                    'date': last_date,
                }
        except Exception:
            pass
    return results


def _compute_symbol_row(symbol: str, df: pd.DataFrame) -> dict:
    """Compute all 12 factor inputs from a single price history DataFrame."""
    price        = float(df['Close'].iloc[-1])
    mom          = get_momentum(df)
    atr          = get_atr(df)
    rsi          = get_rsi(df)
    vol_trend    = get_volume_trend(df)
    vol_ratio    = get_volume_ratio(df)
    macd_data    = get_macd(df)
    ma_data      = get_ma_distances(df)
    realized_vol = get_realized_vol(df)
    max_dd       = get_max_drawdown(df)
    return {
        'symbol':          symbol,
        'price':           price,
        'atr':             atr,
        'rsi':             rsi,
        'volume_trend':    vol_trend,
        'volume_ratio':    vol_ratio,
        'realized_vol_3m': realized_vol,
        'max_drawdown_3m': max_dd,
        **mom,
        **macd_data,
        **ma_data,
    }


def rank_etfs_by_momentum(universe: list = None) -> pd.DataFrame:
    """
    Score all ETFs in the universe across 12 factors.

    Data strategy:
      Primary  — Alpaca StockHistoricalDataClient: single batched API call for all symbols,
                 official SLA, same credentials as the trading client.
      Fallback — yfinance with exponential-backoff retry, used per-symbol if Alpaca
                 omits a symbol or the batch call itself fails.

    Factors returned per row:
      Momentum  — return_1d/5d/1m/3m/6m
      Technical — rsi, macd/macd_signal/macd_hist, pct_above_ma50/ma200
      Risk      — atr, realized_vol_3m, max_drawdown_3m
      Volume    — volume_trend (label), volume_ratio (float)
      Reference — spy_return_3m (for alpha calculation in strategy layer)
    """
    if universe is None:
        universe = ETF_UNIVERSE

    # ── Primary: Alpaca batch (one API call for the entire universe) ──────────
    alpaca_data: dict[str, pd.DataFrame] = {}
    try:
        alpaca_data = _fetch_alpaca_batch(universe)
        missing = [s for s in universe if s not in alpaca_data]
        if missing:
            print(f"  Alpaca missing {len(missing)} symbols, falling back to yfinance: {missing}")
    except Exception as e:
        print(f"  Alpaca batch fetch failed — falling back to yfinance for all symbols: {e}")

    # ── Process each symbol ───────────────────────────────────────────────────
    results = []
    for symbol in universe:
        try:
            if symbol in alpaca_data and len(alpaca_data[symbol]) >= _MIN_BARS_FOR_6M:
                df = alpaca_data[symbol]
            else:
                df = _fetch_full_history(symbol)   # yfinance fallback with retry

            results.append(_compute_symbol_row(symbol, df))
        except Exception as e:
            print(f"  Warning: skipping {symbol} — {e}")

    out = pd.DataFrame(results)
    if out.empty:
        return out

    # SPY benchmark return for cross-sectional alpha calculation
    spy_row = out[out['symbol'] == 'SPY']
    if not spy_row.empty:
        spy_return_3m = float(spy_row['return_3m'].iloc[0])
    else:
        # SPY not in universe (bear regime) — fetch it separately via Alpaca
        try:
            spy_batch     = _fetch_alpaca_batch(['SPY'])
            spy_return_3m = float(get_momentum(spy_batch['SPY'])['return_3m'])
        except Exception:
            try:
                spy_return_3m = float(get_momentum(_fetch_full_history('SPY'))['return_3m'])
            except Exception:
                spy_return_3m = 0.0

    out['spy_return_3m'] = spy_return_3m
    return out
