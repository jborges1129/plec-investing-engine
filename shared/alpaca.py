import os
from dotenv import load_dotenv
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import (
    MarketOrderRequest,
    TakeProfitRequest,
    StopLossRequest,
    TrailingStopOrderRequest,
)
from alpaca.trading.enums import OrderSide, TimeInForce, OrderClass

load_dotenv()


def get_client() -> TradingClient:
    return TradingClient(
        api_key=os.getenv('ALPACA_API_KEY'),
        secret_key=os.getenv('ALPACA_SECRET_KEY'),
        paper=True,
    )


def get_account():
    return get_client().get_account()


def get_positions():
    return get_client().get_all_positions()


def get_position(symbol: str):
    try:
        return get_client().get_open_position(symbol)
    except Exception:
        return None


def place_bracket_order(
    symbol: str,
    qty: float,
    entry_price: float,
    stop_loss_pct: float = 0.06,
    take_profit_pct: float = 0.13,
):
    """Fixed stop-loss + take-profit bracket order."""
    client = get_client()
    stop_price = round(entry_price * (1 - stop_loss_pct), 2)
    take_profit_price = round(entry_price * (1 + take_profit_pct), 2)

    order = MarketOrderRequest(
        symbol=symbol,
        qty=qty,
        side=OrderSide.BUY,
        time_in_force=TimeInForce.DAY,
        order_class=OrderClass.BRACKET,
        take_profit=TakeProfitRequest(limit_price=take_profit_price),
        stop_loss=StopLossRequest(stop_price=stop_price),
    )
    result = client.submit_order(order)
    return result, stop_price, take_profit_price


def place_trailing_stop_order(
    symbol: str,
    qty: float,
    trail_percent: float = 5.0,
):
    """Trailing stop order — stop moves up as price rises."""
    client = get_client()
    order = TrailingStopOrderRequest(
        symbol=symbol,
        qty=qty,
        side=OrderSide.BUY,
        time_in_force=TimeInForce.GTC,
        trail_percent=trail_percent,
    )
    result = client.submit_order(order)
    return result


def close_position(symbol: str):
    return get_client().close_position(symbol)


def cancel_all_orders():
    return get_client().cancel_orders()
