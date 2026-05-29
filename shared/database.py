import os
import json
from datetime import datetime
from sqlalchemy import create_engine, Column, Integer, Float, String, DateTime
from sqlalchemy.orm import declarative_base, sessionmaker

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(BASE_DIR, 'data', 'trading.db')

Base = declarative_base()


class Trade(Base):
    __tablename__ = 'trades'
    id = Column(Integer, primary_key=True)
    order_id = Column(String, unique=True)
    symbol = Column(String)
    module = Column(String)       # 'etf' or 'stocks'
    side = Column(String)         # 'buy' or 'sell'
    qty = Column(Float)
    entry_price = Column(Float)
    entry_time = Column(DateTime, default=datetime.utcnow)
    exit_price = Column(Float, nullable=True)
    exit_time = Column(DateTime, nullable=True)
    stop_loss = Column(Float, nullable=True)
    take_profit = Column(Float, nullable=True)
    trail_percent = Column(Float, nullable=True)
    status = Column(String, default='open')  # 'open', 'closed', 'stopped', 'profit'
    pnl = Column(Float, nullable=True)
    pnl_pct = Column(Float, nullable=True)


class PortfolioSnapshot(Base):
    __tablename__ = 'portfolio_snapshots'
    id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    portfolio_value = Column(Float)
    cash = Column(Float)
    equity = Column(Float)


class DailyBriefing(Base):
    __tablename__ = 'daily_briefings'
    id = Column(Integer, primary_key=True)
    date = Column(String, unique=True)   # YYYY-MM-DD
    symbols_json = Column(String)        # JSON list of recommended symbols
    regime = Column(String)


def get_engine():
    return create_engine(f'sqlite:///{DB_PATH}')


def init_db():
    engine = get_engine()
    Base.metadata.create_all(engine)
    return engine


def get_session():
    engine = init_db()
    Session = sessionmaker(bind=engine)
    return Session()


def log_trade(order_id, symbol, module, side, qty, entry_price,
              stop_loss=None, take_profit=None, trail_percent=None):
    session = get_session()
    trade = Trade(
        order_id=order_id,
        symbol=symbol,
        module=module,
        side=side,
        qty=qty,
        entry_price=entry_price,
        stop_loss=stop_loss,
        take_profit=take_profit,
        trail_percent=trail_percent,
    )
    session.add(trade)
    session.commit()
    session.close()


def close_trade(order_id, exit_price, status='closed'):
    session = get_session()
    trade = session.query(Trade).filter_by(order_id=order_id).first()
    if trade:
        trade.exit_price = exit_price
        trade.exit_time = datetime.utcnow()
        trade.status = status
        trade.pnl = (exit_price - trade.entry_price) * trade.qty
        trade.pnl_pct = (exit_price - trade.entry_price) / trade.entry_price
        session.commit()
    session.close()


def snapshot_portfolio(portfolio_value, cash, equity):
    session = get_session()
    snap = PortfolioSnapshot(
        portfolio_value=portfolio_value,
        cash=cash,
        equity=equity,
    )
    session.add(snap)
    session.commit()
    session.close()


def get_open_trades(module=None):
    session = get_session()
    q = session.query(Trade).filter_by(status='open')
    if module:
        q = q.filter_by(module=module)
    trades = q.all()
    session.close()
    return trades


def get_all_trades(module=None):
    session = get_session()
    q = session.query(Trade)
    if module:
        q = q.filter_by(module=module)
    trades = q.all()
    session.close()
    return trades


def get_snapshots():
    session = get_session()
    snaps = session.query(PortfolioSnapshot).order_by(PortfolioSnapshot.timestamp).all()
    session.close()
    return snaps


def save_daily_briefing(date_str: str, symbols: list, regime: str):
    """Upsert today's recommended symbols — called once per dashboard load."""
    session = get_session()
    existing = session.query(DailyBriefing).filter_by(date=date_str).first()
    if existing:
        existing.symbols_json = json.dumps(symbols)
        existing.regime = regime
    else:
        session.add(DailyBriefing(date=date_str, symbols_json=json.dumps(symbols), regime=regime))
    session.commit()
    session.close()


def get_previous_briefing(today_str: str):
    """Return (date_str, symbols_list, regime) for the most recent past date, or None."""
    session = get_session()
    rec = (
        session.query(DailyBriefing)
        .filter(DailyBriefing.date < today_str)
        .order_by(DailyBriefing.date.desc())
        .first()
    )
    session.close()
    if rec:
        return rec.date, json.loads(rec.symbols_json), rec.regime
    return None
