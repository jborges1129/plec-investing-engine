"""
Intraday scheduler — runs during market hours (9:30–16:00 ET) on weekdays.

Every 30 minutes:
  1. check_closed_positions — detects stop-loss / take-profit fires, logs + emails
  2. check_rotation        — detects when a held ETF slips to rank 4+ while a
                             better ETF is rank 1-2; emails rotation suggestion

No trades are placed automatically. All actions require manual approval.

Usage:
    python run.py scheduler      # runs until Ctrl-C
"""

import time
import logging
from datetime import datetime, timezone

import pytz
from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.cron import CronTrigger

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s  %(levelname)s  %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)
log = logging.getLogger(__name__)

ET = pytz.timezone('America/New_York')
MARKET_OPEN_HOUR   = 9
MARKET_OPEN_MINUTE = 35   # give 5 min after open for prices to settle
MARKET_CLOSE_HOUR  = 15
MARKET_CLOSE_MINUTE = 55  # stop before close


def _is_market_hours() -> bool:
    now_et = datetime.now(ET)
    if now_et.weekday() >= 5:  # Saturday=5, Sunday=6
        return False
    t = now_et.time()
    open_t  = now_et.replace(hour=MARKET_OPEN_HOUR,  minute=MARKET_OPEN_MINUTE,  second=0, microsecond=0).time()
    close_t = now_et.replace(hour=MARKET_CLOSE_HOUR, minute=MARKET_CLOSE_MINUTE, second=0, microsecond=0).time()
    return open_t <= t <= close_t


def _job_snapshot():
    if not _is_market_hours():
        return
    log.info("Snapshotting portfolio...")
    try:
        from shared.alpaca import get_account
        from shared.database import snapshot_portfolio
        a = get_account()
        snapshot_portfolio(float(a.portfolio_value), float(a.cash), float(a.equity))
    except Exception as e:
        log.error(f"snapshot_portfolio failed: {e}")


def _job_closed_positions():
    if not _is_market_hours():
        return
    log.info("Running check_closed_positions...")
    try:
        from etf_module.execute import check_closed_positions
        check_closed_positions()
    except Exception as e:
        log.error(f"check_closed_positions failed: {e}")


def _job_rotation_check():
    if not _is_market_hours():
        return
    log.info("Running check_rotation...")
    try:
        from etf_module.execute import check_rotation
        fired = check_rotation()
        if fired:
            log.info("Rotation alert sent via email.")
        else:
            log.info("No rotation signal.")
    except Exception as e:
        log.error(f"check_rotation failed: {e}")


def run_scheduler():
    scheduler = BlockingScheduler(timezone=ET)

    # Every 30 min on weekdays during market hours (scheduler fires regardless;
    # each job guards itself with _is_market_hours())
    trigger = CronTrigger(
        day_of_week='mon-fri',
        hour=f'{MARKET_OPEN_HOUR}-{MARKET_CLOSE_HOUR}',
        minute='0,15,30,45',  # every 15 min — re-ranks all ETFs each cycle
        timezone=ET,
    )

    scheduler.add_job(_job_snapshot,          trigger, id='portfolio_snapshot')
    scheduler.add_job(_job_closed_positions,  trigger, id='closed_positions')
    scheduler.add_job(_job_rotation_check,    trigger, id='rotation_check')

    log.info("Scheduler started — running every 30 min during market hours (ET).")
    log.info("Press Ctrl-C to stop.")
    try:
        scheduler.start()
    except (KeyboardInterrupt, SystemExit):
        log.info("Scheduler stopped.")
