import os
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from dotenv import load_dotenv

load_dotenv()


def send_alert(subject: str, body: str, is_html: bool = False):
    """Send an email alert. Only call when something is actually actionable."""
    sender = os.getenv('EMAIL_FROM')
    recipient = os.getenv('EMAIL_TO')
    password = os.getenv('EMAIL_PASSWORD')

    if not all([sender, recipient, password]):
        print(f"[Email not configured] {subject}")
        return

    msg = MIMEMultipart('alternative')
    msg['Subject'] = f"[Trading Alert] {subject}"
    msg['From'] = sender
    msg['To'] = recipient

    content_type = 'html' if is_html else 'plain'
    msg.attach(MIMEText(body, content_type))

    try:
        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as server:
            server.login(sender, password)
            server.sendmail(sender, recipient, msg.as_string())
        print(f"Alert sent: {subject}")
    except Exception as e:
        print(f"Failed to send alert: {e}")


def format_regime_alert(regime_data: dict, prev_regime: str) -> tuple[str, str]:
    regime = regime_data['regime']
    subject = f"Regime change: {prev_regime.upper()} → {regime.upper()}"
    body = f"""Market regime has shifted.

Previous: {prev_regime.upper()}
Current:  {regime.upper()}

VIX: {regime_data['vix']:.1f}
SPY: ${regime_data['spy_price']:.2f}
50-day MA: ${regime_data['ma50']:.2f}
200-day MA: ${regime_data['ma200']:.2f}
SPY above 200MA: {regime_data['spy_above_ma200']}

{"→ Consider rotating toward defensive ETFs (TLT, GLD, XLV)." if regime == 'bear' else ""}
{"→ Consider rotating toward momentum ETFs (QQQ, XLK, XLE)." if regime == 'bull' else ""}

Open your trading session to review positions.
"""
    return subject, body


def format_news_alert(symbol: str, article: dict) -> tuple[str, str]:
    subject = f"{symbol} — {article['title'][:60]}"
    body = f"""Relevant news for {symbol}

Sentiment: {article['sentiment_label'].upper()} ({article['sentiment_score']:+.2f})
Source: {article['source']}

{article['title']}

{article['summary']}

Full article: {article['url']}
"""
    return subject, body


def format_rotation_alert(held: list[dict], candidates: list[dict]) -> tuple[str, str]:
    """
    Alert fired when a held ETF slips to rank 4+ while a better ETF is rank 1-2.
    held: list of dicts with symbol, rank, return_3m
    candidates: list of dicts with symbol, rank, return_3m, stop_loss_pct, take_profit_pct, price
    """
    held_str  = ', '.join(f"{h['symbol']} (rank #{h['rank']}, 3m {h['return_3m']*100:+.1f}%)" for h in held)
    cand_str  = '\n'.join(
        f"  #{c['rank']} {c['symbol']}  3m {c['return_3m']*100:+.1f}%  "
        f"~${c['price']:.2f}  stop {c['stop_loss_pct']*100:.1f}%  target {c['take_profit_pct']*100:.1f}%"
        for c in candidates
    )
    subject = f"ETF Rotation Alert — {', '.join(c['symbol'] for c in candidates)} now ranked higher"
    body = f"""Momentum rotation opportunity detected.

Currently held (slipped in rank):
  {held_str}

Top-ranked ETFs not currently held:
{cand_str}

ACTION REQUIRED (you must do this manually):
  1. Review rankings: python run.py briefing
  2. Close the lagging position via Alpaca or: python run.py scan
  3. Enter the new position: python run.py etf-execute

This alert is informational — no trades were placed automatically.
"""
    return subject, body


def format_position_alert(symbol: str, event: str, price: float, pnl: float, pnl_pct: float) -> tuple[str, str]:
    sign = '+' if pnl >= 0 else ''
    subject = f"{symbol} {event} — {sign}{pnl_pct*100:.1f}%"
    body = f"""{symbol} position closed via {event}.

Exit price: ${price:.2f}
P&L: {sign}${pnl:.2f} ({sign}{pnl_pct*100:.1f}%)

Open your trading session to decide whether to re-enter.
"""
    return subject, body
