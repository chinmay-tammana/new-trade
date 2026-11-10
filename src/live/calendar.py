"""NYSE Trading Calendar and Session Helpers for Live Pipeline.

Provides deterministic session hours, 1h bar schedules (with half-day awareness:
e.g. 13:00 early close yields 4 bars), daylight savings transitions, and
next upcoming bar close calculations.
"""

import datetime
import os
from zoneinfo import ZoneInfo
from typing import Dict, Any, List, Optional, Tuple

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import GetCalendarRequest

from configs.config import CONFIG

ET = ZoneInfo("America/New_York")


def get_alpaca_trading_client() -> Optional[TradingClient]:
    """Initialize Alpaca trading client if credentials are configured."""
    api_key = os.environ.get("APCA_API_KEY_ID")
    api_secret = os.environ.get("APCA_API_SECRET_KEY")
    if api_key and api_secret:
        return TradingClient(api_key=api_key, secret_key=api_secret, paper=True)
    return None


def get_trading_session_info(
    date: datetime.date,
    client: Optional[TradingClient] = None
) -> Optional[Dict[str, Any]]:
    """Determine if a given date is a NYSE trading day, and return open/close times."""
    # Weekend check
    if date.weekday() >= 5:
        return None

    # Known fixed holidays (offline fallback)
    holidays = {
        datetime.date(2025, 1, 1), datetime.date(2025, 1, 20), datetime.date(2025, 2, 17),
        datetime.date(2025, 4, 18), datetime.date(2025, 5, 26), datetime.date(2025, 6, 19),
        datetime.date(2025, 7, 4), datetime.date(2025, 9, 1), datetime.date(2025, 11, 27),
        datetime.date(2025, 12, 25), datetime.date(2026, 1, 1), datetime.date(2026, 1, 19),
        datetime.date(2026, 2, 16), datetime.date(2026, 4, 3), datetime.date(2026, 5, 25),
        datetime.date(2026, 6, 19), datetime.date(2026, 7, 3), datetime.date(2026, 9, 7),
        datetime.date(2026, 11, 26), datetime.date(2026, 12, 25)
    }
    if date in holidays:
        return None

    # Known half days (13:00 close)
    half_days = {
        datetime.date(2024, 7, 3), datetime.date(2024, 11, 29), datetime.date(2024, 12, 24),
        datetime.date(2025, 7, 3), datetime.date(2025, 11, 28), datetime.date(2025, 12, 24),
        datetime.date(2026, 11, 27), datetime.date(2026, 12, 24)
    }

    # Attempt query via Alpaca calendar API if client provided
    if client is not None:
        try:
            req = GetCalendarRequest(start=date, end=date)
            cal_days = client.get_calendar(req)
            if not cal_days:
                return None
            day_info = cal_days[0]
            # Convert to localized datetime
            open_dt = datetime.datetime.combine(day_info.date, day_info.open, tzinfo=ET)
            close_dt = datetime.datetime.combine(day_info.date, day_info.close, tzinfo=ET)
            is_half = (close_dt.hour < 16)
            return {
                "date": date,
                "open": open_dt,
                "close": close_dt,
                "is_half_day": is_half
            }
        except Exception:
            pass

    # Offline deterministic schedule
    open_dt = datetime.datetime.combine(date, datetime.time(9, 30), tzinfo=ET)
    is_half = (date in half_days)
    close_time = datetime.time(13, 0) if is_half else datetime.time(16, 0)
    close_dt = datetime.datetime.combine(date, close_time, tzinfo=ET)

    return {
        "date": date,
        "open": open_dt,
        "close": close_dt,
        "is_half_day": is_half
    }


def generate_session_candle_schedule(
    session: Dict[str, Any]
) -> List[Tuple[datetime.datetime, datetime.datetime, bool]]:
    """Return list of (bar_start, bar_end, is_last_bar) for a given session.
    A normal session produces 7 bars:
      09:30-10:30, 10:30-11:30, 11:30-12:30, 12:30-13:30, 13:30-14:30, 14:30-15:30, 15:30-16:00 (half-hour)
    A 13:00 half-day produces 4 bars:
      09:30-10:30, 10:30-11:30, 11:30-12:30, 12:30-13:00 (half-hour)
    """
    bars = []
    curr = session["open"]
    close = session["close"]

    while curr < close:
        next_bar = curr + datetime.timedelta(hours=1)
        if next_bar > close:
            next_bar = close
        is_last = (next_bar == close)
        bars.append((curr, next_bar, is_last))
        curr = next_bar

    return bars


def is_market_open_now(now_dt: Optional[datetime.datetime] = None) -> Tuple[bool, Optional[Dict[str, Any]]]:
    """Check if NYSE market is currently open in ET."""
    if now_dt is None:
        now_dt = datetime.datetime.now(ET)
    else:
        now_dt = now_dt.astimezone(ET)

    session = get_trading_session_info(now_dt.date())
    if session is None:
        return False, None

    is_open = (session["open"] <= now_dt < session["close"])
    return is_open, session


def get_next_bar_close(now_dt: Optional[datetime.datetime] = None) -> Optional[datetime.datetime]:
    """Find the next upcoming bar close timestamp in ET."""
    if now_dt is None:
        now_dt = datetime.datetime.now(ET)
    else:
        now_dt = now_dt.astimezone(ET)

    # Check today's session
    session = get_trading_session_info(now_dt.date())
    if session is not None and now_dt < session["close"]:
        schedule = generate_session_candle_schedule(session)
        for _, bar_end, _ in schedule:
            if bar_end > now_dt:
                return bar_end

    # Look ahead up to 7 days for next session
    curr_date = now_dt.date() + datetime.timedelta(days=1)
    for _ in range(7):
        next_session = get_trading_session_info(curr_date)
        if next_session is not None:
            schedule = generate_session_candle_schedule(next_session)
            if schedule:
                return schedule[0][1]
        curr_date += datetime.timedelta(days=1)

    return None


class NyseCalendar:
    """Convenience class providing calendar and session helper methods."""

    def __init__(self, client: Optional[TradingClient] = None):
        self.client = client or get_alpaca_trading_client()

    def get_session_bounds(self, date: datetime.date) -> Optional[Dict[str, Any]]:
        info = get_trading_session_info(date, client=self.client)
        if not info:
            return None
        sched = generate_session_candle_schedule(info)
        return {
            "date": info["date"],
            "open": info["open"],
            "close": info["close"],
            "is_half_day": info["is_half_day"],
            "bars": [{"start": s, "end": e, "is_last": l} for s, e, l in sched]
        }

    def is_market_open(self, now_dt: Optional[datetime.datetime] = None) -> Tuple[bool, Optional[Dict[str, Any]]]:
        return is_market_open_now(now_dt)

    def get_next_bar_close(self, now_dt: Optional[datetime.datetime] = None) -> Optional[datetime.datetime]:
        return get_next_bar_close(now_dt)
