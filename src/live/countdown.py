"""Pure Countdown Formatter for Live Market Engine and Dashboard (V3.2).

Specifications:
- More than 24 hours: "Xd Yh Zm" (e.g. "2d 7h 12m")
- 1 to 24 hours: "Yh Zm" (e.g. "7h 12m")
- Under 1 hour: "mm:ss" (e.g. "34:21")
- Under or equal to 0 seconds: "00:00" during market open, or "Opens Soon" when closed.
"""

def format_countdown_pure(seconds: int, market_open: bool = False) -> str:
    """Pure countdown formatter adhering strictly to V3.2 display rules."""
    if seconds <= 0:
        return "00:00" if market_open else "Opens Soon"

    if market_open:
        # During session: if under 1 hour, standard mm:ss
        if seconds < 3600:
            mins = seconds // 60
            secs = seconds % 60
            return f"{mins:02d}:{secs:02d}"
        else:
            # Session multi-hour (e.g. early morning or extended)
            hours = seconds // 3600
            mins = (seconds % 3600) // 60
            return f"{hours}h {mins:02d}m"

    # Outside market hours:
    days = seconds // 86400
    rem = seconds % 86400
    hours = rem // 3600
    mins = (rem % 3600) // 60

    if days >= 1:
        return f"{days}d {hours}h {mins}m"
    elif hours >= 1:
        return f"{hours}h {mins}m"
    else:
        # Under 1 hour to market open
        secs = rem % 60
        return f"{mins:02d}:{secs:02d}"
