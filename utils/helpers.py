from datetime import datetime, timezone, timedelta
from typing import Optional


def gmt_now() -> datetime:
    return datetime.now(timezone.utc)


def get_session(gmt_hour: int) -> str:
    if 0 <= gmt_hour < 9:
        return "Asian"
    elif 8 <= gmt_hour < 17:
        return "London"
    elif 13 <= gmt_hour < 22:
        return "New_York"
    else:
        return "Low_Liquidity"


def get_session_overlap(gmt_hour: int) -> Optional[str]:
    if 13 <= gmt_hour < 17:
        return "London × New_York"
    return None


def format_price(price: float) -> str:
    return f"{price:.5f}"
