"""Date/time helpers shared by environments and the benchmark builder.

All simulated time is explicit: environments carry a fixed ``current_date`` in
state and never read the system clock, so every episode is reproducible.
Format conventions: dates ``YYYY-MM-DD``, times ``HH:MM`` (24h), timestamps
``YYYY-MM-DDTHH:MM``.
"""

from __future__ import annotations

import re
from datetime import date as _date

_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_TIME_RE = re.compile(r"^([01]\d|2[0-3]):[0-5]\d$")
_DATETIME_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T([01]\d|2[0-3]):[0-5]\d$")

#: Fixed "today" for all default fixtures (a Monday).
DEFAULT_CURRENT_DATE = "2026-03-02"


class TimeFormatError(ValueError):
    """Raised when a date/time string violates the documented format."""


def is_valid_date(value: str) -> bool:
    if not isinstance(value, str) or not _DATE_RE.match(value):
        return False
    try:
        _date.fromisoformat(value)
    except ValueError:
        return False
    return True


def is_valid_time(value: str) -> bool:
    return isinstance(value, str) and bool(_TIME_RE.match(value))


def is_valid_datetime(value: str) -> bool:
    return isinstance(value, str) and bool(_DATETIME_RE.match(value))


def parse_date(value: str) -> str:
    if not is_valid_date(value):
        raise TimeFormatError(f"Invalid date {value!r}, expected YYYY-MM-DD (e.g. 2026-03-02)")
    return value


def parse_time(value: str) -> str:
    if not is_valid_time(value):
        raise TimeFormatError(f"Invalid time {value!r}, expected HH:MM 24-hour (e.g. 14:30)")
    return value


def parse_datetime(value: str) -> str:
    if not is_valid_datetime(value):
        raise TimeFormatError(f"Invalid datetime {value!r}, expected YYYY-MM-DDTHH:MM")
    return value


def time_to_minutes(value: str) -> int:
    parse_time(value)
    hours, minutes = value.split(":")
    return int(hours) * 60 + int(minutes)


def minutes_to_time(minutes: int) -> str:
    if not 0 <= minutes <= 24 * 60:
        raise TimeFormatError(f"Minutes out of range: {minutes}")
    return f"{minutes // 60:02d}:{minutes % 60:02d}"


def shift_time(value: str, delta_minutes: int) -> str:
    return minutes_to_time(time_to_minutes(value) + delta_minutes)


def add_days(value: str, days: int) -> str:
    """Shift a YYYY-MM-DD date; used by builders for 'tomorrow' etc."""
    parse_date(value)
    shifted = _date.fromordinal(_date.fromisoformat(value).toordinal() + days)
    return shifted.isoformat()


def ranges_overlap(a_start: int, a_end: int, b_start: int, b_end: int) -> bool:
    """Half-open interval overlap: [a_start, a_end) vs [b_start, b_end)."""
    return a_start < b_end and b_start < a_end


def merge_ranges(ranges: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Merge overlapping/touching [start, end) ranges; returns sorted copy."""
    merged: list[tuple[int, int]] = []
    for start, end in sorted(ranges):
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def find_free_ranges(
    busy: list[tuple[int, int]],
    window_start: int,
    window_end: int,
    min_duration: int,
) -> list[tuple[int, int]]:
    """Free gaps of at least ``min_duration`` inside [window_start, window_end).

    Busy ranges are clipped to the window and merged; gaps are returned sorted.
    """
    if min_duration <= 0 or window_end <= window_start:
        return []
    clipped = [
        (max(s, window_start), min(e, window_end))
        for s, e in merge_ranges(busy)
        if e > window_start and s < window_end
    ]
    free: list[tuple[int, int]] = []
    cursor = window_start
    for start, end in clipped:
        if start - cursor >= min_duration:
            free.append((cursor, start))
        cursor = max(cursor, end)
    if window_end - cursor >= min_duration:
        free.append((cursor, window_end))
    return free
