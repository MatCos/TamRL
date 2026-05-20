from __future__ import annotations

import re
from typing import Any


def is_solved(result_str: str) -> bool:
    return result_str.strip().lower() in {"true", "false", "verified", "falsified"}


def is_completed(metrics: Any) -> bool:
    return isinstance(metrics, dict) and metrics.get("completes", 0) > 0


def avg(values: list) -> float:
    return sum(values) / len(values) if values else 0.0


def fmt_time(t: Any) -> str:
    if isinstance(t, (int, float)):
        m, s = divmod(t, 60)
        return f"{int(m)}m {s:.1f}s"
    return t


def parse_time_to_seconds(time_str: Any) -> float | None:
    if time_str is None or str(time_str) == "null":
        return None
    s = str(time_str).strip()
    if s.lower() in {"n/a", "na", "---", "-", ""}:
        return None
    total = 0.0
    h_match = re.search(r"(\d+)h", s)
    m_match = re.search(r"(\d+)m", s)
    s_match = re.search(r"([\d.]+)s", s)
    if h_match:
        total += int(h_match.group(1)) * 3600
    if m_match:
        total += int(m_match.group(1)) * 60
    if s_match:
        total += float(s_match.group(1))
    if not h_match and not m_match and not s_match:
        return None
    return total


def fmt_seconds(seconds: float) -> str:
    if seconds >= 3600:
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        s = seconds % 60
        return f"{h}h {m}m {s:.1f}s"
    elif seconds >= 60:
        m = int(seconds // 60)
        return f"{m}m {seconds % 60:.1f}s"
    return f"{seconds:.1f}s"


def fmt_seconds_compact(t: float) -> str:
    if t >= 3600:
        total_min = int(round(t / 60))
        h, m = divmod(total_min, 60)
        return f"{h}h {m}m" if m else f"{h}h"
    if t >= 60:
        m = int(t // 60)
        s = int(round(t - m * 60))
        if s == 60:
            m += 1
            s = 0
        return f"{m}m {s}s" if s else f"{m}m"
    if t >= 10:
        return f"{int(round(t))}s"
    return fmt_seconds(t)


def normalize(text: str) -> str:
    return "".join(ch for ch in text.lower() if ch.isalnum())


def latex_escape(text: str) -> str:
    return (
        str(text)
        .replace("_", r"\_")
        .replace("&", r"\&")
        .replace("%", r"\%")
        .replace("#", r"\#")
    )
