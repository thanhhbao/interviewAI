from __future__ import annotations

import re


FILLER_PATTERN = re.compile(r"\b(uh|um|er|ah|like|you know|ờ|à|ừm|thì|là)\b", re.IGNORECASE)


def count_filler_words(text: str) -> int:
    return len(FILLER_PATTERN.findall(text))


def compute_wpm(text: str, duration_seconds: float) -> float:
    if duration_seconds <= 0:
        return 0.0
    words = len(text.split())
    return round(words / (duration_seconds / 60.0), 2)


def compute_fluency_score(text: str, duration_seconds: float) -> float:
    wpm = compute_wpm(text, duration_seconds)
    fillers = count_filler_words(text)
    baseline = 1.0
    if wpm < 80 or wpm > 180:
        baseline -= 0.2
    baseline -= min(fillers * 0.03, 0.4)
    return round(max(0.0, min(1.0, baseline)), 2)
