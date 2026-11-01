"""Pure helpers; no game/PPO dependencies needed for adapter checks."""
from __future__ import annotations
import math


def rect_separation(rect, x: float, y: float) -> float:
    """Distance from ball CENTER to axis-aligned rectangle (not its center)."""
    rx, ry, width, height = rect
    dx = max(rx - x, 0., x - (rx + width))
    dy = max(ry - y, 0., y - (ry + height))
    return math.hypot(dx, dy)


def ranked_indices(records, x: float, y: float) -> list[int]:
    """Sort by local geometric relevance, stably breaking ties by native ID order.

    The ranking is the same for full-entity truncation and local observations.
    It is deliberately goal/route independent: no scripted demonstrations.
    """
    return sorted(range(len(records)), key=lambda i: (rect_separation(records[i]['rect'], x, y), i))


def parse_map_selection(spec: str) -> list[int]:
    """Map numbers 1..10; accepts 'all', '1-10', '1,3,5', or '2'."""
    value = str(spec).strip().lower()
    if value in ('all', '1-10'):
        return list(range(1, 11))
    selected = []
    for token in value.split(','):
        token = token.strip()
        if '-' in token:
            parts = token.split('-')
            if len(parts) != 2 or not all(part.strip().isdigit() for part in parts):
                raise ValueError(f'Invalid map range: {token!r}')
            lo, hi = map(int, parts)
            if hi < lo:
                raise ValueError('Map ranges must ascend')
            selected.extend(range(lo, hi + 1))
        else:
            if not token.isdigit():
                raise ValueError(f'Invalid map number: {token!r}')
            selected.append(int(token))
    if not selected or any(number < 1 or number > 10 for number in selected):
        raise ValueError('Mixed RL map numbers must be 1..10')
    return list(dict.fromkeys(selected))
