from dataclasses import dataclass
from typing import Callable, Optional


def narsify(observation):
    """Convert a Gym observation to NARS input"""
    return ",".join(str(x) for x in observation)


def loc(pos: tuple[int, int]) -> str:
    """Turn coordinates into a location string"""
    return f"loc_x{pos[0]}_y{pos[1]}"


def pos(loc: str) -> tuple[int, int]:
    """Turn a location string into coordinates"""
    coord_str = loc[4:]
    x, y = coord_str.split("_")
    x, y = x[1:], y[1:]
    return int(x), int(y)


def ext(s: str) -> str:
    """Just a helper to wrap strings in '{}'"""
    return "{" + s + "}"


def nal_demand(s: str) -> str:
    """Return NAL for demanding something"""
    return f"{s}! :|:"


def nal_now(s: str) -> str:
    """Return NAL statement in the present"""
    return f"{s}. :|:"


@dataclass
class Goal:
    """A goal"""

    symbol: str
    satisfied: Callable
    knowledge: Optional[list[str]] = None
