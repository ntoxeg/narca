from dataclasses import dataclass
from typing import Any, Callable, Optional


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


def parse_truth_value(tv_str: str) -> dict[str, float]:
    splits = tv_str.split(" ")
    freq_s, conf_s = splits[0], splits[1]
    if freq_s[-1] == ",":
        freq_s = freq_s[:-1]

    frequency = float(freq_s.split("=")[1])
    confidence = float(conf_s.split("=")[1])
    return {
        "frequency": frequency,
        "confidence": confidence,
    }


def parse_task(s: str) -> dict[str, Any]:
    M: dict[str, Any] = {"occurrenceTime": "eternal"}
    if " :|:" in s:
        M["occurrenceTime"] = "now"
        s = s.replace(" :|:", "")
        if "occurrenceTime" in s:
            M["occurrenceTime"] = s.split("occurrenceTime=")[1].split(" ")[0]
    sentence = (
        s.split(" occurrenceTime=")[0]
        if " occurrenceTime=" in s
        else s.split(" Priority=")[0]
    )
    M["punctuation"] = sentence[-4] if ":|:" in sentence else sentence[-1]
    M["term"] = (
        sentence.split(" creationTime")[0]
        .split(" occurrenceTime")[0]
        .split(" Truth")[0][:-1]
    )
    if "Truth" in s:
        M["truth"] = parse_truth_value(s.split("Truth: ")[1])
    return M


def parse_reason(sraw: str) -> Optional[dict[str, str]]:
    if "implication: " not in sraw:
        return None
    Implication = parse_task(
        sraw.split("implication: ")[-1].split("precondition: ")[0]
    )  # last reason only (others couldn't be associated currently)
    Precondition = parse_task(sraw.split("precondition: ")[-1].split("\n")[0])
    Implication["occurrenceTime"] = "eternal"
    Precondition["punctuation"] = Implication["punctuation"] = "."
    Reason = {}
    Reason["desire"] = sraw.split("decision expectation=")[-1].split(" ")[0]
    Reason["hypothesis"] = Implication
    Reason["precondition"] = Precondition
    return Reason


def parse_execution(e: str) -> dict[str, Any]:
    splits = e.split(" ")
    if "args " not in e:
        return {"operator": splits[0], "arguments": []}
    return {
        "operator": splits[0],
        "arguments": e.split("args ")[1][1:-1].split(" * "),
    }


@dataclass
class Goal:
    """A goal"""

    symbol: str
    satisfied: Callable
    knowledge: Optional[list[str]] = None

    def __repr__(self) -> str:
        return f"Goal {self.symbol}; {self.knowledge}"
