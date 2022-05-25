import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

from .nar import send_input

NARS_PATH = Path(os.environ["NARS_HOME"])


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


# GRIDDLY RELATED
def last_avatar_event(history: list[dict]) -> Optional[dict]:
    """Return the last avatar event in the history"""
    for event in reversed(history):
        if event["SourceObjectName"] == "avatar":
            return event
    return None


# TODO: agent should be of `NarsAgent`
def object_reached(agent, obj_type: str, env_state: dict, info: dict) -> bool:
    """Check if an object has been reached

    Uses event logs from Griddly environments with `enable_history(True)`.
    """
    # try:
    #     avatar = next(obj for obj in env_state["Objects"] if obj["Name"] == "avatar")
    # except StopIteration:
    #     ic("No avatar found. Goal unsatisfiable.")
    #     return False
    # try:
    #     target = next(obj for obj in env_state["Objects"] if obj["Name"] == obj_type)
    # except StopIteration:
    #     return True
    # return avatar["Location"] == target["Location"]
    history = info["History"]
    if len(history) == 0:
        return False

    last_avent = last_avatar_event(history)
    if last_avent is not None:
        if last_avent["DestinationObjectName"] == obj_type:
            send_input(
                agent.process,
                nal_now(f"<{ext(last_avent['DestinationObjectName'])} --> [reached]>"),
            )
            return True

    return False
