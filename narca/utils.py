import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

from .nar import send_input
from .narsese import *

NARS_PATH = Path(os.environ["NARS_HOME"])


def manhattan_distance(pos1: tuple[int, int], pos2: tuple[int, int]) -> int:
    """Calculate the Manhattan distance between two points"""
    return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])


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
