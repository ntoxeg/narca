import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

from icecream import ic

from .nar import send_input
from .narsese import *

NARS_PATH = Path(os.environ["NARS_HOME"])


def manhattan_distance(pos1: tuple[int, int], pos2: tuple[int, int]) -> int:
    """Calculate the Manhattan distance between two points"""
    return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])


# GRIDDLY RELATED
def last_avatar_event(agent, history: list[dict]) -> Optional[dict]:
    """Return the last avatar event in the history"""
    for event in reversed(history):
        if event["SourceObjectName"] == type(agent).AVATAR_LABEL:
            return event
    return None


def got_rewarded(env_state: dict, _) -> bool:
    return env_state["reward"] > 0


def object_reached(
    agent, move_name: str, obj_type: str, env_state: dict, info: dict
) -> bool:
    """Check if an object has been reached

    Uses event logs from Griddly environments with `enable_history(True)`.
    """
    history = info["History"]
    if len(history) == 0:
        return False

    last_avent = last_avatar_event(agent, history)
    if last_avent is not None:
        if (
            last_avent["DestinationObjectName"] == obj_type
            and last_avent["ActionName"] == move_name
        ):
            send_input(
                agent.process,
                nal_now(f"<{ext(last_avent['DestinationObjectName'])} --> [reached]>"),
            )
            return True

    return False


def in_front(
    viewer_orient: str, viewer_loc: tuple[int, int], obloc: tuple[int, int]
) -> bool:
    """Check if an object is in front of the viewer"""
    match viewer_orient:
        case "UP":
            return obloc[1] < viewer_loc[1]
        case "RIGHT":
            return obloc[0] > viewer_loc[0]
        case "DOWN":
            return obloc[1] > viewer_loc[1]
        case "LEFT":
            return obloc[0] < viewer_loc[0]

    return False


def perpendicular(
    viewer_orient: str, viewer_loc: tuple[int, int], obloc: tuple[int, int]
) -> bool:
    """Check if an object is perpendicular to the viewer

    Note: it is assumed that an object cannot occupy the same space as an agent.
    """
    match viewer_orient:
        case "UP":
            return obloc[1] == viewer_loc[1]
        case "RIGHT":
            return obloc[0] == viewer_loc[0]
        case "DOWN":
            return obloc[1] == viewer_loc[1]
        case "LEFT":
            return obloc[0] == viewer_loc[0]

    return False


def nal_rel_pos(
    obname: str, orient: str, avatar_loc: tuple[int, int], obloc: tuple[int, int]
) -> Optional[str]:
    """Produce NARS statement about relative position of an object w.r.t. avatar

    The object is required to be in front of the avatar.
    """
    match orient:
        case "UP":
            if obloc[1] < avatar_loc[1]:
                if obloc[0] == avatar_loc[0]:
                    return nal_now(f"<{ext(obname)} --> [ahead]>")
                if obloc[0] < avatar_loc[0]:
                    return nal_now(f"<{ext(obname)} --> [leftward]>")
                if obloc[0] > avatar_loc[0]:
                    return nal_now(f"<{ext(obname)} --> [rightward]>")
            if obloc[1] == avatar_loc[1]:
                if obloc[0] < avatar_loc[0]:
                    return nal_now(f"<{ext(obname)} --> [leftward]>")
                if obloc[0] > avatar_loc[0]:
                    return nal_now(f"<{ext(obname)} --> [rightward]>")
        case "RIGHT":
            if obloc[0] > avatar_loc[0]:
                if obloc[1] == avatar_loc[1]:
                    return nal_now(f"<{ext(obname)} --> [ahead]>")
                if obloc[1] < avatar_loc[1]:
                    return nal_now(f"<{ext(obname)} --> [leftward]>")
                if obloc[1] > avatar_loc[1]:
                    return nal_now(f"<{ext(obname)} --> [rightward]>")
            if obloc[0] == avatar_loc[0]:
                if obloc[1] < avatar_loc[1]:
                    return nal_now(f"<{ext(obname)} --> [leftward]>")
                if obloc[1] > avatar_loc[1]:
                    return nal_now(f"<{ext(obname)} --> [rightward]>")
        case "DOWN":
            if obloc[1] > avatar_loc[1]:
                if obloc[0] == avatar_loc[0]:
                    return nal_now(f"<{ext(obname)} --> [ahead]>")
                if obloc[0] < avatar_loc[0]:
                    return nal_now(f"<{ext(obname)} --> [rightward]>")
                if obloc[0] > avatar_loc[0]:
                    return nal_now(f"<{ext(obname)} --> [leftward]>")
            if obloc[1] == avatar_loc[1]:
                if obloc[0] < avatar_loc[0]:
                    return nal_now(f"<{ext(obname)} --> [rightward]>")
                if obloc[0] > avatar_loc[0]:
                    return nal_now(f"<{ext(obname)} --> [leftward]>")
        case "LEFT":
            if obloc[0] < avatar_loc[0]:
                if obloc[1] == avatar_loc[1]:
                    return nal_now(f"<{ext(obname)} --> [ahead]>")
                if obloc[1] < avatar_loc[1]:
                    return nal_now(f"<{ext(obname)} --> [rightward]>")
                if obloc[1] > avatar_loc[1]:
                    return nal_now(f"<{ext(obname)} --> [leftward]>")
            if obloc[0] == avatar_loc[0]:
                if obloc[1] < avatar_loc[1]:
                    return nal_now(f"<{ext(obname)} --> [rightward]>")
                if obloc[1] > avatar_loc[1]:
                    return nal_now(f"<{ext(obname)} --> [leftward]>")

    return None


def nal_distance(
    obname: str, avatar_info: tuple[tuple[int, int], str], obloc: tuple[int, int]
) -> list[str]:
    """Produce NARS statement about distance of an object w.r.t. avatar

    Represents distance as 'far' / 'near', depending on the manhattan distance.
    """

    def ds_str(ds):
        dss = str(abs(ds))
        if ds > 0:  # left
            return "L" + dss
        if ds < 0:
            return "R" + dss
        return dss

    avatar_loc, orient = avatar_info
    if manhattan_distance(avatar_loc, obloc) > 2:
        # return [nal_now(f"<{ext(obname)} --> [far]>")]
        return []

    df, dss = 0, "0"
    match orient:
        case "UP":
            if obloc[1] < avatar_loc[1]:
                df, ds = avatar_loc[1] - obloc[1], avatar_loc[0] - obloc[0]
                dss = ds_str(ds)
        case "RIGHT":
            if obloc[0] > avatar_loc[0]:
                df, ds = obloc[0] - avatar_loc[0], avatar_loc[1] - obloc[1]
                dss = ds_str(ds)
        case "DOWN":
            if obloc[1] > avatar_loc[1]:
                df, ds = obloc[1] - avatar_loc[1], obloc[0] - avatar_loc[0]
                dss = ds_str(ds)
        case "LEFT":
            if obloc[0] < avatar_loc[0]:
                df, ds = avatar_loc[0] - obloc[0], obloc[1] - avatar_loc[1]
                dss = ds_str(ds)

    return [
        nal_now(f"<({ext(obname)} * {df}) --> delta_forward>"),
        nal_now(f"<({ext(obname)} * {dss}) --> delta_sideways>"),
    ]


def abs_to_rel(avatar, op):
    orient_to_num = {
        "UP": 0,
        "RIGHT": 1,
        "DOWN": 2,
        "LEFT": 3,
        "NONE": 0,  # HACK: assuming that NONE is the same as UP
    }
    if avatar["Orientation"] == "NONE":
        ic("Warning: avatar orientation is NONE. Assuming UP.")
    avatar_orient = orient_to_num[avatar["Orientation"]]
    dor = 0
    match op:
        case "^up":
            dor = 4 - avatar_orient if avatar_orient != 0 else 0
        case "^right":
            dor = 5 - avatar_orient if avatar_orient != 1 else 0
        case "^down":
            dor = 6 - avatar_orient if avatar_orient != 2 else 0
        case "^left":
            dor = 7 - avatar_orient if avatar_orient != 3 else 0

    return ["^rotate_right"] * dor + ["^move_forwards"]
