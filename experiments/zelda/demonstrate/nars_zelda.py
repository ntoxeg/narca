import json
import logging
from functools import partial

import griddly  # noqa
import gym
import pexpect
from griddly import gd
from icecream import ic
from tensorboardX import SummaryWriter

from narca.nar import *
from narca.utils import *
from narca.zelda import Runner, ZeldaAgent, ZeldaLevelGenerator

# setup a logger for nars output
logging.basicConfig(filename="nars_zelda.log", filemode="w", level=logging.DEBUG)
logger = logging.getLogger("nars")

NUM_EPISODES = 50
MAX_ITERATIONS = 100
ENV_NAME = "GDY-Zelda-v0"
DIFFICULTY_LEVEL = 1
NUM_DEMOS = 5

with open("difficulty_settings.json") as f:
    difficulty_settings = json.load(f)
LEVELGEN_CONFIG = difficulty_settings[ENV_NAME][str(DIFFICULTY_LEVEL)] | {
    "max_goals": 1,
    "p_key": 1.0,
    "max_spiders": 1,
    "p_spider": 1.0,
}


def object_reached(obj_type: str, env_state: dict, info: dict) -> bool:
    """Check if an object has been reached

    Assumes that if the object does not exist, then it must have been reached.
    """
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


def got_rewarded(env_state: dict, _) -> bool:
    return env_state["reward"] > 0


def make_goal(process: pexpect.spawn, env_state: dict, goal_symbol: str) -> None:
    """Make an explicit goal using the position of an object"""
    goal = next(obj for obj in env_state["Objects"] if obj["Name"] == "goal")
    goal_loc = loc(goal["Location"])
    goal_achievement = f"<<({ext('SELF')} * {goal_loc}) --> at> =/> {goal_symbol}>."
    send_input(process, goal_achievement)


def make_loc_goal(process: pexpect.spawn, pos, goal_symbol):
    """Make a goal for a location"""
    goal_loc = loc(pos)
    goal_achievement = f"<<({ext('SELF')} * {goal_loc}) --> at> =/> {goal_symbol}>."
    send_input(process, goal_achievement)


def key_check(_, info) -> bool:
    history = info["History"]
    if len(history) == 0:
        return False
    last_event = history[-1]
    return (
        last_event["SourceObjectName"] == "avatar"
        and last_event["DestinationObjectName"] == "key"
    )


if __name__ == "__main__":
    env = gym.make(ENV_NAME, player_observer_type=gd.ObserverType.VECTOR)
    env.enable_history(True)  # type: ignore
    levelgen = ZeldaLevelGenerator(LEVELGEN_CONFIG)

    reach_object_knowledge = [
        f"<(<($obj * #location) --> at> &/ <({ext('SELF')} * #location) --> ^goto>) =/> <$obj --> [reached]>>.",
    ]
    rel_pos_knowledge = [
        f"<(<$obj --> [ahead]> &/ ^move_forwards) =/> <$obj --> [reached]>>.",
        f"<(<$obj --> [leftward]> &/ ^move_forwards &/ ^rotate_left) =/> <$obj --> [ahead]>>.",
        f"<(<$obj --> [rightward]> &/ ^move_forwards &/ ^rotate_right) =/> <$obj --> [ahead]>>.",
    ]
    background_knowledge = rel_pos_knowledge

    key_goal_sym = "GOT_KEY"
    reach_key = [f"<({ext('key')} --> [reached]) =/> {key_goal_sym}>."]
    door_goal_sym = "AT_DOOR"
    reach_door = [f"<<{ext('goal')} --> [reached]> =/> {door_goal_sym}>."]
    complete_goal_sym = "COMPLETE"
    complete_goal = [f"<({key_goal_sym} &/ {door_goal_sym}) =/> {complete_goal_sym}>."]

    KEY_GOAL = Goal(key_goal_sym, partial(object_reached, "key"), reach_key)
    DOOR_GOAL = Goal(door_goal_sym, partial(object_reached, "goal"), reach_door)
    COMPLETE_GOAL = Goal(
        complete_goal_sym,
        lambda evst, info: agent.has_key and DOOR_GOAL.satisfied(evst, info),
        complete_goal,
    )
    REWARD_GOAL = Goal("GOT_REWARD", got_rewarded)

    goals = [
        KEY_GOAL,
        DOOR_GOAL,
        COMPLETE_GOAL,
        REWARD_GOAL,
    ]

    agent = ZeldaAgent(
        env,
        COMPLETE_GOAL,
        think_ticks=10,
        background_knowledge=background_knowledge,
    )
    runner = Runner(agent, goals, levelgen)

    # DEMONSTRATE
    for _ in range(NUM_DEMOS):
        plan = [
            "^rotate_left",
            "^move_forwards",
            "^move_forwards",
            "^rotate_right",
            "^move_forwards",
            "^move_forwards",
        ]
        print("Demonstration: completing a level...")
        runner.demo_goal(plan)

    # Run the agent
    runner.run(
        NUM_EPISODES,
        MAX_ITERATIONS,
        log_tb=True,
        comment_suffix="-demonstrate",
    )
