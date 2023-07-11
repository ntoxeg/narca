import logging
from functools import partial

import griddly  # noqa
import gym
from griddly import gd
from narpyn.ona.nar import *

from narca.agent import Runner
from narca.drunk_dwarf import DrunkDwarfAgent
from narca.utils import *

# setup a logger for nars output
logging.basicConfig(filename="nars_drunk_dwarf.log", filemode="w", level=logging.DEBUG)
logger = logging.getLogger("nars")

NUM_EPISODES = 50
MAX_ITERATIONS = 100
ENV_NAME = "GDY-Drunk-Dwarf-v0"
DIFFICULTY_LEVEL = 1
MAIN_TAG = "demonstrate"
NUM_DEMOS = 10

# with open("difficulty_settings.json") as f:
#     difficulty_settings = json.load(f)
# LEVELGEN_CONFIG = difficulty_settings[ENV_NAME][str(DIFFICULTY_LEVEL)] | {
#     "max_goals": 1,
#     "p_key": 1.0,
#     "max_spiders": 1,
#     "p_spider": 1.0,
# }


def key_check(_, info) -> bool:
    history = info["History"]
    if len(history) == 0:
        return False
    last_event = history[-1]
    return (
        last_event["SourceObjectName"] == "drunk_dwarf"
        and last_event["DestinationObjectName"] == "key"
    )


if __name__ == "__main__":
    env = gym.make(ENV_NAME, player_observer_type=gd.ObserverType.VECTOR)
    env.enable_history(True)  # type: ignore

    reach_object_knowledge = [
        f"<(<($obj * #location) --> at> &/ <({ext('SELF')} * #location) --> ^goto>) =/> <$obj --> [reached]>>.",
    ]
    rel_pos_knowledge = [
        "<(<$obj --> [ahead]> &/ ^move_forwards) =/> <$obj --> [reached]>>.",
        # f"<(<$obj --> [leftward]> &/ ^move_forwards &/ ^rotate_left) =/> <$obj --> [ahead]>>.",
        # f"<(<$obj --> [rightward]> &/ ^move_forwards &/ ^rotate_right) =/> <$obj --> [ahead]>>.",
    ]
    background_knowledge = rel_pos_knowledge

    key_goal_sym = "GOT_KEY"
    reach_key = [f"<({ext('key')} --> [reached]) =/> {key_goal_sym}>."]
    door_goal_sym = "DOOR_OPENED"
    open_door = [
        f"<({key_goal_sym} &/ <{ext('door')} --> [reached]>) =/> {door_goal_sym}>."
    ]
    complete_goal_sym = "COMPLETE"
    complete_goal = [
        f"<({door_goal_sym} &/ <{ext('coffin_bed')} --> [reached]>) =/> {complete_goal_sym}>."
    ]

    agent = DrunkDwarfAgent(
        env,
        background_knowledge=background_knowledge,
    )

    KEY_GOAL = Goal(
        key_goal_sym, partial(object_reached, agent, "stumble", "key"), reach_key
    )
    DOOR_GOAL = Goal(
        door_goal_sym,
        lambda evst, info: agent.has_key
        and object_reached(agent, "stumble", "door", evst, info),
        open_door,
    )
    COMPLETE_GOAL = Goal(
        complete_goal_sym,
        partial(object_reached, agent, "stumble", "coffin_bed"),
        complete_goal,
    )
    REWARD_GOAL = Goal("GOT_REWARD", got_rewarded)

    goals = [
        KEY_GOAL,
        DOOR_GOAL,
        COMPLETE_GOAL,
        REWARD_GOAL,
    ]

    agent.setup_goals(COMPLETE_GOAL, goals)
    runner = Runner(agent)

    # DEMONSTRATE
    for _ in range(NUM_DEMOS):
        plan = [
            "^rotate_left",
            "^rotate_left",
            "^move_forwards",
            "^move_forwards",
            "^move_forwards",
            "^rotate_right",
            "^rotate_right",
            "^move_forwards",
            "^rotate_right",
            "^move_forwards",
            "^move_forwards",
            "^move_forwards",
            "^rotate_right",
            "^move_forwards",
        ]
        print("Demonstration: completing a level...")
        runner.demo_goal(plan)

    # Run the agent
    runner.run(
        NUM_EPISODES,
        MAX_ITERATIONS,
        log_tb=True,
        tb_comment_suffix=MAIN_TAG,
    )
