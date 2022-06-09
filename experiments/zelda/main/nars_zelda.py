import logging
import os
from functools import partial

import griddly  # noqa
import gym
import neptune.new as neptune
from griddly import gd
from icecream import ic

from narca.agent import Runner
from narca.nar import *
from narca.utils import *
from narca.zelda import ZeldaAgent

# setup a logger for nars output
logging.basicConfig(filename="nars_zelda.log", filemode="w", level=logging.DEBUG)
logger = logging.getLogger("nars")

NUM_EPISODES = 50
MAX_ITERATIONS = 100
ENV_NAME = "GDY-Zelda-v0"
MAIN_TAG = "main"
DIFFICULTY_LEVEL = 1

THINK_TICKS = 5


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
    try:
        neprun = neptune.init(
            project=os.environ["NEPTUNE_PROJECT"],
            tags=[ENV_NAME, MAIN_TAG, f"difficulty:{DIFFICULTY_LEVEL}"],
        )
    except KeyError:
        neprun = None

    env = gym.make(
        ENV_NAME,
        player_observer_type=gd.ObserverType.VECTOR,
        level=DIFFICULTY_LEVEL - 1,
    )
    env.enable_history(True)  # type: ignore

    reach_object_knowledge = [
        f"<(<($obj * #location) --> at> &/ <({ext('SELF')} * #location) --> ^goto>) =/> <$obj --> [reached]>>.",
    ]
    rel_pos_knowledge = [
        f"<(<$obj --> [ahead]> &/ ^move_forwards) =/> <$obj --> [reached]>>.",
        # f"<(<$obj --> [leftward]> &/ ^move_forwards &/ ^rotate_left) =/> <$obj --> [ahead]>>.",
        # f"<(<$obj --> [rightward]> &/ ^move_forwards &/ ^rotate_right) =/> <$obj --> [ahead]>>.",
    ]
    background_knowledge = rel_pos_knowledge

    key_goal_sym = "GOT_KEY"
    reach_key = [f"<({ext('key')} --> [reached]) =/> {key_goal_sym}>."]
    # door_goal_sym = "AT_DOOR"
    # reach_door = [f"<<{ext('goal')} --> [reached]> =/> {door_goal_sym}>."]
    complete_goal_sym = "COMPLETE"
    complete_goal = [
        f"<({key_goal_sym} &/ <{ext('goal')} --> [reached]>) =/> {complete_goal_sym}>."
    ]

    agent = ZeldaAgent(
        env,
        think_ticks=THINK_TICKS,
        background_knowledge=background_knowledge,
    )

    KEY_GOAL = Goal(
        key_goal_sym, partial(object_reached, agent, "move", "key"), reach_key
    )
    # DOOR_GOAL = Goal(door_goal_sym, partial(object_reached, "goal"), reach_door)
    COMPLETE_GOAL = Goal(
        complete_goal_sym,
        lambda evst, info: agent.has_key
        and object_reached(agent, "move", "goal", evst, info),
        complete_goal,
    )
    REWARD_GOAL = Goal("GOT_REWARD", got_rewarded)

    goals = [
        KEY_GOAL,
        # DOOR_GOAL,
        COMPLETE_GOAL,
        REWARD_GOAL,
    ]

    agent.setup_goals(COMPLETE_GOAL, goals)
    runner = Runner(agent)

    callbacks = []
    if neprun is not None:
        neprun["parameters"] = {
            "goals": [g.symbol for g in goals],
            "think_ticks": THINK_TICKS,
        }

        def nep_ep_callback(run_info: dict):
            neprun["train/episode_reward"].log(run_info["episode_reward"])
            neprun["train/total_reward"].log(run_info["total_reward"])

        def nep_run_callback(run_info: dict):
            neprun["train/avg_ep_reward"] = run_info["avg_ep_reward"]
            neprun[
                "train/avg_completion_rate"
            ] = f"{run_info['avg_completion_rate']*100:.0f}%"
            neprun["train/completed_rate"] = f"{run_info['completed_rate']*100:.0f}%"

        callbacks.append(("on_episode_end", nep_ep_callback))
        callbacks.append(("on_run_end", nep_run_callback))

    # Run the agent
    runner.run(
        NUM_EPISODES,
        MAX_ITERATIONS,
        log_tb=True,
        tb_comment_suffix=f"zelda-{MAIN_TAG}",
        callbacks=callbacks,
    )
    if neprun is not None:
        neprun.stop()
