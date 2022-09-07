import argparse
import logging
import os
from functools import partial

import gym
import hyperstate
import neptune.new as neptune
from griddly import gd
from icecream import ic

from narca.agent import Runner
from narca.drunk_dwarf import DrunkDwarfAgent
from narca.nar import *
from narca.utils import *

# setup a logger for nars output
logging.basicConfig(filename="nars_drunk_dwarf.log", filemode="w", level=logging.DEBUG)
logger = logging.getLogger("nars")


ENV_NAME = "GDY-Drunk-Dwarf-v0"
MAIN_TAG = "main"


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
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None, help="Path to config file")
    parser.add_argument("--hps", nargs="+", help="Override hyperparameter value")
    parser.add_argument(
        "--log_neptune",
        action="store_true",
        default=False,
        help="Enable logging to Neptune",
    )
    parser.add_argument(
        "--log_tensorboard",
        action="store_true",
        default=False,
        help="Enable logging to TensorBoard",
    )
    args = parser.parse_args()
    config = hyperstate.load(Config, file=args.config, overrides=args.hps)
    logger.info("Run configuration: %s", config)

    neprun = (
        neptune.init(
            project=os.environ["NEPTUNE_PROJECT"],
            tags=[ENV_NAME, MAIN_TAG, f"difficulty:{config.difficulty_level}"],
        )
        if args.log_neptune
        else None
    )

    env = gym.make(
        ENV_NAME,
        player_observer_type=gd.ObserverType.VECTOR,
        level=config.difficulty_level - 1,
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
        think_ticks=config.agent.think_ticks,
        view_radius=config.agent.view_radius,
        background_knowledge=background_knowledge,
        motor_babbling=config.nars.motor_babbling,
        decision_threshold=config.nars.decision_threshold,
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

    callbacks = []
    if neprun is not None:
        neprun["parameters"] = {
            "goals": [g.symbol for g in goals],
            "think_ticks": config.agent.think_ticks,
            "view_radius": config.agent.view_radius,
            "num_episodes": config.num_episodes,
            "max_iterations": config.max_steps,
            "motor_babbling": config.nars.motor_babbling,
            "decision_threshold": config.nars.decision_threshold,
        }

        def nep_ep_callback(run_info: dict):
            neprun["train/episode_reward"].log(run_info["episode_reward"])
            neprun["train/total_reward"].log(run_info["total_reward"])

        def nep_run_callback(run_info: dict):
            neprun["train/avg_ep_reward"] = run_info["avg_ep_reward"]
            neprun["train/avg_completion_rate"] = run_info["avg_completion_rate"]
            neprun["train/completed_rate"] = run_info["completed_rate"]

        callbacks.append(("on_episode_end", nep_ep_callback))
        callbacks.append(("on_run_end", nep_run_callback))

    # Run the agent
    runner.run(
        config.num_episodes,
        config.max_steps,
        log_tb=args.log_tensorboard,
        tb_comment_suffix=f"drunk_dwarf-{MAIN_TAG}:{config.difficulty_level}",
        callbacks=callbacks,
    )
    if neprun is not None:
        neprun.stop()
