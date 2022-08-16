import logging
import os
from functools import partial

import griddly  # noqa
import gym
import neptune.new as neptune
from griddly import GymWrapperFactory, gd
from icecream import ic

from narca.agent import Runner
from narca.drunk_dwarf import DrunkDwarfAgent
from narca.nar import *
from narca.utils import *

# setup a logger for nars output
logging.basicConfig(filename="nars_drunk_dwarf.log", filemode="w", level=logging.DEBUG)
logger = logging.getLogger("nars")

NUM_EPISODES = 50
MAX_ITERATIONS = 100
ENV_NAME_STEM = "DrunkDwarfExtended"
ENV_NAME = f"GDY-{ENV_NAME_STEM}-v0"
MAIN_TAG = "extended_curriculum"
DIFFICULTY_LEVEL = 5

THINK_TICKS = 3
VIEW_RADIUS = 1
MOTOR_BABBLING = 0.2
DECISION_THRESHOLD = 0.55


def key_check(_, info) -> bool:
    history = info["History"]
    if len(history) == 0:
        return False
    last_event = history[-1]
    return (
        last_event["SourceObjectName"] == "drunk_dwarf"
        and last_event["DestinationObjectName"] == "key"
    )


wrapper = GymWrapperFactory()
current_dir = os.path.dirname(os.path.abspath(__file__))
wrapper.build_gym_from_yaml(ENV_NAME_STEM, f"{current_dir}/drunk_dwarf_extended.yml")

if __name__ == "__main__":
    try:
        neprun = neptune.init(
            project=os.environ["NEPTUNE_PROJECT"],
            tags=[ENV_NAME, MAIN_TAG, f"difficulty:{DIFFICULTY_LEVEL}"],
        )
    except KeyError:
        neprun = None

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
    complete_goal_sym = "COMPLETE"
    complete_goal = [f"<<{ext('coffin_bed')} --> [reached]> =/> {complete_goal_sym}>."]

    # TODO: add the option to defer environment initialization.
    env = gym.make(
        ENV_NAME,
        player_observer_type=gd.ObserverType.VECTOR,
        level=0,
        new_step_api=True,
    )
    env.enable_history(True)  # type: ignore

    agent = DrunkDwarfAgent(
        env,
        think_ticks=THINK_TICKS,
        view_radius=VIEW_RADIUS,
        background_knowledge=background_knowledge,
        motor_babbling=MOTOR_BABBLING,
        decision_threshold=DECISION_THRESHOLD,
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
            "think_ticks": THINK_TICKS,
            "view_radius": agent.view_radius,
            "num_episodes": NUM_EPISODES,
            "max_iterations": MAX_ITERATIONS,
            "motor_babbling": MOTOR_BABBLING,
            "decision_threshold": DECISION_THRESHOLD,
        }

        def nep_ep_callback(run_info: dict):
            neprun["train/episode_reward"].log(run_info["episode_reward"])
            neprun["train/total_reward"].log(run_info["total_reward"])

        def nep_level_callback(run_info: dict):
            neprun[f"train/{run_info['level']}/avg_ep_reward"] = run_info[
                f"{run_info['level']}/avg_ep_reward"
            ]
            neprun[f"train/{run_info['level']}/avg_completion_rate"] = run_info[
                f"{run_info['level']}/avg_completion_rate"
            ]
            neprun[f"train/{run_info['level']}/completed_rate"] = run_info[
                f"{run_info['level']}/completed_rate"
            ]

        def nep_run_callback(run_info: dict):
            neprun["train/avg_ep_reward"] = run_info["avg_ep_reward"]
            neprun["train/avg_completion_rate"] = run_info["avg_completion_rate"]
            neprun["train/completed_rate"] = run_info["completed_rate"]

        callbacks.append(("on_episode_end", nep_ep_callback))
        callbacks.append(("on_level_end", nep_level_callback))
        callbacks.append(("on_run_end", nep_run_callback))

    # Run the agent
    runner.run_curriculum(
        ENV_NAME,
        DIFFICULTY_LEVEL,
        NUM_EPISODES,
        MAX_ITERATIONS,
        log_tb=True,
        tb_comment_suffix=f"drunk_dwarf-{MAIN_TAG}:{DIFFICULTY_LEVEL}",
        callbacks=callbacks,
    )
    if neprun is not None:
        neprun.stop()
