import logging
import os
import random
from functools import partial
from pathlib import Path
from time import sleep
from typing import Optional

import griddly  # noqa
import gym
import pexpect
from griddly import gd
from icecream import ic
from tensorboardX import SummaryWriter

from narca.astar import *
from narca.nar import *
from narca.narsese import *
from narca.utils import *
from narca.zelda import ZeldaAgent

# setup a logger for nars output
logging.basicConfig(filename="nars_zelda.log", filemode="w", level=logging.DEBUG)
logger = logging.getLogger("nars")

NUM_EPISODES = 50
MAX_ITERATIONS = 100


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
    env = gym.make("GDY-Zelda-v0", player_observer_type=gd.ObserverType.VECTOR)
    env.enable_history(True)  # type: ignore

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
    total_reward = 0.0
    episode_reward = 0.0
    tb_writer = SummaryWriter(comment="-nars-zelda")
    done = False
    # TRAINING LOOP
    for episode in range(NUM_EPISODES):
        agent.reset()

        for i in range(MAX_ITERATIONS):
            agent.observe(complete=i % 10 == 0)

            obs, reward, cumr, done, info = agent.step()
            episode_reward += cumr

            env_state = agent.env.get_state()  # type: ignore
            env_state["reward"] = reward

            satisfied_goals = [g.satisfied(env_state, info) for g in goals]
            for g, sat in zip(goals, satisfied_goals):
                if sat:
                    print(f"{g.symbol} satisfied.")
                    send_input(agent.process, nal_now(g.symbol))
                    get_raw_output(agent.process)

                    if g.symbol == key_goal_sym:
                        agent.has_key = True

            env.render(observer="global")  # type: ignore # Renders the entire environment
            # sleep(1)

            if done:
                break

        print(f"Episode {episode+1} finished with reward {episode_reward}.")
        total_reward += episode_reward
        tb_writer.add_scalar("train/episode_reward", episode_reward, episode)
        tb_writer.add_scalar("train/total_reward", total_reward, episode)
        episode_reward = 0.0
        send_input(agent.process, nal_now("RESET"))

    # tb_writer.add_scalar("train/episode_reward", episode_reward, num_episodes)
    print(f"Average total reward per episode: {total_reward / NUM_EPISODES}.")
    env.close()  # Call explicitly to avoid exception on quit
