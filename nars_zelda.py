import logging
import os
import random
import socket
from pathlib import Path
from time import sleep
from typing import Optional

import griddly  # noqa
import gym
import pexpect
from griddly import gd
from icecream import ic
from tensorboardX import SummaryWriter

from astar import *
from udpnar import *
from utils import *

# setup a logger for nars output
logging.basicConfig(filename="nars_zelda.log", filemode="w", level=logging.DEBUG)
logger = logging.getLogger("nars")

NARS_PATH = Path(os.environ["NARS_HOME"])
NARS_OPERATIONS = {
    "^goto": 1,
    "^attack": 2,
    "^rotate_left": 3,
    "^move_forwards": 4,
    "^rotate_right": 5,
    "^move_backwards": 6,
}


def narsify_from_state(env_state: dict):
    """Produce NARS statements from environment semantic state"""
    # TODO: determine if a single `wall` entity would be preferable
    # TODO: handle the case where every type of an object can be a multiple
    walls = [
        (i, obj["Location"])
        for i, obj in enumerate(env_state["Objects"])
        if obj["Name"] == "wall"
    ]
    object_beliefs = [
        f"<({ext(obj['Name'])} * {loc(obj['Location'])}) --> at>. :|:"
        for obj in env_state["Objects"]
        if obj["Name"] != "wall"
    ]
    wall_beliefs = [
        f"<({ext('wall' + str(i))} * {loc(pos)}) --> at>. :|:" for i, pos in walls
    ]
    return object_beliefs + wall_beliefs


def to_gym_action(nars_output: str, env_state: dict) -> list[int]:
    """Convert NARS output to a Gym action"""
    to_griddly_id = {
        "^rotate_left": 1,
        "^move_forwards": 2,
        "^rotate_right": 3,
        "^move_backwards": 4,
    }

    matches = [target in nars_output for target in NARS_OPERATIONS.keys()]
    matches = zip(NARS_OPERATIONS.keys(), matches)
    for op, match in matches:
        if match:
            if op == "^goto":
                avatar = next(
                    obj for obj in env_state["Objects"] if obj["Name"] == "avatar"
                )

                split_on_args = nars_output.split("args")
                args = split_on_args[1][1:-1].split("*")

                path_ops = pathfind(avatar["Location"], pos(args[1]))
                return [0, to_griddly_id[path_ops[0]]]
            if op == "^attack":
                return [1, 1]
            return [0, to_griddly_id[op]]

    return [0, 0]  # noop


def goal_reached(env_state: dict) -> bool:
    """Check if the goal has been reached"""
    try:
        avatar = next(obj for obj in env_state["Objects"] if obj["Name"] == "avatar")
        goal = next(obj for obj in env_state["Objects"] if obj["Name"] == "goal")
    except StopIteration:
        ic("No avatar or goal found. Goal unsatisfiable.")
        return False
    return avatar["Location"] == goal["Location"]


def got_rewarded(env_state: dict) -> bool:
    return env_state["reward"] > 0


def make_goal(sock: socket.socket, env_state: dict, goal_symbol: str) -> None:
    """Make an explicit goal using the position of an object"""
    goal = next(obj for obj in env_state["Objects"] if obj["Name"] == "goal")
    goal_loc = loc(goal["Location"])
    goal_achievement = f"<<({ext('avatar')} * {goal_loc}) --> at> =/> {goal_symbol}>."
    send_input(sock, goal_achievement)


def send_observation(
    sock: socket.socket, process: pexpect.spawn, env_state: dict, complete=False
) -> None:
    """Send observation to NARS

    Args:
        sock: socket to send observation to
        process: NARS process
        env_state: environment state
        complete: whether to send the complete observation (include wall beliefs)
    """
    # send the observation to NARS
    # send_input(sock, narsify(obs))
    state_narsese = narsify_from_state(env_state)
    # TODO: I am not sure if I can only send single lines or not.
    statements = [st for st in state_narsese if "wall" not in st or complete]
    for statement in statements:
        send_input(sock, statement)
        get_output(process)
    # send_input(sock, narsify_from_state(env_state))


def demo_reach_loc(
    symbol: str, agent_pos: tuple[int, int], pos: tuple[int, int]
) -> None:
    """Demonstrate reaching a location"""
    actions_to_take = pathfind(agent_pos, pos)
    for action in actions_to_take:
        send_input(SOCKET, f"{action}. :|:")
        env_state = env.get_state()  # type: ignore
        obs, _, done, _ = env.step(to_gym_action(action, env_state))
        send_observation(SOCKET, process, env.get_state())  # type: ignore

        env.render(observer="global")  # type: ignore
        sleep(1)
        if done:
            env.reset()
            demo_reach_loc(symbol, agent_pos, pos)
    send_input(SOCKET, f"{symbol}. :|:")


def make_loc_goal(sock, pos, goal_symbol):
    """Make a goal for a location"""
    goal_loc = loc(pos)
    goal_achievement = f"<<({ext('avatar')} * {goal_loc}) --> at> =/> {goal_symbol}>."
    send_input(sock, goal_achievement)


# def check_goals(goals: list[Goal], env_state: dict) -> list[bool]:
#     satisfied = [goal.satisfied(env_state) for goal in goals]
#     for g, sat in zip(goals, satisfied):
#         if sat:
#             send_input(SOCKET, f"{g.symbol}. :|:")
#             get_output(process)
#     return satisfied

if __name__ == "__main__":
    reach_object_knowledge = [
        f"<(<($obj * #location) --> at> &/ <({ext('avatar')} * #location) --> ^goto>) =/> <$obj --> [reached]>>."
    ]
    DOOR_GOAL = Goal("AT_DOOR", goal_reached, reach_object_knowledge)
    REWARD_GOAL = Goal("GOT_REWARD", got_rewarded)

    goals = [
        DOOR_GOAL,
        REWARD_GOAL,
    ]
    persistent_goal = DOOR_GOAL

    # ./NAR UDPNAR IP PORT  timestep(ns per cycle) printDerivations
    process_cmd = [
        (NARS_PATH / "NAR").as_posix(),
        "UDPNAR",
        IP,
        str(PORT),
        "1000000",
        "true",
    ]
    # ./NAR shell
    # process_cmd = [
    #     (NARS_PATH / "NAR").as_posix(),
    #     "shell",
    # ]
    # process = subprocess.Popen(
    #     process_cmd,
    #     stdout=subprocess.PIPE,
    #     universal_newlines=True,
    # )
    process = pexpect.spawn(process_cmd[0], process_cmd[1:])
    sleep(3)  # wait for UDPNAR to make sure early commands don't get lost

    # setup NARS
    setup_nars(SOCKET, NARS_OPERATIONS)
    logger.info(get_output(process))

    env = gym.make("GDY-Zelda-v0", player_observer_type=gd.ObserverType.VECTOR)
    obs = env.reset()
    # For now we will just use `get_state` to get the state
    env_state = env.get_state()  # type: ignore

    # send goal knowledge
    if persistent_goal.knowledge is not None:
        for belief in persistent_goal.knowledge:
            send_input(SOCKET, belief)
    # send first observation (complete)
    send_observation(SOCKET, process, env_state, complete=True)

    # first, show how to reach a location
    # reach_loc1_sym = "REACH_LOC1"
    # reach_loc2_sym = "REACH_LOC2"
    # reach_loc3_sym = "REACH_LOC3"
    # reach_loc1_pos = (3, 2)
    # reach_loc2_pos = (8, 5)
    # reach_loc3_pos = (2, 5)

    # av = next(obj for obj in env_state["Objects"] if obj["Name"] == "avatar")
    # agent_pos = av["Location"]

    # make_loc_goal(sock, reach_loc1_pos, reach_loc1_sym)
    # demo_reach_loc(reach_loc1_sym, agent_pos, reach_loc1_pos)
    # make_loc_goal(sock, reach_loc2_pos, reach_loc2_sym)
    # demo_reach_loc(reach_loc2_sym, agent_pos, reach_loc2_pos)
    # demo_reach_loc(reach_loc3_sym, agent_pos, reach_loc3_pos)

    total_reward = 0.0
    episode_reward = 0.0
    num_episodes = 1
    tb_writer = SummaryWriter(comment="-nars-zelda")
    # TRAINING LOOP
    for s in range(100):
        send_observation(
            SOCKET, process, env_state
        )  # TODO: remove duplicate first observation

        # determine the action to take from NARS
        send_input(SOCKET, nal_demand(persistent_goal.symbol))
        nars_output = expect_output(
            SOCKET, process, list(NARS_OPERATIONS.keys()), goal_reentry=persistent_goal
        )

        if nars_output is None:
            nars_output = random.choice(list(NARS_OPERATIONS.keys()))
            send_input(SOCKET, nal_now(nars_output))

        obs, reward, done, info = env.step(to_gym_action(nars_output, env.get_state()))  # type: ignore
        episode_reward += reward
        env_state = env.get_state()  # type: ignore
        env_state["reward"] = reward

        satisfied_goals = [g.satisfied(env_state) for g in goals]
        for g, sat in zip(goals, satisfied_goals):
            if sat:
                ic(f"{g.symbol} satisfied.")
                send_input(SOCKET, nal_now(g.symbol))
                get_output(process)

        env.render(observer="global")  # type: ignore # Renders the entire environment
        sleep(1)

        if done:
            total_reward += episode_reward
            tb_writer.add_scalar("train/episode_reward", episode_reward, num_episodes)
            env.reset()
            num_episodes += 1
            episode_reward = 0.0

    # tb_writer.add_scalar("train/episode_reward", episode_reward, num_episodes)
    print(f"Average total reward per episode: {total_reward / num_episodes}.")
    env.close()  # Call explicitly to avoid exception on quit
