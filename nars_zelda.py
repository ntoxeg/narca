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

from narca.astar import *
from narca.udpnar import *
from narca.utils import *

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


def to_gym_actions(nars_output: dict[str, Any], env_state: dict) -> list[list[int]]:
    """Convert NARS output to a Gym action

    Now includes simple planning. To be refactored.
    """
    to_griddly_id = {
        "^rotate_left": 1,
        "^move_forwards": 2,
        "^rotate_right": 3,
        "^move_backwards": 4,
    }

    if len(nars_output["executions"]) < 1:
        raise ValueError("No executions found.")
    for exe in nars_output["executions"]:  # for now we will process the first execution
        ic(exe)
        op = exe["operator"]
        args = exe["arguments"]
        if op == "^goto":
            avatar = next(
                obj for obj in env_state["Objects"] if obj["Name"] == "avatar"
            )

            if len(args) < 2:
                ic("Not enough arguments received, assuming random coordinates.")
                args = [
                    "{avatar}",
                    loc((random.randint(0, 6), random.randint(0, 6))),
                ]

            ic("Executing ^goto with args:", args)
            path_ops = pathfind(avatar["Location"], pos(args[1]))
            rel_ops = abs_to_rel(avatar, path_ops[0])
            return [[0, to_griddly_id[op]] for op in rel_ops]
        if op == "^attack":
            return [[1, 1]]
        return [[0, to_griddly_id[op]]]

    return [[0, 0]]  # noop


def abs_to_rel(avatar, op):
    orient_to_num = {
        "UP": 0,
        "RIGHT": 1,
        "DOWN": 2,
        "LEFT": 3,
    }
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
        get_raw_output(process)
    # send_input(sock, narsify_from_state(env_state))


# def demo_reach_loc(
#     symbol: str, agent_pos: tuple[int, int], pos: tuple[int, int]
# ) -> None:
#     """Demonstrate reaching a location"""
#     actions_to_take = pathfind(agent_pos, pos)
#     for action in actions_to_take:
#         send_input(SOCKET, f"{action}. :|:")
#         env_state = env.get_state()  # type: ignore
#         obs, _, done, _ = env.step(to_gym_actions(action, env_state))
#         send_observation(SOCKET, process, env.get_state())  # type: ignore

#         env.render(observer="global")  # type: ignore
#         sleep(1)
#         if done:
#             env.reset()
#             demo_reach_loc(symbol, agent_pos, pos)
#     send_input(SOCKET, f"{symbol}. :|:")


def make_loc_goal(sock, pos, goal_symbol):
    """Make a goal for a location"""
    goal_loc = loc(pos)
    goal_achievement = f"<<({ext('avatar')} * {goal_loc}) --> at> =/> {goal_symbol}>."
    send_input(sock, goal_achievement)


if __name__ == "__main__":
    door_goal_sym = "AT_DOOR"
    reach_object_knowledge = [
        f"<(<($obj * #location) --> at> &/ <({ext('avatar')} * #location) --> ^goto>) =/> <$obj --> [reached]>>.",
        f"<<{ext('goal')} --> [reached]> =/> {door_goal_sym}>.",
    ]
    DOOR_GOAL = Goal(door_goal_sym, goal_reached, reach_object_knowledge)
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
    logger.info(get_raw_output(process))

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
        # send_input(SOCKET, "10")
        nars_output = expect_output(
            SOCKET,
            process,
            list(NARS_OPERATIONS.keys()),
            goal_reentry=persistent_goal,
            think_ticks=20,
        )

        if nars_output is None:
            nars_output = random.choice(
                list(NARS_OPERATIONS.keys())
            )  # FIXME: generate proper random execution
            send_input(SOCKET, nal_now(nars_output))  # FIXME: handle arguments

        reward = 0.0
        done = False
        actions = to_gym_actions(nars_output, env.get_state())  # type: ignore
        for action in actions:
            obs, reward, done, info = env.step(action)
            episode_reward += reward

        env_state = env.get_state()  # type: ignore
        env_state["reward"] = reward

        satisfied_goals = [g.satisfied(env_state) for g in goals]
        for g, sat in zip(goals, satisfied_goals):
            if sat:
                ic(f"{g.symbol} satisfied.")
                send_input(SOCKET, nal_now(g.symbol))
                get_raw_output(process)

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
