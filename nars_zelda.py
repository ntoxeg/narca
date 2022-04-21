import logging
import os
import random
import socket
import subprocess
from pathlib import Path
from time import sleep

import griddly  # noqa
import gym
from griddly import gd
from icecream import ic
from tensorboardX import SummaryWriter

# setup a logger for nars output
logging.basicConfig(filename="nars_zelda.log", filemode="w", level=logging.DEBUG)
logger = logging.getLogger("nars")

NARS_PATH = Path(os.environ["NARS_HOME"])
IP = "127.0.0.1"
PORT = 50000

NARS_OPERATIONS = {
    "^rotate_left": 1,
    "^move_forwards": 2,
    "^rotate_right": 3,
    "^move_backwards": 4,
}


def send_input(socket: socket.socket, input_: str) -> None:
    """Send input to NARS server"""
    socket.sendto((input_ + "\0").encode(), (IP, PORT))


# def send_input_process(process: subprocess.Popen, input_: str):
#     """Send input to NARS process"""
#     stdin = process.stdin
#     stdin.write(input_ + "\n")


def get_output(process: subprocess.Popen) -> str:
    """Get output from NARS server"""
    output = process.stdout.readline().strip()
    logger.debug(output)
    return output


def expect_output(
    sock: socket.socket,
    process: subprocess.Popen,
    targets: list[str],
    patience: int = 10,
    goal_reentry: str = None,
) -> str:
    output = get_output(process)
    while not any(target in output for target in targets):
        if patience <= 0:
            ic("Patience has run out, returning None.")
            return None
        patience -= 1
        # ic("Output is:", output)
        # ic("Waiting for:", targets)
        # sleep(1)
        if goal_reentry is not None:
            send_input(sock, goal_reentry)
        else:
            send_input(sock, "0")
        output = get_output(process)
    ic("Got a valid operation.")
    return output


def narsify(observation):
    """Convert a Gym observation to NARS input"""
    return ",".join(str(x) for x in observation)


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
        f"<{obj['Name']} --> [loc_x{obj['Location'][0]}_y{obj['Location'][1]}]>. :|:"
        for obj in env_state["Objects"]
        if obj["Name"] != "wall"
    ]
    wall_beliefs = [f"<wall{i} --> [loc_x{pos[0]}_y{pos[1]}]>. :|:" for i, pos in walls]
    return object_beliefs + wall_beliefs


def to_gym_action(nars_output: str) -> list[int]:
    """Convert NARS output to a Gym action"""
    matches = [target in nars_output for target in NARS_OPERATIONS.keys()]
    action_id = list(NARS_OPERATIONS.values())[matches.index(True)]
    action = [0, action_id]  # 0 is for "move"

    return action


def setup_nars_ops(socket: socket.socket):
    """Setup NARS operations"""
    for op in NARS_OPERATIONS:
        send_input(socket, f"*setopname {NARS_OPERATIONS[op]} {op}")
    send_input(socket, f"*babblingops={len(NARS_OPERATIONS)}")


# def setup_nars_ops_process(process: subprocess.Popen):
#     """Setup NARS operations"""
#     stdin = process.stdin
#     for op in NARS_OPERATIONS:
#         stdin.write(f"*setopname {NARS_OPERATIONS[op]} {op}\n")


def setup_nars(socket: socket.socket):
    """Send NARS settings"""
    send_input(socket, "*reset")
    setup_nars_ops(socket)
    send_input(socket, "*motorbabbling=0.3")
    # send_input(socket, "*volume=0")


# def setup_nars_process(process: subprocess.Popen):
#     """Setup NARS process"""
#     stdin = process.stdin
#     stdin.write("*reset\n")
#     setup_nars_ops_process(process)
#     stdin.write("*motorbabbling=0.3\n")


def goal_satisfied(env_state: dict) -> bool:
    """Check if the goal has been reached"""
    try:
        avatar = next(obj for obj in env_state["Objects"] if obj["Name"] == "avatar")
        goal = next(obj for obj in env_state["Objects"] if obj["Name"] == "goal")
    except StopIteration:
        ic("No avatar or goal found. Goal unsatisfiable.")
        return False
    return avatar["Location"] == goal["Location"]


if __name__ == "__main__":
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

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
    process = subprocess.Popen(
        process_cmd,
        stdout=subprocess.PIPE,
        universal_newlines=True,
    )
    sleep(3)  # wait for UDPNAR to make sure early commands don't get lost

    # setup NARS
    setup_nars(sock)
    goal_symbol = "AT_DOOR"
    persistent_goal = f"{goal_symbol}! :|:"
    goal_achievement = f"<(<avatar --> [?1]> &/ <goal --> [?1]>) =/> {goal_symbol}>."
    send_input(sock, goal_achievement)
    logger.info(get_output(process))

    env = gym.make("GDY-Zelda-v0", player_observer_type=gd.ObserverType.VECTOR)
    obs = env.reset()
    # For now we will just use `get_state` to get the state
    env_state = env.get_state()

    total_reward = 0.0
    episode_reward = 0.0
    num_episodes = 1
    tb_writer = SummaryWriter(comment="-nars-zelda")
    for s in range(1000):
        # send the observation to NARS
        # send_input(sock, narsify(obs))
        state_narsese = narsify_from_state(env_state)
        # TODO: I am not sure if I can only send single lines or not.
        for statement in state_narsese:
            send_input(sock, statement)
            get_output(process)
        # send_input(sock, narsify_from_state(env_state))

        send_input(sock, persistent_goal)
        get_output(process)

        # determine the action to take from NARS
        nars_output = expect_output(
            sock, process, NARS_OPERATIONS.keys(), goal_reentry=persistent_goal
        )

        if nars_output is None:
            nars_output = random.choice(list(NARS_OPERATIONS.keys()))
            send_input(sock, f"{nars_output}. :|:")

        obs, reward, done, info = env.step(to_gym_action(nars_output))
        episode_reward += reward
        # env.render()  # Renders the environment from the perspective of a single player
        env_state = env.get_state()

        if goal_satisfied(env_state):
            ic("Goal achieved!")
            send_input(sock, f"{goal_symbol}. :|:")
            get_output(process)

        env.render(observer="global")  # Renders the entire environment
        sleep(1)

        if done:
            total_reward += episode_reward
            tb_writer.add_scalar("train/episode_reward", episode_reward, num_episodes)
            env.reset()
            num_episodes += 1
            episode_reward = 0.0

    print(f"Average total reward per episode: {total_reward / num_episodes}.")
    env.close()  # Call explicitly to avoid exception on quit
