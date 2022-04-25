import logging
import os
import random
import socket
from pathlib import Path
from time import sleep

import griddly  # noqa
import gym
import pexpect
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


def get_output(process: pexpect.spawn) -> str:
    """Get output from NARS server"""
    # outlines = process.stdout.readlines()
    # output = "\n".join(outlines)
    # process.sendline("0")
    # HACK: fuck it, just use sock to send input
    send_input(sock, "0")
    process.expect(["done with 0 additional inference steps.", pexpect.EOF])
    # process.expect(pexpect.EOF)
    output = "\n".join(
        [s.strip().decode("utf-8") for s in process.before.split(b"\n")][2:-3]  # type: ignore
    )
    logger.debug(output)
    return output


def expect_output(
    sock: socket.socket,
    process: pexpect.spawn,
    targets: list[str],
    think_ticks: int = 10,
    patience: int = 10,
    goal_reentry: str = None,  # type: ignore TODO: migrate to Python 3.10 ASAP
) -> str:
    output = get_output(process)
    while not any(target in output for target in targets):
        if patience <= 0:
            ic("Patience has run out, returning None.")
            return None  # type: ignore
        patience -= 1

        # ic("Output is:", output)
        # ic("Waiting for:", targets)
        # sleep(1)
        if goal_reentry is not None:
            send_input(sock, goal_reentry)

        send_input(sock, str(think_ticks))
        output = get_output(process)
    ic("Got a valid operation.")
    return output


def narsify(observation):
    """Convert a Gym observation to NARS input"""
    return ",".join(str(x) for x in observation)


def loc(pos: tuple[int, int]) -> str:
    """Turn coordinates into a location string"""
    return f"loc_x{pos[0]}_y{pos[1]}"


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


def ext(s: str) -> str:
    """Just a helper to wrap strings in '{}'"""
    return "{" + s + "}"


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
    # process = subprocess.Popen(
    #     process_cmd,
    #     stdout=subprocess.PIPE,
    #     universal_newlines=True,
    # )
    process = pexpect.spawn(process_cmd[0], process_cmd[1:])
    sleep(3)  # wait for UDPNAR to make sure early commands don't get lost

    # setup NARS
    setup_nars(sock)
    goal_symbol = "AT_DOOR"
    persistent_goal = f"{goal_symbol}! :|:"
    # goal_achievement = f"<(<avatar --> [?1]> &/ <goal --> [?1]>) =/> {goal_symbol}>."
    # send_input(sock, goal_achievement)
    logger.info(get_output(process))

    env = gym.make("GDY-Zelda-v0", player_observer_type=gd.ObserverType.VECTOR)
    obs = env.reset()
    # For now we will just use `get_state` to get the state
    env_state = env.get_state()  # type: ignore

    # generate an explicit position goal
    make_goal(sock, env_state, goal_symbol)

    # send first observation (complete)
    send_observation(sock, process, env_state, complete=True)

    total_reward = 0.0
    episode_reward = 0.0
    num_episodes = 1
    tb_writer = SummaryWriter(comment="-nars-zelda")
    # TRAINING LOOP
    for s in range(1000):
        send_observation(
            sock, process, env_state
        )  # TODO: remove duplicate first observation

        # determine the action to take from NARS
        send_input(sock, persistent_goal)
        nars_output = expect_output(
            sock, process, list(NARS_OPERATIONS.keys()), goal_reentry=persistent_goal
        )

        if nars_output is None:
            nars_output = random.choice(list(NARS_OPERATIONS.keys()))
            send_input(sock, f"{nars_output}. :|:")

        obs, reward, done, info = env.step(to_gym_action(nars_output))
        episode_reward += reward
        # env.render()  # Renders the environment from the perspective of a single player
        env_state = env.get_state()  # type: ignore

        if goal_satisfied(env_state):
            ic("Goal achieved!")
            send_input(sock, f"{goal_symbol}. :|:")
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
