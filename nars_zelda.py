import os
import socket
import subprocess
from pathlib import Path

import griddly  # noqa
import gym
from griddly import gd

NARS_PATH = Path(os.environ["NARS_HOME"])
IP = "127.0.0.1"
PORT = 50000

NARS_OPERATIONS = {
    "rotate_left": 1,
    "move_forwards": 2,
    "rotate_right": 3,
    "move_backwards": 4,
}


def send_input(socket: socket.socket, input_: str) -> None:
    """Send input to NARS server"""
    socket.sendto((input_ + "\0").encode(), ("127.0.0.1", 50000))


def get_output(process: subprocess.Popen) -> str:
    """Get output from NARS server"""
    return process.stdout.readline().strip()


def expect_output(process: subprocess.Popen, targets: list[str]) -> str:
    output = get_output(process)
    while not any(target in output for target in targets):
        output = get_output(process)
    return output


def narsify(observation):
    """Convert a Gym observation to NARS input"""
    return ",".join(str(x) for x in observation)


def narsify_from_state(env_state: dict):
    """Produce NARS statements from environment semantic state"""
    object_beliefs = [
        f"<{obj['Name']} --> [loc_x{obj['Location'][0]}_y{obj['Location'][1]}]>. :|:"
        for obj in env_state["Objects"]
    ]
    return object_beliefs


def to_gym_action(nars_output):
    """Convert NARS output to a Gym action"""
    matches = [target in nars_output for target in NARS_OPERATIONS.keys()]
    action_id = NARS_OPERATIONS.values()[matches.index(True)]
    action = [0, action_id]  # 0 is for "move"

    return action


def setup_nars_ops(socket: socket.socket):
    """Setup NARS operations"""
    for op in NARS_OPERATIONS:
        send_input(socket, f"*setopname {NARS_OPERATIONS[op]} ^{op}")


def setup_nars(socket: socket.socket):
    """Send NARS settings"""
    setup_nars_ops(socket)
    send_input(socket, "*motorbabbling=0.1")
    # send_input(socket, "*volume=0")


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
    process = subprocess.Popen(
        process_cmd, stdout=subprocess.PIPE, universal_newlines=True
    )

    # setup NARS
    setup_nars(sock)

    env = gym.make("GDY-Zelda-v0", player_observer_type=gd.ObserverType.VECTOR)
    obs = env.reset()
    env_state = env.get_state()

    # For now we will just use `get_state` to get the state
    for s in range(100):
        # send the observation to NARS
        # send_input(sock, narsify(obs))
        state_narsese = narsify_from_state(env_state)
        # TODO: I am not sure if I can only send single lines or not.
        for statement in state_narsese:
            send_input(sock, statement)
        # send_input(sock, narsify_from_state(env_state))

        # determine an action to take from NARS
        nars_output = expect_output(process, NARS_OPERATIONS.keys())

        obs, reward, done, info = env.step(to_gym_action(nars_output))
        # env.render()  # Renders the environment from the perspective of a single player
        env_state = env.get_state()

        env.render(observer="global")  # Renders the entire environment

        if done:
            env.reset()
