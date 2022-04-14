import os
import subprocess
from pathlib import Path
from socket import socket

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


def send_input(socket: socket, input_: str) -> None:
    """Send input to NARS server"""
    socket.sendto((input_ + "\0").encode(), ("127.0.0.1", 50000))


def get_output(process: subprocess.Popen) -> str:
    """Get output from NARS server"""
    return process.stdout.readline().decode().strip()


def narsify(observation):
    """Convert a Gym observation to NARS input"""
    return ",".join(str(x) for x in observation)


def to_gym_action(nars_output):
    """Convert NARS output to a Gym action"""
    return NARS_OPERATIONS[nars_output]


if __name__ == "__main__":
    sock = socket(socket.AF_INET, socket.SOCK_DGRAM)

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

    env = gym.make("GDY-Zelda-v0", player_observer_type=gd.ObserverType.VECTOR)
    obs, _ = env.reset()

    # Replace with your own control algorithm!
    for s in range(100):
        # send the observation to NARS
        send_input(sock, narsify(obs))
        # determine an action to take from NARS
        nars_output = get_output(process)

        obs, reward, done, info = env.step(to_gym_action(nars_output))
        # env.render()  # Renders the environment from the perspective of a single player

        env.render(observer="global")  # Renders the entire environment

        if done:
            env.reset()
