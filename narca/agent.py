import subprocess
from abc import ABCMeta, abstractmethod
from typing import Optional

import gym

from .nar import send_input, setup_nars
from .narsese import Goal
from .utils import NARS_PATH


class Agent(metaclass=ABCMeta):
    AVATAR_LABEL = "avatar"

    def __init__(self, env: gym.Env):
        self.env = env

    def reset(self, level_string: Optional[str] = None):
        if level_string is None:
            self.env.reset()
        else:
            self.env.reset(level_string=level_string)  # type: ignore

    @abstractmethod
    def plan(self) -> list[list[int]]:

        return [[]]


class NarsAgent(Agent, metaclass=ABCMeta):
    def __init__(
        self,
        env: gym.Env,
        ops: list[str],
        goal: Goal,
        think_ticks: int = 5,
        background_knowledge: Optional[list[str]] = None,
    ):
        super().__init__(env)

        self.operations = ops
        self.goal = goal
        self.think_ticks = think_ticks
        self.background_knowledge = background_knowledge

        # ./NAR UDPNAR IP PORT  timestep(ns per cycle) printDerivations
        # process_cmd = [
        #     (NARS_PATH / "NAR").as_posix(),
        #     "UDPNAR",
        #     IP,
        #     str(PORT),
        #     "1000000",
        #     "true",
        # ]
        # ./NAR shell
        process_cmd = [
            (NARS_PATH / "NAR").as_posix(),
            "shell",
        ]
        self.process: subprocess.Popen = subprocess.Popen(
            process_cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            universal_newlines=True,
        )
        # sleep(3)  # wait for UDPNAR to make sure early commands don't get lost

        # setup NARS
        setup_nars(self.process, self.operations)
        # logger.info("\n".join(get_raw_output(self.process)))

        # send background knowledge
        if self.background_knowledge is not None:
            for statement in self.background_knowledge:
                send_input(self.process, statement)
        # send goal knowledge
        if self.goal.knowledge is not None:
            for belief in self.goal.knowledge:
                send_input(self.process, belief)

        send_input(self.process, "3")

    @abstractmethod
    def observe(self, complete: bool = False) -> None:
        pass
