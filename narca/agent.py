import subprocess
from abc import ABCMeta, abstractmethod
from time import sleep
from typing import Callable, Optional

import gym
from griddly.util.rllib.environment.level_generator import LevelGenerator
from tensorboardX import SummaryWriter

from .nar import get_raw_output, send_input, setup_nars
from .narsese import Goal, nal_now
from .utils import NARS_PATH


class Agent(metaclass=ABCMeta):
    AVATAR_LABEL = "avatar"
    MAX_EP_REWARD = 1.0

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
        main_goal: Optional[Goal] = None,
        goals: Optional[list[Goal]] = None,
        think_ticks: int = 5,
        background_knowledge: Optional[list[str]] = None,
    ):
        super().__init__(env)

        self.operations = ops
        self.main_goal = main_goal
        self.goals = goals
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
            "InspectionOnExit",
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

        if main_goal is not None and goals is not None:
            self.setup_goals(main_goal, goals)

        send_input(self.process, "3")

    def setup_goals(self, main_goal: Goal, goals: list[Goal]):
        self.main_goal = main_goal
        self.goals = goals

        # send goal knowledge
        for g in goals:
            if g.knowledge is not None:
                for belief in g.knowledge:
                    send_input(self.process, belief)

    @abstractmethod
    def observe(self, complete: bool = False) -> None:
        pass


class Runner:
    """Functionality for running agent interaction episodes"""

    def __init__(
        self,
        agent: NarsAgent,
        levelgen: Optional[LevelGenerator] = None,
    ):
        self.agent = agent
        self.levelgen = levelgen

    def run(
        self,
        num_episodes: int,
        max_iterations: int,
        log_tb: bool = False,
        tb_comment_suffix: str = "",
        callbacks: list[tuple[str, Callable]] = [],
    ) -> None:
        """Run agent interaction episodes"""
        run_info = dict(total_reward=0.0, episode_reward=0.0, num_complete=0)
        done = False
        tb_writer = (
            SummaryWriter(comment=f"-ona-{tb_comment_suffix}") if log_tb else None
        )

        for episode in range(num_episodes):
            lvl_str = self.levelgen.generate() if self.levelgen is not None else None
            self.agent.reset(level_string=lvl_str)

            for i in range(max_iterations):
                self.agent.observe(complete=True)

                _, reward, cumr, done, info = self.agent.step()
                run_info["episode_reward"] += cumr

                env_state = self.agent.env.get_state()  # type: ignore
                env_state["reward"] = reward

                satisfied_goals = [
                    g.satisfied(env_state, info) for g in self.agent.goals
                ]
                for g, sat in zip(self.agent.goals, satisfied_goals):
                    if sat:
                        print(f"{g.symbol} satisfied.")
                        send_input(self.agent.process, nal_now(g.symbol))
                        get_raw_output(self.agent.process)

                        if g.symbol == "GOT_KEY":
                            self.agent.has_key = True

                        if g.symbol == "COMPLETE":
                            run_info["num_complete"] += 1

                self.agent.env.render(observer="global")  # type: ignore # Renders the entire environment

                if done:
                    break

            print(
                f"Episode {episode+1} finished with reward {run_info['episode_reward']}."
            )
            run_info["total_reward"] += run_info["episode_reward"]

            # Performance logging subroutines
            for trigger, callback in callbacks:
                if trigger == "on_episode_end":
                    callback(run_info)
            if tb_writer is not None:
                tb_writer.add_scalar(
                    "train/episode_reward", run_info["episode_reward"], episode
                )
                tb_writer.add_scalar(
                    "train/total_reward", run_info["total_reward"], episode
                )

            # Post-episode wrap up
            run_info["episode_reward"] = 0.0

        # Final wrap up
        run_info["avg_ep_reward"] = run_info["total_reward"] / num_episodes
        run_info["avg_completion_rate"] = (
            run_info["avg_ep_reward"] / self.agent.__class__.MAX_EP_REWARD
        )
        run_info["completed_rate"] = run_info["num_complete"] / num_episodes
        print(
            f"Average total reward per episode: {run_info['avg_ep_reward']}.",
            f"Average completion rate: {run_info['avg_completion_rate']*100:.0f}%.",
            f"Completed rate: {run_info['completed_rate']*100:.0f}%.",
        )

        for trigger, callback in callbacks:
            if trigger == "on_run_end":
                callback(run_info)

        self.agent.env.close()  # Call explicitly to avoid exception on quit

        # export concepts
        send_input(self.agent.process, "*concepts")
        concepts = "\n".join(get_raw_output(self.agent.process))
        with open("ona_concept_export.txt", "w") as f:
            f.write(concepts)

    def demo_goal(self, plan: list[str]) -> None:
        """Demonstrate reaching the goal

        Generates levels that fit a given plan.
        """
        lvl_str = (
            self.levelgen.generate_for_plan(plan) if self.levelgen is not None else None
        )

        self.agent.reset(level_string=lvl_str)
        for action in plan:
            send_input(self.agent.process, f"{action}. :|:")
            gym_actions = self.agent.determine_actions(
                {"executions": [{"operator": action, "arguments": []}]}
            )
            _, reward, done, info = self.agent.env.step(gym_actions[0])
            self.agent.observe()

            env_state: dict[str, Any] = self.agent.env.get_state()  # type: ignore
            env_state["reward"] = reward

            satisfied_goals = [g.satisfied(env_state, info) for g in self.agent.goals]
            for g, sat in zip(self.agent.goals, satisfied_goals):
                if sat:
                    print(f"{g.symbol} satisfied.")
                    send_input(self.agent.process, nal_now(g.symbol))
                    get_raw_output(self.agent.process)

                    if g.symbol == "GOT_KEY":
                        self.agent.has_key = True

            self.agent.env.render(observer="global")  # type: ignore
            sleep(1)
            if reward < 0.0:  # reward is -1.0 in case of avatar's death
                self.demo_goal(plan)
