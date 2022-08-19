import subprocess
from abc import ABCMeta, abstractmethod
from time import sleep
from typing import Any, Callable, Optional

import gym
import numpy as np
from griddly import gd
from griddly.util.rllib.environment.level_generator import LevelGenerator
from icecream import ic
from tensorboardX import SummaryWriter

from .nar import get_output, get_raw_output, send_input, setup_nars
from .narsese import Goal, nal_now
from .utils import NARS_PATH


class Agent(metaclass=ABCMeta):
    AVATAR_LABEL = "avatar"
    MAX_EP_REWARD = 1.0

    def __init__(self, env: gym.Env):
        self.env = env
        self.obj_names: list[str] = env.game.get_object_names()  # type: ignore

    def reset(self, level_string: Optional[str] = None):
        if level_string is None:
            return self.env.reset()
        else:
            return self.env.reset(level_string=level_string)  # type: ignore

    def step(self) -> tuple[Any, float, float, bool, Any]:
        actions = self.plan()
        obs = []
        reward = 0.0
        cumr = 0.0
        done = False
        info = None
        for action in actions:
            obs, reward, done, info = self.env.step(action)
            cumr += reward

        return obs, reward, cumr, done, info

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
        view_radius: int = 1,
        background_knowledge: Optional[list[str]] = None,
        motor_babbling: Optional[float] = None,
        decision_threshold: Optional[float] = None,
    ):
        super().__init__(env)

        self.operations = ops
        self.main_goal = main_goal
        self.goals = goals
        self.think_ticks = think_ticks
        self.view_radius = view_radius
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
        setup_nars(
            self.process,
            self.operations,
            motor_babbling=motor_babbling,
            decision_threshold=decision_threshold,
        )
        # logger.info("\n".join(get_raw_output(self.process)))

        # send background knowledge
        if self.background_knowledge is not None:
            for statement in self.background_knowledge:
                send_input(self.process, statement)
            send_input(self.process, "3")

        if main_goal is not None and goals is not None:
            self.setup_goals(main_goal, goals)

    def setup_goals(self, main_goal: Goal, goals: list[Goal]):
        self.main_goal = main_goal
        self.goals = goals

        # send goal knowledge
        for g in goals:
            if g.knowledge is not None:
                for belief in g.knowledge:
                    send_input(self.process, belief)
        send_input(self.process, "3")

    @abstractmethod
    def observe(self, observation: np.ndarray) -> None:
        raise NotImplementedError

    @abstractmethod
    def determine_actions(self, info: dict[str, Any]) -> list[list[int]]:
        raise NotImplementedError

    def load_concepts(self, concept_file: str):
        """Load concepts from a file"""
        with open(concept_file) as f:
            for line in f:
                send_input(self.process, line)

    def process_goals(self, env_state: dict, info: dict) -> bool:
        """Check for goal satisfaction and inform ONA

        Returns True if the level got completed.
        """

        if self.goals is None:
            return False
        complete = False

        satisfied_goals = [g.satisfied(env_state, info) for g in self.goals]
        for g, sat in zip(self.goals, satisfied_goals):
            if sat:
                print(f"{g.symbol} satisfied.")
                send_input(self.process, nal_now(g.symbol))
                get_raw_output(self.process)

                info[f"{g.symbol}_satisfied"] = True

                if g.symbol == "GOT_KEY":
                    self.has_key = True

                if g.symbol == "COMPLETE":
                    complete = True
            else:
                info[f"{g.symbol}_satisfied"] = False

        return complete

    def step(self, observation: np.ndarray) -> tuple[Any, float, float, bool, Any]:
        self.observe(observation)
        actions = self.plan()
        obs = []
        reward = 0.0
        cumr = 0.0
        done = False
        info = None
        for action in actions:
            obs, reward, done, info = self.env.step(action)
            cumr += reward
            env_state = self.env.get_state()  # type: ignore
            env_state["reward"] = reward
            self.process_goals(env_state, info)

        # Introduce a delay after sending all observation and goal-related statements.
        send_input(self.process, "3")
        return obs, reward, cumr, done, info


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
            obs = self.agent.reset(level_string=lvl_str)

            for _ in range(max_iterations):
                obs, reward, cumr, done, info = self.agent.step(obs)
                run_info["episode_reward"] += cumr

                env_state = self.agent.env.get_state()  # type: ignore
                env_state["reward"] = reward

                if info["COMPLETE_satisfied"]:
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

    def run_curriculum(
        self,
        env_name: str,
        max_difficulty: int,
        num_episodes: int,
        max_iterations: int,
        log_tb: bool = False,
        tb_comment_suffix: str = "",
        callbacks: list[tuple[str, Callable]] = [],
    ):
        """Runs the agent's curriculum loop

        Args:
            env_name (str): The name of the environment.
            max_difficulty (int): The maximum difficulty of the environment.
            num_episodes (int): Number of episodes to run.
            max_iterations (int): Maximum number of iterations to run.
            log_tb (bool): log to TensorBoard or not.
            tb_comment_suffix
            callbacks (list[tuple[str, Callable]]): a list of callbacks to run.
        """
        run_info = dict(total_reward=0.0, episode_reward=0.0, num_complete=0)
        done = False
        tb_writer = (
            SummaryWriter(comment=f"-ona-{tb_comment_suffix}") if log_tb else None
        )

        for difficulty in range(max_difficulty):
            env = gym.make(
                env_name,
                player_observer_type=gd.ObserverType.VECTOR,
                level=difficulty,
                new_step_api=True,
            )
            env.enable_history(True)  # type: ignore
            self.agent.env = env
            run_info["level"] = difficulty + 1
            run_info[f"{difficulty + 1}/total_reward"] = 0.0
            run_info[f"{difficulty + 1}/episode_reward"] = 0.0
            run_info[f"{difficulty + 1}/num_complete"] = 0

            for episode in range(num_episodes):
                lvl_str = (
                    self.levelgen.generate() if self.levelgen is not None else None
                )
                obs = self.agent.reset(level_string=lvl_str)

                for _ in range(max_iterations):
                    obs, reward, cumr, done, info = self.agent.step(obs)
                    run_info["episode_reward"] += cumr
                    run_info[f"{run_info['level']}/episode_reward"] += cumr

                    env_state = self.agent.env.get_state()  # type: ignore
                    env_state["reward"] = reward

                    if info["COMPLETE_satisfied"]:
                        run_info["num_complete"] += 1
                        run_info[f"{run_info['level']}/num_complete"] += 1

                    self.agent.env.render(observer="global")  # type: ignore # Renders the entire environment

                    if done:
                        break

                print(
                    f"Episode {episode+1} finished with reward {run_info['episode_reward']}."
                )
                run_info["total_reward"] += run_info["episode_reward"]
                run_info[f"{run_info['level']}/total_reward"] += run_info[
                    f"{run_info['level']}/episode_reward"
                ]

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

            # Post-level procedure
            run_info[f"{run_info['level']}/avg_ep_reward"] = (
                run_info[f"{run_info['level']}/total_reward"] / num_episodes
            )
            run_info[f"{run_info['level']}/avg_completion_rate"] = (
                run_info[f"{run_info['level']}/avg_ep_reward"]
                / self.agent.__class__.MAX_EP_REWARD
            )
            run_info[f"{run_info['level']}/completed_rate"] = (
                run_info[f"{run_info['level']}/num_complete"] / num_episodes
            )

            for trigger, callback in callbacks:
                if trigger == "on_level_end":
                    callback(run_info)

        # Final wrap up
        run_info["avg_ep_reward"] = run_info["total_reward"] / (
            num_episodes * max_difficulty
        )
        run_info["avg_completion_rate"] = (
            run_info["avg_ep_reward"] / self.agent.__class__.MAX_EP_REWARD
        )
        run_info["completed_rate"] = run_info["num_complete"] / (
            num_episodes * max_difficulty
        )
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

            self.agent.env.render(observer="global")  # type: ignore
            sleep(1)
            if reward < 0.0:  # reward is -1.0 in case of avatar's death
                self.demo_goal(plan)
