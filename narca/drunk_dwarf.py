import random
from functools import partial
from time import sleep

import gym
import numpy as np
from griddly.util.rllib.environment.level_generator import LevelGenerator

from .agent import Agent, NarsAgent
from .astar import pathfind
from .nar import *
from .utils import *


class DrunkDwarfAgent(NarsAgent):
    """Agent for Drunk Dwarf"""

    NARS_OPERATIONS = [
        "^rotate_left",
        "^move_forwards",
        "^rotate_right",
    ]
    AVATAR_LABEL = "drunk_dwarf"
    MAX_EP_REWARD = 3.0

    def __init__(
        self,
        env: gym.Env,
        main_goal: Optional[Goal] = None,
        goals: Optional[list[Goal]] = None,
        think_ticks: int = 5,
        view_radius: int = 1,
        background_knowledge=None,
        motor_babbling: Optional[float] = None,
        decision_threshold: Optional[float] = None,
    ):
        super().__init__(
            env,
            DrunkDwarfAgent.NARS_OPERATIONS,
            main_goal,
            goals,
            think_ticks,
            view_radius,
            background_knowledge,
            motor_babbling,
            decision_threshold,
        )

        # init agent's state
        self.has_key = False
        self.object_info = {"current": {}, "previous": {}}

    def reset(self, level_string: Optional[str] = None):
        obs = super().reset(level_string)

        # reset agent's state
        self.has_key = False
        self.object_info = {"current": {}, "previous": {}}

        # send reset info to NARS
        send_input(self.process, nal_now("RESET"))
        send_input(self.process, "100")

        return obs

    def plan(self) -> list[list[int]]:
        if self.main_goal is None:
            raise RuntimeError("Main goal is not set.")
        # determine the action to take from NARS
        send_input(self.process, nal_demand(self.main_goal.symbol))

        nars_output = expect_output(
            self.process,
            self.operations,
            goal_reentry=self.main_goal,
            think_ticks=self.think_ticks,
            patience=1,
        )

        if nars_output is None:
            # op = random.choice(list(NARS_OPERATIONS.keys()))
            # send_input(self.process, nal_now(op))  # FIXME: handle arguments
            # nars_output = {"executions": [{"operator": op, "arguments": []}]}
            return [[0, 0]]

        return self.determine_actions(nars_output)

    def determine_actions(self, nars_output: dict[str, Any]) -> list[list[int]]:
        """Determine appropriate Gym actions based on NARS output"""
        env_state: dict[str, Any] = self.env.get_state()  # type: ignore
        to_griddly_id = {
            "^rotate_left": 1,
            "^move_forwards": 2,
            "^rotate_right": 3,
        }

        if len(nars_output["executions"]) < 1:
            raise RuntimeError("No operator executions found from NAR.")
        for exe in nars_output[
            "executions"
        ]:  # for now we will process the first execution
            ic(exe)
            op = exe["operator"]
            args = exe["arguments"]
            if op == "^goto":
                avatar = next(
                    obj
                    for obj in env_state["Objects"]
                    if obj["Name"] == DrunkDwarfAgent.AVATAR_LABEL
                )

                if len(args) < 2:
                    ic("Not enough arguments received, assuming random coordinates.")
                    args = [
                        "{SELF}",
                        loc((random.randint(0, 6), random.randint(0, 6))),
                    ]

                ic("Executing ^goto with args:", args)
                path_ops = pathfind(avatar["Location"], pos(args[1]))
                if len(path_ops) == 0:
                    return [[0, 0]]
                rel_ops = abs_to_rel(avatar, path_ops[0])
                return [[0, to_griddly_id[op]] for op in rel_ops]
            if op == "^attack":
                return [[1, 1]]
            return [[0, to_griddly_id[op]]]

        return [[0, 0]]  # noop

    def _pos_beliefs(
        self, avatar_loc: tuple[int, int], avatar_orient: str
    ) -> list[str]:
        """Produce NARS statements from held object information"""

        return [
            (f"<{ext('SELF')} --> [orient-{avatar_orient.lower()}]>. :|:")
        ] + self._relative_beliefs(avatar_loc, avatar_orient)

    def _relative_beliefs(
        self, avatar_loc: tuple[int, int], avatar_orient: str
    ) -> list[str]:
        """Produce NARS statements about relative positions of objects"""
        beliefs: list[str] = []
        if avatar_orient == "NONE":
            return []

        for typ in self.object_info["current"]:
            for obname, obloc in self.object_info["current"][typ].items():
                relpos = nal_rel_pos(obname, avatar_orient, avatar_loc, obloc)
                if relpos is not None:
                    beliefs.append(relpos)
                    beliefs.extend(
                        nal_distance(obname, (avatar_loc, avatar_orient), obloc)
                    )

        return beliefs

    def _diff_beliefs(self, avatar_loc: tuple[int, int]) -> list[str]:
        beliefs: list[str] = []

        # send info about what got closer and what further away
        for typ, obinfo in self.object_info["previous"].items():
            for obj_name, obj_loc in obinfo.items():
                if (
                    typ in self.object_info["current"]
                    and obj_name in self.object_info["current"][typ]
                ):
                    new_obj_loc = self.object_info["current"][typ][obj_name]
                    if manhattan_distance(avatar_loc, new_obj_loc) < manhattan_distance(
                        avatar_loc, obj_loc
                    ):
                        beliefs.append(nal_now(f"<{ext(obj_name)} --> [closer]>"))
                    elif manhattan_distance(
                        avatar_loc, new_obj_loc
                    ) > manhattan_distance(avatar_loc, obj_loc):
                        beliefs.append(nal_now(f"<{ext(obj_name)} --> [further]>"))
                else:
                    beliefs.append(nal_now(f"<{ext(obj_name)} --> [gone]>"))

        return beliefs

    def _send_beliefs(self, state_narsese: list[str]) -> None:
        for statement in state_narsese:
            send_input(self.process, statement)
            get_raw_output(self.process)

    def observe(self, observation: np.ndarray) -> None:
        env_state: dict[str, Any] = self.env.get_state()  # type: ignore
        num_sent_beliefs = 0

        def where_obj_type(obj_type: str, obs: np.ndarray) -> list[tuple[int, int]]:
            xs, ys = obs[self.obj_names.index(obj_type)].nonzero()
            return list(zip(xs, ys))

        # no threats in the environment, can assume avatar exists.
        avatar_loc = where_obj_type(__class__.AVATAR_LABEL, observation)[0]

        def obj_label(obj_type: str, obloc: tuple[int, int]) -> Optional[str]:
            x, y = obloc
            avx, avy = avatar_loc
            if y < avy:  # objects in front of the agent
                if x < avx:  # leftward
                    return f"L{obj_type}"
                elif x > avx:  # rightward
                    return f"R{obj_type}"
                else:
                    return f"A{obj_type}"  # Ahead
            else:
                return None

        visible_objects = {
            obj_type: where_obj_type(obj_type, observation)
            for obj_type in self.obj_names
            if obj_type != __class__.AVATAR_LABEL
        }
        obj_type_labels = [
            [obj_label(obj_type, obloc) for obloc in visible_objects[obj_type]]
            for obj_type in visible_objects.keys()
        ]
        obj_labels: list[str] = []
        for labels in obj_type_labels:
            obj_labels.extend(
                [
                    label + str(i + 1)
                    for i, label in enumerate(labels)
                    if label is not None
                ]
            )

        obj_concurrent_belief = (
            f"({obj_labels[0]} &| {obj_labels[1]})"
            if len(obj_labels) > 1
            else obj_labels[0]
        )
        if len(obj_labels) > 2:
            for i in range(2, len(obj_labels)):
                obj_concurrent_belief = f"({obj_concurrent_belief} &| {obj_labels[i]})"

        # nal_now("<" + " &| ".join(obj_labels) + ">")
        # ic(obj_concurrent_belief)
        self._send_beliefs([nal_now(obj_concurrent_belief)])
        num_sent_beliefs += 1

        # try:
        #     avatar = next(
        #         obj
        #         for obj in env_state["Objects"]
        #         if obj["Name"] == __class__.AVATAR_LABEL
        #     )
        # except StopIteration:
        #     return

        # avatar_loc, avatar_orient = avatar["Location"], avatar["Orientation"]

        # self.object_info["previous"] = self.object_info["current"]
        # obtypes = set(
        #     obj["Name"]
        #     for obj in env_state["Objects"]
        #     if obj["Name"] != __class__.AVATAR_LABEL
        # )

        # self.object_info["current"] = {
        #     typ: {
        #         f"{obj['Name']}{i+1}": obj["Location"]
        #         for i, obj in enumerate(env_state["Objects"])
        #         if obj["Name"] == typ
        #         and manhattan_distance(avatar_loc, obj["Location"]) <= self.view_radius
        #         and in_front(avatar_orient, avatar_loc, obj["Location"])
        #     }
        #     for typ in obtypes
        # }

        # # search for high-importance objects that are not in the current view
        # important_types = ["key", "door", "coffin_bed"]
        # for typ in important_types:
        #     self.object_info["current"][typ] = {
        #         f"{obj['Name']}{i+1}": obj["Location"]
        #         for i, obj in enumerate(env_state["Objects"])
        #         if obj["Name"] == typ
        #     }

        # # HACK: for singletons change the name to type's name
        # for typ in self.object_info["current"]:
        #     typ_info = self.object_info["current"][typ]
        #     if len(typ_info) == 1:
        #         self.object_info["current"][typ] = {
        #             typ: obj for obj in typ_info.values()
        #         }

        # for typ, obinfo in self.object_info["current"].items():
        #     for obj_name in obinfo.keys():
        #         if (
        #             typ in self.object_info["previous"]
        #             and obj_name not in self.object_info["previous"][typ]
        #         ):
        #             send_input(self.process, nal_now(f"<{ext(obj_name)} --> [new]>"))
        #             num_sent_beliefs += 1

        # pos_beliefs = self._pos_beliefs(avatar_loc, avatar_orient)
        # diff_beliefs = self._diff_beliefs(avatar_loc)

        # num_sent_beliefs += len(pos_beliefs) + len(diff_beliefs)
        # self._send_beliefs(pos_beliefs + diff_beliefs)
        # ic("Sent pos beliefs:", pos_beliefs)
        # ic("Sent diff beliefs:", diff_beliefs)
        ic("Total beliefs sent:", num_sent_beliefs)
        # sleep(1)


class DrunkDwarfRandom(Agent):
    """An agent that chooses random actions"""

    AVATAR_LABEL = "drunk_dwarf"
    MAX_EP_REWARD = 3.0

    def __init__(self, env: gym.Env):
        super().__init__(env)

    def plan(self) -> list[list[int]]:
        return [self.env.action_space.sample()]


def demo_reach_key(symbol: str, agent: DrunkDwarfAgent) -> None:
    """Demonstrate reaching the key"""
    reach_key = [f"<({ext('key')} --> [reached]) =/> {symbol}>."]
    goals = [Goal(symbol, partial(object_reached, agent, "key"), reach_key)]
    actions_to_take = ["^rotate_left"] + (["^move_forwards"] * 4)
    for action in actions_to_take:
        send_input(agent.process, f"{action}. :|:")
        gym_actions = agent.determine_actions(
            {"executions": [{"operator": action, "arguments": []}]}
        )
        obs, reward, done, info = agent.env.step(gym_actions[0])
        agent.observe(obs)

        env_state = agent.env.get_state()  # type: ignore
        env_state["reward"] = reward

        satisfied_goals = [g.satisfied(env_state, info) for g in goals]
        for g, sat in zip(goals, satisfied_goals):
            if sat:
                print(f"{g.symbol} satisfied.")
                send_input(agent.process, nal_now(g.symbol))
                get_raw_output(agent.process)

                if g.symbol == symbol:
                    agent.has_key = True

        agent.env.render(observer="global")  # type: ignore
        sleep(1)
        if done:
            agent.reset()
            demo_reach_key(symbol, agent)
