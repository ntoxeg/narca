import random
from functools import partial
from time import sleep

import gym
import numpy as np
from griddly.util.rllib.environment.level_generator import LevelGenerator

from .agent import NarsAgent
from .astar import pathfind
from .nar import *
from .utils import *


class ZeldaLevelGenerator(LevelGenerator):
    KEY = "+"
    GOAL = "g"

    AGENT = "A"

    WALL = "w"
    SPIDER = "3"

    def __init__(self, config):
        super().__init__(config)
        self._width = config.get("width", 10)
        self._height = config.get("height", 10)
        self._p_key = config.get("p_key", 0.1)
        self._max_goals = config.get("max_goals", 3)
        self._p_spider = config.get("p_spider", 0.1)
        self._max_spiders = config.get("max_spiders", 3)

    def _place_walls(self, map_: np.chararray) -> np.chararray:

        # top/bottom wall
        wall_y = np.array([0, self._height - 1])
        map_[:, wall_y] = ZeldaLevelGenerator.WALL

        # left/right wall
        wall_x = np.array([0, self._width - 1])
        map_[wall_x, :] = ZeldaLevelGenerator.WALL

        return map_

    def _place_keys_goals(
        self,
        map_: np.chararray,
        possible_locations: list[tuple[int, int]],
        probability: float,
        key_char: str,
        goal_char: str,
        max_keys: int,
    ) -> tuple[np.chararray, list[tuple[int, int]]]:
        # TODO: turn into a generic method
        for _ in range(max_keys):
            if np.random.random() < probability:
                key_location_idx = np.random.choice(len(possible_locations))
                key_location = possible_locations.pop(key_location_idx)
                map_[key_location[0], key_location[1]] = key_char

                goal_location_idx = np.random.choice(len(possible_locations))
                goal_location = possible_locations.pop(goal_location_idx)
                map_[goal_location[0], goal_location[1]] = goal_char

        return map_, possible_locations

    def _place_spiders(
        self, map_: np.chararray, possible_locations: list[tuple[int, int]]
    ) -> tuple[np.chararray, list[tuple[int, int]]]:
        for _ in range(self._max_spiders):
            if np.random.random() < self._p_spider:
                spider_location_idx = np.random.choice(len(possible_locations))
                spider_location = possible_locations.pop(spider_location_idx)
                map_[
                    spider_location[0], spider_location[1]
                ] = ZeldaLevelGenerator.SPIDER

        return map_, possible_locations

    def _level_string(self, map_: np.chararray) -> str:
        level_string = ""
        for h in range(0, self._height):
            for w in range(0, self._width):
                level_string += map_[w, h].decode().ljust(4)
            level_string += "\n"

        return level_string

    def _rotation_transition(self, orient: str, action: str) -> str:
        match action:
            case "^rotate_left":
                match orient:
                    case "UP":
                        return "LEFT"
                    case "RIGHT":
                        return "UP"
                    case "DOWN":
                        return "RIGHT"
                    case "LEFT":
                        return "DOWN"
            case "^rotate_right":
                match orient:
                    case "UP":
                        return "RIGHT"
                    case "RIGHT":
                        return "DOWN"
                    case "DOWN":
                        return "LEFT"
                    case "LEFT":
                        return "UP"

        ic("Warning: invalid action/orientation combination.")
        return orient

    def _action_transition(
        self, state: tuple[int, int, str], action: str
    ) -> tuple[int, int, str]:
        # state is (x, y, orientation)
        match action:
            case "^move_forwards":
                match state[2]:
                    case "UP":
                        return state[0], state[1] - 1, state[2]
                    case "RIGHT":
                        return state[0] + 1, state[1], state[2]
                    case "DOWN":
                        return state[0], state[1] + 1, state[2]
                    case "LEFT":
                        return state[0] - 1, state[1], state[2]
            case "^move_backwards":
                match state[2]:
                    case "UP":
                        return state[0], state[1] + 1, state[2]
                    case "RIGHT":
                        return state[0] - 1, state[1], state[2]
                    case "DOWN":
                        return state[0], state[1] - 1, state[2]
                    case "LEFT":
                        return state[0] + 1, state[1], state[2]
            case "^rotate_left":
                return state[0], state[1], self._rotation_transition(state[2], action)
            case "^rotate_right":
                return state[0], state[1], self._rotation_transition(state[2], action)

        return state

    def generate(self) -> str:
        map_ = np.chararray((self._width, self._height), itemsize=2)
        map_[:] = "."

        # Generate walls
        map_ = self._place_walls(map_)

        # all possible locations
        possible_locations = [
            (w, h)
            for h in range(1, self._height - 1)
            for w in range(1, self._width - 1)
        ]

        # Place keys and goals
        map_, possible_locations = self._place_keys_goals(
            map_,
            possible_locations,
            self._p_key,
            ZeldaLevelGenerator.KEY,
            ZeldaLevelGenerator.GOAL,
            self._max_goals,
        )

        # Place spiders
        map_, possible_locations = self._place_spiders(map_, possible_locations)

        # Place Agent
        agent_location_idx = np.random.choice(len(possible_locations))
        agent_location = possible_locations[agent_location_idx]
        map_[agent_location[0], agent_location[1]] = ZeldaLevelGenerator.AGENT

        return self._level_string(map_)

    def generate_for_plan(self, plan: list[str]) -> str:
        map_ = np.chararray((self._width, self._height), itemsize=2)
        map_[:] = "."

        # Generate walls
        map_ = self._place_walls(map_)

        # all possible locations
        possible_locations = [
            (w, h)
            for h in range(1, self._height - 1)
            for w in range(1, self._width - 1)
        ]

        # Place spiders
        map_, possible_locations = self._place_spiders(map_, possible_locations)

        # Restrict locations based on the length of the plan
        possible_locations = [
            (x, y)
            for x, y in possible_locations
            if x in range(1 + len(plan), self._width - 1 - len(plan))
            and y in range(1 + len(plan), self._height - 1 - len(plan))
        ]

        # Place Agent
        agent_location_idx = np.random.choice(len(possible_locations))
        agent_location = possible_locations[agent_location_idx]
        map_[agent_location[0], agent_location[1]] = ZeldaLevelGenerator.AGENT

        # Given the list of actions, place the key and goal
        agent_state = (agent_location[0], agent_location[1], "UP")

        actions_pre, actions_post = plan[: len(plan) // 2], plan[len(plan) // 2 :]
        for action in actions_pre:
            agent_state = self._action_transition(agent_state, action)
        map_[agent_state[0], agent_state[1]] = ZeldaLevelGenerator.KEY

        for action in actions_post:
            agent_state = self._action_transition(agent_state, action)
        map_[agent_state[0], agent_state[1]] = ZeldaLevelGenerator.GOAL

        return self._level_string(map_)


class ZeldaAgent(NarsAgent):
    """Agent for Zelda"""

    NARS_OPERATIONS = [
        "^rotate_left",
        "^move_forwards",
        "^rotate_right",
        "^move_backwards",
        "^attack",
        # "^goto",
    ]
    VIEW_RADIUS = 1
    AVATAR_LABEL = "avatar"
    MAX_EP_REWARD = 2.0

    def __init__(
        self,
        env: gym.Env,
        main_goal: Optional[Goal] = None,
        goals: Optional[list[Goal]] = None,
        think_ticks: int = 5,
        background_knowledge=None,
    ):
        super().__init__(
            env,
            ZeldaAgent.NARS_OPERATIONS,
            main_goal,
            goals,
            think_ticks,
            background_knowledge,
        )

        # init agent's state
        self.has_key = False
        self.object_info = {"current": {}, "previous": {}}

    def reset(self, level_string: Optional[str] = None):
        super().reset(level_string)

        # reset agent's state
        self.has_key = False
        self.object_info = {"current": {}, "previous": {}}

        # send reset info to NARS
        send_input(self.process, nal_now("RESET"))
        send_input(self.process, "100")

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
            "^move_backwards": 4,
            "^attack": 1,
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
                    if obj["Name"] == ZeldaAgent.AVATAR_LABEL
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

    def step(self):
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

    def observe(self, complete=False) -> None:
        env_state: dict[str, Any] = self.env.get_state()  # type: ignore

        try:
            avatar = next(
                obj
                for obj in env_state["Objects"]
                if obj["Name"] == __class__.AVATAR_LABEL
            )
        except StopIteration:
            return

        avatar_loc, avatar_orient = avatar["Location"], avatar["Orientation"]

        self.object_info["previous"] = self.object_info["current"]
        self.object_info["current"] = {
            obj["Name"]: obj["Location"]
            for obj in env_state["Objects"]
            if obj["Name"] != __class__.AVATAR_LABEL
        }  # FIXME: narrow down to only objects in front of agent, make this a set of labels

        self.object_info["current"] = {
            typ: {
                f"{obj['Name']}{i+1}": obj["Location"]
                for i, obj in enumerate(env_state["Objects"])
                if obj["Name"] == typ
                and manhattan_distance(avatar_loc, obj["Location"])
                <= __class__.VIEW_RADIUS
            }
            for typ in self.object_info["current"].keys()
        }

        # HACK: for singletons change the name to type's name
        for typ in self.object_info["current"]:
            typ_info = self.object_info["current"][typ]
            if len(typ_info) == 1:
                self.object_info["current"][typ] = {
                    typ: obj for obj in typ_info.values()
                }

        for typ, obinfo in self.object_info["current"].items():
            for obj_name in obinfo.keys():
                if (
                    typ in self.object_info["previous"]
                    and obj_name not in self.object_info["previous"][typ]
                ):
                    send_input(self.process, nal_now(f"<{ext(obj_name)} --> [new]>"))

        self._send_beliefs(self._pos_beliefs(avatar_loc, avatar_orient))

        self._send_beliefs(self._diff_beliefs(avatar_loc))

        send_input(self.process, "3")


def demo_reach_key(symbol: str, agent: ZeldaAgent) -> None:
    """Demonstrate reaching the key"""
    reach_key = [f"<({ext('key')} --> [reached]) =/> {symbol}>."]
    goals = [Goal(symbol, partial(object_reached, agent, "key"), reach_key)]
    actions_to_take = ["^rotate_left"] + (["^move_forwards"] * 4)
    for action in actions_to_take:
        send_input(agent.process, f"{action}. :|:")
        gym_actions = agent.determine_actions(
            {"executions": [{"operator": action, "arguments": []}]}
        )
        _, reward, done, info = agent.env.step(gym_actions[0])
        agent.observe()

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
