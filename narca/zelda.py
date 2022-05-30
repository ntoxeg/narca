import random
from functools import partial
from time import sleep

import gym
import numpy as np
from griddly.util.rllib.environment.level_generator import LevelGenerator

from .agent import Agent
from .astar import pathfind
from .nar import *
from .utils import NARS_PATH, object_reached

NARS_OPERATIONS = {
    "^rotate_left": 1,
    "^move_forwards": 2,
    "^rotate_right": 3,
    "^move_backwards": 4,
    "^attack": 5,
    # "^goto": 6,
}


def abs_to_rel(avatar, op):
    orient_to_num = {
        "UP": 0,
        "RIGHT": 1,
        "DOWN": 2,
        "LEFT": 3,
        "NONE": 0,  # HACK: assuming that NONE is the same as UP
    }
    if avatar["Orientation"] == "NONE":
        ic("Warning: avatar orientation is NONE. Assuming UP.")
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


def narsify_from_state(env_state: dict[str, Any]) -> list[str]:
    """Produce NARS statements from environment semantic state"""
    # TODO: handle the case where every type of an object can be a multiple
    special_types = ["wall", "avatar"]

    walls = [
        (i, obj["Location"])
        for i, obj in enumerate(env_state["Objects"])
        if obj["Name"] == "wall"
    ]

    avatar = next(obj for obj in env_state["Objects"] if obj["Name"] == "avatar")
    avatar_loc = f"<({ext('SELF')} * {loc(avatar['Location'])}) --> at>. :|:"
    avatar_orient = f"<{ext('SELF')} --> [orient-{avatar['Orientation'].lower()}]>. :|:"
    avatar_beliefs = [avatar_loc, avatar_orient]

    object_beliefs = [
        f"<({ext(obj['Name'])} * {loc(obj['Location'])}) --> at>. :|:"
        for obj in env_state["Objects"]
        if obj["Name"] not in special_types
    ]
    wall_beliefs = [
        f"<({ext('wall' + str(i))} * {loc(pos)}) --> at>. :|:" for i, pos in walls
    ]

    # NEXT: remove absolute position beliefs
    return relative_beliefs(env_state)


def send_observation(process: pexpect.spawn, env_state: dict, complete=False) -> None:
    """Send observation to NARS

    Args:
        process: NARS process
        env_state: environment state
        complete: whether to send the complete observation (include wall beliefs)
    """
    # send the observation to NARS
    # send_input(sock, narsify(obs))
    state_narsese = narsify_from_state(env_state)
    statements = [st for st in state_narsese if "wall" not in st or complete]
    for statement in statements:
        send_input(process, statement)
        get_raw_output(process)
    # send_input(sock, narsify_from_state(env_state))


def relative_beliefs(env_state: dict) -> list[str]:
    """Produce NARS statements about relative positions of objects"""
    beliefs = []
    # we need to process the key, the spider and the goal (door)
    try:
        key = next(obj for obj in env_state["Objects"] if obj["Name"] == "key")
    except StopIteration:
        key = None
    try:
        spider = next(obj for obj in env_state["Objects"] if obj["Name"] == "spider")
    except StopIteration:
        spider = None
    try:
        goal = next(obj for obj in env_state["Objects"] if obj["Name"] == "goal")
    except StopIteration:
        return []
    try:
        avatar = next(obj for obj in env_state["Objects"] if obj["Name"] == "avatar")
    except StopIteration:
        return []

    # we need to know the orientation of the avatar
    orient = avatar["Orientation"]
    if orient == "NONE":
        return []

    # we need to know the location of the avatar
    avatar_loc = avatar["Location"]

    # check if the key in in 180 degree arc in front
    if key is not None:
        key_loc = key["Location"]
        relpos = nal_rel_pos("key", orient, avatar_loc, key_loc)
        if relpos is not None:
            beliefs.append(relpos)

    # check if the spider is in front
    if spider is not None:
        spider_loc = spider["Location"]
        relpos = nal_rel_pos("spider", orient, avatar_loc, spider_loc)
        if relpos is not None:
            beliefs.append(relpos)

    # check if the goal is in front
    goal_loc = goal["Location"]
    relpos = nal_rel_pos("goal", orient, avatar_loc, goal_loc)
    if relpos is not None:
        beliefs.append(relpos)

    return beliefs


def nal_rel_pos(obname, orient, avatar_loc, obloc) -> Optional[str]:
    """Produce NARS statement about relative position of an object w.r.t. avatar

    The object is required to be in front of the avatar.
    """
    match orient:
        case "UP":
            if obloc[1] < avatar_loc[1]:
                if obloc[0] == avatar_loc[0]:
                    return nal_now(f"<{ext(obname)} --> [frontward ahead]>")
                if obloc[0] < avatar_loc[0]:
                    return nal_now(f"<{ext(obname)} --> [frontward leftward]>")
                if obloc[0] > avatar_loc[0]:
                    return nal_now(f"<{ext(obname)} --> [frontward rightward]>")
        case "RIGHT":
            if obloc[0] > avatar_loc[0]:
                if obloc[1] == avatar_loc[1]:
                    return nal_now(f"<{ext(obname)} --> [frontward ahead]>")
                if obloc[1] < avatar_loc[1]:
                    return nal_now(f"<{ext(obname)} --> [frontward leftward]>")
                if obloc[1] > avatar_loc[1]:
                    return nal_now(f"<{ext(obname)} --> [frontward rightward]>")
        case "DOWN":
            if obloc[1] > avatar_loc[1]:
                if obloc[0] == avatar_loc[0]:
                    return nal_now(f"<{ext(obname)} --> [frontward ahead]>")
                if obloc[0] < avatar_loc[0]:
                    return nal_now(f"<{ext(obname)} --> [frontward rightward]>")
                if obloc[0] > avatar_loc[0]:
                    return nal_now(f"<{ext(obname)} --> [frontward leftward]>")
        case "LEFT":
            if obloc[0] < avatar_loc[0]:
                if obloc[1] == avatar_loc[1]:
                    return nal_now(f"<{ext(obname)} --> [frontward ahead]>")
                if obloc[1] < avatar_loc[1]:
                    return nal_now(f"<{ext(obname)} --> [frontward rightward]>")
                if obloc[1] > avatar_loc[1]:
                    return nal_now(f"<{ext(obname)} --> [frontward leftward]>")

    return None


class ZeldaAgent(Agent):
    """Agent for Zelda"""

    def __init__(
        self,
        env: gym.Env,
        goal: Goal,
        think_ticks: int = 5,
        background_knowledge=None,
    ):
        super().__init__(env)
        self.goal = goal
        self.has_key = False
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
        # process = subprocess.Popen(
        #     process_cmd,
        #     stdout=subprocess.PIPE,
        #     universal_newlines=True,
        # )
        self.process: pexpect.spawn = pexpect.spawn(process_cmd[0], process_cmd[1:])
        # sleep(3)  # wait for UDPNAR to make sure early commands don't get lost

        # setup NARS
        setup_nars(self.process, NARS_OPERATIONS)
        # logger.info("\n".join(get_raw_output(self.process)))

        # send background knowledge
        if self.background_knowledge is not None:
            for statement in self.background_knowledge:
                send_input(self.process, statement)
        # send goal knowledge
        if self.goal.knowledge is not None:
            for belief in self.goal.knowledge:
                send_input(self.process, belief)

    def reset(self, level_string: Optional[str] = None):
        if level_string is None:
            self.env.reset()
        else:
            self.env.reset(level_string=level_string)
        self.has_key = False

    def plan(self) -> list[list[int]]:
        # determine the action to take from NARS
        send_input(self.process, nal_demand(self.goal.symbol))

        nars_output = expect_output(
            self.process,
            list(NARS_OPERATIONS.keys()),
            goal_reentry=self.goal,
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
        env_state = self.env.get_state()
        to_griddly_id = {
            "^rotate_left": 1,
            "^move_forwards": 2,
            "^rotate_right": 3,
            "^move_backwards": 4,
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
                    obj for obj in env_state["Objects"] if obj["Name"] == "avatar"
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

    def observe(self, complete=False):
        env_state = self.env.get_state()
        send_observation(self.process, env_state, complete)


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


def demo_goal(goal: Goal, agent: ZeldaAgent, plan: list[str]) -> None:
    """Demonstrate reaching the key"""
    goals = [goal]
    for action in plan:
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

                if g.symbol == "GOT_KEY":
                    agent.has_key = True

        agent.env.render(observer="global")  # type: ignore
        sleep(1)
        if done:
            agent.reset()  # TODO: track level string in agent
            demo_goal(goal, agent, plan)
