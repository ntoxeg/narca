import logging
import os
import random
from functools import partial
from pathlib import Path
from time import sleep
from typing import Optional

import griddly  # noqa
import gym
import pexpect
from griddly import gd
from icecream import ic
from tensorboardX import SummaryWriter

from narca.agent import Agent
from narca.astar import *
from narca.nar import *
from narca.utils import *

# setup a logger for nars output
logging.basicConfig(filename="nars_zelda.log", filemode="w", level=logging.DEBUG)
logger = logging.getLogger("nars")

NARS_PATH = Path(os.environ["NARS_HOME"])
NARS_OPERATIONS = {
    "^rotate_left": 1,
    "^move_forwards": 2,
    "^rotate_right": 3,
    "^move_backwards": 4,
    "^attack": 5,
    # "^goto": 6,
}
NUM_EPISODES = 50
MAX_ITERATIONS = 100


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

    def reset(self):
        self.env.reset()
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


def object_reached(obj_type: str, env_state: dict, info: dict) -> bool:
    """Check if an object has been reached

    Assumes that if the object does not exist, then it must have been reached.
    """
    # try:
    #     avatar = next(obj for obj in env_state["Objects"] if obj["Name"] == "avatar")
    # except StopIteration:
    #     ic("No avatar found. Goal unsatisfiable.")
    #     return False
    # try:
    #     target = next(obj for obj in env_state["Objects"] if obj["Name"] == obj_type)
    # except StopIteration:
    #     return True
    # return avatar["Location"] == target["Location"]
    history = info["History"]
    if len(history) == 0:
        return False

    last_event = history[-1]
    if (
        last_event["SourceObjectName"] == "avatar"
        and last_event["DestinationObjectName"] == obj_type
    ):
        send_input(
            agent.process,
            nal_now(f"<{ext(last_event['DestinationObjectName'])} --> [reached]>"),
        )
        return True

    return False


def got_rewarded(env_state: dict, _) -> bool:
    return env_state["reward"] > 0


def make_goal(process: pexpect.spawn, env_state: dict, goal_symbol: str) -> None:
    """Make an explicit goal using the position of an object"""
    goal = next(obj for obj in env_state["Objects"] if obj["Name"] == "goal")
    goal_loc = loc(goal["Location"])
    goal_achievement = f"<<({ext('SELF')} * {goal_loc}) --> at> =/> {goal_symbol}>."
    send_input(process, goal_achievement)


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


def demo_reach_key(symbol: str) -> None:
    """Demonstrate reaching the key"""
    actions_to_take = ["^rotate_left"] + (["^move_forwards"] * 4)
    for action in actions_to_take:
        send_input(agent.process, f"{action}. :|:")
        gym_actions = agent.determine_actions(
            {"executions": [{"operator": action, "arguments": []}]}
        )
        _, _, done, _ = agent.env.step(gym_actions[0])
        send_observation(agent.process, agent.env.get_state())  # type: ignore

        env.render(observer="global")  # type: ignore
        sleep(1)
        if done:
            env.reset()
            demo_reach_key(symbol)
    send_input(agent.process, nal_now(symbol))


def make_loc_goal(process: pexpect.spawn, pos, goal_symbol):
    """Make a goal for a location"""
    goal_loc = loc(pos)
    goal_achievement = f"<<({ext('SELF')} * {goal_loc}) --> at> =/> {goal_symbol}>."
    send_input(process, goal_achievement)


def key_check(_, info) -> bool:
    history = info["History"]
    if len(history) == 0:
        return False
    last_event = history[-1]
    return (
        last_event["SourceObjectName"] == "avatar"
        and last_event["DestinationObjectName"] == "key"
    )


if __name__ == "__main__":
    env = gym.make("GDY-Zelda-v0", player_observer_type=gd.ObserverType.VECTOR)
    env.enable_history(True)  # type: ignore

    reach_object_knowledge = [
        f"<(<($obj * #location) --> at> &/ <({ext('SELF')} * #location) --> ^goto>) =/> <$obj --> [reached]>>.",
    ]
    rel_pos_knowledge = [
        f"<(<$obj --> [ahead]> &/ ^move_forwards) =/> <$obj --> [reached]>>.",
        f"<(<$obj --> [leftward]> &/ ^move_forwards &/ ^rotate_left) =/> <$obj --> [ahead]>>.",
        f"<(<$obj --> [rightward]> &/ ^move_forwards &/ ^rotate_right) =/> <$obj --> [ahead]>>.",
    ]
    background_knowledge = rel_pos_knowledge

    key_goal_sym = "GOT_KEY"
    reach_key = [f"<({ext('key')} --> [reached]) =/> {key_goal_sym}>."]
    door_goal_sym = "AT_DOOR"
    reach_door = [f"<<{ext('goal')} --> [reached]> =/> {door_goal_sym}>."]
    complete_goal_sym = "COMPLETE"
    complete_goal = [f"<({key_goal_sym} &/ {door_goal_sym}) =/> {complete_goal_sym}>."]

    KEY_GOAL = Goal(key_goal_sym, partial(object_reached, "key"), reach_key)
    DOOR_GOAL = Goal(door_goal_sym, partial(object_reached, "goal"), reach_door)
    COMPLETE_GOAL = Goal(
        complete_goal_sym,
        lambda evst, info: agent.has_key and DOOR_GOAL.satisfied(evst, info),
        complete_goal,
    )
    REWARD_GOAL = Goal("GOT_REWARD", got_rewarded)

    goals = [
        KEY_GOAL,
        DOOR_GOAL,
        COMPLETE_GOAL,
        REWARD_GOAL,
    ]

    agent = ZeldaAgent(
        env,
        COMPLETE_GOAL,
        think_ticks=10,
        background_knowledge=background_knowledge,
    )
    total_reward = 0.0
    episode_reward = 0.0
    tb_writer = SummaryWriter(comment="-nars-zelda")
    done = False
    # TRAINING LOOP
    for episode in range(NUM_EPISODES):
        agent.reset()

        for i in range(MAX_ITERATIONS):
            agent.observe(complete=i % 10 == 0)

            obs, reward, cumr, done, info = agent.step()
            episode_reward += cumr

            env_state = agent.env.get_state()  # type: ignore
            env_state["reward"] = reward

            satisfied_goals = [g.satisfied(env_state, info) for g in goals]
            for g, sat in zip(goals, satisfied_goals):
                if sat:
                    print(f"{g.symbol} satisfied.")
                    send_input(agent.process, nal_now(g.symbol))
                    get_raw_output(agent.process)

                    if g.symbol == key_goal_sym:
                        agent.has_key = True

            env.render(observer="global")  # type: ignore # Renders the entire environment
            # sleep(1)

            if done:
                break

        print(f"Episode {episode+1} finished with reward {episode_reward}.")
        total_reward += episode_reward
        tb_writer.add_scalar("train/episode_reward", episode_reward, episode)
        tb_writer.add_scalar("train/total_reward", total_reward, episode)
        episode_reward = 0.0
        send_input(agent.process, nal_now("RESET"))

    # tb_writer.add_scalar("train/episode_reward", episode_reward, num_episodes)
    print(f"Average total reward per episode: {total_reward / NUM_EPISODES}.")
    env.close()  # Call explicitly to avoid exception on quit
