import logging
import os

import griddly  # noqa
import gym
import neptune.new as neptune
from griddly import gd

from narca.drunk_dwarf import DrunkDwarfRandom
from narca.nar import *
from narca.utils import *

# setup a logger for nars output
logging.basicConfig(filename="nars_drunk_dwarf.log", filemode="w", level=logging.DEBUG)
logger = logging.getLogger("nars")

NUM_EPISODES = 50
MAX_ITERATIONS = 100
ENV_NAME = "GDY-Drunk-Dwarf-v0"
MAIN_TAG = "random"
DIFFICULTY_LEVEL = 1


if __name__ == "__main__":
    try:
        neprun = neptune.init(
            project=os.environ["NEPTUNE_PROJECT"],
            tags=[ENV_NAME, MAIN_TAG, f"difficulty:{DIFFICULTY_LEVEL}"],
        )
    except KeyError:
        neprun = None

    env = gym.make(
        ENV_NAME,
        player_observer_type=gd.ObserverType.VECTOR,
        level=DIFFICULTY_LEVEL - 1,
    )

    agent = DrunkDwarfRandom(env)
    callbacks = []
    if neprun is not None:
        neprun["parameters"] = {
            "num_episodes": NUM_EPISODES,
            "max_iterations": MAX_ITERATIONS,
        }

        def nep_ep_callback(run_info: dict):
            neprun["train/episode_reward"].log(run_info["episode_reward"])
            neprun["train/total_reward"].log(run_info["total_reward"])

        def nep_run_callback(run_info: dict):
            neprun["train/avg_ep_reward"] = run_info["avg_ep_reward"]
            neprun["train/avg_completion_rate"] = run_info["avg_completion_rate"]
            neprun["train/completed_rate"] = run_info["completed_rate"]

        callbacks.append(("on_episode_end", nep_ep_callback))
        callbacks.append(("on_run_end", nep_run_callback))

    # Run the agent
    run_info = dict(total_reward=0.0, episode_reward=0.0, num_complete=0)
    done = False

    for episode in range(NUM_EPISODES):
        agent.reset()

        for i in range(MAX_ITERATIONS):
            _, reward, cumr, done, info = agent.step()
            run_info["episode_reward"] += cumr

            if run_info["episode_reward"] == DrunkDwarfRandom.MAX_EP_REWARD:
                run_info["num_complete"] += 1

            agent.env.render(observer="global")  # type: ignore # Renders the entire environment

            if done:
                break

        print(f"Episode {episode+1} finished with reward {run_info['episode_reward']}.")
        run_info["total_reward"] += run_info["episode_reward"]

        # Performance logging subroutines
        for trigger, callback in callbacks:
            if trigger == "on_episode_end":
                callback(run_info)

        # Post-episode wrap up
        run_info["episode_reward"] = 0.0

    # Final wrap up
    run_info["avg_ep_reward"] = run_info["total_reward"] / NUM_EPISODES
    run_info["avg_completion_rate"] = (
        run_info["avg_ep_reward"] / agent.__class__.MAX_EP_REWARD
    )
    run_info["completed_rate"] = run_info["num_complete"] / NUM_EPISODES
    print(
        f"Average total reward per episode: {run_info['avg_ep_reward']}.",
        f"Average completion rate: {run_info['avg_completion_rate']*100:.0f}%.",
        f"Completed rate: {run_info['completed_rate']*100:.0f}%.",
    )

    for trigger, callback in callbacks:
        if trigger == "on_run_end":
            callback(run_info)

    agent.env.close()  # Call explicitly to avoid exception on quit

    if neprun is not None:
        neprun.stop()
