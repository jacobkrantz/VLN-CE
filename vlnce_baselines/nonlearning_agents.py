import json
from collections import defaultdict

import numpy as np
from habitat import Env, logger
from habitat.config.default import Config
from habitat.core.agent import Agent
from habitat.sims.habitat_simulator.actions import HabitatSimActions
from tqdm import tqdm, trange

from vlnce_baselines.common.environments import VLNCEInferenceEnv


def evaluate_agent(config: Config) -> None:
    split = config.EVAL.SPLIT
    config.defrost()
    # turn off RGBD rendering as neither RandomAgent nor HandcraftedAgent use it.
    config.TASK_CONFIG.SIMULATOR.AGENT_0.SENSORS = []
    config.TASK_CONFIG.TASK.SENSORS = []
    config.TASK_CONFIG.ENVIRONMENT.ITERATOR_OPTIONS.SHUFFLE = False
    config.TASK_CONFIG.ENVIRONMENT.ITERATOR_OPTIONS.MAX_SCENE_REPEAT_STEPS = -1
    config.TASK_CONFIG.DATASET.SPLIT = split
    config.TASK_CONFIG.TASK.NDTW.SPLIT = split
    config.TASK_CONFIG.TASK.SDTW.SPLIT = split
    config.freeze()

    env = Env(config=config.TASK_CONFIG)

    assert config.EVAL.NONLEARNING.AGENT in [
        "RandomAgent",
        "HandcraftedAgent",
    ], "EVAL.NONLEARNING.AGENT must be either RandomAgent or HandcraftedAgent."

    if config.EVAL.NONLEARNING.AGENT == "RandomAgent":
        agent = RandomAgent()
    else:
        agent = HandcraftedAgent()

    stats = defaultdict(float)
    num_episodes = min(config.EVAL.EPISODE_COUNT, len(env.episodes))
    for _ in trange(num_episodes):
        obs = env.reset()
        agent.reset()

        while not env.episode_over:
            action = agent.act(obs)
            obs = env.step(action)

        for m, v in env.get_metrics().items():
            stats[m] += v

    stats = {k: v / num_episodes for k, v in stats.items()}

    logger.info(f"Averaged benchmark for {config.EVAL.NONLEARNING.AGENT}:")
    for stat_key in stats.keys():
        logger.info("{}: {:.3f}".format(stat_key, stats[stat_key]))

    with open(f"stats_{config.EVAL.NONLEARNING.AGENT}_{split}.json", "w") as f:
        json.dump(stats, f, indent=4)


def nonlearning_inference(config: Config) -> None:
    split = config.INFERENCE.SPLIT
    config.defrost()
    # turn off RGBD rendering as neither RandomAgent nor HandcraftedAgent use it.
    config.TASK_CONFIG.SIMULATOR.AGENT_0.SENSORS = []
    config.TASK_CONFIG.DATASET.SPLIT = config.INFERENCE.SPLIT
    config.TASK_CONFIG.ENVIRONMENT.ITERATOR_OPTIONS.SHUFFLE = False
    config.TASK_CONFIG.ENVIRONMENT.ITERATOR_OPTIONS.MAX_SCENE_REPEAT_STEPS = -1
    config.TASK_CONFIG.TASK.MEASUREMENTS = []
    config.TASK_CONFIG.TASK.SENSORS = []
    config.freeze()

    env = VLNCEInferenceEnv(config=config)

    assert config.INFERENCE.NONLEARNING.AGENT in [
        "RandomAgent",
        "HandcraftedAgent",
    ], "INFERENCE.NONLEARNING.AGENT must be either RandomAgent or HandcraftedAgent."

    if config.INFERENCE.NONLEARNING.AGENT == "RandomAgent":
        agent = RandomAgent()
    else:
        agent = HandcraftedAgent()

    episode_predictions = defaultdict(list)
    for _ in tqdm(range(len(env.episodes)), desc=f"[inference:{split}]"):
        env.reset()
        obs = agent.reset()

        episode_id = env.current_episode.episode_id
        episode_predictions[episode_id].append(env.get_info(obs))

        while not env.get_done(obs):
            obs = env.step(agent.act(obs))
            episode_predictions[episode_id].append(env.get_info(obs))

    with open(config.INFERENCE.PREDICTIONS_FILE, "w") as f:
        json.dump(episode_predictions, f, indent=2)

    logger.info(f"Predictions saved to: {config.INFERENCE.PREDICTIONS_FILE}")


class RandomAgent(Agent):
    """Selects an action at each time step by sampling from the oracle action
    distribution of the training set.
    """

    def __init__(self, probs=None):
        self.actions = [
            HabitatSimActions.STOP,
            HabitatSimActions.MOVE_FORWARD,
            HabitatSimActions.TURN_LEFT,
            HabitatSimActions.TURN_RIGHT,
        ]
        if probs is not None:
            self.probs = probs
        else:
            self.probs = [0.02, 0.68, 0.15, 0.15]

    def reset(self):
        pass

    def act(self, observations):
        return {"action": np.random.choice(self.actions, p=self.probs)}


class HandcraftedAgent(Agent):
    """Agent picks a random heading and takes 37 forward actions (average
    oracle path length) before calling stop.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        # 9.27m avg oracle path length in Train.
        # Fwd step size: 0.25m. 9.25m/0.25m = 37
        self.forward_steps = 37
        self.turns = np.random.randint(0, int(360 / 15) + 1)

    def act(self, observations):
        if self.turns > 0:
            self.turns -= 1
            return {"action": HabitatSimActions.TURN_RIGHT}
        if self.forward_steps > 0:
            self.forward_steps -= 1
            return {"action": HabitatSimActions.MOVE_FORWARD}
        return {"action": HabitatSimActions.STOP}
