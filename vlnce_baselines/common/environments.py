from typing import Optional

import habitat
import numpy as np
from habitat import Config, Dataset
from habitat.tasks.utils import cartesian_to_polar
from habitat.utils.geometry_utils import quaternion_rotate_vector
from habitat_baselines.common.baseline_registry import baseline_registry


@baseline_registry.register_env(name="VLNCEDaggerEnv")
class VLNCEDaggerEnv(habitat.RLEnv):
    def __init__(self, config: Config, dataset: Optional[Dataset] = None):
        self._success_distance = config.TASK_CONFIG.TASK.SUCCESS_DISTANCE
        super().__init__(config.TASK_CONFIG, dataset)

    def get_reward_range(self):
        # We don't use a reward for DAgger, but the baseline_registry requires
        # we inherit from habitat.RLEnv.
        return (0.0, 0.0)

    def get_reward(self, observations):
        return 0.0

    def _distance_target(self):
        current_position = self._env.sim.get_agent_state().position.tolist()
        target_position = self._env.current_episode.goals[0].position
        distance = self._env.sim.geodesic_distance(current_position, target_position)
        return distance

    def get_done(self, observations):
        episode_success = (
            self._env.task.is_stop_called
            and self._distance_target() < self._success_distance
        )
        return self._env.episode_over or episode_success

    def get_info(self, observations):
        return self.habitat_env.get_metrics()


@baseline_registry.register_env(name="VLNCEInferenceEnv")
class VLNCEInferenceEnv(habitat.RLEnv):
    def __init__(self, config: Config, dataset: Optional[Dataset] = None):
        super().__init__(config.TASK_CONFIG, dataset)

    def get_reward_range(self):
        return (0.0, 0.0)

    def get_reward(self, observations):
        return 0.0

    def get_done(self, observations):
        return self._env.episode_over

    def get_info(self, observations):
        agent_state = self._env.sim.get_agent_state()
        heading_vector = quaternion_rotate_vector(
            agent_state.rotation.inverse(), np.array([0, 0, -1])
        )
        heading = cartesian_to_polar(-heading_vector[2], heading_vector[0])[1]
        return {
            "position": agent_state.position.tolist(),
            "heading": heading,
            "stop": self._env.task.is_stop_called,
        }
