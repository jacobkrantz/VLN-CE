from typing import Optional

import habitat
from habitat import Config, Dataset
from habitat.tasks.nav.shortest_path_follower import ShortestPathFollower
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
