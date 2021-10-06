from typing import Any, Dict, Optional, Tuple, Union

import habitat
import numpy as np
from habitat import Config, Dataset
from habitat.core.simulator import Observations
from habitat.tasks.utils import cartesian_to_polar
from habitat.utils.geometry_utils import quaternion_rotate_vector
from habitat_baselines.common.baseline_registry import baseline_registry

from habitat_extensions.discrete_planner import DiscretePathPlanner
from habitat_extensions.utils import generate_video, navigator_video_frame


@baseline_registry.register_env(name="VLNCEDaggerEnv")
class VLNCEDaggerEnv(habitat.RLEnv):
    def __init__(self, config: Config, dataset: Optional[Dataset] = None):
        super().__init__(config.TASK_CONFIG, dataset)

    def get_reward_range(self) -> Tuple[float, float]:
        # We don't use a reward for DAgger, but the baseline_registry requires
        # we inherit from habitat.RLEnv.
        return (0.0, 0.0)

    def get_reward(self, observations: Observations) -> float:
        return 0.0

    def get_done(self, observations: Observations) -> bool:
        return self._env.episode_over

    def get_info(self, observations: Observations) -> Dict[Any, Any]:
        return self.habitat_env.get_metrics()


@baseline_registry.register_env(name="VLNCEInferenceEnv")
class VLNCEInferenceEnv(habitat.RLEnv):
    def __init__(self, config: Config, dataset: Optional[Dataset] = None):
        super().__init__(config.TASK_CONFIG, dataset)

    def get_reward_range(self):
        return (0.0, 0.0)

    def get_reward(self, observations: Observations):
        return 0.0

    def get_done(self, observations: Observations):
        return self._env.episode_over

    def get_info(self, observations: Observations):
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


@baseline_registry.register_env(name="VLNCEWaypointEnv")
class VLNCEWaypointEnv(habitat.RLEnv):
    def __init__(
        self, config: Config, dataset: Optional[Dataset] = None
    ) -> None:
        self._rl_config = config.RL
        self._reward_measure_name = self._rl_config.REWARD_MEASURE
        self._success_measure_name = self._rl_config.SUCCESS_MEASURE
        super().__init__(config.TASK_CONFIG, dataset)

    def get_reward_range(self) -> Tuple[float, float]:
        return (
            np.finfo(np.float).min,
            np.finfo(np.float).max,
        )

    def get_reward(self, observations: Observations) -> float:
        return self._env.get_metrics()[self._reward_measure_name]

    def _episode_success(self) -> bool:
        return self._env.get_metrics()[self._success_measure_name]

    def get_done(self, observations: Observations) -> bool:
        return self._env.episode_over or self._episode_success()

    def get_info(self, observations: Observations) -> Dict[str, Any]:
        return self.habitat_env.get_metrics()

    def get_num_episodes(self) -> int:
        return len(self.episodes)


@baseline_registry.register_env(name="VLNCEWaypointEnvDiscretized")
class VLNCEWaypointEnvDiscretized(VLNCEWaypointEnv):
    def __init__(
        self, config: Config, dataset: Optional[Dataset] = None
    ) -> None:
        self.video_option = config.VIDEO_OPTION
        self.video_dir = config.VIDEO_DIR
        self.video_frames = []

        step_size = config.TASK_CONFIG.SIMULATOR.FORWARD_STEP_SIZE
        self.discrete_planner = DiscretePathPlanner(
            forward_distance=step_size,
            turn_angle=np.deg2rad(config.TASK_CONFIG.SIMULATOR.TURN_ANGLE),
            goal_radius=round(step_size / 2, 2) + 0.01,  # 0.13m for 0.25m step
        )
        super().__init__(config, dataset)

    def get_reward(self, *args: Any, **kwargs: Any) -> float:
        return 0.0

    def reset(self) -> Observations:
        observations = self._env.reset()
        if self.video_option:
            agent_state = self._env.sim.get_agent_state()
            start_pos = agent_state.position
            start_heading = agent_state.rotation

            info = self.get_info(observations)
            self.video_frames = [
                navigator_video_frame(
                    observations, info, start_pos, start_heading
                )
            ]

        return observations

    def step(
        self, action: Union[int, str, Dict[str, Any]], *args, **kwargs
    ) -> Tuple[Observations, Any, bool, dict]:
        observations = None
        start_pos, start_heading = None, None

        if self.video_option:
            agent_state = self._env.sim.get_agent_state()
            start_pos = agent_state.position
            start_heading = agent_state.rotation

        if action != "STOP":
            plan = self.discrete_planner.plan(
                r=action["action_args"]["r"],
                theta=action["action_args"]["theta"],
            )
            if len(plan) == 0:
                agent_state = self._env.sim.get_agent_state()
                observations = self._env.sim.get_observations_at(
                    agent_state.position, agent_state.rotation
                )

            for discrete_action in plan:
                observations = self._env.step(discrete_action, *args, **kwargs)
                if self.video_option:
                    info = self.get_info(observations)
                    self.video_frames.append(
                        navigator_video_frame(
                            observations,
                            info,
                            start_pos,
                            start_heading,
                            action,
                        )
                    )

                if self._env.episode_over:
                    break
        else:
            observations = self._env.step(action, *args, **kwargs)
            if self.video_option:
                info = self.get_info(observations)
                self.video_frames.append(
                    navigator_video_frame(
                        observations,
                        info,
                        start_pos,
                        start_heading,
                        action,
                    )
                )

        reward = self.get_reward(observations)
        done = self.get_done(observations)
        info = self.get_info(observations)

        if self.video_option and done:
            generate_video(
                video_option=self.video_option,
                video_dir=self.video_dir,
                images=self.video_frames,
                episode_id=self._env.current_episode.episode_id,
                checkpoint_idx=0,
                metrics={"SPL": round(info["spl"], 5)},
                tb_writer=None,
                fps=8,
            )

        return observations, reward, done, info
