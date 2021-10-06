from copy import deepcopy
from typing import Any

import numpy as np
from gym import Space, spaces
from habitat.config import Config
from habitat.core.registry import registry
from habitat.core.simulator import Sensor, SensorTypes, Simulator
from habitat.sims.habitat_simulator.actions import HabitatSimActions
from habitat.tasks.nav.shortest_path_follower import ShortestPathFollower
from numpy import ndarray

from habitat_extensions.shortest_path_follower import (
    ShortestPathFollowerCompat,
)
from habitat_extensions.task import VLNExtendedEpisode


@registry.register_sensor(name="GlobalGPSSensor")
class GlobalGPSSensor(Sensor):
    """Current agent location in global coordinate frame"""

    cls_uuid: str = "globalgps"

    def __init__(
        self, *args: Any, sim: Simulator, config: Config, **kwargs: Any
    ):
        self._sim = sim
        self._dimensionality = config.DIMENSIONALITY
        super().__init__(config=config)

    def _get_uuid(self, *args: Any, **kwargs: Any):
        return self.cls_uuid

    def _get_sensor_type(self, *args: Any, **kwargs: Any):
        return SensorTypes.POSITION

    def _get_observation_space(self, *args: Any, **kwargs: Any):
        return spaces.Box(
            low=np.finfo(np.float).min,
            high=np.finfo(np.float).max,
            shape=(self._dimensionality,),
            dtype=np.float,
        )

    def get_observation(self, *args: Any, **kwargs: Any):
        agent_position = self._sim.get_agent_state().position
        if self._dimensionality == 2:
            agent_position = np.array([agent_position[0], agent_position[2]])
        return agent_position.astype(np.float32)


@registry.register_sensor
class VLNOracleProgressSensor(Sensor):
    """Relative progress towards goal"""

    cls_uuid: str = "progress"

    def __init__(
        self, sim: Simulator, config: Config, *args: Any, **kwargs: Any
    ) -> None:
        self._sim = sim
        super().__init__(config=config)

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return self.cls_uuid

    def _get_sensor_type(self, *args: Any, **kwargs: Any) -> SensorTypes:
        return SensorTypes.MEASUREMENT

    def _get_observation_space(self, *args: Any, **kwargs: Any) -> Space:
        return spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float)

    def get_observation(self, *args: Any, episode, **kwargs: Any) -> float:
        distance_to_target = self._sim.geodesic_distance(
            self._sim.get_agent_state().position.tolist(),
            episode.goals[0].position,
        )

        # just in case the agent ends up somewhere it shouldn't
        if not np.isfinite(distance_to_target):
            return np.array([0.0])

        distance_from_start = episode.info["geodesic_distance"]
        return np.array(
            [(distance_from_start - distance_to_target) / distance_from_start]
        )


@registry.register_sensor
class AngleFeaturesSensor(Sensor):
    """Returns a fixed array of features describing relative camera poses based
    on https://arxiv.org/abs/1806.02724. This encodes heading angles but
    assumes a single elevation angle.
    """

    cls_uuid: str = "angle_features"

    def __init__(self, *args: Any, config: Config, **kwargs: Any) -> None:
        self.cameras = config.CAMERA_NUM
        super().__init__(config)
        orient = [np.pi * 2 / self.cameras * i for i in range(self.cameras)]
        self.angle_features = np.stack(
            [np.array([np.sin(o), np.cos(o), 0.0, 1.0]) for o in orient]
        )

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return self.cls_uuid

    def _get_sensor_type(self, *args: Any, **kwargs: Any) -> SensorTypes:
        return SensorTypes.HEADING

    def _get_observation_space(self, *args: Any, **kwargs: Any) -> Space:
        return spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self.cameras, 4),
            dtype=np.float,
        )

    def get_observation(self, *args: Any, **kwargs: Any) -> ndarray:
        return deepcopy(self.angle_features)


@registry.register_sensor
class ShortestPathSensor(Sensor):
    """Provides the next action to follow the shortest path to the goal."""

    cls_uuid: str = "shortest_path_sensor"

    def __init__(
        self, *args: Any, sim: Simulator, config: Config, **kwargs: Any
    ):
        super().__init__(config=config)
        cls = ShortestPathFollower
        if config.USE_ORIGINAL_FOLLOWER:
            cls = ShortestPathFollowerCompat
        self.follower = cls(sim, config.GOAL_RADIUS, return_one_hot=False)

    def _get_uuid(self, *args: Any, **kwargs: Any):
        return self.cls_uuid

    def _get_sensor_type(self, *args: Any, **kwargs: Any):
        return SensorTypes.TACTILE

    def _get_observation_space(self, *args: Any, **kwargs: Any):
        return spaces.Box(low=0.0, high=100, shape=(1,), dtype=np.float)

    def get_observation(self, *args: Any, episode, **kwargs: Any):
        best_action = self.follower.get_next_action(episode.goals[0].position)
        if best_action is None:
            best_action = HabitatSimActions.STOP
        return np.array([best_action])


@registry.register_sensor
class RxRInstructionSensor(Sensor):
    """Loads pre-computed intruction features from disk in the baseline RxR
    BERT file format.
    https://github.com/google-research-datasets/RxR/tree/7a6b87ba07959f5176aa336192a8c5dc85ca1b8e#downloading-bert-text-features
    """

    cls_uuid: str = "rxr_instruction"

    def __init__(self, *args: Any, config: Config, **kwargs: Any):
        self.features_path = config.features_path
        super().__init__(config=config)

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return self.cls_uuid

    def _get_sensor_type(self, *args: Any, **kwargs: Any):
        return SensorTypes.MEASUREMENT

    def _get_observation_space(self, *args: Any, **kwargs: Any):
        return spaces.Box(
            low=np.finfo(np.float).min,
            high=np.finfo(np.float).max,
            shape=(512, 768),
            dtype=np.float,
        )

    def get_observation(
        self, *args: Any, episode: VLNExtendedEpisode, **kwargs
    ):
        features = np.load(
            self.features_path.format(
                split=episode.instruction.split,
                id=int(episode.instruction.instruction_id),
                lang=episode.instruction.language.split("-")[0],
            )
        )
        feats = np.zeros((512, 768), dtype=np.float32)
        s = features["features"].shape
        feats[: s[0], : s[1]] = features["features"]
        return feats
