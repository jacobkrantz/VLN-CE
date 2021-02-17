from typing import Any, Dict

import numpy as np
from gym import spaces
from habitat.config import Config
from habitat.core.registry import registry
from habitat.core.simulator import Observations, Sensor, SensorTypes, Simulator
from habitat.sims.habitat_simulator.actions import HabitatSimActions
from habitat.tasks.nav.shortest_path_follower import ShortestPathFollower

from habitat_extensions.shortest_path_follower import (
    ShortestPathFollowerCompat,
)
from habitat_extensions.task import VLNExtendedEpisode


@registry.register_sensor(name="GlobalGPSSensor")
class GlobalGPSSensor(Sensor):
    r"""The agents current location in the global coordinate frame

    Args:
        sim: reference to the simulator for calculating task observations.
        config: Contains the DIMENSIONALITY field for the number of dimensions
                to express the agents position
    Attributes:
        _dimensionality: number of dimensions used to specify the agents position
    """

    cls_uuid: str = "globalgps"

    def __init__(
        self, sim: Simulator, config: Config, *args: Any, **kwargs: Any
    ):
        self._sim = sim
        self._dimensionality = getattr(config, "DIMENSIONALITY", 2)
        assert self._dimensionality in [2, 3]
        super().__init__(config=config)

    def _get_uuid(self, *args: Any, **kwargs: Any):
        return self.cls_uuid

    def _get_sensor_type(self, *args: Any, **kwargs: Any):
        return SensorTypes.POSITION

    def _get_observation_space(self, *args: Any, **kwargs: Any):
        return spaces.Box(
            low=np.finfo(np.float32).min,
            high=np.finfo(np.float32).max,
            shape=(self._dimensionality,),
            dtype=np.float32,
        )

    def get_observation(self, *args: Any, **kwargs: Any):
        return self._sim.get_agent_state().position.astype(np.float32)


@registry.register_sensor
class ShortestPathSensor(Sensor):
    r"""Sensor for observing the action to take that follows the shortest path
    to the goal.

    Args:
        sim: reference to the simulator for calculating task observations.
        config: config for the sensor.
    """

    cls_uuid: str = "shortest_path_sensor"

    def __init__(
        self, sim: Simulator, config: Config, *args: Any, **kwargs: Any
    ):
        super().__init__(config=config)
        if config.USE_ORIGINAL_FOLLOWER:
            self.follower = ShortestPathFollowerCompat(
                sim, config.GOAL_RADIUS, return_one_hot=False
            )
            self.follower.mode = "geodesic_path"
        else:
            self.follower = ShortestPathFollower(
                sim, config.GOAL_RADIUS, return_one_hot=False
            )

    def _get_uuid(self, *args: Any, **kwargs: Any):
        return self.cls_uuid

    def _get_sensor_type(self, *args: Any, **kwargs: Any):
        return SensorTypes.TACTILE

    def _get_observation_space(self, *args: Any, **kwargs: Any):
        return spaces.Box(low=0.0, high=100, shape=(1,), dtype=np.float)

    def get_observation(self, *args: Any, episode, **kwargs: Any):
        best_action = self.follower.get_next_action(episode.goals[0].position)
        return np.array(
            [
                best_action
                if best_action is not None
                else HabitatSimActions.STOP
            ]
        )


@registry.register_sensor
class VLNOracleProgressSensor(Sensor):
    r"""Sensor for observing how much progress has been made towards the goal.

    Args:
        sim: reference to the simulator for calculating task observations.
        config: config for the sensor.
    """

    cls_uuid: str = "progress"

    def __init__(
        self, sim: Simulator, config: Config, *args: Any, **kwargs: Any
    ):
        self._sim = sim
        super().__init__(config=config)

    def _get_uuid(self, *args: Any, **kwargs: Any):
        return self.cls_uuid

    def _get_sensor_type(self, *args: Any, **kwargs: Any):
        # TODO: what is the correct sensor type?
        return SensorTypes.MEASUREMENT

    def _get_observation_space(self, *args: Any, **kwargs: Any):
        return spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float)

    def get_observation(
        self, observations, *args: Any, episode, **kwargs: Any
    ):
        current_position = self._sim.get_agent_state().position.tolist()

        distance_to_target = self._sim.geodesic_distance(
            current_position, episode.goals[0].position
        )

        distance_from_start = episode.info["geodesic_distance"]

        return (distance_from_start - distance_to_target) / distance_from_start


@registry.register_sensor
class RxRInstructionSensor(Sensor):

    cls_uuid: str = "rxr_instruction"

    def __init__(
        self, sim: Simulator, config: Config, *args: Any, **kwargs: Any
    ):
        self.features_path = config.features_path
        super().__init__(config=config)

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return self.cls_uuid

    def _get_sensor_type(self, *args: Any, **kwargs: Any):
        return SensorTypes.MEASUREMENT

    def _get_observation_space(self, *args: Any, **kwargs: Any):
        return spaces.Box(
            low=np.finfo(np.float32).min,
            high=np.finfo(np.float32).max,
            shape=(512, 768),
            dtype=np.float32,
        )

    def get_observation(
        self,
        observations: Dict[str, "Observations"],
        episode: VLNExtendedEpisode,
        **kwargs,
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
