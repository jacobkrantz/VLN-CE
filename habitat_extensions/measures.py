import gzip
import json
from typing import Any

import numpy as np
from dtw import dtw
from fastdtw import fastdtw
from habitat.config import Config
from habitat.core.embodied_task import EmbodiedTask, Measure
from habitat.core.registry import registry
from habitat.core.simulator import Simulator


@registry.register_measure
class PathLength(Measure):
    r"""Path Length (PL)

    PL = sum(geodesic_distance(agent_prev_position, agent_position)
            over all agent positions.
    """

    def __init__(self, sim: Simulator, config: Config, *args: Any, **kwargs: Any):
        self._previous_position = None
        self._start_end_episode_distance = None
        self._agent_episode_distance = None
        self._sim = sim
        self._config = config

        super().__init__(**kwargs)

    def reset_metric(self, episode, *args: Any, **kwargs: Any):
        self._previous_position = self._sim.get_agent_state().position.tolist()
        self._start_end_episode_distance = self._sim.geodesic_distance(
            self._previous_position, episode.goals[0].position
        )
        self._agent_episode_distance = 0.0
        self._metric = None

    def _euclidean_distance(self, position_a, position_b):
        return np.linalg.norm(np.array(position_b) - np.array(position_a), ord=2)

    def update_metric(self, episode, action, *args: Any, **kwargs: Any):
        current_position = self._sim.get_agent_state().position.tolist()

        distance_to_target = self._sim.geodesic_distance(
            current_position, episode.goals[0].position
        )

        self._agent_episode_distance += self._euclidean_distance(
            current_position, self._previous_position
        )

        self._previous_position = current_position

        self._metric = self._agent_episode_distance

    @staticmethod
    def _get_uuid(*args: Any, **kwargs: Any):
        return "path_length"


@registry.register_measure
class OracleNavigationError(Measure):
    r"""Oracle Navigation Error (ONE)

    ONE = min(geosdesic_distance(agent_pos, goal))
            over all agent_pos in agent path.

    This computes oracle navigation error for every update regardless of
    whether or not the end of the episode has been reached.
    """

    def __init__(self, *args: Any, sim: Simulator, config: Config, **kwargs: Any):
        self._sim = sim
        self._config = config
        super().__init__()

    def reset_metric(self, *args: Any, episode, **kwargs: Any):
        self._metric = float("inf")

    def update_metric(self, *args: Any, episode, action, **kwargs: Any):
        current_position = self._sim.get_agent_state().position.tolist()
        distance_to_target = self._sim.geodesic_distance(
            current_position, episode.goals[0].position
        )
        if distance_to_target < self._metric:
            self._metric = distance_to_target

    @staticmethod
    def _get_uuid(*args: Any, **kwargs: Any):
        return "oracle_navigation_error"


@registry.register_measure
class OracleSuccess(Measure):
    r"""Oracle Success Rate (OSR)

    OSR = I(ONE <= goal_radius),
    where ONE is Oracle Navigation Error.
    """

    def __init__(self, *args: Any, sim: Simulator, config: Config, **kwargs: Any):
        self._sim = sim
        self._config = config
        super().__init__()

    def reset_metric(self, *args: Any, episode, **kwargs: Any):
        self._metric = 0

    def update_metric(
        self, *args: Any, episode, action, task: EmbodiedTask, **kwargs: Any
    ):
        if self._metric:
            # skip, already had oracle success
            return

        current_position = self._sim.get_agent_state().position.tolist()
        distance_to_target = self._sim.geodesic_distance(
            current_position, episode.goals[0].position
        )

        if distance_to_target < self._config.SUCCESS_DISTANCE:
            self._metric = 1

    @staticmethod
    def _get_uuid(*args: Any, **kwargs: Any):
        return "oracle_success"


@registry.register_measure
class OracleSPL(Measure):
    r"""OracleSPL (Oracle Success weighted by Path Length)

    OracleSPL = max(SPL) over all points in the agent path
    """

    def __init__(self, *args: Any, sim: Simulator, config: Config, **kwargs: Any):
        self._previous_position = None
        self._start_end_episode_distance = None
        self._agent_episode_distance = None
        self._ep_success = None
        self._sim = sim
        self._config = config
        super().__init__()

    def reset_metric(self, *args: Any, episode, **kwargs: Any):
        self._previous_position = self._sim.get_agent_state().position.tolist()
        self._start_end_episode_distance = episode.info["geodesic_distance"]
        self._agent_episode_distance = 0.0
        self._ep_success = 0
        self._metric = 0.0

    def _euclidean_distance(self, position_a, position_b):
        return np.linalg.norm(np.array(position_b) - np.array(position_a), ord=2)

    def update_metric(
        self, *args: Any, episode, action, task: EmbodiedTask, **kwargs: Any
    ):
        if self._ep_success:  # shortest path already found
            return

        current_position = self._sim.get_agent_state().position.tolist()

        self._agent_episode_distance += self._euclidean_distance(
            current_position, self._previous_position
        )
        self._previous_position = current_position

        distance_to_target = self._sim.geodesic_distance(
            current_position, episode.goals[0].position
        )
        if distance_to_target < self._config.SUCCESS_DISTANCE:
            self._ep_success = 1
            self._metric = self._ep_success * (
                self._start_end_episode_distance
                / max(self._start_end_episode_distance, self._agent_episode_distance)
            )

    @staticmethod
    def _get_uuid(*args: Any, **kwargs: Any):
        return "oracle_spl"


@registry.register_measure
class StepsTaken(Measure):
    r"""Counts the number of times update_metric() is called. This is equal to
    the number of times that the agent takes an action. STOP counts as an
    action.
    """

    def __init__(self, *args: Any, sim: Simulator, config: Config, **kwargs: Any):
        self._sim = sim
        self._config = config
        self._metric = 0
        super().__init__()

    def reset_metric(self, *args: Any, episode, **kwargs: Any):
        self._metric = 0

    def update_metric(
        self, *args: Any, episode, action, task: EmbodiedTask, **kwargs: Any
    ):
        self._metric += 1

    @staticmethod
    def _get_uuid(*args: Any, **kwargs: Any):
        return "steps_taken"


@registry.register_measure
class NDTW(Measure):
    r"""NDTW (Normalized Dynamic Time Warping)

    ref: Effective and General Evaluation for Instruction
        Conditioned Navigation using Dynamic Time
        Warping - Magalhaes et. al
    https://arxiv.org/pdf/1907.05446.pdf
    """

    def __init__(self, *args: Any, sim: Simulator, config: Config, **kwargs: Any):
        self._sim = sim
        self._config = config
        self.locations = []
        self.gt_locations = []
        self.dtw_func = fastdtw if config.FDTW else dtw

        gt_path = config.GT_PATH.format(split=config.SPLIT)
        with gzip.open(gt_path, "rt") as f:
            self.gt_json = json.load(f)
        super().__init__()

    @staticmethod
    def _get_uuid(*args: Any, **kwargs: Any):
        return "ndtw"

    def reset_metric(self, *args: Any, episode, **kwargs: Any):
        self.locations.clear()
        self.gt_locations = self.gt_json[str(episode.episode_id)]["locations"]
        self._metric = None

    def _euclidean_distance(self, position_a, position_b):
        return np.linalg.norm(np.array(position_b) - np.array(position_a), ord=2)

    def update_metric(
        self, *args: Any, episode, action, task: EmbodiedTask, **kwargs: Any
    ):
        current_position = self._sim.get_agent_state().position.tolist()
        if len(self.locations) == 0:
            self.locations.append(current_position)
        else:
            if current_position == self.locations[-1]:
                return
            self.locations.append(current_position)

        dtw_distance = self.dtw_func(
            self.locations, self.gt_locations, dist=self._euclidean_distance
        )[0]

        nDTW = np.exp(
            -dtw_distance / (len(self.gt_locations) * self._config.SUCCESS_DISTANCE)
        )
        self._metric = nDTW


@registry.register_measure
class SDTW(Measure):
    r"""SDTW (Success Weighted be nDTW)

    ref: Effective and General Evaluation for Instruction
        Conditioned Navigation using Dynamic Time
        Warping - Magalhaes et. al
    https://arxiv.org/pdf/1907.05446.pdf
    """

    def __init__(self, *args: Any, sim: Simulator, config: Config, **kwargs: Any):
        self._sim = sim
        self._config = config
        self.locations = []
        self.gt_locations = []
        self.dtw_func = fastdtw if config.FDTW else dtw

        gt_path = config.GT_PATH.format(split=config.SPLIT)
        with gzip.open(gt_path, "rt") as f:
            self.gt_json = json.load(f)
        super().__init__()

    @staticmethod
    def _get_uuid(*args: Any, **kwargs: Any):
        return "sdtw"

    def reset_metric(self, *args: Any, episode, **kwargs: Any):
        self.locations.clear()
        self.gt_locations = self.gt_json[str(episode.episode_id)]["locations"]
        self._metric = None

    def _euclidean_distance(self, position_a, position_b):
        return np.linalg.norm(np.array(position_b) - np.array(position_a), ord=2)

    def update_metric(
        self, *args: Any, episode, action, task: EmbodiedTask, **kwargs: Any
    ):
        current_position = self._sim.get_agent_state().position.tolist()
        if len(self.locations) == 0:
            self.locations.append(current_position)
        else:
            if current_position != self.locations[-1]:
                self.locations.append(current_position)

        dtw_distance = self.dtw_func(
            self.locations, self.gt_locations, dist=self._euclidean_distance
        )[0]

        nDTW = np.exp(
            -dtw_distance / (len(self.gt_locations) * self._config.SUCCESS_DISTANCE)
        )

        distance_to_target = self._sim.geodesic_distance(
            current_position, episode.goals[0].position
        )
        if task.is_stop_called and distance_to_target < self._config.SUCCESS_DISTANCE:
            ep_success = 1
        else:
            ep_success = 0

        self._metric = ep_success * nDTW
