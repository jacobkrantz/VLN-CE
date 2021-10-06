from typing import Any

import numpy as np
from gym import spaces
from habitat.core.registry import registry
from habitat.core.simulator import Observations
from habitat.tasks.nav.nav import TeleportAction

from habitat_extensions.utils import (
    compute_heading_to,
    rtheta_to_global_coordinates,
)


@registry.register_task_action
class GoTowardPoint(TeleportAction):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """This waypoint action is parameterized by (r, theta) and simulates
        straight-line movement toward a waypoint, stopping upon collision or
        reaching the specified point.
        """
        super().__init__(*args, **kwargs)
        self._rotate_agent = self._config.rotate_agent

    def step(
        self,
        *args: Any,
        r: float,
        theta: float,
        **kwargs: Any,
    ) -> Observations:
        y_delta = kwargs["y_delta"] if "y_delta" in kwargs else 0.0
        pos = rtheta_to_global_coordinates(
            self._sim, r, theta, y_delta=y_delta, dimensionality=3
        )

        agent_pos = self._sim.get_agent_state().position
        new_pos = np.array(self._sim.step_filter(agent_pos, pos))
        new_rot = self._sim.get_agent_state().rotation
        if np.any(np.isnan(new_pos)) or not self._sim.is_navigable(new_pos):
            new_pos = agent_pos
            if self._rotate_agent:
                new_rot, _ = compute_heading_to(agent_pos, pos)
        else:
            new_pos = np.array(self._sim.pathfinder.snap_point(new_pos))
            if np.any(np.isnan(new_pos)) or not self._sim.is_navigable(
                new_pos
            ):
                new_pos = agent_pos
            if self._rotate_agent:
                new_rot, _ = compute_heading_to(agent_pos, pos)

        assert np.all(np.isfinite(new_pos))
        return self._sim.get_observations_at(
            position=new_pos, rotation=new_rot, keep_agent_at_new_pose=True
        )

    @property
    def action_space(self) -> spaces.Dict:
        coord_range = self.COORDINATE_MAX - self.COORDINATE_MIN
        return spaces.Dict(
            {
                "r": spaces.Box(
                    low=np.array([0.0]),
                    high=np.array([np.sqrt(2 * (coord_range ** 2))]),
                    dtype=np.float,
                ),
                "theta": spaces.Box(
                    low=np.array([0.0]),
                    high=np.array([2 * np.pi]),
                    dtype=np.float,
                ),
            }
        )
