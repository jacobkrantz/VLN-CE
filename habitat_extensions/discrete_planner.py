from typing import List, Tuple

import numpy as np
from habitat.sims.habitat_simulator.actions import HabitatSimActions
from numpy import ndarray


class DiscretePathPlanner:
    """Runs a path planning algorithm that greedily minimizes the Euclidean
    distance to a goal location as given in polar coordinates. The path plan
    is created assuming discrete actions MOVE_FORWARD, TURN_LEFT, and
    TURN_RIGHT. The plan further assumes infinite space devoid of obstacles.
    """

    RAD_15DEG = np.deg2rad(15.0)

    def __init__(
        self,
        forward_distance: float = 0.25,
        turn_angle: float = RAD_15DEG,
        goal_radius: float = 0.13,
        step_limit: int = 200,
    ) -> None:
        """
        Args:
            forward_distance: forward step size in meters
            turn_angle: radians of Left and Right actions
            goal_radius: planner stops when within this radius in meters
            step_limit: maximum number of steps in a path plan.
        """
        assert np.isclose((np.pi * 2) % turn_angle, 0.0)
        self._forward_distance = forward_distance
        self.turn_angle = turn_angle
        self.num_turns_in_circle = int((np.pi * 2) / self.turn_angle)
        self.goal_radius = goal_radius
        self.step_limit = step_limit

    def plan(self, r: float, theta: float) -> List[int]:
        """(r, theta) is a relative waypoint in polar coordinates where r is
        in meters and theta is in radians.
        """
        start_position = np.array([0.0, 0.0])
        current_position = np.array([0.0, 0.0])
        current_heading = 0.0
        goal_position = self.pol2cart_habitat(r, theta)
        path_plan = []

        def distance_to_goal(pos: ndarray) -> float:
            return np.linalg.norm(pos - goal_position)

        while round(distance_to_goal(current_position), 3) > self.goal_radius:
            # generate all possible next forward positions
            # pick the candidate that minimizes distance to goal
            next_position = min(
                self.generate_candidate_positions(
                    current_position, current_heading
                ),
                key=lambda p: distance_to_goal(p[0]),
            )

            # make an action plan for it and add to total action plan
            current_position, current_heading, num_turns = next_position

            if num_turns > self.num_turns_in_circle // 2:
                right_turns = self.num_turns_in_circle - num_turns
                path_plan.extend([HabitatSimActions.TURN_RIGHT] * right_turns)
            else:
                path_plan.extend([HabitatSimActions.TURN_LEFT] * num_turns)

            path_plan.append(HabitatSimActions.MOVE_FORWARD)

            assert len(path_plan) < self.step_limit, "reached step limit"

        # heading should face away from the starting point
        ideal_heading = self.heading_to(start_position, goal_position)
        while current_heading - ideal_heading > (self.turn_angle / 2):
            current_heading -= self.turn_angle
            path_plan.append(HabitatSimActions.TURN_RIGHT)
        while ideal_heading - current_heading > (self.turn_angle / 2):
            current_heading += self.turn_angle
            path_plan.append(HabitatSimActions.TURN_LEFT)

        return path_plan

    def generate_candidate_positions(
        self, position: ndarray, heading: float
    ) -> List[Tuple[ndarray, float, int]]:
        """
        returns:
            [(position, heading, num_left_turns), ...]
        """
        candidates = []
        angle = heading
        for i in range(round((np.pi * 2) / self.turn_angle)):
            pos_delta = self.pol2cart_habitat(self._forward_distance, angle)
            candidates.append((position + pos_delta, angle, i))
            angle = (angle + self.turn_angle) % (np.pi * 2)

        return candidates

    @staticmethod
    def heading_to(position_from: ndarray, position_to: ndarray) -> float:
        # different than utils.compute_heading_to
        delta_x = position_to[0] - position_from[0]
        delta_z = position_to[-1] - position_from[-1]
        xz_angle = np.arctan2(delta_z, delta_x)
        return (xz_angle + np.pi) % (2 * np.pi)

    @staticmethod
    def pol2cart_habitat(rho: float, phi: float) -> ndarray:
        return rho * np.array([-np.cos(phi), -np.sin(phi)])
