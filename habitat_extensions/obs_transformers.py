import copy
import numbers
from typing import Dict, List, Sequence, Tuple, Union

import numpy as np
import torch
from gym import Space, spaces
from habitat.config import Config
from habitat.core.logging import logger
from habitat.core.simulator import Observations
from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_baselines.common.obs_transformers import ObservationTransformer
from habitat_baselines.utils.common import (
    center_crop,
    get_image_height_width,
    overwrite_gym_box_shape,
)
from torch import Tensor


@baseline_registry.register_obs_transformer()
class CenterCropperPerSensor(ObservationTransformer):
    """Center crop the input on a per-sensor basis"""

    sensor_crops: Dict[str, Union[int, Tuple[int, int]]]
    channels_last: bool

    def __init__(
        self,
        sensor_crops: List[Tuple[str, Union[int, Tuple[int, int]]]],
        channels_last: bool = True,
    ) -> None:
        super().__init__()

        self.sensor_crops = dict(sensor_crops)
        for k in self.sensor_crops:
            size = self.sensor_crops[k]
            if isinstance(size, numbers.Number):
                self.sensor_crops[k] = (int(size), int(size))
            assert len(size) == 2, "forced input size must be len of 2 (h, w)"

        self.channels_last = channels_last

    def transform_observation_space(
        self,
        observation_space: Space,
    ) -> Space:
        observation_space = copy.deepcopy(observation_space)
        for key in observation_space.spaces:
            if (
                key in self.sensor_crops
                and observation_space.spaces[key].shape[-3:-1]
                != self.sensor_crops[key]
            ):
                h, w = get_image_height_width(
                    observation_space.spaces[key], channels_last=True
                )
                logger.info(
                    "Center cropping observation size of %s from %s to %s"
                    % (key, (h, w), self.sensor_crops[key])
                )

                observation_space.spaces[key] = overwrite_gym_box_shape(
                    observation_space.spaces[key], self.sensor_crops[key]
                )
        return observation_space

    @torch.no_grad()
    def forward(self, observations: Dict[str, Tensor]) -> Dict[str, Tensor]:
        observations.update(
            {
                sensor: center_crop(
                    observations[sensor],
                    self.sensor_crops[sensor],
                    channels_last=self.channels_last,
                )
                for sensor in self.sensor_crops
                if sensor in observations
            }
        )
        return observations

    @classmethod
    def from_config(cls, config: Config):
        cc_config = config.RL.POLICY.OBS_TRANSFORMS.CENTER_CROPPER_PER_SENSOR
        return cls(cc_config.SENSOR_CROPS)


@baseline_registry.register_obs_transformer()
class ObsStack(ObservationTransformer):
    """Stack multiple sensors into a single sensor observation."""

    def __init__(
        self, sensor_rewrites: List[Tuple[str, Sequence[str]]]
    ) -> None:
        """Args:
        sensor_rewrites: a tuple of rewrites where a rewrite is a list of
        sensor names to be combined into one sensor.
        """
        self.rewrite_dict: Dict[str, Sequence[str]] = dict(sensor_rewrites)
        super(ObsStack, self).__init__()

    def transform_observation_space(
        self,
        observation_space: Space,
    ) -> Space:
        observation_space = copy.deepcopy(observation_space)
        for target_uuid, sensors in self.rewrite_dict.items():
            orig_space = observation_space.spaces[sensors[0]]
            for k in sensors:
                del observation_space.spaces[k]

            low = (
                orig_space.low
                if np.isscalar(orig_space.low)
                else np.min(orig_space.low)
            )
            high = (
                orig_space.high
                if np.isscalar(orig_space.high)
                else np.max(orig_space.high)
            )
            shape = (len(sensors),) + (orig_space.shape)

            observation_space.spaces[target_uuid] = spaces.Box(
                low=low, high=high, shape=shape, dtype=orig_space.dtype
            )

        return observation_space

    @torch.no_grad()
    def forward(self, observations: Observations) -> Observations:
        for new_obs_keys, old_obs_keys in self.rewrite_dict.items():
            new_obs = torch.stack(
                [observations[k] for k in old_obs_keys], axis=1
            )
            for k in old_obs_keys:
                del observations[k]

            observations[new_obs_keys] = new_obs
        return observations

    @classmethod
    def from_config(cls, config: Config):
        return cls(config.RL.POLICY.OBS_TRANSFORMS.OBS_STACK.SENSOR_REWRITES)
