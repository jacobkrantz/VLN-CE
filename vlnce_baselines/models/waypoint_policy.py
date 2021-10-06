from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from gym import Space
from habitat.config.default import Config
from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_baselines.rl.ppo.policy import Policy
from torch import Tensor

from vlnce_baselines.models.utils import (
    CustomFixedCategorical,
    TruncatedNormal,
    batched_index_select,
)
from vlnce_baselines.models.waypoint_predictors import WaypointPredictionNet


@baseline_registry.register_policy
class WaypointPolicy(Policy):
    def __init__(
        self,
        observation_space: Space,
        action_space: Space,
        model_config: Config,
    ) -> None:
        super().__init__(
            WaypointPredictionNet(
                observation_space=observation_space,
                model_config=model_config,
            ),
            1,  # ignore the action dimension
        )
        self._config = model_config
        self.wypt_cfg = model_config.WAYPOINT
        self._offset_limit = np.pi / self._config.num_panos

    def forward(self, *x):
        raise NotImplementedError

    def _create_distance_distribution(
        self, var1: Tensor, var2: Optional[Tensor], pano: Tensor
    ) -> Union["CustomFixedCategorical", "TruncatedNormal"]:
        """Creates a distribution for the distance prediction, either discrete
        (CustomFixedCategorical) or continuous (TruncatedNormal).
        """
        if self.wypt_cfg.continuous_distance:
            # var1: mode, var2: variance
            distance_distribution = TruncatedNormal(
                loc=torch.gather(var1, dim=1, index=pano),
                scale=torch.sqrt(torch.gather(var2, dim=1, index=pano)),
                smin=self.wypt_cfg.min_distance_prediction,
                smax=self.wypt_cfg.max_distance_prediction,
            )
        else:
            distance_distribution = CustomFixedCategorical(
                logits=batched_index_select(var1, dim=1, index=pano)
            )

        return distance_distribution

    def _create_offset_distribution(
        self, var1: Tensor, var2: Optional[Tensor], pano: Tensor
    ) -> Union["CustomFixedCategorical", "TruncatedNormal"]:
        """Creates a distribution for the offset prediction, either discrete
        (CustomFixedCategorical) or continuous (TruncatedNormal).
        """
        if self.wypt_cfg.continuous_offset:
            # var1: mode, var2: variance
            offset_distribution = TruncatedNormal(
                loc=torch.gather(var1, dim=1, index=pano),
                scale=torch.sqrt(torch.gather(var2, dim=1, index=pano)),
                smin=-self._offset_limit,
                smax=self._offset_limit,
            )
        else:
            offset_distribution = CustomFixedCategorical(
                logits=batched_index_select(var1, dim=1, index=pano)
            )

        return offset_distribution

    def get_offset_prediction(
        self, offset_distribution, deterministic: bool = False
    ) -> Tuple[Tensor, Tensor, Tensor]:
        if deterministic:
            offset = offset_distribution.mode()
        else:
            offset = offset_distribution.sample()

        offset_log_prob = offset_distribution.log_prob(offset)
        action_offset = self.net.offset_to_continuous(offset)
        variance = offset_distribution.variance
        mode = offset_distribution.mode()

        # ablate the offset prediction
        if not self.wypt_cfg.predict_offset:
            action_offset = torch.zeros_like(action_offset)
            offset = torch.zeros_like(offset)
            if offset.dtype == torch.int64:
                offset *= self.wypt_cfg.discrete_offsets // 2
            variance = torch.zeros_like(variance)

        return offset, action_offset, offset_log_prob, variance, mode

    def get_distance_prediction(
        self, distance_distribution, deterministic: bool = False
    ) -> Tuple[Tensor, Tensor, Tensor]:
        if deterministic:
            distance = distance_distribution.mode()
        else:
            distance = distance_distribution.sample()

        distance_log_prob = distance_distribution.log_prob(distance)
        action_distance = self.net.distance_to_continuous(distance)
        variance = distance_distribution.variance
        mode = distance_distribution.mode()

        # ablate distance prediction to a fixed 0.25m
        if not self.wypt_cfg.predict_distance:
            action_distance = torch.zeros_like(action_distance) + 0.25
            distance = torch.zeros_like(distance)
            if distance.dtype != torch.int64:
                distance = torch.zeros_like(distance) + 0.25
            variance = torch.zeros_like(variance)

        return distance, action_distance, distance_log_prob, variance, mode

    def act(
        self,
        observations: Dict[str, Tensor],
        rnn_states: Tensor,
        prev_actions: Dict[str, Tensor],
        masks: Tensor,
        deterministic: bool = False,
    ) -> Tuple[
        Tensor,
        List[Dict[str, Any]],
        Dict[str, Tensor],
        Dict[str, Tensor],
        Tensor,
        Tensor,
        Tensor,
    ]:
        output = self.net(
            observations,
            rnn_states,
            prev_actions,
            masks,
        )
        (
            pano_stop_distribution,
            offset_variable1,
            offset_variable2,
            distance_variable1,
            distance_variable2,
            x,
            rnn_states_out,
        ) = output

        pano_stop = (
            pano_stop_distribution.mode()
            if deterministic
            else pano_stop_distribution.sample()
        )
        stop = (pano_stop == self._config.num_panos).to(torch.uint8)
        pano = pano_stop % self._config.num_panos

        distance_distribution = self._create_distance_distribution(
            distance_variable1, distance_variable2, pano
        )
        offset_distribution = self._create_offset_distribution(
            offset_variable1, offset_variable2, pano
        )

        (
            distance,
            action_distance,
            distance_log_probs,
            dist_var,
            dist_mode,
        ) = self.get_distance_prediction(distance_distribution, deterministic)
        (
            offset,
            action_offset,
            offset_log_probs,
            ofst_var,
            ofst_mode,
        ) = self.get_offset_prediction(offset_distribution, deterministic)

        actions = []
        radians_per_pano = 2 * np.pi / self._config.num_panos
        theta = (pano * radians_per_pano + action_offset) % (2 * np.pi)
        for i in range(pano_stop.shape[0]):
            if stop[i]:
                actions.append({"action": "STOP"})
            else:
                actions.append(
                    {
                        "action": {
                            "action": "GO_TOWARD_POINT",
                            "action_args": {
                                "r": action_distance[i].item(),
                                "theta": theta[i].item(),
                            },
                        }
                    }
                )

        action_log_probs = pano_stop_distribution.log_prob(pano_stop)

        # only include distance and offset log probs if action != STOP
        pano_mask = (pano_stop != self._config.num_panos).to(
            action_log_probs.dtype
        )
        if self.wypt_cfg.predict_distance:
            action_log_probs = (
                action_log_probs
                + pano_mask
                * self.wypt_cfg.predict_distance
                * distance_log_probs
            )
        if self.wypt_cfg.predict_offset:
            action_log_probs = (
                action_log_probs
                + pano_mask * self.wypt_cfg.predict_offset * offset_log_probs
            )

        value = self.critic(x)
        action_elements = {
            "pano": pano_stop,
            "offset": offset,
            "distance": distance,
        }
        variances = {"distance": dist_var, "offset": ofst_var}
        modes = {"offset": ofst_mode, "distance": dist_mode}

        return (
            value,
            actions,
            action_elements,
            modes,
            variances,
            action_log_probs,
            rnn_states_out,
            pano_stop_distribution,
        )

    def get_value(
        self,
        observations: Dict[str, Tensor],
        rnn_states: Tensor,
        prev_actions: Dict[str, Tensor],
        masks: Tensor,
    ) -> Tensor:
        output = self.net(
            observations,
            rnn_states,
            prev_actions,
            masks,
        )
        hidden_state = output[5]
        return self.critic(hidden_state)

    def evaluate_actions(
        self,
        observations: Dict[str, Tensor],
        rnn_states: Tensor,
        prev_actions: Dict[str, Tensor],
        masks: Tensor,
        action_components: Dict[str, Tensor],
    ) -> Tuple[Tensor, Tensor, Dict[str, Tensor], Tensor]:
        output = self.net(
            observations,
            rnn_states,
            prev_actions,
            masks,
        )

        (
            pano_stop_distribution,
            offset_variable1,
            offset_variable2,
            distance_variable1,
            distance_variable2,
            x,
            rnn_states_out,
        ) = output

        value = self.critic(x)
        pano_log_probs = pano_stop_distribution.log_prob(
            action_components["pano"]
        )

        idx = (
            action_components["pano"].to(torch.int64) % self._config.num_panos
        )

        distance_distribution = self._create_distance_distribution(
            distance_variable1, distance_variable2, idx
        )
        offset_distribution = self._create_offset_distribution(
            offset_variable1, offset_variable2, idx
        )

        # only include distance and offset log probs if the action included them
        pano_mask = (action_components["pano"] != self._config.num_panos).to(
            pano_log_probs.dtype
        )
        d_mask = pano_mask * self.wypt_cfg.predict_distance
        o_mask = pano_mask * self.wypt_cfg.predict_offset

        distance_log_probs = d_mask * distance_distribution.log_prob(
            action_components["distance"]
        )
        offset_log_probs = o_mask * offset_distribution.log_prob(
            action_components["offset"]
        )
        action_log_probs = (
            pano_log_probs + distance_log_probs + offset_log_probs
        )
        entropy = {
            "pano": pano_stop_distribution.entropy(),
            "offset": (o_mask * offset_distribution.entropy()).squeeze(1),
            "distance": (d_mask * distance_distribution.entropy()).squeeze(1),
        }

        return (
            value,
            action_log_probs,
            entropy,
            rnn_states_out,
        )

    @classmethod
    def from_config(
        cls, config: Config, observation_space: Space, action_space: Space
    ):
        config.defrost()
        config.MODEL.num_panos = config.TASK_CONFIG.TASK.PANO_ROTATIONS
        config.freeze()

        return cls(
            observation_space=observation_space,
            action_space=action_space,
            model_config=config.MODEL,
        )
