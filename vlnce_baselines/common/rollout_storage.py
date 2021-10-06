from collections import defaultdict
from typing import DefaultDict, Dict, Iterator, Tuple

import torch
from gym import Space
from habitat.core.simulator import Observations
from habitat_baselines.common.rollout_storage import RolloutStorage
from torch import Tensor


class ActionDictRolloutStorage(RolloutStorage):
    """A RolloutStorage container for actions consisting of pano, offset, and
    distance components.
    """

    def __init__(
        self,
        num_steps: int,
        num_envs: int,
        observation_space: Space,
        recurrent_hidden_state_size: int,
        num_recurrent_layers: int = 1,
        continuous_offset: bool = True,
        continuous_distance: bool = True,
    ) -> None:
        self.observations = {}

        for sensor in observation_space.spaces:
            self.observations[sensor] = torch.zeros(
                num_steps + 1,
                num_envs,
                *observation_space.spaces[sensor].shape,
            )

        self.recurrent_hidden_states = torch.zeros(
            num_steps + 1,
            num_envs,
            num_recurrent_layers,
            recurrent_hidden_state_size,
        )

        self.rewards = torch.zeros(num_steps, num_envs, 1)
        self.value_preds = torch.zeros(num_steps + 1, num_envs, 1)
        self.returns = torch.zeros(num_steps + 1, num_envs, 1)

        self.action_log_probs = torch.zeros(num_steps, num_envs, 1)

        self.actions = {
            k: torch.zeros(num_steps, num_envs, 1)
            for k in ["pano", "offset", "distance"]
        }
        self.prev_actions = {
            k: torch.zeros(num_steps + 1, num_envs, 1)
            for k in ["pano", "offset", "distance"]
        }
        self.prev_actions["pano"] = self.prev_actions["pano"].long()
        if not continuous_distance:
            self.prev_actions["distance"] = self.prev_actions[
                "distance"
            ].long()
        if not continuous_offset:
            self.prev_actions["offset"] = self.prev_actions["offset"].long()

        self.masks = torch.zeros(num_steps + 1, num_envs, 1, dtype=torch.uint8)

        self.num_steps = num_steps
        self.step = 0

    def to(self, device: torch.device) -> None:
        for sensor in self.observations:
            self.observations[sensor] = self.observations[sensor].to(device)

        self.recurrent_hidden_states = self.recurrent_hidden_states.to(device)
        self.rewards = self.rewards.to(device)
        self.value_preds = self.value_preds.to(device)
        self.returns = self.returns.to(device)

        for k in self.actions:
            self.actions[k] = self.actions[k].to(device)
            self.prev_actions[k] = self.prev_actions[k].to(device)

        self.action_log_probs = self.action_log_probs.to(device)
        self.masks = self.masks.to(device)

    def insert(
        self,
        observations: Observations,
        recurrent_hidden_states: Tensor,
        action: Dict[str, Tensor],
        action_log_probs: Tensor,
        value_preds: Tensor,
        rewards: Tensor,
        masks: Tensor,
    ) -> None:
        for sensor in observations:
            self.observations[sensor][self.step + 1].copy_(
                observations[sensor]
            )
        self.recurrent_hidden_states[self.step + 1].copy_(
            recurrent_hidden_states
        )

        for k in action:
            self.actions[k][self.step].copy_(action[k])
            self.prev_actions[k][self.step + 1].copy_(action[k])

        self.action_log_probs[self.step].copy_(action_log_probs)
        self.value_preds[self.step].copy_(value_preds)
        self.rewards[self.step].copy_(rewards)
        self.masks[self.step + 1].copy_(masks)
        self.step = self.step + 1

    def after_update(self) -> None:
        for sensor in self.observations:
            self.observations[sensor][0].copy_(
                self.observations[sensor][self.step]
            )

        self.recurrent_hidden_states[0].copy_(
            self.recurrent_hidden_states[self.step]
        )
        self.masks[0].copy_(self.masks[self.step])
        for k in self.prev_actions:
            self.prev_actions[k][0].copy_(self.prev_actions[k][self.step])
        self.step = 0

    def compute_returns(
        self, next_value: Tensor, use_gae: bool, gamma: float, tau: float
    ) -> None:
        if use_gae:
            self.value_preds[self.step] = next_value
            gae = 0
            for step in reversed(range(self.step)):
                delta = (
                    self.rewards[step]
                    + gamma * self.value_preds[step + 1] * self.masks[step + 1]
                    - self.value_preds[step]
                )
                gae = delta + gamma * tau * self.masks[step + 1] * gae
                self.returns[step] = gae + self.value_preds[step]
                assert not torch.isnan(self.returns[step]).any(), (
                    f"Return is NaN.\nreward:\t{self.rewards[step]}"
                    f"\ngae:\t{gae}\ndelta:\t{delta}"
                    f"\nvalue_preds: {self.value_preds[step]}"
                )
        else:
            self.returns[self.step] = next_value
            for step in reversed(range(self.step)):
                self.returns[step] = (
                    self.returns[step + 1] * gamma * self.masks[step + 1]
                    + self.rewards[step]
                )

    def recurrent_generator(
        self, advantages: Tensor, num_mini_batch: int
    ) -> Iterator[
        Tuple[
            DefaultDict[str, Tensor],
            Tensor,
            DefaultDict[str, Tensor],
            DefaultDict[str, Tensor],
            Tensor,
            Tensor,
            Tensor,
            Tensor,
            Tensor,
            DefaultDict[str, Tensor],
        ]
    ]:
        """The yielded `actions_batch` and `prev_actions_batch` are
        dictionaries with keys ["pano", "offset", "distance"] whose
        values are batched action elements.
        """
        num_processes = self.rewards.size(1)
        assert num_processes >= num_mini_batch, (
            "Trainer requires the number of processes ({}) "
            "to be greater than or equal to the number of "
            "trainer mini batches ({}).".format(num_processes, num_mini_batch)
        )
        num_envs_per_batch = num_processes // num_mini_batch
        perm = torch.randperm(num_processes)
        for start_ind in range(0, num_processes, num_envs_per_batch):
            observations_batch = defaultdict(list)

            recurrent_hidden_states_batch = []
            old_action_log_probs_batch = []
            prev_actions_batch = defaultdict(list)
            actions_batch = defaultdict(list)
            value_preds_batch = []
            return_batch = []
            masks_batch = []
            adv_targ = []

            for offset in range(num_envs_per_batch):
                ind = perm[start_ind + offset]

                for sensor in self.observations:
                    observations_batch[sensor].append(
                        self.observations[sensor][: self.step, ind]
                    )

                recurrent_hidden_states_batch.append(
                    self.recurrent_hidden_states[0, ind]
                )

                for k in self.actions:
                    actions_batch[k].append(self.actions[k][: self.step, ind])
                    prev_actions_batch[k].append(
                        self.prev_actions[k][: self.step, ind]
                    )

                old_action_log_probs_batch.append(
                    self.action_log_probs[: self.step, ind]
                )
                value_preds_batch.append(self.value_preds[: self.step, ind])
                return_batch.append(self.returns[: self.step, ind])
                masks_batch.append(self.masks[: self.step, ind])

                adv_targ.append(advantages[: self.step, ind])

            T, N = self.step, num_envs_per_batch

            # These are all tensors of size (T, N, -1)
            for sensor in observations_batch:
                observations_batch[sensor] = torch.stack(
                    observations_batch[sensor], 1
                )

            for k in self.actions:
                actions_batch[k] = torch.stack(actions_batch[k], 1)
                prev_actions_batch[k] = torch.stack(prev_actions_batch[k], 1)

            old_action_log_probs_batch = torch.stack(
                old_action_log_probs_batch, 1
            )
            value_preds_batch = torch.stack(value_preds_batch, 1)
            return_batch = torch.stack(return_batch, 1)
            masks_batch = torch.stack(masks_batch, 1)
            adv_targ = torch.stack(adv_targ, 1)

            # States is just a (num_recurrent_layers, N, -1) tensor
            recurrent_hidden_states_batch = torch.stack(
                recurrent_hidden_states_batch, 0
            )

            # Flatten the (T, N, ...) tensors to (T * N, ...)
            for sensor in observations_batch:
                observations_batch[sensor] = self._flatten_helper(
                    T, N, observations_batch[sensor]
                )

            for k in self.actions:
                actions_batch[k] = self._flatten_helper(T, N, actions_batch[k])
                prev_actions_batch[k] = self._flatten_helper(
                    T, N, prev_actions_batch[k]
                )

            old_action_log_probs_batch = self._flatten_helper(
                T, N, old_action_log_probs_batch
            )
            value_preds_batch = self._flatten_helper(T, N, value_preds_batch)
            return_batch = self._flatten_helper(T, N, return_batch)
            masks_batch = self._flatten_helper(T, N, masks_batch)
            adv_targ = self._flatten_helper(T, N, adv_targ)

            yield (
                observations_batch,
                recurrent_hidden_states_batch,
                actions_batch,
                prev_actions_batch,
                value_preds_batch,
                return_batch,
                masks_batch,
                old_action_log_probs_batch,
                adv_targ,
            )

    @staticmethod
    def _flatten_helper(t: int, n: int, tensor: Tensor) -> Tensor:
        """Given a tensor of size (t, n, ..), flatten it to size (t*n, ...).
        Args:
            t: first dimension of tensor.
            n: second dimension of tensor.
            tensor: target tensor to be flattened.
        Returns:
            flattened tensor of size (t*n, ...)
        """
        return tensor.view(t * n, *tensor.size()[2:])
