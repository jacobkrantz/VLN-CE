import abc
from typing import Any

from habitat_baselines.rl.ppo.policy import Policy
from habitat_baselines.utils.common import CategoricalNet

from vlnce_baselines.models.utils import CustomFixedCategorical


class ILPolicy(Policy, metaclass=abc.ABCMeta):
    def __init__(self, net, dim_actions):
        """Defines an imitation learning policy as having functions act() and
        build_distribution().
        """
        super(Policy, self).__init__()
        self.net = net
        self.dim_actions = dim_actions

        self.action_distribution = CategoricalNet(
            self.net.output_size, self.dim_actions
        )

    def forward(self, *x):
        raise NotImplementedError

    def act(
        self,
        observations,
        rnn_states,
        prev_actions,
        masks,
        deterministic=False,
    ):
        features, rnn_states = self.net(
            observations, rnn_states, prev_actions, masks
        )
        distribution = self.action_distribution(features)

        if deterministic:
            action = distribution.mode()
        else:
            action = distribution.sample()

        return action, rnn_states

    def get_value(self, *args: Any, **kwargs: Any):
        raise NotImplementedError

    def evaluate_actions(self, *args: Any, **kwargs: Any):
        raise NotImplementedError

    def build_distribution(
        self, observations, rnn_states, prev_actions, masks
    ) -> CustomFixedCategorical:
        features, rnn_states = self.net(
            observations, rnn_states, prev_actions, masks
        )
        return self.action_distribution(features)
