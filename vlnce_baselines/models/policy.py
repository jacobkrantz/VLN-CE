from habitat_baselines.common.utils import CustomFixedCategorical
from habitat_baselines.rl.ppo.policy import Policy


class BasePolicy(Policy):
    def build_distribution(
        self, observations, rnn_hidden_states, prev_actions, masks
    ) -> CustomFixedCategorical:
        features, rnn_hidden_states = self.net(
            observations, rnn_hidden_states, prev_actions, masks
        )
        distribution = self.action_distribution(features)

        return distribution
