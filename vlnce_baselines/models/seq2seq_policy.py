import torch
import torch.nn as nn
import torch.nn.functional as F
from gym import Space
from habitat import Config
from habitat_baselines.common.baseline_registry import BaselineRegistry
from habitat_baselines.rl.models.rnn_state_encoder import (
    build_rnn_state_encoder,
)
from habitat_baselines.rl.ppo.policy import Net

from vlnce_baselines.common.aux_losses import AuxLosses
from vlnce_baselines.models.encoders import resnet_encoders
from vlnce_baselines.models.encoders.instruction_encoder import (
    InstructionEncoder,
)
from vlnce_baselines.models.policy import ILPolicy


@BaselineRegistry.register_policy
class Seq2SeqPolicy(ILPolicy):
    def __init__(
        self,
        observation_space: Space,
        action_space: Space,
        model_config: Config,
    ):
        super().__init__(
            Seq2SeqNet(
                observation_space=observation_space,
                model_config=model_config,
                num_actions=action_space.n,
            ),
            action_space.n,
        )

    @classmethod
    def from_config(
        cls, config: Config, observation_space: Space, action_space: Space
    ):
        config.defrost()
        config.MODEL.TORCH_GPU_ID = config.TORCH_GPU_ID
        config.freeze()

        return cls(
            observation_space=observation_space,
            action_space=action_space,
            model_config=config.MODEL,
        )


class Seq2SeqNet(Net):
    """A baseline sequence to sequence network that performs single modality
    encoding of the instruction, RGB, and depth observations. These encodings
    are concatentated and fed to an RNN. Finally, a distribution over discrete
    actions (FWD, L, R, STOP) is produced.
    """

    def __init__(
        self, observation_space: Space, model_config: Config, num_actions: int
    ):
        super().__init__()
        self.model_config = model_config

        # Init the instruction encoder
        self.instruction_encoder = InstructionEncoder(
            model_config.INSTRUCTION_ENCODER
        )

        # Init the depth encoder
        assert model_config.DEPTH_ENCODER.cnn_type in ["VlnResnetDepthEncoder"]
        self.depth_encoder = getattr(
            resnet_encoders, model_config.DEPTH_ENCODER.cnn_type
        )(
            observation_space,
            output_size=model_config.DEPTH_ENCODER.output_size,
            checkpoint=model_config.DEPTH_ENCODER.ddppo_checkpoint,
            backbone=model_config.DEPTH_ENCODER.backbone,
            trainable=model_config.DEPTH_ENCODER.trainable,
        )

        # Init the RGB visual encoder
        assert model_config.RGB_ENCODER.cnn_type in [
            "TorchVisionResNet18",
            "TorchVisionResNet50",
        ]
        self.rgb_encoder = getattr(
            resnet_encoders, model_config.RGB_ENCODER.cnn_type
        )(
            model_config.RGB_ENCODER.output_size,
            normalize_visual_inputs=model_config.normalize_rgb,
            trainable=model_config.RGB_ENCODER.trainable,
            spatial_output=False,
        )

        if model_config.SEQ2SEQ.use_prev_action:
            self.prev_action_embedding = nn.Embedding(num_actions + 1, 32)

        # Init the RNN state decoder
        rnn_input_size = (
            self.instruction_encoder.output_size
            + model_config.DEPTH_ENCODER.output_size
            + model_config.RGB_ENCODER.output_size
        )

        if model_config.SEQ2SEQ.use_prev_action:
            rnn_input_size += self.prev_action_embedding.embedding_dim

        self.state_encoder = build_rnn_state_encoder(
            input_size=rnn_input_size,
            hidden_size=model_config.STATE_ENCODER.hidden_size,
            rnn_type=model_config.STATE_ENCODER.rnn_type,
            num_layers=1,
        )

        self.progress_monitor = nn.Linear(
            self.model_config.STATE_ENCODER.hidden_size, 1
        )

        self._init_layers()

        self.train()

    @property
    def output_size(self):
        return self.model_config.STATE_ENCODER.hidden_size

    @property
    def is_blind(self):
        return self.rgb_encoder.is_blind or self.depth_encoder.is_blind

    @property
    def num_recurrent_layers(self):
        return self.state_encoder.num_recurrent_layers

    def _init_layers(self):
        nn.init.kaiming_normal_(
            self.progress_monitor.weight, nonlinearity="tanh"
        )
        nn.init.constant_(self.progress_monitor.bias, 0)

    def forward(self, observations, rnn_states, prev_actions, masks):
        instruction_embedding = self.instruction_encoder(observations)
        depth_embedding = self.depth_encoder(observations)
        rgb_embedding = self.rgb_encoder(observations)

        if self.model_config.ablate_instruction:
            instruction_embedding = instruction_embedding * 0
        if self.model_config.ablate_depth:
            depth_embedding = depth_embedding * 0
        if self.model_config.ablate_rgb:
            rgb_embedding = rgb_embedding * 0

        x = torch.cat(
            [instruction_embedding, depth_embedding, rgb_embedding], dim=1
        )

        if self.model_config.SEQ2SEQ.use_prev_action:
            prev_actions_embedding = self.prev_action_embedding(
                ((prev_actions.float() + 1) * masks).long().view(-1)
            )
            x = torch.cat([x, prev_actions_embedding], dim=1)

        x, rnn_states_out = self.state_encoder(x, rnn_states, masks)

        if self.model_config.PROGRESS_MONITOR.use and AuxLosses.is_active():
            progress_hat = torch.tanh(self.progress_monitor(x))
            progress_loss = F.mse_loss(
                progress_hat.squeeze(1),
                observations["progress"],
                reduction="none",
            )
            AuxLosses.register_loss(
                "progress_monitor",
                progress_loss,
                self.model_config.PROGRESS_MONITOR.alpha,
            )

        return x, rnn_states_out
