import abc

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from gym import Space
from habitat import Config
from habitat_baselines.rl.models.rnn_state_encoder import RNNStateEncoder
from habitat_baselines.rl.ppo.policy import Net

from vlnce_baselines.common.aux_losses import AuxLosses
from vlnce_baselines.models.encoders.instruction_encoder import InstructionEncoder
from vlnce_baselines.models.encoders.rcm_state_encoder import RCMStateEncoder
from vlnce_baselines.models.encoders.resnet_encoders import (
    TorchVisionResNet50,
    VlnResnetDepthEncoder,
)
from vlnce_baselines.models.policy import BasePolicy


class CMAPolicy(BasePolicy):
    def __init__(
        self, observation_space: Space, action_space: Space, model_config: Config
    ):
        super().__init__(
            CMANet(
                observation_space=observation_space,
                model_config=model_config,
                num_actions=action_space.n,
            ),
            action_space.n,
        )


class CMANet(Net):
    r""" A cross-modal attention (CMA) network that contains:
        Instruction encoder
        Depth encoder
        RGB encoder
        RNN state encoder or CMA state encoder
    """

    def __init__(self, observation_space: Space, model_config: Config, num_actions):
        super().__init__()
        self.model_config = model_config
        model_config.defrost()
        model_config.INSTRUCTION_ENCODER.final_state_only = False
        model_config.freeze()

        # Init the instruction encoder
        self.instruction_encoder = InstructionEncoder(model_config.INSTRUCTION_ENCODER)

        # Init the depth encoder
        assert model_config.DEPTH_ENCODER.cnn_type in [
            "VlnResnetDepthEncoder"
        ], "DEPTH_ENCODER.cnn_type must be VlnResnetDepthEncoder"
        self.depth_encoder = VlnResnetDepthEncoder(
            observation_space,
            output_size=model_config.DEPTH_ENCODER.output_size,
            checkpoint=model_config.DEPTH_ENCODER.ddppo_checkpoint,
            backbone=model_config.DEPTH_ENCODER.backbone,
            spatial_output=True,
        )

        # Init the RGB encoder
        assert model_config.RGB_ENCODER.cnn_type in [
            "TorchVisionResNet50"
        ], "RGB_ENCODER.cnn_type must be TorchVisionResNet50'."

        device = (
            torch.device("cuda", model_config.TORCH_GPU_ID)
            if torch.cuda.is_available()
            else torch.device("cpu")
        )
        self.rgb_encoder = TorchVisionResNet50(
            observation_space,
            model_config.RGB_ENCODER.output_size,
            device,
            spatial_output=True,
        )

        self.prev_action_embedding = nn.Embedding(num_actions + 1, 32)

        self.rcm_state_encoder = model_config.CMA.rcm_state_encoder

        hidden_size = model_config.STATE_ENCODER.hidden_size
        self._hidden_size = hidden_size

        if self.rcm_state_encoder:
            self.state_encoder = RCMStateEncoder(
                self.rgb_encoder.output_shape[0],
                self.depth_encoder.output_shape[0],
                model_config.STATE_ENCODER.hidden_size,
                self.prev_action_embedding.embedding_dim,
            )
        else:
            self.rgb_linear = nn.Sequential(
                nn.AdaptiveAvgPool1d(1),
                nn.Flatten(),
                nn.Linear(
                    self.rgb_encoder.output_shape[0],
                    model_config.RGB_ENCODER.output_size,
                ),
                nn.ReLU(True),
            )
            self.depth_linear = nn.Sequential(
                nn.Flatten(),
                nn.Linear(
                    np.prod(self.depth_encoder.output_shape),
                    model_config.DEPTH_ENCODER.output_size,
                ),
                nn.ReLU(True),
            )

            # Init the RNN state decoder
            rnn_input_size = model_config.DEPTH_ENCODER.output_size
            rnn_input_size += model_config.RGB_ENCODER.output_size
            rnn_input_size += self.prev_action_embedding.embedding_dim

            self.state_encoder = RNNStateEncoder(
                input_size=rnn_input_size,
                hidden_size=model_config.STATE_ENCODER.hidden_size,
                num_layers=1,
                rnn_type=model_config.STATE_ENCODER.rnn_type,
            )

        self._output_size = (
            model_config.STATE_ENCODER.hidden_size
            + model_config.RGB_ENCODER.output_size
            + model_config.DEPTH_ENCODER.output_size
            + self.instruction_encoder.output_size
        )

        self.rgb_kv = nn.Conv1d(
            self.rgb_encoder.output_shape[0],
            hidden_size // 2 + model_config.RGB_ENCODER.output_size,
            1,
        )

        self.depth_kv = nn.Conv1d(
            self.depth_encoder.output_shape[0],
            hidden_size // 2 + model_config.DEPTH_ENCODER.output_size,
            1,
        )

        self.state_q = nn.Linear(hidden_size, hidden_size // 2)
        self.text_k = nn.Conv1d(
            self.instruction_encoder.output_size, hidden_size // 2, 1
        )
        self.text_q = nn.Linear(self.instruction_encoder.output_size, hidden_size // 2)

        self.register_buffer("_scale", torch.tensor(1.0 / ((hidden_size // 2) ** 0.5)))

        self.second_state_compress = nn.Sequential(
            nn.Linear(
                self._output_size + self.prev_action_embedding.embedding_dim,
                self._hidden_size,
            ),
            nn.ReLU(True),
        )

        self.second_state_encoder = RNNStateEncoder(
            input_size=self._hidden_size,
            hidden_size=self._hidden_size,
            num_layers=1,
            rnn_type=model_config.STATE_ENCODER.rnn_type,
        )
        self._output_size = model_config.STATE_ENCODER.hidden_size

        self.progress_monitor = nn.Linear(self.output_size, 1)

        self._init_layers()

        self.train()

    @property
    def output_size(self):
        return self._output_size

    @property
    def is_blind(self):
        return self.rgb_encoder.is_blind or self.depth_encoder.is_blind

    @property
    def num_recurrent_layers(self):
        return self.state_encoder.num_recurrent_layers + (
            self.second_state_encoder.num_recurrent_layers
        )

    def _init_layers(self):
        if self.model_config.PROGRESS_MONITOR.use:
            nn.init.kaiming_normal_(self.progress_monitor.weight, nonlinearity="tanh")
            nn.init.constant_(self.progress_monitor.bias, 0)

    def _attn(self, q, k, v, mask=None):
        logits = torch.einsum("nc, nci -> ni", q, k)

        if mask is not None:
            logits = logits - mask.float() * 1e8

        attn = F.softmax(logits * self._scale, dim=1)

        return torch.einsum("ni, nci -> nc", attn, v)

    def forward(self, observations, rnn_hidden_states, prev_actions, masks):
        r"""
        instruction_embedding: [batch_size x INSTRUCTION_ENCODER.output_size]
        depth_embedding: [batch_size x DEPTH_ENCODER.output_size]
        rgb_embedding: [batch_size x RGB_ENCODER.output_size]
        """
        instruction_embedding = self.instruction_encoder(observations)
        depth_embedding = self.depth_encoder(observations)
        depth_embedding = torch.flatten(depth_embedding, 2)

        rgb_embedding = self.rgb_encoder(observations)
        rgb_embedding = torch.flatten(rgb_embedding, 2)

        prev_actions = self.prev_action_embedding(
            ((prev_actions.float() + 1) * masks).long().view(-1)
        )

        if self.model_config.ablate_instruction:
            instruction_embedding = instruction_embedding * 0
        if self.model_config.ablate_depth:
            depth_embedding = depth_embedding * 0
        if self.model_config.ablate_rgb:
            rgb_embedding = rgb_embedding * 0

        if self.rcm_state_encoder:
            (
                state,
                rnn_hidden_states[0 : self.state_encoder.num_recurrent_layers],
            ) = self.state_encoder(
                rgb_embedding,
                depth_embedding,
                prev_actions,
                rnn_hidden_states[0 : self.state_encoder.num_recurrent_layers],
                masks,
            )
        else:
            rgb_in = self.rgb_linear(rgb_embedding)
            depth_in = self.depth_linear(depth_embedding)

            state_in = torch.cat([rgb_in, depth_in, prev_actions], dim=1)
            (
                state,
                rnn_hidden_states[0 : self.state_encoder.num_recurrent_layers],
            ) = self.state_encoder(
                state_in,
                rnn_hidden_states[0 : self.state_encoder.num_recurrent_layers],
                masks,
            )

        text_state_q = self.state_q(state)
        text_state_k = self.text_k(instruction_embedding)
        text_mask = (instruction_embedding == 0.0).all(dim=1)
        text_embedding = self._attn(
            text_state_q, text_state_k, instruction_embedding, text_mask
        )

        rgb_k, rgb_v = torch.split(
            self.rgb_kv(rgb_embedding), self._hidden_size // 2, dim=1
        )
        depth_k, depth_v = torch.split(
            self.depth_kv(depth_embedding), self._hidden_size // 2, dim=1
        )

        text_q = self.text_q(text_embedding)
        rgb_embedding = self._attn(text_q, rgb_k, rgb_v)
        depth_embedding = self._attn(text_q, depth_k, depth_v)

        x = torch.cat(
            [state, text_embedding, rgb_embedding, depth_embedding, prev_actions], dim=1
        )
        x = self.second_state_compress(x)
        (
            x,
            rnn_hidden_states[self.state_encoder.num_recurrent_layers :],
        ) = self.second_state_encoder(
            x, rnn_hidden_states[self.state_encoder.num_recurrent_layers :], masks
        )

        if self.model_config.PROGRESS_MONITOR.use and AuxLosses.is_active():
            progress_hat = torch.tanh(self.progress_monitor(x))
            progress_loss = F.mse_loss(
                progress_hat.squeeze(1), observations["progress"], reduction="none"
            )
            AuxLosses.register_loss(
                "progress_monitor",
                progress_loss,
                self.model_config.PROGRESS_MONITOR.alpha,
            )

        return x, rnn_hidden_states


if __name__ == "__main__":
    from vlnce_baselines.config.default import get_config
    from gym import spaces

    config = get_config("habitat_baselines/config/vln/il_vln.yaml")

    device = (
        torch.device("cuda", config.TORCH_GPU_ID)
        if torch.cuda.is_available()
        else torch.device("cpu")
    )

    observation_space = spaces.Dict(
        dict(
            rgb=spaces.Box(low=0, high=0, shape=(224, 224, 3), dtype=np.float32),
            depth=spaces.Box(low=0, high=0, shape=(256, 256, 1), dtype=np.float32),
        )
    )

    # Add TORCH_GPU_ID to VLN config for a ResNet layer
    config.defrost()
    config.MODEL.TORCH_GPU_ID = config.TORCH_GPU_ID
    config.freeze()

    action_space = spaces.Discrete(4)

    policy = CMAPolicy(observation_space, action_space, config.MODEL).to(device)

    dummy_instruction = torch.randint(1, 4, size=(4 * 2, 8), device=device)
    dummy_instruction[:, 5:] = 0
    dummy_instruction[0, 2:] = 0

    obs = dict(
        rgb=torch.randn(4 * 2, 224, 224, 3, device=device),
        depth=torch.randn(4 * 2, 256, 256, 1, device=device),
        instruction=dummy_instruction,
        progress=torch.randn(4 * 2, 1, device=device),
    )

    hidden_states = torch.randn(
        policy.net.state_encoder.num_recurrent_layers,
        2,
        policy.net._hidden_size,
        device=device,
    )
    prev_actions = torch.randint(0, 3, size=(4 * 2, 1), device=device)
    masks = torch.ones(4 * 2, 1, device=device)

    AuxLosses.activate()

    policy.evaluate_actions(obs, hidden_states, prev_actions, masks, prev_actions)
