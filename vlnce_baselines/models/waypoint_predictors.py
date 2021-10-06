from typing import Dict, Tuple

import numpy as np
import torch
from gym import Space
from habitat import Config
from habitat_baselines.rl.models.rnn_state_encoder import (
    build_rnn_state_encoder,
)
from habitat_baselines.rl.ppo.policy import Net
from torch import Tensor, nn

from vlnce_baselines.models.encoders import resnet_encoders
from vlnce_baselines.models.encoders.instruction_encoder import (
    InstructionEncoder,
)
from vlnce_baselines.models.utils import (
    CustomFixedCategorical,
    DotProductAttention,
    MultiHeadDotProductAttention,
    TemperatureTanh,
)

PREV_ACTION_DIM = 4
PANO_ATTN_KEY_DIM = 128
ANGLE_FEATURE_SIZE = 4


class WaypointPredictionNet(Net):
    def __init__(self, observation_space: Space, model_config: Config) -> None:
        super().__init__()
        self.model_config = model_config
        self.wypt_cfg = model_config.WAYPOINT
        self._hidden_size = model_config.STATE_ENCODER.hidden_size
        self._num_panos = self.model_config.num_panos

        # Init the instruction encoder
        self.instruction_encoder = InstructionEncoder(
            model_config.INSTRUCTION_ENCODER
        )

        # Init the depth encoder
        cnn_type = model_config.DEPTH_ENCODER.cnn_type
        assert cnn_type in ["VlnResnetDepthEncoder"]
        self.depth_encoder = getattr(resnet_encoders, cnn_type)(
            observation_space,
            output_size=model_config.DEPTH_ENCODER.output_size,
            checkpoint=model_config.DEPTH_ENCODER.ddppo_checkpoint,
            backbone=model_config.DEPTH_ENCODER.backbone,
            spatial_output=True,
        )

        # Init the RGB encoder
        cnn_type = model_config.RGB_ENCODER.cnn_type
        assert cnn_type in ["TorchVisionResNet18", "TorchVisionResNet50"]
        self.rgb_encoder = getattr(resnet_encoders, cnn_type)(
            model_config.RGB_ENCODER.output_size,
            normalize_visual_inputs=self.model_config.normalize_rgb,
            spatial_output=True,
            single_spatial_filter=False,
        )

        # Init the visual history layers
        input_size = model_config.RGB_ENCODER.output_size + PREV_ACTION_DIM
        # observation history
        input_size += model_config.DEPTH_ENCODER.output_size
        input_size += model_config.RGB_ENCODER.output_size

        self.visual_rnn = build_rnn_state_encoder(
            input_size=input_size,
            hidden_size=self._hidden_size,
            rnn_type=model_config.STATE_ENCODER.rnn_type,
            num_layers=1,
        )

        self.rgb_pool_linear = nn.Linear(
            self.rgb_encoder.resnet_layer_size,
            model_config.RGB_ENCODER.output_size,
        )

        self.rgb_hist_linear = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(
                self.rgb_encoder.output_shape[0],
                model_config.RGB_ENCODER.output_size,
            ),
            nn.ReLU(True),
        )
        self.depth_hist_linear = nn.Sequential(
            nn.Flatten(),
            nn.Linear(
                np.prod(self.depth_encoder.output_shape),
                model_config.DEPTH_ENCODER.output_size,
            ),
            nn.ReLU(True),
        )

        # Init the instruction attention
        dk_inst = self._hidden_size // 2
        self.inst_attn_q = nn.Sequential(
            nn.Linear(self._hidden_size, dk_inst), nn.ReLU(True)
        )
        self.inst_attn_k = nn.Conv1d(
            self.instruction_encoder.output_size, dk_inst, 1
        )
        self.inst_attn = DotProductAttention(dk_inst)

        # Init: spatial attention (single frame)
        self.text_q_linear = nn.Linear(
            self.instruction_encoder.output_size, self._hidden_size // 2
        )
        self.rgb_kv_spatial = nn.Conv1d(
            self.rgb_encoder.output_shape[0],
            self._hidden_size // 2 + model_config.RGB_ENCODER.output_size,
            1,
        )
        self.rgb_spatial_attn = DotProductAttention(self._hidden_size // 2)
        self.depth_kv_spatial = nn.Conv1d(
            self.depth_encoder.output_shape[0],
            self._hidden_size // 2 + model_config.DEPTH_ENCODER.output_size,
            1,
        )
        self.depth_spatial_attn = DotProductAttention(self._hidden_size // 2)

        # Init: panorama attention (attend over pano frames)
        d_kv_in = (
            model_config.RGB_ENCODER.output_size
            + model_config.DEPTH_ENCODER.output_size
            + ANGLE_FEATURE_SIZE
        )
        self.pano_attn = MultiHeadDotProductAttention(
            d_q_in=self.instruction_encoder.output_size,
            d_k_in=d_kv_in,
            d_v_in=d_kv_in,
            d_qk=PANO_ATTN_KEY_DIM,
            d_v=PANO_ATTN_KEY_DIM,
            num_heads=1,
            d_out=d_kv_in,
        )

        # Init the recurrence layer
        self.main_state_compress = nn.Sequential(
            nn.Linear(
                (
                    self.instruction_encoder.output_size
                    + model_config.RGB_ENCODER.output_size
                    + model_config.DEPTH_ENCODER.output_size
                    + ANGLE_FEATURE_SIZE
                    + self._hidden_size
                    + PREV_ACTION_DIM
                ),
                self._hidden_size,
            ),
            nn.ReLU(True),
        )
        self.main_state_encoder = build_rnn_state_encoder(
            input_size=self._hidden_size,
            hidden_size=self._hidden_size,
            rnn_type=model_config.STATE_ENCODER.rnn_type,
            num_layers=1,
        )

        # Init: output generation layers
        final_feature_size = (
            model_config.RGB_ENCODER.output_size
            + model_config.DEPTH_ENCODER.output_size
            + ANGLE_FEATURE_SIZE
        )

        self.stop_linear = nn.Linear(self._hidden_size, 1)
        nn.init.constant_(self.stop_linear.bias, 0)

        self.compress_x_linear = nn.Sequential(
            nn.Linear(self._hidden_size, final_feature_size), nn.ReLU(True)
        )

        in_dim = self._hidden_size + final_feature_size
        self._init_distance_linear(in_dim, final_feature_size)
        self._init_offset_linear(in_dim, final_feature_size)

        self.train()

    def distance_to_continuous(self, distance: Tensor) -> Tensor:
        """Maps a distance prediction to a continuous radius r in meters."""
        if self.wypt_cfg.continuous_distance:
            return distance

        range_dist = (
            self.wypt_cfg.max_distance_prediction
            - self.wypt_cfg.min_distance_prediction
        )
        meters_per_distance = range_dist / (
            self.wypt_cfg.discrete_distances - 1
        )
        return self.wypt_cfg.min_distance_prediction + (
            distance * meters_per_distance
        )

    def offset_to_continuous(self, offset: Tensor) -> Tensor:
        """Maps an offset prediction to a continuous offset in radians."""
        if self.wypt_cfg.continuous_offset:
            return offset

        radians_per_pano = 2 * np.pi / self._num_panos
        rad_per_offset = radians_per_pano / (
            self.wypt_cfg.discrete_offsets - 1
        )
        return (-radians_per_pano / 2) + (offset * rad_per_offset)

    @property
    def num_recurrent_layers(self):
        return (
            self.main_state_encoder.num_recurrent_layers
            + self.visual_rnn.num_recurrent_layers
        )

    @property
    def is_blind(self):
        return self.rgb_encoder.is_blind and self.depth_encoder.is_blind

    @property
    def output_size(self):
        return self._hidden_size

    def _init_distance_linear(
        self, in_dim: int, final_feature_size: int
    ) -> None:
        """Initialize the distance output to be either discrete or
        continuous. If continuous, the distribution is TruncatedNormal.
        """
        if self.wypt_cfg.continuous_distance:
            self.distance_linear = nn.Sequential(
                nn.Linear(in_dim, 1), nn.Sigmoid()
            )
            self.distance_var_linear = nn.Sequential(
                nn.Linear(self._hidden_size + final_feature_size, 1),
                nn.Sigmoid(),
            )
        else:
            self.distance_linear = nn.Linear(
                in_dim, self.wypt_cfg.discrete_distances
            )

    def _init_offset_linear(
        self, in_dim: int, final_feature_size: int
    ) -> None:
        """Initialize the offset output to be either discrete or continuous.
        If continuous, the distribution is TruncatedNormal.
        """
        if self.wypt_cfg.continuous_offset:
            self.offset_linear = nn.Sequential(
                nn.Linear(in_dim, 1),
                TemperatureTanh(temperature=self.wypt_cfg.offset_temperature),
            )
            self.offset_scale = np.pi / self._num_panos
            self.offset_var_linear = nn.Sequential(
                nn.Linear(self._hidden_size + final_feature_size, 1),
                nn.Sigmoid(),
            )
        else:
            self.offset_linear = nn.Linear(
                in_dim, self.wypt_cfg.discrete_offsets
            )

    def _map_pano_to_heading_features(self, pano: Tensor) -> Tensor:
        """Maps a tensor of pano ids to heading features
        Args:
            pano: size [B, 1]
        """
        delta_rot = (np.pi * 2) / self._num_panos
        heading = pano * delta_rot
        return torch.cat([torch.sin(heading), torch.cos(heading)], dim=1)

    def _mean_pool_rgb_features(self, features: Tensor) -> Tensor:
        """[B, 12, 2112, 16] -> [B, 256]"""
        x = features

        # remove spatial features
        x = x[:, :, : self.rgb_encoder.resnet_layer_size]

        x = torch.mean(x, dim=3)
        x = self.rgb_pool_linear(x)
        return torch.mean(x, dim=1)

    def forward(
        self,
        observations: Dict[str, Tensor],
        rnn_states: Tensor,
        prev_actions: Dict[str, Tensor],
        masks: Tensor,
    ) -> Tuple[
        CustomFixedCategorical,
        Tensor,
        Tensor,
        Tensor,
        Tensor,
        Tensor,
        Tensor,
    ]:
        """
        Returns:
            pano_stop_distribution: [B, p+1] with range [0, inf]
            offsets: [B, p] with range: [-offset_scale, offset_scale]
            distances: [B, p] with range: [min_distance_prediction, max_distance_prediction]
            offsets_vars: [B, p] with range: [min_offset_var, max_offset_var]
            distances_vars: [B, p] with range: [min_distance_var, max_distance_var]
            x: [B, 512]
            rnn_states: [B, 512]
        """
        assert "rgb" in observations
        assert "depth" in observations
        assert "instruction" in observations
        assert "rgb_history" in observations
        assert "depth_history" in observations
        assert "angle_features" in observations

        assert observations["rgb"].shape[1] == self._num_panos
        assert observations["depth"].shape[1] == self._num_panos

        rnn_states_out = torch.zeros_like(rnn_states)

        # ===========================
        #  Single Modality Encoding
        # ===========================

        instruction_embedding = self.instruction_encoder(observations)

        # encode rgb observations and history
        rgb_obs = torch.cat(
            [
                observations["rgb"],
                (
                    observations["rgb_history"].permute(1, 2, 3, 0)
                    * masks.squeeze(1)
                )
                .permute(3, 0, 1, 2)
                .unsqueeze(1),
            ],
            dim=1,
        )

        rgb_size = rgb_obs.size()
        rgb_obs = rgb_obs.view((rgb_size[0] * rgb_size[1], *rgb_size[2:5]))
        rgb_embedding = self.rgb_encoder({"rgb": rgb_obs})
        rgb_embedding = torch.flatten(
            rgb_embedding.view(*rgb_size[0:2], *rgb_embedding.shape[1:]), 3
        )

        # encode depth observations and history
        depth_obs = torch.cat(
            [
                observations["depth"],
                (
                    observations["depth_history"].permute(1, 2, 3, 0)
                    * masks.squeeze(1)
                )
                .permute(3, 0, 1, 2)
                .unsqueeze(1),
            ],
            dim=1,
        )

        depth_size = depth_obs.size()
        depth_obs = depth_obs.view(
            (depth_size[0] * depth_size[1],) + depth_size[2:5]
        )
        depth_embedding = self.depth_encoder({"depth": depth_obs})
        depth_embedding = torch.flatten(
            depth_embedding.view(depth_size[0:2] + depth_embedding.shape[1:]),
            3,
        )

        # split time t embeddings from time t-1 embeddings
        rgb_history = rgb_embedding[:, self._num_panos]
        rgb_embedding = rgb_embedding[:, : self._num_panos].contiguous()
        depth_history = depth_embedding[:, self._num_panos]
        depth_embedding = depth_embedding[:, : self._num_panos].contiguous()

        if len(prev_actions["pano"].shape) == 1:
            for k in prev_actions:
                prev_actions[k] = prev_actions[k].unsqueeze(1)

        prev_actions = (
            torch.cat(
                [
                    self._map_pano_to_heading_features(prev_actions["pano"]),
                    self.offset_to_continuous(prev_actions["offset"]),
                    self.distance_to_continuous(prev_actions["distance"]),
                ],
                dim=1,
            ).float()
            * masks
        )

        # ===========================
        #     Modality Ablations
        # ===========================

        if self.model_config.ablate_instruction:
            instruction_embedding *= 0
        if self.model_config.ablate_rgb:
            rgb_embedding = rgb_embedding * 0
            rgb_history *= 0
        if self.model_config.ablate_depth:
            depth_embedding *= 0
            depth_history *= 0

        # ===========================
        #     Visual History: GRU
        # ===========================

        rnn_inputs = [
            self._mean_pool_rgb_features(rgb_embedding),
            prev_actions,
            self.rgb_hist_linear(rgb_history),
            self.depth_hist_linear(depth_history),
        ]

        (
            visual_hist_feats,
            rnn_states_out[:, 0 : self.visual_rnn.num_recurrent_layers],
        ) = self.visual_rnn(
            torch.cat(rnn_inputs, dim=1),
            rnn_states[:, 0 : self.visual_rnn.num_recurrent_layers],
            masks,
        )

        # ===========================
        #    Instruction Attention
        # ===========================

        text_embedding = self.inst_attn(
            Q=self.inst_attn_q(visual_hist_feats),
            K=self.inst_attn_k(instruction_embedding),
            V=instruction_embedding,
            mask=(instruction_embedding == 0.0).all(dim=1),
        )

        # ===========================
        #     Spatial Attention
        # ===========================

        # flatten pano frames for spatial attention
        batch_size = rgb_embedding.shape[0]
        flat_rgb_embedding = rgb_embedding.view(
            rgb_embedding.shape[0] * rgb_embedding.shape[1],
            *rgb_embedding.shape[2:],
        )
        flat_depth_embedding = depth_embedding.view(
            depth_embedding.shape[0] * depth_embedding.shape[1],
            *depth_embedding.shape[2:],
        )

        # prepare text query
        text_q_spatial = self.text_q_linear(text_embedding)  # [B, 256]
        text_q_spatial = text_q_spatial.repeat_interleave(
            self._num_panos
        ).view(
            text_q_spatial.shape[0] * self._num_panos,
            *text_q_spatial.shape[1:],
        )  # [B*12, 256]

        # split K, V for dot product attention
        rgb_kv_in = self.rgb_kv_spatial(flat_rgb_embedding)  # [B*12, 512, 16]
        rgb_k_spatial, rgb_v_spatial = torch.split(  # k: [B*12, 256, 16]
            rgb_kv_in, self._hidden_size // 2, dim=1  # v: [B*12, 256, 16]
        )
        depth_kv_in = self.depth_kv_spatial(
            flat_depth_embedding
        )  # [B*12, 384, 16]
        depth_k_spatial, depth_v_spatial = torch.split(  # k: [B*12, 256, 16]
            depth_kv_in, self._hidden_size // 2, dim=1  # v: [B*12, 128, 16]
        )

        # perform scaled dot product attention
        spatial_attended_rgb = self.rgb_spatial_attn(  # [B*12, 256]
            text_q_spatial, rgb_k_spatial, rgb_v_spatial
        )
        spatial_attended_depth = self.depth_spatial_attn(  # [B*12, 128]
            text_q_spatial, depth_k_spatial, depth_v_spatial
        )

        # un-flatten spatial features to [B, 12, _]
        spatial_attended_rgb = spatial_attended_rgb.view(
            batch_size,
            spatial_attended_rgb.shape[0] // batch_size,
            *spatial_attended_rgb.shape[1:],
        )
        spatial_attended_depth = spatial_attended_depth.view(
            batch_size,
            spatial_attended_depth.shape[0] // batch_size,
            *spatial_attended_depth.shape[1:],
        )

        # ===========================
        #     Panorama Attention
        # ===========================

        shared_spatial_features = torch.cat(
            [
                spatial_attended_rgb,
                spatial_attended_depth,
                observations["angle_features"],
            ],
            dim=2,
        ).permute(
            0, 2, 1
        )  # [B, _, 12]

        attended_pano_features = self.pano_attn(
            Q=text_embedding,
            K=shared_spatial_features,
            V=shared_spatial_features,
        )

        # ===========================
        #     RNN State Encoder
        # ===========================

        x = torch.cat(
            [
                text_embedding,
                attended_pano_features,
                visual_hist_feats,
                prev_actions,
            ],
            dim=1,
        )

        x = self.main_state_compress(x)
        (
            x,
            rnn_states_out[
                :,
                self.visual_rnn.num_recurrent_layers : self.num_recurrent_layers,
            ],
        ) = self.main_state_encoder(
            x,
            rnn_states[
                :,
                self.visual_rnn.num_recurrent_layers : self.num_recurrent_layers,
            ],
            masks,
        )

        # ===========================
        # Action Distribution Outputs
        # ===========================

        attended_visual_features = torch.cat(
            [
                spatial_attended_rgb,
                spatial_attended_depth,
                observations["angle_features"],
            ],
            dim=2,
        )  # [B, 12, d]

        x_small = self.compress_x_linear(x)  # [B, d]
        x_small = x_small.unsqueeze(1).repeat(
            1, attended_visual_features.size(1), 1
        )  # [B, 12, d]

        dotted_features = (attended_visual_features * x_small).sum(2)
        pano_stop_distribution = CustomFixedCategorical(
            logits=torch.cat([dotted_features, self.stop_linear(x)], dim=1)
        )

        catted_features = torch.cat(
            [
                attended_visual_features,
                x.unsqueeze(1).repeat(1, attended_visual_features.size(1), 1),
            ],
            dim=2,
        )

        # ===========================
        #     Distance Prediction
        # ===========================

        if self.wypt_cfg.continuous_distance:
            distance_variable1 = self.distance_linear(catted_features)
            distance_variable1 = distance_variable1.squeeze(2)
            distance_variable1 = (
                self.wypt_cfg.max_distance_prediction
                - self.wypt_cfg.min_distance_prediction
            ) * distance_variable1 + self.wypt_cfg.min_distance_prediction

            distance_variable2 = (
                self.wypt_cfg.max_distance_var - self.wypt_cfg.min_distance_var
            ) * self.distance_var_linear(catted_features).squeeze(
                2
            ) + self.wypt_cfg.min_distance_var
        else:
            distance_variable1 = self.distance_linear(catted_features)
            distance_variable1 = distance_variable1.squeeze(2)
            distance_variable2 = None

        # ===========================
        #      Offset Prediction
        # ===========================

        if self.wypt_cfg.continuous_offset:
            offset_variable1 = self.offset_scale * self.offset_linear(
                catted_features
            ).squeeze(2)
            offset_variable2 = (
                self.wypt_cfg.max_offset_var - self.wypt_cfg.min_offset_var
            ) * self.offset_var_linear(catted_features).squeeze(
                2
            ) + self.wypt_cfg.min_offset_var
        else:
            offset_variable1 = self.offset_linear(catted_features).squeeze(2)
            offset_variable2 = None

        return (
            pano_stop_distribution,
            offset_variable1,
            offset_variable2,
            distance_variable1,
            distance_variable2,
            x,
            rnn_states_out,
        )
