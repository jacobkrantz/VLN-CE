import torch
import torch.nn as nn
from gym import Space
from habitat import Config
from habitat_baselines.common.baseline_registry import BaselineRegistry
from habitat_baselines.rl.ppo.policy import Net
from vlnce_baselines.models.encoders.min_gpt import GPT, NewGELU
from vlnce_baselines.models.encoders import resnet_encoders
from vlnce_baselines.models.encoders.instruction_encoder import (
    InstructionEncoder, Word2VecEmbeddings, InstructionEncoderWithTransformer
)
from vlnce_baselines.models.policy import ILPolicy
import numpy as np
from torch import Tensor

from vlnce_baselines.models.utils import PositionalEncoding, VanillaMultiHeadAttention


@BaselineRegistry.register_policy
class DecisionTransformerPolicy(ILPolicy):
    def __init__(
        self,
        observation_space: Space,
        action_space: Space,
        model_config: Config,
    ):
        net = "DecisionTransformerNet"
        if hasattr(model_config.DECISION_TRANSFORMER, "net"):
            net = model_config.DECISION_TRANSFORMER.net
        assert net in model_config.DECISION_TRANSFORMER.allowed_models
        print("Training with:", net)
        super().__init__(
            eval(net)(
                observation_space=observation_space,
                model_config=model_config,
                num_actions=action_space.n,
            ),
            action_space.n,
        )

    def act(
        self,
        observations,
        rnn_states,
        prev_actions,
        masks,
        deterministic=False,
    ):
        actions, rnn_states = super().act(observations, rnn_states, prev_actions, masks, deterministic)
        # We just want to return the last action of the transformer sequence...
        return actions[:, -1, :], rnn_states

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


class AbstractDecisionTransformerNet(Net):
    # Decision Transformer where each time step is fed into a GPT backbone.
    # Finally, a distribution over discrete actions (FWD, L, R, STOP) is produced.
    def __init__(
        self, observation_space: Space, model_config: Config, num_actions: int
    ):
        """

        :param observation_space: Delivered by the Habitat Framework
        :param model_config: General config
        :param num_actions: 4 discrete actions (FWD, L, R, STOP)
        """
        super().__init__()
        self.model_config = model_config
        assert model_config.DEPTH_ENCODER.cnn_type in ["VlnResnetDepthEncoder"]
        assert model_config.RGB_ENCODER.cnn_type in [
            "TorchVisionResNet18",
            "TorchVisionResNet50",
        ]

        assert model_config.DECISION_TRANSFORMER.reward_type in model_config.DECISION_TRANSFORMER.allowed_rewards

        n = self.initialize_transformer_step_size()
        self.set_transformer_step_size(n)
        # Init the Depth visual encoder
        self.depth_encoder = getattr(
            resnet_encoders, model_config.DEPTH_ENCODER.cnn_type
        )(
            observation_space,
            output_size=model_config.DEPTH_ENCODER.output_size,
            checkpoint=model_config.DEPTH_ENCODER.ddppo_checkpoint,
            backbone=model_config.DEPTH_ENCODER.backbone,
            trainable=model_config.DEPTH_ENCODER.trainable,
            spatial_output=model_config.DECISION_TRANSFORMER.spatial_output
        )
        # Init the RGB visual encoder
        self.rgb_encoder = getattr(
            resnet_encoders, model_config.RGB_ENCODER.cnn_type
        )(
            model_config.RGB_ENCODER.output_size,
            normalize_visual_inputs=model_config.DECISION_TRANSFORMER.normalize_rgb,
            trainable=model_config.RGB_ENCODER.trainable,
            spatial_output=model_config.DECISION_TRANSFORMER.spatial_output
        )

        if model_config.DECISION_TRANSFORMER.spatial_output:
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

        self.dim_not_included_for_predictions = 2  # Action and reward
        self.exclude_past_action_for_prediction = model_config.DECISION_TRANSFORMER.exclude_past_action_for_prediction
        if not self.exclude_past_action_for_prediction:
            self.dim_not_included_for_predictions = 1
        self.return_to_go_inference = model_config.DECISION_TRANSFORMER.return_to_go_inference

        self.initialize_instruction_encoder()

        self.reward_type = model_config.DECISION_TRANSFORMER.reward_type
        self.action_activation = nn.Sequential(
            nn.Dropout(p=self.model_config.DECISION_TRANSFORMER.activation_action_drop), NewGELU())
        self.instruction_activation = nn.Sequential(
            nn.Dropout(p=self.model_config.DECISION_TRANSFORMER.activation_instruction_drop), NewGELU())
        self.rgb_activation = nn.Sequential(
            nn.Dropout(p=self.model_config.DECISION_TRANSFORMER.activation_rgb_drop), NewGELU())
        self.depth_activation = nn.Sequential(
            nn.Dropout(p=self.model_config.DECISION_TRANSFORMER.activation_depth_drop), NewGELU())
        self.gpt_encoder = GPT(self.model_config.DECISION_TRANSFORMER)
        self.transformer_step_size = self.model_config.DECISION_TRANSFORMER.step_size
        self.embed_timestep = nn.Embedding(model_config.DECISION_TRANSFORMER.episode_horizon,
                                           model_config.DECISION_TRANSFORMER.hidden_dim)
        self.embed_return = nn.Linear(1, model_config.DECISION_TRANSFORMER.hidden_dim)
        self.embed_action = nn.Embedding(num_actions + 1, model_config.DECISION_TRANSFORMER.hidden_dim)
        self.embed_ln = nn.LayerNorm(model_config.DECISION_TRANSFORMER.hidden_dim)
        self.initialize_other_layers()
        self.train()

    def _prepare_embeddings(self, observations):
        """
        read the relevant features from observation and returns it
        :param observations:
        :return: instruction_embedding, depth_embedding, rgb_embedding
        """
        # for all the following keys, we need tto merge the first 2 dimensions
        # [batch, sequence length, all other dimensions] to [batch * sequence length, all other dimensions]

        original_batch_shape = observations["instruction"].shape[0:2]  # excluding the embedding dimentions
        batch_size, seq_length = original_batch_shape
        # the observations were flattened for rnn processing
        # the first dimension is actually equal to sequence length * original batch size.
        # we also retrieve all other dimensions starting at index 1
        shape = lambda tensor: tuple([s for s in original_batch_shape] + [s for s in tensor.shape[1:]])

        # Transpose dimension 0 and 1 and let the last one untouched
        # resize_tensor = lambda tensor: tensor.reshape(shape(tensor)).permute(1,0,-1).contiguous()
        resize_tensor = lambda tensor: tensor.reshape(shape(tensor))

        self._flatten_batch(observations, "rgb")
        self._flatten_batch(observations, "depth")
        self._flatten_batch(observations, "rgb_features")
        self._flatten_batch(observations, "depth_features")
        if not self.model_config.DECISION_TRANSFORMER.use_transformer_encoded_instruction:
            self._flatten_batch(observations, "instruction")

        depth_embedding = self.depth_encoder(observations)
        rgb_embedding = self.rgb_encoder(observations)
        if self.model_config.DECISION_TRANSFORMER.spatial_output:
            depth_embedding = self.depth_linear(
                torch.flatten(depth_embedding, 2))
            rgb_embedding = self.rgb_linear(
                torch.flatten(rgb_embedding, 2))
        depth_embedding = self.depth_activation(depth_embedding)
        rgb_embedding = self.rgb_activation(rgb_embedding)
        # we just undo the permutation made in the original implementation
        instruction_embedding = self.handle_instruction_embeddings(observations, resize_tensor, batch_size, seq_length)

        depth_embedding = resize_tensor(depth_embedding)
        rgb_embedding = resize_tensor(rgb_embedding)

        if self.model_config.ablate_instruction:
            instruction_embedding = instruction_embedding * 0
        if self.model_config.ablate_depth:
            depth_embedding = depth_embedding * 0
        if self.model_config.ablate_rgb:
            rgb_embedding = rgb_embedding * 0

        return instruction_embedding, depth_embedding, rgb_embedding

    def handle_instruction_embeddings(self, observations, resize_tensor, batch_size, seq_length):
        raise not NotImplementedError("Depending, if you get a sentence embedding or the whole word sequence!")

    def initialize_instruction_encoder(self):
        raise not NotImplementedError("Should set instruction encoder used by your model!")

    def initialize_other_layers(self):
        raise not NotImplementedError("Should set the layers used by your model!")

    def initialize_transformer_step_size(self):
        raise not NotImplementedError("Should return the value needed for set_transformer_step_size(self, n) ")

    def create_tensors_for_gpt_as_tuple(self, prev_actions, returns_to_go, instruction_embedding,
                                        depth_embedding, rgb_embedding, timesteps, batch_size,
                                        seq_length):
        raise not NotImplementedError(
            "do the mo del specific work and return a tuple of tensors like (Action, S1, ... Sn, Reward)")

    def set_transformer_step_size(self, n):
        self.model_config.defrost()
        # a step has a size of 2 + n
        # Actions, State 1, State 2... State n, Reward
        self.model_config.DECISION_TRANSFORMER.step_size = n
        self.model_config.freeze()

    def _flatten_batch(self, observations: Tensor, sensor_type: str):

        # quit silently
        if not sensor_type in observations.keys():
            return

        dims = observations[sensor_type].size()
        if len(dims) > 2:
            observations[sensor_type] = observations[sensor_type].view(-1, *dims[2:])

    @property
    def output_size(self):
        steps = max(1, (self.transformer_step_size - self.dim_not_included_for_predictions))

        return self.model_config.DECISION_TRANSFORMER.hidden_dim * steps  # - 2 because we exclude reward / actions for categorical layer

    def create_timesteps(self, sequence_length, batch_size):

        # TODO: use buffer?
        timesteps = [torch.arange(0, sequence_length, dtype=torch.long) for _ in range(batch_size)]
        timesteps = torch.stack(timesteps, dim=0).to(self.embed_ln.weight.device)

        return timesteps

    @property
    def is_blind(self):
        return self.rgb_encoder.is_blind or self.depth_encoder.is_blind

    @property
    def num_recurrent_layers(self):
        return self.state_encoder.num_recurrent_layers

    def forward(self, observations, rnn_states, prev_actions, masks):
        original_batch_shape = observations["instruction"].shape[0:2]  # excluding the embedding dimentions
        batch_size, seq_length = original_batch_shape

        instruction_embedding, depth_embedding, rgb_embedding = self._prepare_embeddings(observations)

        if self.reward_type in observations.keys() and self.training:
            returns_to_go = observations[self.reward_type]
        else:
            # If we don t have any rewards from the environment, just take one
            # as mentioned in the paper during evaluation.
            returns_to_go = torch.ones_like(prev_actions, dtype=torch.float).unsqueeze(
                dim=-1) * self.return_to_go_inference
        if "timesteps" in observations.keys():
            timesteps = observations["timesteps"]
        else:
            timesteps = self.create_timesteps(seq_length, batch_size)

        # squeeze to output the same shape as other embeddings
        # after  the operation with embedding layer
        if len(timesteps.shape) > 2:
            timesteps = timesteps.squeeze(-1)

        tensor_tuples = self.create_tensors_for_gpt_as_tuple(prev_actions, returns_to_go, instruction_embedding,
                                                             depth_embedding, rgb_embedding, timesteps, batch_size,
                                                             seq_length)

        stacked = (
            torch.stack(tensor_tuples, dim=1).permute(0, 2, 1, 3).reshape(batch_size,
                                                                          self.transformer_step_size * seq_length, -1)
        )

        output = self.gpt_encoder(self.embed_ln(stacked))
        output = output.reshape(batch_size, seq_length, self.transformer_step_size, -1).permute(0, 2, 1, 3)

        start_dim = 1
        if not self.exclude_past_action_for_prediction:
            start_dim = 0


        end_dim = max(start_dim + 1, self.transformer_step_size - 1)
        # get predictions
        action_preds = output[:, start_dim:end_dim].permute(0, 2, 1, 3).reshape(batch_size,
                                                                                                       seq_length,
                                                                                                       -1)

        return action_preds, None


class DecisionTransformerNet(AbstractDecisionTransformerNet):
    """Decision Transformer, where RGB, DEPTH and Instructions are concatenated into one state.
    """

    def __init__(
        self, observation_space: Space, model_config: Config, num_actions: int
    ):
        super().__init__(observation_space, model_config, num_actions)

    def handle_instruction_embeddings(self, observations, resize_tensor, batch_size, seq_length):

        instruction_embedding = self.instruction_activation(self.instruction_encoder(observations))

        if not self.model_config.DECISION_TRANSFORMER.use_transformer_encoded_instruction:
            instruction_embedding = resize_tensor(instruction_embedding)
        else:
            instruction_embedding = self.sentence_encoding(instruction_embedding.permute(0, 2, 1)).permute(0, 2,
                                                                                                           1).repeat(
                (1, seq_length, 1))
        return instruction_embedding

    def initialize_other_layers(self):
        # size due to concatenation of instruction, depth, and rgb features
        input_state_size = self.instruction_encoder.output_size + self.model_config.DEPTH_ENCODER.output_size + self.model_config.RGB_ENCODER.output_size
        self.embed_state = nn.Linear(input_state_size, self.model_config.DECISION_TRANSFORMER.hidden_dim)

    def create_tensors_for_gpt_as_tuple(self, prev_actions, returns_to_go, instruction_embedding,
                                        depth_embedding, rgb_embedding, timesteps, batch_size,
                                        seq_length):
        states = torch.cat(
            [instruction_embedding, depth_embedding, rgb_embedding], dim=2
        )
        # embed each modality with a different head
        state_embeddings = self.embed_state(states)
        action_embeddings = self.action_activation(self.embed_action(prev_actions))
        returns_embeddings = self.embed_return(returns_to_go)
        time_embeddings = self.embed_timestep(timesteps)

        state_embeddings2 = state_embeddings + time_embeddings
        action_embeddings2 = action_embeddings + time_embeddings
        returns_embeddings2 = returns_embeddings + time_embeddings

        return returns_embeddings2, state_embeddings2, action_embeddings2

    def initialize_instruction_encoder(self):
        if not self.model_config.DECISION_TRANSFORMER.use_transformer_encoded_instruction:
            # Init the instruction encoder
            self.instruction_encoder = InstructionEncoder(
                self.model_config.INSTRUCTION_ENCODER
            )
        else:
            self.instruction_encoder = InstructionEncoderWithTransformer(self.model_config)
            if self.model_config.DECISION_TRANSFORMER.ENCODER.use_sentence_encoding:
                self.sentence_encoding = nn.AdaptiveAvgPool1d(1)

    def initialize_transformer_step_size(self):
        # reward, state, action
        return 3


class DecisionTransformerEnhancedNet(DecisionTransformerNet):

    def __init__(
        self, observation_space: Space, model_config: Config, num_actions: int
    ):
        super().__init__(observation_space, model_config, num_actions)

    def initialize_other_layers(self):
        out_dim = self.model_config.DECISION_TRANSFORMER.hidden_dim
        self.instruction_embed_state = nn.Linear(self.instruction_encoder.output_size,
                                                 out_dim)
        self.rgb_embed_state = nn.Linear(self.model_config.RGB_ENCODER.output_size,
                                         out_dim)
        self.depth_embed_state = nn.Linear(self.model_config.DEPTH_ENCODER.output_size,
                                           out_dim)

    def initialize_transformer_step_size(self):
        return 5

    def create_tensors_for_gpt_as_tuple(self, prev_actions, returns_to_go, instruction_embedding,
                                        depth_embedding, rgb_embedding, timesteps, batch_size,
                                        seq_length):
        instruction_state_embeddings = self.instruction_embed_state(instruction_embedding)
        rgb_state_embeddings = self.rgb_embed_state(rgb_embedding)
        depth_state_embeddings = self.depth_embed_state(depth_embedding)

        action_embeddings = self.embed_action(prev_actions)
        returns_embeddings = self.embed_return(returns_to_go)
        time_embeddings = self.embed_timestep(timesteps)

        # print(state_embeddings.shape, action_embeddings.shape, returns_embeddings.shape, time_embeddings.shape)
        # time embeddings are treated similar to positional embeddings
        instruction_state_embeddings2 = instruction_state_embeddings + time_embeddings
        rgb_state_embeddings2 = rgb_state_embeddings + time_embeddings
        depth_state_embeddings2 = depth_state_embeddings + time_embeddings
        action_embeddings2 = action_embeddings + time_embeddings
        returns_embeddings2 = returns_embeddings + time_embeddings

        return returns_embeddings2, instruction_state_embeddings2, rgb_state_embeddings2, depth_state_embeddings2, action_embeddings2


class FullDecisionTransformerNet(AbstractDecisionTransformerNet):
    def __init__(
        self, observation_space: Space, model_config: Config, num_actions: int
    ):
        model_config.defrost()
        # We do use Transformer encoding, but it is done to force to flatten the entry for instructions.
        model_config.DECISION_TRANSFORMER.use_transformer_encoded_instruction = False
        model_config.freeze()
        super().__init__(observation_space, model_config, num_actions)

    def prepare_transformer_layer(self, model_config):
        return nn.Transformer(d_model=model_config.DECISION_TRANSFORMER.hidden_dim
                              , nhead=model_config.DECISION_TRANSFORMER.n_head
                              , num_encoder_layers=model_config.DECISION_TRANSFORMER.ENCODER.n_layer
                              , num_decoder_layers=model_config.DECISION_TRANSFORMER.n_layer
                              , dim_feedforward=model_config.DECISION_TRANSFORMER.hidden_dim * 2
                              , activation="gelu"
                              , batch_first=True)

    def handle_instruction_embeddings(self, observations, resize_tensor, batch_size, seq_length):
        instruction_embedding = self.instruction_encoder(observations).permute(0, 2, 1)
        instruction_embedding = resize_tensor(instruction_embedding)
        return instruction_embedding

    def initialize_instruction_encoder(self):
        self.instruction_encoder = Word2VecEmbeddings(
            self.model_config.INSTRUCTION_ENCODER
        )

    def initialize_other_layers(self):
        self.positional_encoding_for_instruction = PositionalEncoding(self.model_config.DECISION_TRANSFORMER.hidden_dim)
        self.instruction_embed_state = nn.Linear(self.instruction_encoder.output_size,
                                                 self.model_config.DECISION_TRANSFORMER.hidden_dim)
        self.rgb_embed_state = nn.Linear(self.model_config.RGB_ENCODER.output_size,
                                         self.model_config.DECISION_TRANSFORMER.hidden_dim)
        self.depth_embed_state = nn.Linear(self.model_config.DEPTH_ENCODER.output_size,
                                           self.model_config.DECISION_TRANSFORMER.hidden_dim)

        self.encoder_instruction_to_rgb = self.prepare_transformer_layer(self.model_config)
        self.encoder_instruction_to_depth = self.prepare_transformer_layer(self.model_config)
        self.encoder_rgb_to_instruction = self.prepare_transformer_layer(self.model_config)
        self.encoder_depth_to_instruction = self.prepare_transformer_layer(self.model_config)
        self.visual_to_sentence_embed = nn.AdaptiveAvgPool1d(1)

    def initialize_transformer_step_size(self):
        step_size = 2
        c = self.model_config.DECISION_TRANSFORMER.ENCODER
        if c.use_rgb_state_embeddings is True:
            step_size += 1
        if c.use_depth_state_embeddings is True:
            step_size += 1
        if c.use_output_rgb_instructions is True:
            step_size += 1
        if c.use_output_depth_instructions is True:
            step_size += 1
        if c.use_output_rgb is True:
            step_size += 1
        if c.use_output_depth is True:
            step_size += 1
        return step_size

    def create_tensors_for_gpt_as_tuple(self, prev_actions, returns_to_go, instruction_embedding,
                                        depth_embedding, rgb_embedding, timesteps, batch_size,
                                        seq_length):
        single_instruction_states = instruction_embedding[:, 0, :, :]
        # embed each modality with a different head
        instruction_state_embeddings = self.positional_encoding_for_instruction(
            self.instruction_embed_state(single_instruction_states.permute(0, 2, 1)))

        # only 2D allowed in Pytorch Implementation
        vision_causal_mask = VanillaMultiHeadAttention.create_causal_mask(seq_length, rgb_embedding.device)[0][0]
        text_mask = VanillaMultiHeadAttention.create_padded_mask(instruction_state_embeddings)

        rgb_state_embeddings = self.rgb_activation(self.embed_ln(self.rgb_embed_state(rgb_embedding)))
        depth_state_embeddings = self.depth_activation(self.embed_ln(self.depth_embed_state(depth_embedding)))

        action_embeddings = self.action_activation(self.embed_action(prev_actions))
        returns_embeddings = self.embed_return(returns_to_go)
        time_embeddings = self.embed_timestep(timesteps)

        # print(state_embeddings.shape, action_embeddings.shape, returns_embeddings.shape, time_embeddings.shape)
        # time embeddings are treated similar to positional embeddings
        rgb_state_embeddings2 = self.embed_ln(rgb_state_embeddings) + time_embeddings
        depth_state_embeddings2 = self.embed_ln(depth_state_embeddings) + time_embeddings
        action_embeddings2 = action_embeddings + time_embeddings
        returns_embeddings2 = returns_embeddings + time_embeddings

        causal_text_mask = VanillaMultiHeadAttention.create_causal_mask(instruction_state_embeddings.shape[1],
                                                                        instruction_state_embeddings.device)[0][0]
        rgb_mask = VanillaMultiHeadAttention.create_padded_mask(rgb_state_embeddings2)
        depth_mask = VanillaMultiHeadAttention.create_padded_mask(depth_state_embeddings2)

        output_rgb_instructions = self.encoder_instruction_to_rgb(src=instruction_state_embeddings,
                                                                  tgt=rgb_state_embeddings2,
                                                                  src_key_padding_mask=text_mask,
                                                                  tgt_mask=vision_causal_mask)

        output_depth_instructions = self.encoder_instruction_to_depth(src=instruction_state_embeddings,
                                                                      tgt=depth_state_embeddings2,
                                                                      src_key_padding_mask=text_mask,
                                                                      tgt_mask=vision_causal_mask)

        output_rgb = self.encoder_rgb_to_instruction(src=rgb_state_embeddings2, tgt=instruction_state_embeddings,
                                                     src_key_padding_mask=rgb_mask, tgt_mask=causal_text_mask)
        output_depth = self.encoder_depth_to_instruction(src=depth_state_embeddings2, tgt=instruction_state_embeddings,
                                                         src_key_padding_mask=depth_mask, tgt_mask=causal_text_mask)

        output_rgb = self.instruction_activation(
            self.visual_to_sentence_embed(output_rgb.permute(0, 2, 1)).permute(0, 2, 1)) + time_embeddings
        output_depth = self.instruction_activation(
            self.visual_to_sentence_embed(output_depth.permute(0, 2, 1)).permute(0, 2, 1)) + time_embeddings

        c = self.model_config.DECISION_TRANSFORMER.ENCODER
        t = [returns_embeddings2]
        if c.use_rgb_state_embeddings is True:
            t.append(rgb_state_embeddings2)
        if c.use_depth_state_embeddings is True:
            t.append(depth_state_embeddings2)
        if c.use_output_rgb_instructions is True:
            t.append(output_rgb_instructions)
        if c.use_output_depth_instructions is True:
            t.append(output_depth_instructions)
        if c.use_output_rgb is True:
            t.append(output_rgb)
        if c.use_output_depth is True:
            t.append(output_depth)
        t.append(action_embeddings2)
        assert len(t) >= 3

        return tuple(t)


class FullDecisionTransformerSingleVisionStateNet(FullDecisionTransformerNet):
    def __init__(
        self, observation_space: Space, model_config: Config, num_actions: int
    ):
        super().__init__(observation_space, model_config, num_actions)

    def initialize_other_layers(self):
        self.positional_encoding_for_instruction = PositionalEncoding(self.model_config.DECISION_TRANSFORMER.hidden_dim)
        self.instruction_embed_state = nn.Linear(self.instruction_encoder.output_size,
                                                 self.model_config.DECISION_TRANSFORMER.hidden_dim)
        self.rgb_embed_state = nn.Linear(self.model_config.RGB_ENCODER.output_size,
                                         self.model_config.DECISION_TRANSFORMER.hidden_dim)
        self.depth_embed_state = nn.Linear(self.model_config.DEPTH_ENCODER.output_size,
                                           self.model_config.DECISION_TRANSFORMER.hidden_dim)

        self.embed_state = nn.Linear(self.rgb_embed_state.out_features + self.depth_embed_state.out_features,
                                     self.model_config.DECISION_TRANSFORMER.hidden_dim)

        self.encoder_instruction_to_state = self.prepare_transformer_layer(self.model_config)
        self.encoder_state_to_instruction = self.prepare_transformer_layer(self.model_config)
        self.visual_to_sentence_embed = nn.AdaptiveAvgPool1d(1)

    def initialize_transformer_step_size(self):
        step_size = 3
        c = self.model_config.DECISION_TRANSFORMER.ENCODER
        if c.use_output_state_instructions is True:  # like a different representation of instructions at each time steps
            step_size += 1
        if c.use_output_state is True:  # A single state representation of the whole sequence...
            step_size += 1
        return step_size

    def create_tensors_for_gpt_as_tuple(self, prev_actions, returns_to_go, instruction_embedding,
                                        depth_embedding, rgb_embedding, timesteps, batch_size,
                                        seq_length):
        single_instruction_states = instruction_embedding[:, 0, :, :]
        # embed each modality with a different head
        instruction_state_embeddings = self.positional_encoding_for_instruction(
            self.instruction_embed_state(single_instruction_states.permute(0, 2, 1)))

        # only 2D allowed in Pytorch Implementation
        vision_causal_mask = VanillaMultiHeadAttention.create_causal_mask(seq_length, rgb_embedding.device)[0][0]
        text_mask = VanillaMultiHeadAttention.create_padded_mask(instruction_state_embeddings)

        rgb_state_embeddings = self.rgb_activation(self.embed_ln(self.rgb_embed_state(rgb_embedding)))
        depth_state_embeddings = self.depth_activation(self.embed_ln(self.depth_embed_state(depth_embedding)))

        states = torch.cat(
            [rgb_state_embeddings, depth_state_embeddings], dim=2
        )
        # embed each modality with a different head
        state_embeddings = self.embed_ln(self.embed_state(states))

        action_embeddings = self.action_activation(self.embed_action(prev_actions))
        returns_embeddings = self.embed_return(returns_to_go)
        time_embeddings = self.embed_timestep(timesteps)

        # print(state_embeddings.shape, action_embeddings.shape, returns_embeddings.shape, time_embeddings.shape)
        # time embeddings are treated similar to positional embeddings
        state_embeddings2 = state_embeddings + time_embeddings
        action_embeddings2 = action_embeddings + time_embeddings
        returns_embeddings2 = returns_embeddings + time_embeddings

        causal_text_mask = VanillaMultiHeadAttention.create_causal_mask(instruction_state_embeddings.shape[1],
                                                                        instruction_state_embeddings.device)[0][0]
        state_mask = VanillaMultiHeadAttention.create_padded_mask(state_embeddings2)

        output_state_instructions = self.encoder_instruction_to_state(src=instruction_state_embeddings,
                                                                      tgt=state_embeddings2,
                                                                      src_key_padding_mask=text_mask,
                                                                      tgt_mask=vision_causal_mask)

        output_state = self.encoder_state_to_instruction(src=state_embeddings2, tgt=instruction_state_embeddings,
                                                         src_key_padding_mask=state_mask, tgt_mask=causal_text_mask)

        output_state = self.instruction_activation(
            self.visual_to_sentence_embed(output_state.permute(0, 2, 1)).permute(0, 2, 1)) + time_embeddings

        c = self.model_config.DECISION_TRANSFORMER.ENCODER
        t = [returns_embeddings2, state_embeddings2]
        if c.use_output_state_instructions is True:  # like a different representation of instructions at each time steps
            t.append(output_state_instructions)
        if c.use_output_state is True:  # A single state representation of the whole sequence...
            t.append(output_state)
        t.append(action_embeddings2)

        return tuple(t)
