import gzip
import json

import torch
import torch.nn as nn
from habitat import Config
from habitat.core.simulator import Observations
from torch import Tensor
from vlnce_baselines.models.utils import VanillaMultiHeadAttention
from vlnce_baselines.models.utils import PositionalEncoding


class Word2VecEmbeddings(nn.Module):
    def __init__(self, config: Config) -> None:
        """
        A layer that allows to us the pretrained Word2Vec embeddings
        without further handling.
        :param config:
        """
        super().__init__()

        self.config = config
        self.padding_idx = 0

        if config.sensor_uuid == "instruction":
            if self.config.use_pretrained_embeddings:
                self.embedding_layer = nn.Embedding.from_pretrained(
                    embeddings=self._load_embeddings(),
                    freeze=not self.config.fine_tune_embeddings,
                )
            else:  # each embedding initialized to sampled Gaussian
                self.embedding_layer = nn.Embedding(
                    num_embeddings=config.vocab_size,
                    embedding_dim=config.embedding_size,
                    padding_idx=self.padding_idx,
                )

    @property
    def output_size(self):
        return self.config.embedding_size

    def _load_embeddings(self) -> Tensor:
        """Loads word embeddings from a pretrained embeddings file.
        PAD: index 0. [0.0, ... 0.0]
        UNK: index 1. mean of all R2R word embeddings: [mean_0, ..., mean_n]
        why UNK is averaged: https://bit.ly/3u3hkYg
        Returns:
            embeddings tensor of size [num_words x embedding_dim]
        """
        with gzip.open(self.config.embedding_file, "rt") as f:
            embeddings = torch.tensor(json.load(f))
        return embeddings

    def forward(self, observations: Observations) -> Tensor:
        """
        Tensor sizes after computation:
            instruction: [batch_size x seq_length]
            lengths: [batch_size]
            hidden_state: [batch_size x hidden_size]
        """
        assert self.config.sensor_uuid == "instruction"
        instruction = observations["instruction"].long()
        instruction = self.embedding_layer(instruction)

        return instruction

class InstructionEncoderWithTransformer(nn.Module):
    def __init__(self, config: Config) -> None:
        """An encoder that uses RNN to encode an instruction. Returns
        the final hidden state after processing the instruction sequence.

        Args:
            config: must have
                embedding_size: The dimension of each embedding vector
                hidden_size: The hidden (output) size
                rnn_type: The RNN cell type.  Must be GRU or LSTM
                final_state_only: If True, return just the final state
        """
        super().__init__()

        self.config = config
        self.word2vec = Word2VecEmbeddings(config.INSTRUCTION_ENCODER)
        encoder_layer = nn.TransformerEncoderLayer(d_model=config.DECISION_TRANSFORMER.hidden_dim, nhead=config.DECISION_TRANSFORMER.n_head, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=config.DECISION_TRANSFORMER.n_layer)
        self.instruction_embed_state = nn.Linear(self.word2vec.output_size,
                                                 config.DECISION_TRANSFORMER.hidden_dim)
        self.positional_encoding = PositionalEncoding(config.DECISION_TRANSFORMER.hidden_dim)

    def forward(self, observations: Observations) -> Tensor:
        """
        Tensor sizes after computation:
            instruction: [batch_size x seq_length]
            lengths: [batch_size]
            hidden_state: [batch_size x hidden_size]
        """
        pretrained_embeddings = self.word2vec(observations)
        # Instructions are repeated at each time step, we want only one instructions
        embeddings = self.instruction_embed_state(pretrained_embeddings)[:,0,:,:]
        embeddings = self.positional_encoding(embeddings)
        padded_mask = VanillaMultiHeadAttention.create_padded_mask(embeddings)
        encoded_instructions = self.transformer_encoder(embeddings, src_key_padding_mask=padded_mask)
        return encoded_instructions

    @property
    def output_size(self):
        return self.config.DECISION_TRANSFORMER.hidden_dim

class InstructionEncoder(nn.Module):
    def __init__(self, config: Config) -> None:
        """An encoder that uses RNN to encode an instruction. Returns
        the final hidden state after processing the instruction sequence.

        Args:
            config: must have
                embedding_size: The dimension of each embedding vector
                hidden_size: The hidden (output) size
                rnn_type: The RNN cell type.  Must be GRU or LSTM
                final_state_only: If True, return just the final state
        """
        super().__init__()

        self.config = config

        rnn = nn.GRU if self.config.rnn_type == "GRU" else nn.LSTM
        self.encoder_rnn = rnn(
            input_size=config.embedding_size,
            hidden_size=config.hidden_size,
            bidirectional=config.bidirectional,
        )

        if config.sensor_uuid == "instruction":
            if self.config.use_pretrained_embeddings:
                self.embedding_layer = nn.Embedding.from_pretrained(
                    embeddings=self._load_embeddings(),
                    freeze=not self.config.fine_tune_embeddings,
                )
            else:  # each embedding initialized to sampled Gaussian
                self.embedding_layer = nn.Embedding(
                    num_embeddings=config.vocab_size,
                    embedding_dim=config.embedding_size,
                    padding_idx=0,
                )

    @property
    def output_size(self):
        return self.config.hidden_size * (1 + int(self.config.bidirectional))

    def _load_embeddings(self) -> Tensor:
        """Loads word embeddings from a pretrained embeddings file.
        PAD: index 0. [0.0, ... 0.0]
        UNK: index 1. mean of all R2R word embeddings: [mean_0, ..., mean_n]
        why UNK is averaged: https://bit.ly/3u3hkYg
        Returns:
            embeddings tensor of size [num_words x embedding_dim]
        """
        with gzip.open(self.config.embedding_file, "rt") as f:
            embeddings = torch.tensor(json.load(f))
        return embeddings

    def forward(self, observations: Observations) -> Tensor:
        """
        Tensor sizes after computation:
            instruction: [batch_size x seq_length]
            lengths: [batch_size]
            hidden_state: [batch_size x hidden_size]
        """
        if self.config.sensor_uuid == "instruction":
            instruction = observations["instruction"].long()
            lengths = (instruction != 0.0).long().sum(dim=1)
            instruction = self.embedding_layer(instruction)
        else:
            instruction = observations["rxr_instruction"]

        lengths = (instruction != 0.0).long().sum(dim=2)
        lengths = (lengths != 0.0).long().sum(dim=1).cpu()

        packed_seq = nn.utils.rnn.pack_padded_sequence(
            instruction, lengths, batch_first=True, enforce_sorted=False
        )

        output, final_state = self.encoder_rnn(packed_seq)

        if self.config.rnn_type == "LSTM":
            final_state = final_state[0]

        if self.config.final_state_only:
            return final_state.squeeze(0)
        else:
            return nn.utils.rnn.pad_packed_sequence(output, batch_first=True)[
                0
            ].permute(0, 2, 1)
