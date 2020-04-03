import numpy as np
import torch
import torch.jit as jit
import torch.nn as nn
import torch.nn.functional as F
from habitat_baselines.rl.models.rnn_state_encoder import RNNStateEncoder


def _unflatten_helper(x, t: int, n: int):
    sz = list(x.size())
    new_sz = [t, n] + sz[1:]

    return x.view(new_sz)


class RCMStateEncoder(RNNStateEncoder):
    r"""A cross-modal state encoder akin to the reinforced cross-modal
    matching state encoder (https://arxiv.org/abs/1811.10092).
    """

    def __init__(
        self,
        rgb_input_channels: int,
        depth_input_channels: int,
        hidden_size: int,
        action_embedding_size: int,
        num_layers: int = 1,
        rnn_type: str = "GRU",
    ):
        nn.Module.__init__(self)
        self._num_recurrent_layers = num_layers
        self._rnn_type = rnn_type

        self.rgb_kv = nn.Conv1d(rgb_input_channels, hidden_size, kernel_size=1)
        self.depth_kv = nn.Conv1d(depth_input_channels, hidden_size, kernel_size=1)
        self.q_net = nn.Linear(hidden_size, hidden_size // 2)

        self.register_buffer("_scale", torch.tensor(1.0 / ((hidden_size // 2) ** 0.5)))

        self.rnn = getattr(nn, rnn_type)(
            input_size=hidden_size + action_embedding_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
        )

        self.layer_init()

    @property
    def num_recurrent_layers(self):
        return super().num_recurrent_layers + 1

    def layer_init(self):
        for param in self.parameters():
            if param.dim() > 1:
                nn.init.orthogonal_(param)
            else:
                nn.init.zeros_(param)

    def _attn(self, q, k, v):
        logits = torch.einsum("nc, nci -> ni", q, k)

        attn = F.softmax(logits * self._scale, dim=1)

        return torch.einsum("ni, nci -> nc", attn, v)

    def forward(
        self, rgb_embedding, depth_embedding, prev_actions, hidden_states, masks
    ):
        # x is a (T, N, -1) tensor flattened to (T * N, -1)
        n = hidden_states.size(1)
        t = int(rgb_embedding.size(0) / n)

        hidden_states, last_output = hidden_states[0:-1], hidden_states[-1]
        hidden_states = self._unpack_hidden(hidden_states)

        rgb_embedding = self.rgb_kv(rgb_embedding)
        depth_embedding = self.depth_kv(depth_embedding)

        # unflatten
        rgb_embedding = _unflatten_helper(rgb_embedding, t, n)
        depth_embedding = _unflatten_helper(depth_embedding, t, n)
        masks = masks.view(t, n)
        prev_actions = prev_actions.view(t, n, -1)

        outputs = []
        for it in range(t):
            rgb = rgb_embedding[it]
            depth = depth_embedding[it]

            rgb_k, rgb_v = torch.chunk(rgb, chunks=2, dim=1)
            depth_k, depth_v = torch.chunk(depth, chunks=2, dim=1)

            last_output = last_output * masks[it].view(n, 1)

            q = self.q_net(last_output)

            rgb_attn = self._attn(q, rgb_k, rgb_v)
            depth_attn = self._attn(q, depth_k, depth_v)

            rnn_input = torch.cat([rgb_attn, depth_attn, prev_actions[it]], dim=1).view(
                1, n, -1
            )

            last_output, hidden_states = self.rnn(
                rnn_input, self._mask_hidden(hidden_states, masks[it].view(1, n, 1))
            )
            last_output = last_output.view(n, -1)

            outputs.append(last_output)

        hidden_states = self._pack_hidden(hidden_states)
        hidden_states = torch.cat([hidden_states, last_output.unsqueeze(0)], dim=0)

        return torch.stack(outputs, dim=0).view(t * n, -1), hidden_states


if __name__ == "__main__":
    rcm = RCMStateEncoder(2048, 1024, 256, 32)

    rgb_input = torch.randn(2 * 4, 2048, 7 * 7)
    depth_input = torch.randn(2 * 4, 1024, 4 * 4)
    prev_actions = torch.randn(2 * 4, 32)
    masks = torch.randint(1, size=(2 * 4,)).float()

    hidden_states = torch.randn(rcm.num_recurrent_layers, 4, 256)

    rcm(rgb_input, depth_input, prev_actions, hidden_states, masks)
