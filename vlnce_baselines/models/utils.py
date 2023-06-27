import math
from numbers import Number
from typing import Any, Optional, Union

import numpy as np
import torch
import torch.nn as nn
from torch import Size, Tensor, nn as nn
from torch.distributions import constraints
from torch.distributions.normal import Normal
from torch.nn import functional as F


class TemperatureTanh(nn.Module):
    def __init__(self, temperature: float = 1.0) -> None:
        """The hyperbolic tangent with an optional temperature."""
        super().__init__()
        assert temperature != 0.0, "temperature must be nonzero."
        self._T = temperature
        self.tanh = torch.nn.Tanh()

    def forward(self, x: Tensor) -> Tensor:
        return self.tanh(x / self._T)


class TruncatedNormal(nn.Module):
    """The truncated normal distribution is derived from the normal
    distribution and is bounded above, below, or by both. It is parameterized
    by the mean and variance of the untruncated normal distrubtion. This is
    a custom implementation because it doesn't exist in pytorch.
    https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
    https://en.wikipedia.org/wiki/Truncated_normal_distribution
    """

    def __init__(
        self,
        loc: Tensor,
        scale: Union[float, Tensor],
        smin: float = -np.inf,
        smax: float = np.inf,
        validate_args: Optional[Any] = None,
    ) -> None:
        super().__init__()
        assert smin < smax, "smin must be less than smax"
        assert np.isfinite(smin) and np.isfinite(
            smax
        ), "two-sided truncation is required for now. Set both `smin` and `smax`."
        assert (loc >= smin).all() and (
            loc <= smax
        ).all(), f"loc is out of range ({smin}, {smax}): {loc}"
        if isinstance(scale, Number):
            assert scale >= 0.0, "scale is negative"
        else:
            assert (scale >= 0.0).all(), "scale is negative"

        self._normal = Normal(loc, scale, validate_args=validate_args)
        self._loc = loc
        self._scale = scale
        self._smin = smin
        self._smax = smax
        self._unbounded = self._smin == -np.inf and self._smax == np.inf
        self.A = 1 / (self._scale * np.sqrt(2 * np.pi))
        self.Z = self._normal.cdf(self._smax) - self._normal.cdf(self._smin)
        self.support = constraints.interval(self._smin, self._smax)
        self._init_mean_variance_entropy()

    def _init_mean_variance_entropy(self) -> None:
        """References for entropy:
        https://github.com/olmjo/RcppTN
        https://en.wikipedia.org/wiki/Truncated_normal_distribution
        """
        standard_normal = Normal(0.0, 1.0)
        standard_normal.pdf = lambda x: (np.e ** (-0.5 * (x ** 2))) / np.sqrt(
            2 * np.pi
        )
        alpha = (self._smin - self._loc) / self._scale
        beta = (self._smax - self._loc) / self._scale

        alpha_pdf = standard_normal.pdf(alpha)
        beta_pdf = standard_normal.pdf(beta)

        alpha_cdf = standard_normal.cdf(alpha)
        beta_cdf = standard_normal.cdf(beta)
        standard_Z = beta_cdf - alpha_cdf

        self._mean = self._loc - self._scale * (
            (beta_pdf - alpha_pdf) / standard_Z
        )

        t1 = (beta * beta_pdf - alpha * alpha_pdf) / standard_Z
        t2 = ((beta_pdf - alpha_pdf) / standard_Z) ** 2
        self._variance = (self._scale ** 2) * (1 - t1 - t2)

        self._entropy = 0.5 * np.log(2 * np.pi * np.e)
        self._entropy += torch.log(self._scale * standard_Z)
        self._entropy += (alpha * alpha_pdf - beta * beta_pdf) / (
            2 * standard_Z
        )

    @property
    def mean(self) -> Tensor:
        return self._mean

    @property
    def variance(self) -> Tensor:
        return self._variance

    def sample(self, resample_limit: int = 10000) -> Tensor:
        if self._unbounded:
            return self._normal.sample()

        samples = self._normal.sample()
        do_resample = (samples < self._smin).logical_or(samples > self._smax)
        num_resamples = 0
        while do_resample.any():
            assert (
                num_resamples < resample_limit
            ), f"Hit resample limit of {resample_limit} for bounds [{self._smin}, {self._smax}]"
            num_resamples += 1

            samples[do_resample] = self._normal.sample()[do_resample]
            do_resample = (samples < self._smin).logical_or(
                samples > self._smax
            )

        return samples

    def log_prob(self, value: Union[float, Tensor]) -> Tensor:
        if self._unbounded:
            return self._normal.log_prob(value)

        msg = "value is out of truncation range and has an undefined log_prob."
        if isinstance(value, Number):
            assert value >= self._smin and value <= self._smax, msg
        else:
            assert (value >= self._smin).all() and (
                value <= self._smax
            ).all(), msg

        normal_prob_density = self.A * np.e ** (
            -0.5 * ((value - self._loc) / self._scale) ** 2
        )
        truncated_prob_density = normal_prob_density / self.Z

        if isinstance(truncated_prob_density, Number):
            return np.log(truncated_prob_density)
        else:
            return truncated_prob_density.log()

    def mode(self):
        return self._loc

    def entropy(self):
        return self._entropy


class DotProductAttention(nn.Module):
    def __init__(self, key_dimension: int) -> None:
        super().__init__()
        self.scale = torch.tensor(1.0 / ((key_dimension) ** 0.5))
        self.softmax = nn.Softmax(dim=2)

    def forward(
        self, Q: Tensor, K: Tensor, V: Tensor, mask: Optional[Tensor] = None
    ) -> Tensor:
        """Scaled dot-product attention with an optional mask.
        2X speed improvement over `torch.einsum`.
        Args:
            query: [Batch, Dk]
            key: [Batch, Dk, P]
            value: [Batch, Dv, P]
        Returns:
            tensor of dimension [Batch, Dv]
        """
        energy = torch.bmm(Q.unsqueeze(1), K)
        if mask is not None:
            energy *= mask.unsqueeze(1).float()

        attn = self.softmax(energy * self.scale)
        return torch.bmm(attn, V.permute(0, 2, 1)).squeeze(1)


class MultiHeadDotProductAttention(nn.Module):
    def __init__(
        self,
        d_q_in: int,
        d_k_in: int,
        d_v_in: int,
        d_qk: int,
        d_v: int,
        num_heads: int,
        d_out: int,
        normalize: bool = True,
        dropout_p: float = 0.0,
    ) -> None:
        """The residual connection of Vaswani et al is not used here. The
        residual makes sense if self-attention is being used.
        Args:
            d_q_in (int): dimension of the query vector input
            d_k_in (int): dimension of the key vector input
            d_v_in (int): dimension of the value vector input
            d_qk (int): dimension to map queries & keys to prior to attention
            d_v (int): dimension to map values to prior to attention
            num_heads (int): number of attention heads
            d_out (int): output dimension of this module (final linear layer)
        """
        super().__init__()
        self.num_heads = num_heads
        self.normalize = normalize
        self.q_linear = nn.Linear(d_q_in, d_qk * num_heads, bias=False)
        self.k_linear = nn.Linear(d_k_in, d_qk * num_heads, bias=False)
        self.v_linear = nn.Linear(d_v_in, d_v * num_heads, bias=False)

        self.attn = DotProductAttention(d_qk)
        self.final_linear = nn.Linear(d_v * num_heads, d_out, bias=False)

        self.dropout = None
        if dropout_p > 0.0:
            self.dropout = nn.Dropout(dropout_p)

        if self.normalize:
            self.layer_norm = nn.LayerNorm(d_out, eps=1e-6)

    def forward(
        self, Q: Tensor, K: Tensor, V: Tensor, mask: None = None
    ) -> Tensor:
        """Performs multihead scaled dot product attention for some Q, K, V.
        Args:
            Q: [Batch, d_q_in]
            K: [Batch, d_k_in, P]
            V: [Batch, d_v_in, P]
        """
        assert K.shape[2] == V.shape[2], "keys must be the same size as values"

        Q = self.q_linear(Q)
        K = self.k_linear(K.permute(0, 2, 1)).permute(0, 2, 1).contiguous()
        V = self.v_linear(V.permute(0, 2, 1)).permute(0, 2, 1).contiguous()

        Q = Q.view(Q.shape[0] * self.num_heads, Q.shape[1] // self.num_heads)
        K = K.view(
            K.shape[0] * self.num_heads,
            K.shape[1] // self.num_heads,
            K.shape[2],
        )
        V = V.view(
            V.shape[0] * self.num_heads,
            V.shape[1] // self.num_heads,
            V.shape[2],
        )

        attended_V = self.attn(Q, K, V, mask=mask)

        attended_V = attended_V.view(
            attended_V.shape[0] // self.num_heads,
            self.num_heads,
            attended_V.shape[1],
        )

        attended_V = attended_V.view(
            attended_V.shape[0], attended_V.shape[1] * attended_V.shape[2]
        )

        out = self.final_linear(attended_V)
        if self.dropout is not None:
            out = self.dropout(out)
        if self.normalize:
            out = self.layer_norm(out)
        return out


class CustomFixedCategorical(torch.distributions.Categorical):
    """Same as the CustomFixedCategorical in hab-lab, but renames log_probs
    to log_prob. All the torch distributions use log_prob.
    """

    def sample(
        self, sample_shape: Size = torch.Size()  # noqa: B008
    ) -> Tensor:
        return super().sample(sample_shape).unsqueeze(-1)

    def log_prob(self, actions: Tensor) -> Tensor:
        return (
            super()
            .log_prob(actions.squeeze(-1))
            .view(actions.size(0), -1)
            .sum(-1)
            .unsqueeze(-1)
        )

    def mode(self):
        return self.probs.argmax(dim=-1, keepdim=True)


def batched_index_select(
    x: torch.Tensor, dim: int, index: torch.LongTensor
) -> torch.Tensor:
    """A batched index_select where each batch selects different indices.

    Args:
        x: size [B, d0, d1, ..., dn]
        dim: int where 0 <= dim < len(x.size())
        index: size [B, d0, d1, ..., dn]

    Returns:
        torch.Tensor where the selected dimension has been squeezed.

    Example:
        >>> x = torch.randn(2,3,4)
        >>> index = torch.randint(0,3, (2,))
        >>> result = batched_index_select(x, 1, index)  # size: [2, 4]
    """
    views = [x.shape[0]] + [
        1 if i != dim else -1 for i in range(1, len(x.shape))
    ]
    expanse = list(x.shape)
    expanse[0] = -1
    expanse[dim] = -1
    index = index.view(views).expand(expanse)
    return torch.gather(x, dim, index).squeeze(dim)


class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 200):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.permute(1,0,2))

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [batch_size, seq_length, embedding_dim]
        """
        x = x + self.pe[:x.size(1)]
        return self.dropout(x)


class VanillaMultiHeadAttention(nn.Module):
    """
    A vanilla multi-head masked attention layer with a projection at the end.
    Depending on the mask and entries provided, it can serve as
    Causal Multihead Self Attention.
    """

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.k_attn = nn.Linear(config.n_embd, config.n_embd)
        self.q_attn = nn.Linear(config.n_embd, config.n_embd)
        self.v_attn = nn.Linear(config.n_embd, config.n_embd)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        # regularization
        self.attn_dropout = nn.Dropout(config.attn_pdrop)
        self.resid_dropout = nn.Dropout(config.resid_pdrop)

        self.n_head = config.n_head
        self.n_embd = config.n_embd

    @staticmethod
    def create_causal_mask(size: int, device="cpu"):
        return (torch.tril(torch.ones(size, size)).view(1, 1, size, size) == 0).to(device)

    @staticmethod
    def create_padded_mask(t: Tensor):
        return (t == 0.0).all(dim=-1).to(t.device)

    def forward(self, q, k, v, mask=None):
        q_B, q_T, q_C = q.size() # batch size, sequence length, embedding dimensionality (n_embd)
        k_B, k_T, k_C = k.size()
        v_B, v_T, v_C = v.size()

        # all entries must have the same batch size. Keys and values must have the same sequence length
        assert q_B == k_B
        assert k_B == v_B
        assert k_C == v_C

        q = self.k_attn(q)
        k = self.k_attn(k)
        v = self.k_attn(v)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q = q.view(q_B, q_T, self.n_head, q_C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        k = k.view(q_B, k_T, self.n_head, k_C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(q_B, v_T, self.n_head, v_C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        if mask is not None:
            num_dim_att = len(att.shape)
            num_dim_mask = len(mask.shape)
            dim_to_add = num_dim_att - num_dim_mask
            assert dim_to_add >= 0
            if dim_to_add > 0:
                for i in range(1, dim_to_add+1):
                    mask = mask.unsqueeze(i)
            att = att.masked_fill(mask, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(q_B, q_T, q_C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y
