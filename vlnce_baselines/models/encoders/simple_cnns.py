import numpy as np
import torch
import torch.nn as nn
from habitat_baselines.common.utils import Flatten
from habitat_baselines.rl.models.simple_cnn import SimpleCNN


class Contiguous(nn.Module):
    r"""Converts a tensor to be stored contiguously if it is not already so.
    """

    def __init__(self):
        super(Contiguous, self).__init__()

    def forward(self, x):
        return x.contiguous()


class SimpleAllCNN(SimpleCNN):
    r"""A Simple 3-Conv CNN followed by a fully connected layer. Takes in
    observations and produces an embedding of the rgb and/or depth components
    if they are present in the provided observations.

    Args:
        observation_space: The observation_space of the agent
        output_size: The size of the embedding vector
    """

    def __init__(self, observation_space, output_size):
        nn.Module.__init__(self)
        if "rgb" in observation_space.spaces:
            self._n_input_rgb = observation_space.spaces["rgb"].shape[2]
        else:
            self._n_input_rgb = 0

        if "depth" in observation_space.spaces:
            self._n_input_depth = observation_space.spaces["depth"].shape[2]
        else:
            self._n_input_depth = 0

        cnn_dims = None
        if self._n_input_rgb > 0:
            cnn_dims = np.array(
                observation_space.spaces["rgb"].shape[:2], dtype=np.float32
            )
        elif self._n_input_depth > 0:
            cnn_dims = np.array(
                observation_space.spaces["depth"].shape[:2], dtype=np.float32
            )
        self._init_model(cnn_dims, output_size)

    def _init_model(self, cnn_dims, output_size):
        r"""cnn_dims: initial cnn dimensions.
        """
        if self.is_blind:
            self.cnn = nn.Sequential()
            return

        # kernel size for different CNN layers
        self._cnn_layers_kernel_size = [(8, 8), (4, 4), (3, 3)]

        # strides for different CNN layers
        self._cnn_layers_stride = [(4, 4), (2, 2), (1, 1)]

        for kernel_size, stride in zip(
            self._cnn_layers_kernel_size, self._cnn_layers_stride
        ):
            cnn_dims = self._conv_output_dim(
                dimension=cnn_dims,
                padding=np.array([0, 0], dtype=np.float32),
                dilation=np.array([1, 1], dtype=np.float32),
                kernel_size=np.array(kernel_size, dtype=np.float32),
                stride=np.array(stride, dtype=np.float32),
            )

        self.cnn = nn.Sequential(
            nn.Conv2d(
                in_channels=self._n_input_rgb + self._n_input_depth,
                out_channels=32,
                kernel_size=self._cnn_layers_kernel_size[0],
                stride=self._cnn_layers_stride[0],
            ),
            nn.ReLU(True),
            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=self._cnn_layers_kernel_size[1],
                stride=self._cnn_layers_stride[1],
            ),
            nn.ReLU(True),
            nn.Conv2d(
                in_channels=64,
                out_channels=32,
                kernel_size=self._cnn_layers_kernel_size[2],
                stride=self._cnn_layers_stride[2],
            ),
            Contiguous(),
            Flatten(),
            nn.Linear(32 * cnn_dims[0] * cnn_dims[1], output_size),
            nn.ReLU(True),
        )
        self.layer_init()


class SimpleDepthCNN(SimpleAllCNN):
    r""" SimpleAllCNN where the only allowed input is a depth observation
    regardless of what other observation modalities are provided.
    """

    def __init__(self, observation_space, output_size):
        nn.Module.__init__(self)
        assert (
            "depth" in observation_space.spaces
        ), "Depth input required to use SimpleDepthCNN"
        self._n_input_depth = observation_space.spaces["depth"].shape[2]
        self._n_input_rgb = 0

        cnn_dims = np.array(
            observation_space.spaces["depth"].shape[:2], dtype=np.float32
        )
        self._init_model(cnn_dims, output_size)

    def forward(self, observations):
        # permute tensor to dimension [BATCH x CHANNEL x HEIGHT X WIDTH]
        depth_observations = observations["depth"].permute(0, 3, 1, 2)
        return self.cnn(depth_observations)


class SimpleRGBCNN(SimpleAllCNN):
    r""" SimpleAllCNN where the only allowed input is an RGB observation
    regardless of what other observation modalities are provided.
    """

    def __init__(self, observation_space, output_size):
        nn.Module.__init__(self)
        assert (
            "rgb" in observation_space.spaces
        ), "RGB input required to use SimpleRGBCNN"
        self._n_input_depth = 0
        self._n_input_rgb = observation_space.spaces["rgb"].shape[2]

        cnn_dims = np.array(observation_space.spaces["rgb"].shape[:2], dtype=np.float32)
        self._init_model(cnn_dims, output_size)

    def forward(self, observations):
        # permute tensor to dimension [BATCH x CHANNEL x HEIGHT X WIDTH]
        rgb_observations = observations["rgb"].permute(0, 3, 1, 2)
        rgb_observations = rgb_observations / 255.0  # normalize RGB
        return self.cnn(rgb_observations)
