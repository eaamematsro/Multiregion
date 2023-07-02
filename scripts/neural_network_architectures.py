import torch.nn as nn
import torch
import numpy as np
import pdb
import  matplotlib.pyplot as plt


class CNN(nn.Module):
    def __init__(self, layer_sizes: list = None, conv_sizes: list[dict] = None,
                 non_linearity: nn.functional = None, pool_size: int = 24):
        super().__init__()
        if conv_sizes is None:
            conv_sizes = [
                {
                    'in_channels': 3,
                    'out_channels': 10,
                    'kernel_size': (5, 5),
                    'padding': (2, 2),
                    'stride': 1
                },
                {
                    'in_channels': 10,
                    'out_channels': 3,
                    'kernel_size': (3, 3),
                    'padding': (1, 1),
                    'stride': 1
                },
            ]

        if layer_sizes is None:
            layer_sizes = [100, 50, 25]
        self.layer_sizes = layer_sizes

        if non_linearity is None:
            non_linearity = nn.ReLU()
        self.non_linearity = non_linearity

        convs = []
        for conv_dict in conv_sizes:
            convs.append(nn.Conv2d(**conv_dict))
            convs.append(self.non_linearity)
            convs.append(nn.MaxPool2d(kernel_size=(3, 3), stride=2))
        self.conv_layer = nn.Sequential(*convs)
        self.avg_pool = nn.AdaptiveAvgPool2d((pool_size, pool_size))

        modules = []
        previous_size = pool_size ** 2 * conv_sizes[-1]['out_channels']
        for layer_size in layer_sizes:
            modules.append(
                nn.Linear(previous_size, layer_size)
            )
            modules.append(self.non_linearity)
            previous_size = layer_size
        self.linear = nn.Sequential(*modules)

    def forward(self, x):
        """
        :param x: expects image with dimensions [batch_size, channels, height, width]
        :return: returns outputs with dimensions [batch_size, sample]
        """
        x = self.conv_layer(x)
        x = self.avg_pool(x)
        x = x.view(x.shape[0], -1)
        y = self.linear(x)
        return y


class MLP(nn.Module):
    def __init__(self, layer_sizes: list = None, non_linearity: nn.functional = None, input_size: int = 150):
        super().__init__()
        if layer_sizes is None:
            layer_sizes = [100, 50, 25]
        self.layer_sizes = layer_sizes

        if non_linearity is None:
            non_linearity = nn.ReLU()
        self.non_linearity = non_linearity

        modules = []
        previous_size = input_size
        for m_idx, layer_size in enumerate(layer_sizes):
            modules.append(
                nn.Linear(previous_size, layer_size)
            )
            modules.append(self.non_linearity)
            previous_size = layer_size
        self.mlp = nn.Sequential(*modules)

    def forward(self, x):
        """
        :param x: [ batch, input]
        :return: [batch, output]
        """
        y = self.mlp(x)
        return y


class RNN(nn.Module):
    def __init__(self, hidden_layer_size: int = 100, input_size: int = 50,
                 tau: float = .15,
                 dt: float = 0.01, non_linearity: nn.functional = None, grid_width: int = 10,
                 min_g: float = 0.5, max_g: float = 2):
        super().__init__()
        j_mat = np.zeros((hidden_layer_size, hidden_layer_size)).astype(np.float32)

        for idx in range(hidden_layer_size):
            j_mat[:, idx] = (np.random.randn(hidden_layer_size) / np.sqrt(hidden_layer_size)
                             * ((max_g - min_g) / grid_width * (idx % grid_width) + min_g)).astype(np.float32)

        if non_linearity is None:
            non_linearity = nn.Tanh()
        self.non_linearity = non_linearity

        self.J = nn.Parameter(torch.from_numpy(j_mat))
        self.B = nn.Parameter(torch.randn(hidden_layer_size))
        self.I = nn.Parameter(torch.randn((input_size, hidden_layer_size)))
        self.dt = dt
        self.tau = tau
        self.n = hidden_layer_size
        self.x = None
        self.r = None
        self.reset_state()

    def reset_state(self, batch_size: int = 5):
        self.x = torch.randn((batch_size, self.n)) / torch.sqrt(torch.tensor(self.n))
        self.r = self.non_linearity(self.x)

    def forward(self, u):
        """

        :param u: [batch, inputs]
        :return: [batch, outputs]
        """
        x = self.x + self.dt / self.tau * (
            -self.x + u @ self.I + self.r @ self.J + self.B
        )
        r = self.non_linearity(self.x)

        return x, r


if __name__ == '__main__':
    pass
