import torch.nn as nn
import torch
import numpy as np
import pdb


class CNN(nn.Module):
    def __init__(self):
        super().__init__()


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
        # expects dims = [samples, batch]
        y = self.mlp(x.T)
        return y.T


class RNN(nn.Module):
    def __init__(self):
        super().__init__()


if __name__ == '__main__':
    pass