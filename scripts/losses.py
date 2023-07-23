import pdb

import numpy as np
import torch.nn as nn
import torch
from typing import Tuple, Callable


class TargetLoss(nn.Module):
    def __init__(self):
        super(TargetLoss, self).__init__()
        raise NotImplementedError


class EnergyLoss(nn.Module):
    def __init__(self):
        super(EnergyLoss, self).__init__()
        raise NotImplementedError


class DistanceLoss(nn.Module):
    def __init__(self, x_locations: np.ndarray = None, y_locations: np.ndarray = None,
                 weight: float = 1e-3, size: int = 250, sigma: Tuple[float, float] = (1, 5), power: int = 2,
                 dist_fun: Callable[[np.ndarray], float] = None, device: torch.device = None,
                 ):
        super(DistanceLoss, self).__init__()
        if device is None:

            ncudas = torch.cuda.device_count()
            if ncudas > 1:
                device = torch.device(f"cuda:{0}" if torch.cuda.is_available() else "cpu")
            else:
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if x_locations is None:
            x_locations = np.zeros(size)

        if y_locations is None:
            y_locations = np.zeros(size)

        if dist_fun is None:
            dist_fun = lambda dists, sig: 1 / (1 + np.exp(-dists / (sig ** 2) - 1))

        assert y_locations.shape == x_locations.shape

        x_dists = np.abs(x_locations[:, None] - x_locations[None, :]) ** power
        y_dists = np.abs(y_locations[:, None] - y_locations[None, :]) ** power

        distance = (dist_fun(x_dists, sigma[0]) + dist_fun(y_dists, sigma[1])) ** (1 / power)
        self.dist_mat = torch.from_numpy(distance.astype(np.float32)).to(device)
        self.weight = weight
        self.eps = np.finfo(float).eps

    def forward(self, matrix):
        return self.weight * (self.dist_mat * torch.abs(matrix + self.eps)).sum()
