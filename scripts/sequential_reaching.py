import pdb

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from neural_network_architectures import (RNN, CNN, MLP)


class SequentialReachingNetwork(nn.Module):
    def __init__(self, duration: int = 100, n_clusters: int = 3, n_reaches: int = 3, use_cnn: bool = False,
                 device: torch.device = None, load_cnn: bool = False, base_lr: float = 1e-3,
                 wd: float = 0, n_hidden: int = 100):
        super().__init__()

        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.device = device

        if use_cnn:
            raise NotImplementedError
            # self.model_inputs = CNN()
        else:
            self.model_inputs = MLP(layer_sizes=[25, 50, 10], input_size=(n_reaches + 1) * 2)
            input_size = 10
        self.rnn = RNN(input_size=input_size, hidden_layer_size=n_hidden)
        self.Wout = nn.Parameter(
            torch.from_numpy(
                np.random.randn(n_hidden, 2).astype(np.float32)
            )
        )

        modules = [self.model_inputs, self.rnn]
        for module in modules:
            module.to(device)

        if load_cnn:
            self.optimizer = torch.optim.Adam(
                [
                    {'params': self.rnn.parameters()},
                    {'params': self.model_inputs.parameters(), 'lr': 1e-1 * base_lr}
                ], lr=base_lr, weight_decay=wd
            )

        else:
            self.optimizer = torch.optim.Adam(
                self.parameters(), lr=base_lr, weight_decay=wd
            )
        self.duration = duration
        self.Loss = []
        self.probs = None
        self.initialize_targets(n_clusters=n_clusters)
        self.reaches = n_reaches



    def initialize_targets(self, n_clusters: int = 3, max_bound: float = 100, sigma_factor: float = .1):
        vals = np.linspace(0, max_bound)
        sigma = (max_bound * sigma_factor)

        if n_clusters == 1:
            mean = max_bound / 2
            self.probs = 1/(np.sqrt(2 * np.pi * sigma ** 2)) * np.exp(-1/2 * ((vals - mean) / sigma) ** 2)
        elif n_clusters > 1:
            target_locations = np.linspace(0, max_bound, n_clusters)
            self.probs = np.zeros_like(vals)
            for target_location in target_locations:
                self.probs += (1 / (np.sqrt(2 * np.pi * sigma ** 2)) *
                               np.exp(-1 / 2 * ((vals - target_location) / sigma) ** 2) / n_clusters)

        self.probs /= self.probs.sum()

    def sample_targets(self, batch_size: int = 25):
        targets = []
        for i in range(batch_size):
            targets.append(np.random.choice(range(self.probs.shape[0]), p=self.probs, size=(self.reaches, 2)))

        return torch.from_numpy(np.stack(targets).astype(np.float32))

    def forward(self, targets):
        target = targets.reshape(targets.shape[0], -1)
        position = torch.zeros(targets.shape[0], 2)
        input = torch.hstack([target, position])
        self.rnn.reset_state(batch_size=target.shape[0])
        position_store = torch.zeros((self.duration, target.shape[0], 2))
        for time in range(self.duration):
            rnn_input = self.model_inputs(input)
            rnn_hidden, rnn_activation = self.rnn(rnn_input)
            position = rnn_activation @ self.Wout
            position_store[time] = position

        return position_store

    def calculate_loss(self, targets, positions, epsilon: float = 1):
        duration, batch_size, _ = positions.shape
        loss = torch.zeros((batch_size, duration))
        for batch in range(batch_size):
            distances = []
            t_idxs = []
            for idx, reach in enumerate(range(self.reaches)):
                distance = torch.sqrt(((positions[:, batch] - targets[batch, idx]) ** 2).sum(axis=1))
                distances.append(distance)
                tidx = next((t_idx for t_idx in range(duration) if distance[t_idx] < epsilon), None)
                t_idxs.append(tidx)

            loss[batch] = distances[0]
            previous_val = 0
            for distance, tidx in zip(distances[1:], t_idxs[:-1]):
                if (tidx is not None) and (tidx > previous_val):
                    loss[batch, tidx:] = distance
                    previous_val = tidx
        return loss.mean()

    def fit(self, n_loops: int = 500, batch_size: int = 25):
        optim = self.optimizer
        for loop in range(n_loops):
            optim.zero_grad()
            targets = self.sample_targets(batch_size=batch_size)
            positions = self.forward(targets)
            Loss = self.calculate_loss(targets, positions)
            self.Loss.append(Loss.item())
            Loss.backward()
            optim.step()

def main():
    if torch.cuda.is_available():
        torch.set_default_tensor_type(torch.cuda.FloatTensor)

    ncudas = torch.cuda.device_count()
    if ncudas > 1:
        device = torch.device(f"cuda:{0}" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = SequentialReachingNetwork()
    model.fit()

if __name__ == '__main__':
    main()
