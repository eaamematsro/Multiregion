import abc
import pdb
import warnings
import pickle
import os
import re
import json
import attr
import cattr
import logging
import numpy as np
import torch
import torch.nn as nn
from losses import DistanceLoss
from pathlib import Path
from datetime import date


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
    def __init__(self, layer_sizes: dict = None, non_linearity: nn.functional = None, input_size: int = 150,
                 xavier_init: bool = False, output_size: int = 250):
        super().__init__()
        default_cfg = {
            'target_cfg': [50, 25, 12],
            'feedback': [10, 5],
            'trigger': [1]
        }
        if layer_sizes is None or not (type(layer_sizes) == dict):
            layer_sizes = default_cfg

        for key, default_value in default_cfg.items():
            curr_val = layer_sizes.get(key, None)
            if curr_val is None:
                warnings.warn(f'Missing value for {key}. Using default values instead.')
                layer_sizes.update({key: default_value})

        self.layer_sizes = layer_sizes
        if non_linearity is None:
            non_linearity = nn.Tanh()
        self.non_linearity = non_linearity
        input_sizes = {
            'target_cfg': input_size,
            'feedback': 2,
            'trigger': 1
        }

        self.mlp = {}
        for layer_name, layer_size in layer_sizes.items():
            modules = []
            previous_size = input_sizes[layer_name]
            layer_size.append(output_size)
            for m_idx, size in enumerate(layer_size):
                modules.append(
                    nn.Linear(previous_size, size)
                )
                modules.append(self.non_linearity)
                previous_size = size
            if xavier_init:
                self.weights_init(modules)
            self.mlp.update({layer_name: nn.Sequential(*modules)})

        # modules = []
        # previous_size = input_size
        # for m_idx, layer_size in enumerate(layer_sizes):
        #     modules.append(
        #         nn.Linear(previous_size, layer_size)
        #     )
        #     modules.append(self.non_linearity)
        #     previous_size = layer_size
        # self.weights_init(modules)

    def forward(self, target_cfg, cur_position, trigger_sig):
        """
        :param x: [ batch, input]
        :return: [batch, output]
        """
        inputs = [target_cfg, cur_position, trigger_sig]
        outputs = []
        for network, input in zip(self.mlp.values(), inputs):
            y = network(input)
            outputs.append(y)
        outputs = tuple(outputs)
        return outputs

    def weights_init(self, modules):
        for m in modules:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('sigmoid'))


class RNN(nn.Module):
    def __init__(self, hidden_layer_size: int = 100, input_size: int = 50,
                 tau: float = .05, position_size: int = 2,
                 dt: float = 0.01, non_linearity: nn.functional = None, grid_width: int = 10,
                 min_g: float = 0.5, max_g: float = 2):
        super().__init__()
        j_mat = np.zeros((hidden_layer_size, hidden_layer_size)).astype(np.float32)

        self.x_loc, self.y_loc = np.zeros(hidden_layer_size), np.zeros(hidden_layer_size)
        for idx in range(hidden_layer_size):
            self.x_loc[idx] = idx % grid_width
            self.y_loc[idx] = idx // grid_width
            j_mat[:, idx] = (np.random.randn(hidden_layer_size) / np.sqrt(hidden_layer_size)
                             * ((max_g - min_g) / grid_width * (idx % grid_width) + min_g)).astype(np.float32)

        if non_linearity is None:
            non_linearity = nn.Tanh()
        self.non_linearity = non_linearity

        self.J = nn.Parameter(torch.from_numpy(j_mat))
        self.B = nn.Parameter(torch.randn(hidden_layer_size) / torch.sqrt(torch.tensor(hidden_layer_size)))
        self.dt = dt
        self.tau = tau
        self.n = hidden_layer_size
        self.x = None
        self.r = None
        self.grid_width = grid_width
        self.reset_state()

    def reset_state(self, batch_size: int = 5):
        self.x = torch.randn((batch_size, self.n)) / torch.sqrt(torch.tensor(self.n))
        self.r = self.non_linearity(self.x)

    def forward(self, targ_cfg, position, trigger, noise_scale: float = .1, mask: torch.Tensor = None):
        """

        :param targ_cfg:
        :param position:
        :param trigger:
        :param noise_scale:
        :param mask:
        :return: [batch, outputs]
        """
        x = self.x + self.dt / self.tau * (
                -self.x + targ_cfg + position + trigger + self.r @ self.J + self.B
                + noise_scale * torch.randn(self.x.shape)
        )
        r = self.non_linearity(self.x)
        if mask is not None:
            x *= mask
            r *= mask
        self.x = x
        self.r = r

        return x, r


class MultiAreaNetwork(nn.Module, metaclass=abc.ABCMeta):
    def __init__(self, duration: int = 500, use_cnn: bool = False, input_size: int = 10,
                 device: torch.device = None, wd: float = 1e-3, n_hidden: int = 500, **kwargs):
        super().__init__()

        if use_cnn:
            raise NotImplementedError
        else:
            layer_size = {
                'target_cfg': [12, 6, 4],
                'feedback': [6, 3],
                'trigger': [1]
            }
            self.model_inputs = MLP(input_size=input_size, output_size=n_hidden)

        self.rnn = RNN(hidden_layer_size=n_hidden)
        self.dist_loss = DistanceLoss(x_locations=self.rnn.x_loc, y_locations=self.rnn.y_loc, device=device,
                                      weight=wd)

        modules = [self.model_inputs, self.rnn]
        for module in modules:
            module.to(device)

        self.duration = duration
        self.Loss = []
        self.save_path = None
        self.text_path = None
        self.set_savepath()
        self.set_logger()

    @abc.abstractmethod
    def initialize_targets(self):
        """ Sets target information"""

    @abc.abstractmethod
    def sample_targets(self):
        """Draw targets"""

    @abc.abstractmethod
    def get_noise(self):
        """Set noise level of system"""

    @abc.abstractmethod
    def calculate_loss(self):
        """Calculate loss"""

    @abc.abstractmethod
    def burn_in(self, n_loops: int = 500, batch_size: int = 10, max_norm: float = 1):
        """Initial model pretraining"""

    @abc.abstractmethod
    def fit(self):
        """Fit network"""

    @abc.abstractmethod
    def reset_optimizer(self):
        """Reset optimizer"""

    def save_model(self):
        state_dict = self.state_dict()
        data_dict = {'model_state': state_dict, 'full_model': self}
        with open(self.save_path, 'wb') as handle:
            pickle.dump(data_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def set_savepath(self):
        cwd = os.getcwd()
        cwd_path = Path(cwd)
        save_folder = cwd_path / 'data'
        save_folder.mkdir(exist_ok=True)
        date_str = date.today().strftime("%Y-%m-%d")
        date_save_path = save_folder / date_str
        date_save_path.mkdir(exist_ok=True)

        reg_exp = '_'.join(['model', '\d+'])
        files = [x for x in date_save_path.iterdir() if x.is_dir() and re.search(reg_exp, str(x.stem))]
        folder_path = date_save_path / f"model_{len(files)}"
        folder_path.mkdir(exist_ok=True)
        self.save_path = folder_path / 'model.pickle'
        self.text_path = folder_path / 'params.json'

    def set_logger(self):
        logging_path = self.save_path.parents[0] / 'log.log'
        logging.basicConfig(
            filename=logging_path,
            filemode='a',
            format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
            datefmt='%H:%M:%S',
            level=logging.INFO)

        logging.info('Initializing model')

    def save_checkpoint(self, checkpoint: int = 0):
        state_dict = self.state_dict()
        data_dict = {'model_state': state_dict, 'full_model': self,
                     'checkpoint': checkpoint}
        path = self.save_path.parents[0]

        check_path = path / 'checkpoint_model.pickle'
        with open(check_path, 'wb') as handle:
            pickle.dump(data_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
        logging.info(f'Saving model at checkpoint {checkpoint}')


class SequentialReachingNetwork(MultiAreaNetwork):
    def __init__(self, duration: int = 500, n_clusters: int = 3, n_reaches: int = 3, use_cnn: bool = False,
                 device: torch.device = None, load_cnn: bool = False, base_lr: float = 1e-4,
                 control_type: str = 'acceleration', max_speed: float = 1,
                 wd: float = 1e-3, n_hidden: int = 500, optimizer: torch.optim.Optimizer = None, **kwargs):
        super().__init__(duration=duration, use_cnn=use_cnn,
                         device=device, wd=wd, n_hidden=n_hidden, input_size=(n_reaches * 2))

        assert control_type in ['acceleration', 'position']

        params = {
            'wd': wd, 'units': n_hidden
        }
        self.params = params

        self.Wout = nn.Parameter(
            torch.from_numpy(
                (np.random.randn(n_hidden, 2) / n_hidden).astype(np.float32)
            )
        )

        if optimizer is None:
            optimizer = torch.optim.Adam

        if load_cnn:
            self.optimizer = optimizer(
                [
                    {'params': self.rnn.parameters()},
                    {'params': self.model_inputs.parameters(), 'lr': 1e-1 * base_lr}
                ], lr=base_lr, weight_decay=wd
            )
        else:
            self.optimizer = optimizer(
                self.parameters(), lr=base_lr, weight_decay=wd
            )
        self.opt_fun = optimizer
        self.opt_config = {'lr': base_lr, 'wd': wd}
        self.load_cnn = load_cnn
        self.probs = None
        self.max_bound = None
        self.fig, self.ax = None, None
        self.initialize_targets(n_clusters=n_clusters)
        self.reaches = n_reaches
        self.control = control_type
        self.max_speed = max_speed
        self.difficulty = 0

    def initialize_targets(self, n_clusters: int = 3, max_bound: int = 1, sigma_factor: float = .05):
        vals = np.linspace(0, max_bound, 100)
        sigma = (max_bound * sigma_factor)
        self.max_bound = max_bound
        if n_clusters == 1:
            mean = max_bound / 2
            self.probs = 1 / (np.sqrt(2 * np.pi * sigma ** 2)) * np.exp(-1 / 2 * ((vals - mean) / sigma) ** 2)
        elif n_clusters > 1:
            target_locations = np.linspace(0, max_bound, n_clusters)
            self.probs = np.zeros_like(vals)
            for target_location in target_locations:
                self.probs += (1 / (np.sqrt(2 * np.pi * sigma ** 2)) *
                               np.exp(-1 / 2 * ((vals - target_location) / sigma) ** 2) / n_clusters)

        self.probs /= self.probs.sum()

    def sample_targets(self, batch_size: int = 25,
                       close_targets: bool = False, difficulty: float = .1):
        targets = []
        for i in range(batch_size):
            if not close_targets:
                targets.append(
                    np.random.choice(range(self.probs.shape[0]), p=self.probs, size=(self.reaches, 2)) / 100)
            else:
                init_target = np.random.choice(range(self.probs.shape[0]), p=self.probs, size=2) / 100
                target_array = np.zeros((self.reaches, 2))
                target_array[0] = init_target
                for reach in range(self.reaches - 1):
                    target_array[reach + 1] = target_array[reach] + difficulty * self.max_bound * np.random.randn(1, 2)
                targets.append(target_array)
        return torch.from_numpy(np.stack(targets).astype(np.float32)).to(self.device)

    def forward(self, targets, pause_duration: int = 10, hold_duration: int = 50,
                tolerance: float = 5, noise_scale: float = .2, mask: torch.Tensor = None):
        self.reach_num = np.zeros(targets.shape[0], dtype=int)
        target = targets.reshape(targets.shape[0], -1) / self.max_bound
        self.rnn.reset_state(batch_size=target.shape[0])
        position_store = torch.zeros((self.duration, target.shape[0], 2))
        rnn_store = torch.zeros((self.duration, *self.rnn.x.shape))
        velocity = None
        trigger_sig = torch.zeros((self.duration, target.shape[0]))
        trigger_sig[int(.8 * hold_duration): int(1.2 * hold_duration)] = 1
        seg_time = int(self.duration / self.reaches)
        for reach in range(self.reaches):
            trigger_sig[int(.8 * hold_duration) + ((reach + 1) * (seg_time + pause_duration))
                        :int(1.2 * hold_duration) + ((reach + 1) * (seg_time + pause_duration))] = 1

        if self.control == 'acceleration':
            velocity = torch.zeros((target.shape[0], 2))
            position = self.max_bound / 2 * torch.ones(targets.shape[0], 2)
        elif self.control == 'position':
            position = torch.zeros(targets.shape[0], 2)
        self.get_reach_number(targets, position, reset=True)
        for time in range(self.duration):
            # trigger_sig = self.get_trigger_signal(targets, position_store[time - pause_duration], time=time,
            #                                       hold_duration=hold_duration, eps=tolerance)
            # reach_nums = (torch.from_numpy(self.reach_num.astype(np.float32))[:, None]).to(self.device)

            # input = torch.hstack([target, position / self.max_bound, trigger_sig[:, None], reach_nums])
            rnn_inputs = self.model_inputs(target, position / self.max_bound, trigger_sig[time, :, None])
            # pdb.set_trace()
            rnn_hidden, rnn_activation = self.rnn(*rnn_inputs, noise_scale=noise_scale, mask=mask)

            if self.control == 'position':
                position = rnn_activation @ self.Wout + torch.tensor(
                    [self.max_bound / 2, self.max_bound / 2])
            elif self.control == 'acceleration':
                velocity = velocity + self.rnn.dt * (rnn_activation @ self.Wout)
                position = position + velocity
            position_store[time] = position  # + noise_scale * torch.randn_like(position)
            rnn_store[time] = rnn_activation
            # self.get_reach_number(targets, position, eps=tolerance)
        # rnn_act = rnn_store[:, 0, :5].detach().cpu().numpy()
        # plt.figure()
        # plt.plot(rnn_act)
        # plt.pause(0.1)
        # pdb.set_trace()
        return position_store, rnn_store

    def get_noise(self, trial: int = 0, max_trials: int = 5000, sig_0: float = 1, min_noise: float = 1e-2):
        return np.exp(-trial / max_trials) * sig_0 + min_noise

    def set_difficulty(self, trial: int = 0, max_trials: int = 5000):
        self.difficulty = 1 / (1 + np.exp(-max_trials * (trial / max_trials ** 2 - 1 / 4)))

    def brachistichrone(self, targets, hold_duration: int = 50, pause_duration: int = 50):
        with torch.no_grad():
            batch_size, nreaches, _ = targets.shape
            optimal_trajectory = torch.zeros((self.duration, batch_size, 2)) + self.max_bound / 2
            optimal_loss = torch.zeros((self.duration, batch_size))
            for tidx, target in enumerate(targets):
                previous_target = torch.zeros(2) + self.max_bound / 2
                time_to_targ = hold_duration
                for reach in range(nreaches):
                    while time_to_targ < self.duration:
                        displacement_vector = -(previous_target - target[reach] + 1e-8)
                        unit_vector = displacement_vector / torch.norm(
                            displacement_vector) * self.max_speed * self.rnn.dt
                        if unit_vector[0] != 0:
                            duration = int((displacement_vector[0] / unit_vector[0]) / self.rnn.dt)
                        elif unit_vector[1] != 0:
                            duration = int((displacement_vector[1] / unit_vector[1]) / self.rnn.dt)
                        else:
                            duration = 0

                        duration = np.minimum(self.duration, duration)
                        # if time_to_targ + duration < self.duration:

                        final_time = np.minimum(self.duration, time_to_targ + duration)
                        try:
                            time_vals = (torch.linspace(
                                time_to_targ, final_time,
                                final_time - time_to_targ) - time_to_targ) * self.rnn.dt
                        except:
                            pdb.set_trace()
                        optimal_trajectory[time_to_targ: final_time, tidx, :] = (
                                unit_vector[:, None] * time_vals[None, :] + previous_target[:, None]).T

                        optimal_loss[time_to_targ: final_time, tidx] = ((
                                                                                ((optimal_trajectory[
                                                                                  time_to_targ: final_time, tidx, :] -
                                                                                  target[reach]) ** 2) ** .5)
                                                                        .sum(axis=1) * torch.arange(
                            final_time - time_to_targ) ** 2 / self.duration ** 2
                                                                        )

                        if reach == (nreaches - 1):
                            optimal_trajectory[final_time:, tidx] = target[reach]
                        else:
                            optimal_trajectory[final_time:final_time + pause_duration, tidx, :] = target[reach]
                        time_to_targ += duration + pause_duration
                        previous_target = target[reach]

        return optimal_trajectory, optimal_loss

    def calculate_loss(self, targets, positions, rnn_act, **kwargs):
        dist_loss = self.dist_loss(self.rnn.J)
        target_loss = self.get_target_loss(targets, positions, **kwargs)
        energy_loss = self.get_energy_loss(rnn_act, **kwargs)

        return target_loss + 1e-5 * energy_loss + dist_loss

    def get_target_loss(self, targets, positions, epsilon: float = 5, method: str = 'sequenced',
                        pause: int = 50, normalize: bool = True, hold_duration: int = 50, **kwargs):
        eps = 1e-8
        assert method in ['naive', 'sequenced', 'optimal'], NotImplementedError
        duration, batch_size, _ = positions.shape
        loss = torch.zeros((batch_size, duration))
        hold_loss = torch.zeros(batch_size)
        time_loss = torch.zeros((batch_size, duration))
        optimal_losses = None
        origin = self.max_bound / 2 * torch.ones((1, 2))
        seg_time = int(duration / self.reaches)
        target_position = torch.zeros_like(positions)
        target_position[:hold_duration] = self.max_bound / 2
        loss_fn = nn.MSELoss()

        for reach in range(self.reaches):
            target_position[hold_duration + (reach * (seg_time + pause))
                            : hold_duration + ((reach + 1) * (seg_time + pause))] = targets[:, reach]
        loss = loss_fn(positions, target_position)
        return loss

    def get_target_loss_old(self, targets, positions, epsilon: float = 5, method: str = 'sequenced',
                        pause: int = 50, normalize: bool = True, hold_duration: int = 50, **kwargs):
        eps = 1e-8
        assert method in ['naive', 'sequenced', 'optimal'], NotImplementedError
        duration, batch_size, _ = positions.shape
        loss = torch.zeros((batch_size, duration))
        hold_loss = torch.zeros(batch_size)
        time_loss = torch.zeros((batch_size, duration))
        optimal_losses = None
        origin = self.max_bound / 2 * torch.ones((1, 2))

        if method == 'optimal':
            optimal_trajectory, optimal_losses = self.brachistichrone(targets, hold_duration=hold_duration,
                                                                      pause_duration=pause)
            loss = ((optimal_trajectory - positions) ** 2).sum(axis=2)
            hold_loss = torch.zeros(1)
            time_loss = torch.zeros((1, 1))
        else:
            if normalize:
                _, optimal_losses = self.brachistichrone(targets, hold_duration=hold_duration,
                                                         pause_duration=pause)
            for batch in range(batch_size):
                distances = torch.zeros((self.reaches, duration))
                t_idxs = []
                for reach in range(self.reaches):
                    distances[reach] = (torch.sqrt(((positions[:, batch] - targets[batch, reach]) ** 2 + eps) /
                                                   self.max_bound).sum(axis=1))
                    tidx = next((t_idx for t_idx in range(duration) if distances[reach, t_idx] < epsilon), None)
                    t_idxs.append(tidx)

                weighting = torch.zeros((self.reaches, duration))
                previous_val = hold_duration
                hold_times = []
                if method == 'sequenced':
                    weighting[0] = torch.arange(duration) ** 2
                    if self.reaches > 1:
                        with torch.no_grad():
                            for reach, t_idx in zip(range(self.reaches), t_idxs):
                                if (t_idx is not None) and (previous_val is not None) and (t_idx > previous_val):
                                    weighting[reach, t_idx + pause:] = 0
                                    weighting[reach,
                                    previous_val: np.minimum(t_idx + pause, duration)] = (
                                                                                            torch.arange(
                                                                                             0,
                                                                                             np.minimum(
                                                                                                 t_idx + pause,
                                                                                                 duration) - previous_val) ** 2) / self.duration ** 2
                                    # time_loss[batch, previous_val: np.minimum(t_idx + pause, duration)] = torch.arange(
                                    #     0, np.minimum(t_idx + pause, duration) - previous_val) ** 2 / duration ** 2
                                    previous_val = np.minimum(t_idx + pause, duration)
                                    hold_times.append(t_idx)
                                else:
                                    previous_val = None
                        loss[batch] = (weighting * distances).sum(axis=0)
                    else:
                        loss[batch] = (weighting * distances).sum(axis=0)
                elif method == 'naive':
                    if self.reaches > 1:
                        ts = np.linspace(0, duration, self.reaches).astype(int)
                        # with torch.no_grad():
                        for reach in range(self.reaches):
                            time = int(duration / self.reaches * (reach + 1))
                            weighting[reach, previous_val: np.minimum(time + pause, duration)] = torch.arange(
                                0, np.minimum(time + pause, duration) - previous_val) ** 2 / self.duration ** 2
                            previous_val = np.minimum(time + pause, duration)
                            hold_times.append(time)
                        loss[batch] = (weighting * distances).sum(axis=0)
                    else:
                        loss[batch] = (weighting * distances).sum(axis=0)
                loss[batch, :hold_duration] = torch.pow(
                    torch.pow((positions[:hold_duration, batch] - origin), 2).sum(axis=1), .5)
                deriv = torch.diff(positions[:, batch], dim=0) / self.rnn.dt

                hold_loss[batch] = hold_loss[batch] + torch.linalg.norm(deriv[: hold_duration], axis=1).sum()
                for hold in hold_times:
                    hold_loss[batch] = hold_loss[batch] + torch.linalg.norm(deriv[hold: hold + pause], axis=1).sum()

        if normalize and not (method == 'optimal'):
            return (torch.log(loss.sum(axis=1) / optimal_losses.sum(axis=0))
                    + time_loss.sum(axis=1) + hold_loss).mean()
        else:
            return (loss.sum(axis=1) + time_loss.sum(axis=1) + hold_loss).mean()

    def get_energy_loss(cls, rnn_activation, **kwargs):
        return torch.norm(rnn_activation, dim=[0, 2]).mean()

    def get_trigger_signal(self, targets, positions, time: int = 0, hold_duration: int = 50,
                           eps: float = 5):
        batch_size, reaches, _ = targets.shape
        if time == hold_duration:
            trigger_sig = torch.ones(batch_size)
        else:
            trigger_sig = torch.zeros(batch_size)
            reach_num = self.reach_num
            for batch, reach in enumerate(reach_num):
                try:
                    trigger_sig[batch] = (
                            torch.pow(
                                torch.pow(
                                    (positions[batch] - targets[batch, reach]), 2
                                ).sum(), .5
                            ) <= eps
                    )
                except IndexError:
                    pdb.set_trace()
        return trigger_sig

    def get_reach_number(self, targets, positions, eps: float = 1, reset: bool = False):
        if reset:
            curr_reach_num = np.zeros(targets.shape[0], dtype=int)
        else:
            curr_reach_num = self.reach_num
        for batch, reach_num in enumerate(curr_reach_num):
            if ((positions[batch] - targets[batch, reach_num]) ** 2).sum() ** .5 <= eps:
                curr_reach_num[batch] += 1
        self.reach_num = np.minimum(curr_reach_num, self.reaches - 1)

    def burn_in(self, n_loops: int = 500, batch_size: int = 10, max_norm: float = 1):
        self.reset_optimizer()
        optimizer = self.optimizer
        for loop in range(n_loops):
            optimizer.zero_grad()
            targets = self.sample_targets(batch_size=batch_size)
            positions, rnn_act = self.forward(targets)
            Loss = self.calculate_loss(targets, positions, rnn_act, method='naive')
            Loss.backward()
            nn.utils.clip_grad_norm_(self.parameters(), max_norm=max_norm)
            optimizer.step()

        self.reset_optimizer()

    def fit(self, n_loops: list[int] = None, methods: list[int] = None, batch_size: int = 128, eval_freq: int = 50,
            pause: int = 50, eps: float = 1, hold_duration: int = 50, max_norm: float = 1, max_batch: int = 15):
        if n_loops is None:
            n_loops = [5000, ]
        if methods is None:
            methods = ['naive', ]
        optim = self.optimizer
        epochs = np.ceil( batch_size / max_batch).astype('int')
        for loop_dir, method in zip(n_loops, methods):
            self.reset_optimizer()
            for loop in range(loop_dir):
                optim.zero_grad()
                sigma = self.get_noise(loop, loop_dir, sig_0=0, min_noise=0)
                # self.set_difficulty(loop, loop_dir)
                sum_loss = 0
                for epoch in range(epochs):
                    targets = self.sample_targets(batch_size=max_batch, difficulty=self.difficulty)
                    positions, rnn_act = self.forward(targets, noise_scale=sigma, tolerance=eps)
                    Loss = self.calculate_loss(targets, positions, rnn_act, method=method, epsilon=eps,
                                               hold_duration=hold_duration, pause=pause)
                    Loss.backward()
                    sum_loss += Loss.item()
                self.Loss.append(sum_loss)
                nn.utils.clip_grad_norm_(self.parameters(), max_norm=max_norm)
                optim.step()

                if (loop % eval_freq) == 0:
                    logging.info(f"Iteration {loop}: Loss: {self.Loss[-1]}")
                    self.save_checkpoint(checkpoint=(loop // eval_freq))
                    self.evaluate_network()
                    self.plot_activity()

    def reset_optimizer(self):
        if self.load_cnn:
            self.optimizer = self.opt_fun(
                [
                    {'params': self.rnn.parameters()},
                    {'params': self.model_inputs.parameters(), 'lr': 1e-1 * self.opt_config['lr']}
                ], lr=self.opt_config['lr'], weight_decay=self.opt_config['wd']
            )
        else:
            self.optimizer = self.opt_fun(
                self.parameters(), lr=self.opt_config['lr'], weight_decay=self.opt_config['wd']
            )






