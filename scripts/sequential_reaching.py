import pdb
import copy
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib as mpl
from networks import (RNN, CNN, MLP, SequentialReachingNetwork)
from losses import DistanceLoss
from matplotlib.animation import FuncAnimation
from IPython import display


class Network(SequentialReachingNetwork):
    def __init__(self, duration: int = 300, n_clusters: int = 2, n_reaches: int = 2, use_cnn: bool = False,
                 device: torch.device = None, load_cnn: bool = False, base_lr: float = 1e-3,
                 control_type: str = 'acceleration',
                 wd: float = 1e-4, n_hidden: int = 250, **kwargs):
        super().__init__(duration=duration, n_clusters=n_clusters, n_reaches=n_reaches, use_cnn=use_cnn,
                 device=device, load_cnn=load_cnn, base_lr=base_lr,
                 control_type=control_type, wd=wd, n_hidden=n_hidden, **kwargs)

        assert control_type in ['acceleration', 'position']
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.device = device
        self.draw_colorbar = None
        self.ani = None
        self.make_fig_layout()

    def evaluate_network(self):
        targets = self.sample_targets(batch_size=1)
        with torch.no_grad():
            positions, rnn_activation = self.forward(targets, noise_scale=0)
            Loss = self.calculate_loss(targets, positions, rnn_activation)

        position = positions.detach().cpu().numpy().squeeze()
        targs = targets.cpu().numpy().squeeze()
        plt.title(Loss.cpu().numpy())
        # self.fig.clf()
        for ax in self.ax:
            ax.cla()
        self.ax[0].scatter(self.max_bound / 2, self.max_bound / 2, label=f'Start', s=50)
        if self.reaches > 1:
            for idx, target in enumerate(targs):
                self.ax[0].scatter(target[0], target[1], label=f'Target {idx + 1}', s=75, alpha=.75)
        else:
            self.ax[0].scatter(targs[0], targs[1], label=f'Target 1', s=75, alpha=.75)
        self.ax[0].plot(position[:, 0], position[:, 1], ls='--')
        self.ax[0].set_xlim([-.2 * self.max_bound, 1.2 * self.max_bound])
        self.ax[0].set_ylim([-.2 * self.max_bound, 1.2 * self.max_bound])
        self.ax[0].legend()
        self.ax[1].plot(self.Loss)
        self.ax[1].set_yscale('log')
        plt.pause(1)

    def plot_activity(self, cmap: mpl.colors.ListedColormap = None,
                      vmin: float = -1, vmax: float = 1, fps: int = 50):
        if cmap is None:
            cmap = mpl.cm.plasma

        normalization = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
        grid_width = self.rnn.grid_width
        targets = self.sample_targets(batch_size=1)

        with torch.no_grad():
            positions, rnn_activation = self.forward(targets, noise_scale=0)
        targets = targets.squeeze().cpu().numpy()
        activity = rnn_activation.squeeze().cpu().numpy()
        position = positions.squeeze().cpu().numpy()
        nrows = np.ceil(activity.shape[1] / grid_width).astype(int)
        imaged_activity = np.zeros((activity.shape[0], nrows, grid_width))
        for neu_idx in range(activity.shape[1]):
            col_idx = neu_idx % grid_width
            row_idx = neu_idx // grid_width
            imaged_activity[:, row_idx, col_idx] = activity[:, neu_idx]
        fig = self.fig
        axes = self.ax[-2:]

        for ax in axes:
            ax.clear()

        a = imaged_activity[0]
        axes[0].set_title('RNN Activity')
        axes[1].set_title('Trajectory')
        axes[1].set_ylabel('Y position')
        axes[1].set_xlabel('X position')
        axes[0].set_xlabel('Region')

        for reach in range(self.reaches):
            axes[1].scatter(targets[reach, 0], targets[reach, 0], label=f'Target {reach +1}')
        axes[1].legend()
        im = axes[0].imshow(a, cmap=cmap, norm=normalization, aspect='auto')
        tj, = axes[1].plot(position[0, 0], position[0, 1])
        axes[1].set_ylim([-.2 * self.max_bound, 1.2 * self.max_bound])
        axes[1].set_xlim([-.2 * self.max_bound, 1.2 * self.max_bound])
        fig.tight_layout()

        if self.draw_colorbar:
            plt.colorbar(im, label='Activity')
            self.draw_colorbar = False
        def AnimationFunction(time):
            im.set_array(imaged_activity[time])
            tj.set_data(position[:time, 0], position[:time, 0])
            return im, tj

        anim = FuncAnimation(
            fig,
            AnimationFunction,
            frames=activity.shape[0],
            interval=1000/fps,
            blit=False
        )

        self.ani = anim
        plt.pause(15)

    def test_sensitivity(self, samples: int = 100):
        constant = []
        vars = []
        position = self.max_bound / 2 * torch.ones(samples, 2)
        plt.close('all')
        for reach in range(self.reaches):
            targets = self.sample_targets(batch_size=samples)
            targets[:, reach] = targets[0, reach]
            self.get_reach_number(targets, position, reset=True)
            target = targets.reshape(targets.shape[0], -1) / self.max_bound
            trigger_sig = self.get_trigger_signal(targets, position)
            reach_nums = (torch.from_numpy(self.reach_num.astype(np.float32))[:, None]).to(self.device)
            input = torch.hstack([target, position / self.max_bound, trigger_sig[:, None], reach_nums])
            with torch.no_grad():
                rnn_input = self.model_inputs(input).cpu().numpy()
            vars.append((rnn_input.var(axis=0)).mean())
            constant.append(reach)

        ## Vary trigger sig
        targets = self.sample_targets(batch_size=samples)
        for reach in range(self.reaches):
            targets[:, reach] = targets[0, 0]
        self.get_reach_number(targets, position, reset=True)
        target = targets.reshape(targets.shape[0], -1) / self.max_bound
        trigger_sig = torch.linspace(0, 1, samples)
        reach_nums = (torch.from_numpy(self.reach_num.astype(np.float32))[:, None]).to(self.device)
        input = torch.hstack([target, position / self.max_bound, trigger_sig[:, None], reach_nums])
        with torch.no_grad():
            rnn_input = self.model_inputs(input).cpu().numpy()
        vars.append((rnn_input.var(axis=0)).mean())
        constant.append(self.reaches)

        ## Vary reach signal
        targets = self.sample_targets(batch_size=samples)
        for reach in range(self.reaches):
            targets[:, reach] = targets[0, 0]
        self.get_reach_number(targets, position, reset=True)
        target = targets.reshape(targets.shape[0], -1) / self.max_bound
        trigger_sig = self.get_trigger_signal(targets, position)
        reach_nums = torch.linspace(0, self.reaches, samples).to(self.device)[:, None]
        input = torch.hstack([target, position / self.max_bound, trigger_sig[:, None], reach_nums])
        with torch.no_grad():
            rnn_input = self.model_inputs(input).cpu().numpy()
        vars.append((rnn_input.var(axis=0)).mean())
        constant.append(self.reaches + 1)

        targ_labels = [f"Target {i+1}" for i in range(self.reaches)]
        labels = targ_labels + ['All Targets & Reach number', 'All Targets & trigger']
        plt.figure()
        plt.scatter(constant, vars)
        plt.xlabel('Constant Input')
        plt.ylabel('Mean variance')
        plt.xticks(ticks=np.arange(len(constant)), labels=labels, rotation=90)
        plt.pause(0.1)

    def make_fig_layout(self):
        self.fig, self.ax = plt.subplots(1, 4, figsize=(25, 12))
        self.draw_colorbar = True
        # fig, ax = plt.subplots(1, 2, figsize=(30, 12))
        # self.ani_fig = (fig, ax)

    def reset_optimizer(self):
        self.Loss = []
        if self.load_cnn:
            self.optimizer = torch.optim.AdamW(
                [
                    {'params': self.rnn.parameters()},
                    {'params': self.model_inputs.parameters(), 'lr': 1e-1 * self.opt_config['lr']}
                ], lr=self.opt_config['lr'], weight_decay=self.opt_config['wd']
            )

        else:
            self.optimizer = torch.optim.AdamW(
                self.parameters(), lr=self.opt_config['lr'], weight_decay=self.opt_config['wd']
            )


def main(gpu: int = 0):
    if torch.cuda.is_available():
        torch.set_default_tensor_type(torch.cuda.FloatTensor)

    ncudas = torch.cuda.device_count()
    if ncudas > 1:
        device = torch.device(f"cuda:{0}" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Network(device=device, max_speed=2)
    model.to(device)
    model.burn_in(n_loops=100)
    model.fit(eps=1e-1)
    # model.test_sensitivity()
    model.evaluate_network()
    model.plot_activity()
    pdb.set_trace()



if __name__ == '__main__':
    main()
