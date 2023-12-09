import pdb
import copy
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib as mpl
from networks import (RNN, CNN, MLP, SequentialReachingNetwork)
from matplotlib.animation import FuncAnimation, PillowWriter
from sklearn.decomposition import PCA
from sklearn.metrics import r2_score
from itertools import combinations
# TODO: Check if regions have different abstraction levels


class Network(SequentialReachingNetwork):
    def __init__(self, duration: int = 300, n_clusters: int = 3, n_reaches: int = 2, use_cnn: bool = False,
                 device: torch.device = None, load_cnn: bool = False, base_lr: float = 1e-3,
                 control_type: str = 'acceleration',
                 wd: float = 1e-5, n_hidden: int = 250, **kwargs):
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

        def animation_function(time):
            im.set_array(imaged_activity[time])
            tj.set_data(position[:time, 0], position[:time, 0])
            return im, tj

        anim = FuncAnimation(
            fig,
            animation_function,
            frames=activity.shape[0],
            interval=1000/fps,
            blit=False
        )
        handle = self.save_path.parents[0] / 'animation.gif'
        FFwriter = PillowWriter(fps=fps)
        anim.save(handle, writer=FFwriter)
        plt.pause(15)
        # pdb.set_trace()

    def test_sensitivity(self, samples: int = 10, cmap: mpl.cm.ScalarMappable = None,
                         plot_3d: bool = True, save_path: str = None):

        if cmap is None:
            cmap = mpl.cm.plasma

        plt.close('all')

        normalization = mpl.colors.Normalize(vmin=0, vmax=samples)
        grid_width = self.rnn.grid_width
        position = self.max_bound / 2 * torch.ones(samples, 2)
        # for reach in range(self.reaches):
        reach = 0
        targets = self.sample_targets(batch_size=samples)
        targets[:, reach] = targets[0, reach]
        sorted_probs = np.argsort(self.probs)
        sorted_probs = np.flip(sorted_probs)
        for sample in range(samples):
            target = sorted_probs[sample]
            targets[sample, 1:] = target * torch.ones((self.reaches - 1)) / 100

        with torch.no_grad():
            positions, rnn_activation = self.forward(targets, noise_scale=0)
        activity = rnn_activation.squeeze().cpu().numpy()
        position = positions.squeeze().cpu().numpy()
        nrows = np.ceil(activity.shape[2] / grid_width).astype(int)
        imaged_activity = np.zeros((activity.shape[0], samples, nrows, grid_width))
        for neu_idx in range(activity.shape[2]):
            col_idx = neu_idx % grid_width
            row_idx = neu_idx // grid_width
            imaged_activity[:, :, row_idx, col_idx] = activity[:, :, neu_idx]

        pcs = np.zeros((grid_width, int(samples * imaged_activity.shape[0] / 2), 3))
        duration = int(imaged_activity.shape[0] / 2)
        for region in range(grid_width):
            region_activity = np.swapaxes(
                imaged_activity[:int(imaged_activity.shape[0] / 2), :, :, region], 0, -1
            )
            flattened_activity = region_activity.reshape(region_activity.shape[0], -1)
            pca = PCA(n_components=3)
            pcs[region] = pca.fit_transform(flattened_activity.T)

        if plot_3d:
            fig, ax = plt.subplots(1, grid_width, figsize=(5 * grid_width, 5),
                                   subplot_kw={"projection": "3d"})
        else:
            fig, ax = plt.subplots(1, grid_width, figsize=(5 * grid_width, 5))

        for region, axis in zip(range(grid_width), ax):
            axis.set_title(f"Region {region}")
            for sample in range(samples):
                if plot_3d:
                    axis.scatter(pcs[region, sample * duration, 0], pcs[region, sample * duration, 1],
                                 pcs[region, sample * duration, 2],
                                 color=cmap(normalization(sample)))
                    axis.plot(pcs[region, sample * duration: (sample + 1) * duration, 0],
                                    pcs[region, sample * duration: (sample + 1) * duration, 1],
                              pcs[region, sample * duration: (sample + 1) * duration, 2],
                              color=cmap(normalization(sample)))
                else:
                    axis.scatter(pcs[region, sample * duration, 0], pcs[region, sample * duration, 1],
                                 color=cmap(normalization(sample)))
                    axis.plot(pcs[region, sample * duration: (sample + 1) * duration, 0],
                              pcs[region, sample * duration: (sample + 1) * duration, 1],
                              color=cmap(normalization(sample)))

        fig.tight_layout()
        for axis in ax:
            axis.set_xlabel('PC 1')
            axis.set_ylabel('PC 2')
            if plot_3d:
                axis.set_zlabel('PC 3')
        if save_path is not None:
            file_name = save_path / 'pc_plot'
            fig.savefig(file_name)
        plt.pause(1)

    def make_fig_layout(self):
        self.fig, self.ax = plt.subplots(1, 4, figsize=(30, 8))
        self.draw_colorbar = True
        # fig, ax = plt.subplots(1, 2, figsize=(30, 12))
        # self.ani_fig = (fig, ax)

    def evaluate_contribution(self, cmap: mpl.cm.ScalarMappable = None, plot: bool = False):
    # TODO: implemnt linear probes for initial target location information
        def gini(x: list[float]) -> float:
            x = np.array(x, dtype=np.float32)
            n = len(x)
            diffs = sum(abs(i - j) for i, j in combinations(x, r=2))
            return diffs / (2 * n ** 2 * x.mean())

        plt.close('all')
        if cmap is None:
            cmap = mpl.cm.plasma

        grid_width = self.rnn.grid_width
        position = self.max_bound / 2 * torch.ones(1, 2)
        targets = self.sample_targets(batch_size=100)

        mask = torch.ones(self.rnn.J.shape[0])
        with torch.no_grad():
            original_position, _ = self.forward(targets, noise_scale=0, mask=mask)

        position_store = []
        normalization = mpl.colors.Normalize(vmin=0, vmax=grid_width)

        for idx in range(grid_width):
            for n_idx in range(self.rnn.J.shape[0]):
                if n_idx % grid_width == idx:
                    mask[n_idx] = 0
            with torch.no_grad():
                positions, rnn_activation = self.forward(targets, noise_scale=0, mask=mask)
            position_store.append(positions)

        if plot:
            fig, ax = plt.subplots(1, 2)

        results = []

        for idx, position in enumerate(position_store):
            if plot:
                ax[0].plot(position[:, 0, 0], color=cmap(normalization(idx)))
                ax[1].scatter(idx, ((position - original_position)**2).mean())
            vaf = r2_score(original_position.flatten().numpy(), position.flatten().numpy())
            results.append({'Regions Removed': idx + 1,
                           'Error': ((position.numpy() - original_position.numpy()) ** 2).mean(),
                            'VAF': vaf, 'gini': np.nan,
                            'Type': 'Cumulative', 'Preserved Region': np.nan,
                            'Final Loss': self.Loss[-1],
                            'VAF Ratio': np.nan,
                            **self.params})
        if plot:
            ax[0].plot(original_position[:, 0, 0], color='k', ls='--', label='Unperturbed')
            ax[0].legend()
            ax[1].set_xlabel('Removed regions')
            ax[1].set_yscale('log')

        position_store = []
        if plot:
            fig2, ax2 = plt.subplots(1, 2)

        for idx in range(grid_width):
            mask = torch.ones(self.rnn.J.shape[0])
            for n_idx in range(self.rnn.J.shape[0]):
                if n_idx % grid_width == idx:
                    mask[n_idx] = 0
            with torch.no_grad():
                positions, rnn_activation = self.forward(targets, noise_scale=0, mask=mask)
            position_store.append(positions)

        for idx, position in enumerate(position_store):
            if plot:
                ax2[0].plot(position[:, 0, 0], color=cmap(normalization(idx)))
                ax2[1].scatter(idx, ((position - original_position)**2).mean())
            vaf = r2_score(original_position.flatten().numpy(), position.flatten().numpy())
            results.append({'Regions Removed': idx, 'Error': ((position.numpy() - original_position.numpy()) ** 2).mean(),
                            **self.params,
                            'Type': 'Single', 'Preserved Region': np.nan,
                            'VAF': vaf, 'gini': np.nan,
                            'Final Loss': self.Loss[-1],
                            'VAF Ratio': np.nan,
                            })

        for sample in range(25):
            vafs = []
            curr_results = []
            targets = self.sample_targets(batch_size=24)
            with torch.no_grad():
                original_position, _ = self.forward(targets, noise_scale=0, mask=mask)
            for idx in range(grid_width):
                plt.close('all')
                mask = torch.zeros(self.rnn.J.shape[0])
                for n_idx in range(self.rnn.J.shape[0]):
                    if n_idx % grid_width == idx:
                        mask[n_idx] = 1
                with torch.no_grad():
                    _, rnn_activation = self.forward(targets, noise_scale=0, mask=mask)
                rnn_activity = rnn_activation.numpy()
                positions = original_position.numpy()
                if self.control == 'position':
                    all_features = rnn_activity.reshape(rnn_activity.shape[0] * rnn_activity.shape[1], -1)
                    target = positions.reshape(positions.shape[0] * positions.shape[1], -1)
                    features = np.hstack([all_features[:, np.abs(all_features.sum(axis=0)) > np.finfo('float').eps],
                                          np.ones((all_features.shape[0], 1))])
                elif self.control == 'acceleration':
                    velocity = np.diff(positions, axis=0)
                    acceleration = np.diff(velocity, axis=0)
                    target = np.swapaxes(acceleration, 0, 1).reshape(acceleration.shape[0] * acceleration.shape[1], -1)
                    rnn_activity = rnn_activity[:-2]
                    all_features = np.swapaxes(rnn_activity, 0, 1).reshape(rnn_activity.shape[0] * rnn_activity.shape[1], -1)
                    features = np.hstack([all_features[:, np.abs(all_features.sum(axis=0)) > np.finfo('float').eps],
                                          np.ones((all_features.shape[0], 1))])
                k = 1e-3
                coeffs = np.linalg.inv(features.T @ features + k * np.eye(features.shape[1])) @ features.T @ target
                testing = features @ coeffs
                r2 = r2_score(target, testing)
                vafs.append(r2)
                if hasattr(self, 'target_loss'):
                    final_loss = float(self.target_loss[-1])
                else:
                    final_loss = self.Loss[-1]
                curr_results.append(
                    {'Regions Removed': np.nan, 'Error': ((testing - target) ** 2).mean(),
                     **self.params,
                     'Type': 'Single Linear', 'Preserved Region': idx,
                     'VAF': r2,
                     'Final Loss': final_loss,}
                    )
            gini_coeff = gini(vafs)
            [result.update({'gini': gini_coeff}) for result in curr_results]
            [result.update({'VAF Ratio': vafs[-1]/vafs[0]}) for result in curr_results]
            results.extend(curr_results)
        if plot:
            ax2[0].plot(original_position[:, 0, 0], color='k', ls='--', label='Unperturbed')
            ax2[0].legend()
            ax2[1].set_xlabel('Removed regions')
            ax2[1].set_yscale('log')
            plt.pause(.01)
        return results


def main(gpu: int = 2):
    if torch.cuda.is_available():
        torch.set_default_tensor_type(torch.cuda.FloatTensor)

    ncudas = torch.cuda.device_count()
    if ncudas > 1:
        device = torch.device(f"cuda:{gpu}" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    samples = 10
    decay = np.logspace(-5, -3, samples, base=10)
    for sample, wd in zip(range(samples), decay):
        plt.close('all')
        model = Network(device=device, max_speed=2, wd=wd)
        model.to(device)
        model.burn_in(n_loops=100)
        model.fit(eps=1e-1)
        model.save_model()
    model.test_sensitivity()
    model.evaluate_network()
    model.plot_activity()



if __name__ == '__main__':
    repeats = 5
    for _ in range(repeats):
        main()
