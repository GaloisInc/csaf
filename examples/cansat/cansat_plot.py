# can sat animation creator
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation
from matplotlib.colors import LinearSegmentedColormap
import typing as typ

import matplotlib.pyplot as plt
from scipy.spatial.qhull import Delaunay

from plot import plot_component


def plot_sats(trajs) -> plt.Figure:
    # pack states into convenient data structure
    states = [np.array(trajs['can' + str(idx)].states) for idx in range(4)]

    # create a plot of initial and final state
    def _plot_satellites(ax, sats, use_tri=True):
        ax.set_xlim([-20, 20])
        ax.set_ylim([-20, 20])
        points = np.array([(0, 0)] + [s[:2] for s in sats])
        for idx, p in enumerate(points):
            ax.scatter(*p)
            ax.text(p[0] - 0.5, p[1] + 0.5,
                    f"{'chief ' if idx == 0 else 'chaser '}{idx}")
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        if use_tri:
            tri = Delaunay(points)
            ax.triplot(points[:, 0], points[:, 1], tri.simplices, '--')

    fig, ax = plt.subplots(figsize=(16, 8), ncols=2, sharey=True, sharex=True)
    _plot_satellites(ax[0], [s[0, :] for s in states])
    _plot_satellites(ax[1], [s[-1, :] for s in states])
    ax[0].set_title("Initial State")
    ax[1].set_title("Final State")
    return fig


def plot_sats_anim(states, frames=1000,
                   interval=50) -> matplotlib.animation.FuncAnimation:
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xlabel('X (m)', size=12)
    ax.set_ylabel('Y (m)', size=12)
    ax.set_xlim([-20, 20])
    ax.set_ylim([-20, 20])
    ax.set_aspect('equal')
    x_vals = []
    y_vals = []
    intensity = []

    colors = [[0, 0, 1, 0], [0, 0, 1, 0.5], [0, 0.2, 0.4, 1]]
    cmap = LinearSegmentedColormap.from_list("", colors)
    scatter = ax.scatter(x_vals,
                         y_vals,
                         c=[],
                         cmap=cmap,
                         vmin=0,
                         vmax=1,
                         s=5,
                         label="Chaser Satellites")
    ax.scatter(0, 0, c='r', label="Chief Satellite")
    a_circle = plt.Circle((0, 0), 3.0, color='k', fill=False)
    ax.add_artist(a_circle)
    ax.legend()

    def get_new_vals(i):
        x = [s[i, 0] for s in states]
        y = [s[i, 1] for s in states]
        return list(x), list(y)

    def update(t):
        nonlocal x_vals, y_vals, intensity
        new_xvals, new_yvals = get_new_vals(t)
        x_vals.extend(new_xvals)
        y_vals.extend(new_yvals)

        scatter.set_offsets(np.c_[x_vals, y_vals])

        intensity = np.concatenate(
            (np.array(intensity) * 0.96, np.ones(len(new_xvals))))
        scatter.set_array(intensity)
        xb = [
            np.min(np.concatenate([s[t - 100:, 0] for s in states])),
            np.max(np.concatenate([s[t - 100:, 0] for s in states]))
        ]
        yb = [
            np.min(np.concatenate([s[t - 100:, 1] for s in states])),
            np.max(np.concatenate([s[t - 100:, 1] for s in states]))
        ]
        if t > 100:
            ax.set_xlim(xb[0] - 0.1 * np.abs(xb[0]),
                        xb[1] + 0.1 * np.abs(xb[1]))
            ax.set_ylim(yb[0] - 0.1 * np.abs(yb[0]),
                        yb[1] + 0.1 * np.abs(yb[1]))

        ax.set_title('Can Satellite Constellation Rejoin (Epoch: %d)' % t)

    return matplotlib.animation.FuncAnimation(fig,
                                              update,
                                              frames=frames,
                                              interval=interval)
