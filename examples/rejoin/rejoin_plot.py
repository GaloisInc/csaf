import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation
from matplotlib.colors import LinearSegmentedColormap


def plot_air(states, frames=1000, interval=50):
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xlabel('X (m)', size=12)
    ax.set_ylabel('Y (m)', size=12)
    ax.set_xlim([-200, 200])
    ax.set_ylim([-250, 100])
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
                         label="Aircraft")

    def get_new_vals(i):
        x = [s[i, 0] * 10 for s in states]
        y = [s[i, 1] * 10 for s in states]
        return list(x), list(y)

    def update(t):
        nonlocal x_vals, y_vals, intensity

        # Get intermediate points
        new_xvals, new_yvals = get_new_vals(t)
        x_vals.extend(new_xvals)
        y_vals.extend(new_yvals)

        scatter.set_offsets(np.c_[x_vals, y_vals])

        #calculate new color values
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
            ax.set_xlim(xb[0] * 10 - 0.1 * np.abs(xb[0] * 10),
                        xb[1] * 10 + 0.1 * np.abs(xb[1] * 10))
            ax.set_ylim(yb[0] * 10 - 0.1 * np.abs(yb[0] * 10),
                        yb[1] * 10 + 0.1 * np.abs(yb[1] * 10))

        ax.set_title('Dubin\'s 2D Plane Rejoin (Epoch: %d)' % t)

    return matplotlib.animation.FuncAnimation(fig,
                                              update,
                                              frames=300,
                                              interval=50)
