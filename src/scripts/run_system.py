import re

import csaf.system as csys
import csaf.trace as ctc
import matplotlib.pyplot as plt


def plot_device(ax: plt.Axes, trajs: ctc.TimeTrace,
                dname: str, fieldname: str, index: int, label=None):
    """convenience method to plot one axis object"""
    import numpy as np
    t = trajs[dname].times
    x = np.array(getattr(trajs[dname], fieldname))[:, index]
    ax.plot(t, x, label=label.split('(')[0])
    ax.grid()
    ax.set_xlim([min(t), max(t)])
    if label:
        ax.set_ylabel(re.search(r'\((.*?)\)',label).group(1))
    ax.legend()

## build and simulate system
my_system = csys.System.from_toml("../../examples/config/config.toml")
trajs = my_system.simulate_tspan([0, 10.0], show_status=True)

## Plot Results
fig, ax = plt.subplots(figsize=(15, 15), nrows=4, ncols=3, sharex=True)

ax[0][0].set_title("F16 Plant")
plot_device(ax[0][0], trajs, "plant", "states", 11, "height (ft)")
plot_device(ax[1][0], trajs, "plant", "states", 0, "airspeed (ft/s)")
plot_device(ax[2][0], trajs, "plant", "states", 3, "roll (degrees)")
plot_device(ax[2][0], trajs, "plant", "states", 4, "pitch (degrees)")
plot_device(ax[2][0], trajs, "plant", "states", 5, "yaw (degrees)")
plot_device(ax[3][0], trajs, "plant", "states", 12, "power (%)")

ax[0][1].set_title("Low Level Controller")
plot_device(ax[0][1], trajs, "controller", "outputs", 0, "s0 ()")
plot_device(ax[1][1], trajs, "controller", "outputs", 1, "s1 ()")
plot_device(ax[2][1], trajs, "controller", "outputs", 2, "s2 ()")
plot_device(ax[3][1], trajs, "controller", "outputs", 3, "s3 ()")

ax[0][2].set_title("Autopilot")
plot_device(ax[0][2], trajs, "autopilot", "outputs", 0, "a0 ()")
plot_device(ax[1][2], trajs, "autopilot", "outputs", 1, "a1 ()")
plot_device(ax[2][2], trajs, "autopilot", "outputs", 2, "a2 ()")
plot_device(ax[3][2], trajs, "autopilot", "outputs", 3, "a3 ()")

[ax[3][idx].set_xlabel('Time (s)') for idx in range(3)]
plt.show()
