import re

import csaf.system as csys
import csaf.trace as ctc
import matplotlib.pyplot as plt


def plot_device(ax: plt.Axes, trajs: ctc.TimeTrace,
                dname: str, fieldname: str, index: int,
                label=None, do_integrate=False):
    """convenience method to plot one axis object"""
    import numpy as np
    t = trajs[dname].times
    t_plant = np.array(trajs["plant"].times)
    x = np.array(getattr(trajs[dname], fieldname))[:, index]

    # pretty plot for zero order hold (ZOH)
    x_match = t_plant.copy()
    x_match[:] = x[1]
    idx_max = 0
    for idx, ti in enumerate(t):
        if idx < len(t) - 1:
            idxs = np.logical_and(t_plant >= ti, t_plant < t[idx+1])
            x_match[idxs] = x[idx+1]
            barr = np.where(idxs)[0]
            if len(barr) > 0:
                idx_max = max(barr)
        else:
            x_match[idx_max:] = x[-1]
    if do_integrate:
        import scipy.integrate
        x_match = scipy.integrate.cumtrapz(x_match, t_plant)
        t_plant = t_plant[:-1]
    ax.plot(t_plant, x_match, label=label.split('(')[0])
    ax.grid()
    ax.set_xlim([min(t_plant), max(t_plant)])
    if label:
        ax.set_ylabel(re.search(r'\((.*?)\)',label).group(1))
    ax.legend()


## build and simulate system
config_filename = "../../examples/config/f16_simple_config.toml"
#config_filename = "../../examples/config/inv_pendulum_config.toml"
my_system = csys.System.from_toml(config_filename)
trajs = my_system.simulate_tspan([0, 15.0], show_status=True)

if "pendulum" in config_filename:
    fig, ax = plt.subplots(figsize=(12, 8), nrows=2, ncols=3, sharex=True)
    ax[0][0].set_title("Pendulum Plant")
    plot_device(ax[0][0], trajs, "plant", "states", 0, "position (m)", do_integrate=True)
    plot_device(ax[1][0], trajs, "plant", "states", 2, "angle (rad)", do_integrate=True)
    ax[1][0].set_xlabel("Time (s)")

    ax[0][1].set_title("Controller")
    plot_device(ax[0][1], trajs, "controller", "outputs", 0, "Force (N)")
    ax[0][1].set_xlabel("Time (s)")
    ax[1][1].axis('off')

    ax[0][2].set_title("Maneuver")
    plot_device(ax[0][2], trajs, "maneuver", "outputs", 0, "Setpoint ()")
    ax[1][2].axis('off')
    ax[0][2].set_xlabel("Time (s)")
    plt.show()

if "f16" in config_filename:
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
