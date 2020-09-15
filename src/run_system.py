""" Demo System Plot

Ethan Lew
08/03/2020

Runs demo systems and plots the results to matplotlib axes

Supports the Stanley Bak's f16 model, GCAS shield f16 and
cart inverted pendulum (CIP)
"""
import re
import os
import sys

import matplotlib.pyplot as plt

import csaf.system as csys
import csaf.config as cconf
import csaf.trace as ctc


def plot_schedule(ax, t, x):
    """schedule (Gantt) chart

    :param ax: plt.Axes object to plot
    :param t: vector of times
    :param x: symbols corresponding to time t
    """

    # unique elements of x
    xu = list(set(x))

    # create a Mealy state machine to draw schedule
    for idx, xui in enumerate(xu):
        xb = [xi == xui for xi in x]
        last_xbi = False
        width = 0.0
        t_start = 0.0
        for xidx, xbi in enumerate(xb):
            if (width > 0.0 and xidx >= (len(xb) -1)):
                # if at end of time, plot remaining
                ax.barh(idx,width=width,left=t_start,color='C0')
            if xbi and last_xbi: # level high
                # continue - increase width
                width += t[xidx] - (t[xidx-1] if xidx > 0 else t_start)
            elif (xbi) and (not last_xbi): # rising edge
                # start condition - start timer and increase width
                width += t[xidx] - (t[xidx-1] if xidx > 0 else t_start)
                t_start = t[xidx]
            elif ((not xbi) and last_xbi): # falling edge
                # plot action
                ax.barh(idx,width=width,left=t_start,color='C0')
                width = 0
                t_start = 0.0
            last_xbi = xbi

    # populate labels and tick marks
    y_ticks = list(range(len(xu)))
    y_labels = xu
    ax.set_yticks(y_ticks)
    ax.set_yticklabels([y.title() for y in y_labels])


def plot_component(ax: plt.Axes, trajs: ctc.TimeTrace,
                   dname: str, fieldname: str, index: int,
                   label=None,
                   do_integrate=False,
                   do_schedule=False,
                   mult=1.0,
                   fix=False):
    """convenience method to plot one axis object
    :param ax: plt.Axes to plot
    :param trajs: csaf simulation output
    :param dname: component name to access
    :param fieldname: name of field to plot
    :param index: index in vector where field is located
    :param label: string containing "plot label (units)"
    :param do_integrate: integrate results (useful for transforming state variable)
    :param do_schedule: use a schedule plot (plot_schedule)
    :param mult: apply a multiplier (simple unit conversion)
    :param fix: center steady state at zero (easier to see features line overshoot, settle time, etc)
    """
    import numpy as np

    # times component
    t = trajs[dname].times
    # variable to plot
    x = np.array(getattr(trajs[dname], fieldname))[:, index]

    if do_schedule:
        plot_schedule(ax, t, x)
    else:
        if do_integrate:
            import scipy.integrate
            x = scipy.integrate.cumtrapz(x, t)
            t = t[:-1]
        if fix:
            x = x - x[-1]
        # pretty plot for zero order hold (ZOH)
        # times of plant
        t_plant = np.array(trajs["plant"].times)
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

        # times now match up with t_plant
        t = t_plant
        ax.plot(t_plant, mult*np.array(x_match), label=label.split('(')[0])
        ax.grid()

    ax.set_xlim([min(t), max(t)])
    if label and not do_schedule:
        ax.set_ylabel(re.search(r'\((.*?)\)',label).group(1))
    ax.legend()


## build and simulate system
csaf_dir=sys.argv[1]
csaf_config=sys.argv[2]
config_filename = csaf_dir + "/" + csaf_config

my_conf = cconf.SystemConfig.from_toml(config_filename)
my_system = csys.System.from_config(my_conf)
trajs = my_system.simulate_tspan([0, 35.0], show_status=True)
my_system.unbind()

if "pendulum" in config_filename:
    fig, ax = plt.subplots(figsize=(12, 8), nrows=2, ncols=3, sharex=True)
    ax[0][0].set_title("Pendulum Plant")
    plot_component(ax[0][0], trajs, "plant", "states", 0, "position (m)", do_integrate=True)
    plot_component(ax[1][0], trajs, "plant", "states", 2, "angle (rad)", do_integrate=True, fix=True)
    ax[1][0].set_xlabel("Time (s)")

    ax[0][1].set_title("Controller")
    plot_component(ax[0][1], trajs, "controller", "outputs", 0, "Force (N)")
    ax[0][1].set_xlabel("Time (s)")
    ax[1][1].axis('off')

    ax[0][2].set_title("Maneuver")
    plot_component(ax[0][2], trajs, "maneuver", "outputs", 0, "Setpoint ()", do_integrate=True, mult=-1.0)
    ax[1][2].axis('off')
    ax[0][2].set_xlabel("Time (s)")

if "pid" in config_filename:
    fig, ax = plt.subplots(figsize=(12, 8), nrows=2, ncols=2, sharex=True)
    ax[0][0].set_title("Pendulum Plant")
    plot_component(ax[0][0], trajs, "plant", "states", 0, "Angle (rad)")
    plot_component(ax[1][0], trajs, "plant", "states", 1, "Angular Rate (rad/s)")
    ax[1][0].set_xlabel("Time (s)")

    ax[0][1].set_title("Controller")
    plot_component(ax[0][1], trajs, "controller", "outputs", 0, "Torque (Nm)")
    plot_component(ax[1][1], trajs, "controller", "states", 0, "Error (rad s)")
    ax[1][1].set_xlabel("Time (s)")

if "shield" in config_filename:
    ## Plot Results
    fig, ax = plt.subplots(figsize=(25, 15), nrows=4, ncols=3, sharex=True)
    ax[0][0].set_title("F16 Plant")
    plot_component(ax[0][0], trajs, "plant", "states", 11, "height (ft)")
    plot_component(ax[1][0], trajs, "plant", "states", 0, "airspeed (ft/s)")
    plot_component(ax[2][0], trajs, "plant", "states", 3, "roll (degrees)")
    plot_component(ax[2][0], trajs, "plant", "states", 4, "pitch (degrees)")
    plot_component(ax[2][0], trajs, "plant", "states", 5, "yaw (degrees)")
    plot_component(ax[3][0], trajs, "plant", "states", 12, "power (%)")

    ax[0][1].set_title("Low Level Controller")
    plot_component(ax[0][1], trajs, "controller", "outputs", 0, "e ()")
    plot_component(ax[1][1], trajs, "controller", "outputs", 1, "a ()")
    plot_component(ax[2][1], trajs, "controller", "outputs", 2, "r ()")
    plot_component(ax[3][1], trajs, "controller", "outputs", 3, "throttle ()")

    ax[0][2].set_title("Autopilots")
    plot_component(ax[0][2], trajs, "monitor", "outputs", 0, "autopilot selected ()", do_schedule=True)
    plot_component(ax[1][2], trajs, "autopilot", "fdas", 0, "GCAS State ()", do_schedule=True)
    ax[1][2].set_title("GCAS Finite Discrete State")
    ax[2][2].axis('off')
    ax[3][2].axis('off')
    ax[1][2].set_xlabel('Time (s)')

    [ax[3][idx].set_xlabel('Time (s)') for idx in range(2)]

elif "f16" in config_filename:
    ## Plot Results
    fig, ax = plt.subplots(figsize=(25, 15), nrows=4, ncols=3, sharex=True)
    ax[0][0].set_title("F16 Plant")
    plot_component(ax[0][0], trajs, "plant", "states", 11, "height (ft)")
    plot_component(ax[1][0], trajs, "plant", "states", 0, "airspeed (ft/s)")
    plot_component(ax[2][0], trajs, "plant", "states", 3, "roll (degrees)")
    plot_component(ax[2][0], trajs, "plant", "states", 4, "pitch (degrees)")
    plot_component(ax[2][0], trajs, "plant", "states", 5, "yaw (degrees)")
    plot_component(ax[3][0], trajs, "plant", "states", 12, "power (%)")

    ax[0][1].set_title("Low Level Controller")
    plot_component(ax[0][1], trajs, "controller", "outputs", 0, "s0 ()")
    plot_component(ax[1][1], trajs, "controller", "outputs", 1, "s1 ()")
    plot_component(ax[2][1], trajs, "controller", "outputs", 2, "s2 ()")
    plot_component(ax[3][1], trajs, "controller", "outputs", 3, "s3 ()")

    ax[0][2].set_title("Autopilot")
    plot_component(ax[0][2], trajs, "autopilot", "outputs", 0, "a0 ()")
    plot_component(ax[1][2], trajs, "autopilot", "outputs", 1, "a1 ()")
    plot_component(ax[2][2], trajs, "autopilot", "outputs", 2, "a2 ()")
    plot_component(ax[3][2], trajs, "autopilot", "outputs", 3, "a3 ()")

    [ax[3][idx].set_xlabel('Time (s)') for idx in range(3)]

# save plot of pub/sub diagram
my_conf.plot_config()
# save plot of demo
plt.savefig(os.path.join(my_conf.output_directory, f"{my_conf.name}-run.png"))
plt.show()
