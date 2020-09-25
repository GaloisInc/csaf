"""
Stanley Bak
Python code for F-16 animation video output
"""
import re
import os
import numpy as np

import matplotlib.pyplot as plt

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
        ax.grid(True)

    ax.set_xlim([min(t), max(t)])
    if label and not do_schedule:
        ax.set_ylabel(re.search(r'\((.*?)\)',label).group(1))
    ax.legend()

def plot_results(config_filename, trajs, filename=None):
    config_filename = os.path.basename(config_filename)

    if "inv_pendulum" in config_filename:
        from inv_plot import plot_pendulum
        plot_pendulum(trajs)

    elif "f16_shield" in config_filename:
        from f16_plot import plot_shield
        plot_shield(trajs)

    elif "f16_simple" in config_filename or "f16_llc_nn" in config_filename:
        from f16_plot import plot_simple
        plot_simple(trajs)
    elif "llc_analyze" in config_filename:
        from f16_plot import plot_llc
        plot_llc(trajs)
    else:
        raise NotImplementedError("Plot options for {} not implemented.".format(config_filename))

    # save plot of demo
    if filename:
        plt.savefig(filename)
    plt.show()