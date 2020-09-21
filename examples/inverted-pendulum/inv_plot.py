import matplotlib.pyplot as plt

from plot import plot_component

def plot_pendulum(trajs):
    """
    Plot trajectory of inverted pendulum
    """
    fig, ax = plt.subplots(figsize=(12, 8), nrows=2, ncols=2, sharex=True)
    ax[0][0].set_title("Pendulum Plant")
    plot_component(ax[0][0], trajs, "plant", "states", 0, "position (m)")
    plot_component(ax[0][0], trajs, "maneuver", "outputs", 0, "Setpoint ()", mult=-1.0)
    plot_component(ax[1][0], trajs, "plant", "states", 2, "angle (rad)")
    ax[1][0].set_xlabel("Time (s)")

    ax[0][1].set_title("Controller")
    plot_component(ax[0][1], trajs, "controller", "outputs", 0, "Force (N)")

    ax[1][1].set_title("Maneuver")
    plot_component(ax[1][1], trajs, "maneuver", "outputs", 0, "Setpoint ()", mult=-1.0)
    ax[1][1].set_xlabel("Time (s)")

    return fig
