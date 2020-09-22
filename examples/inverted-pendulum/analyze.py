import numpy as np

import csaf.system as csys
from csaf.analyze import analyze_overshoot, analyze_rise_time, analyze_steady_state_error, analyze_settling_time

from plot import plot_results

def controller_analyzer(model_conf, job_conf, config_filename):
    """
    Inverted pendulum specific version of controller analyzer

    The only available output is the desired cart position, we are controlling pendulum position to zero.

    Taken from :http://ctms.engin.umich.edu/CTMS/index.php?example=InvertedPendulum&section=ControlStateSpace

    The design criteria for this system for a 0.2-m step in desired cart position $x$ are as follows:
    * Settling time for $x$ and $\theta$ of less than 5 seconds
    * Rise time for $x$ of less than 0.5 seconds
    * Pendulum angle $\theta$ never more than 20 degrees (0.35 radians) from the vertical
    * Steady-state error of less than 2% for $x$ and $\theta$

    Our intention is to examine the controller under different initial conditions, and check whether our design
    criteria hold. 
    """
    tspan = job_conf.get('tspan', [0, 35.0])
    show_status = job_conf.get('show_status', True)
    do_plot = job_conf.get('plot', True)

    max_iterations = job_conf.get('max_iterations', 1)
    xequil = job_conf['initial_conditions']

    # Which analysis to run?
    max_overshoot = job_conf.get('max_overshoot', None)
    rise_time = job_conf.get('rise_time', None)
    steady_state_error = job_conf.get('steady_state_error', None)
    settling_time = job_conf.get('settling_time', None)

    print("Limits:")
    print("max overshoot: {} [%]".format(max_overshoot*100.0))
    print("rise time: {} [s]".format(rise_time))
    print("settling time: {} [s]".format(settling_time))
    print("steady_state_error: {} [%]".format(steady_state_error*100.0))

    # TODO: for now, raise an error, this needs to be fixed
    raise NotImplementedError("Not yet implemented.")
