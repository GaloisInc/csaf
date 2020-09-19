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

    should_continue = True

    for idx in range(max_iterations):
        print("Round {}".format(idx))
        # 0. Build system
        # TODO: better to have means to reset an existing system
        my_system = csys.System.from_config(model_conf)

        # 1.permute initial conditions
        x0 = permute_initial_conditions(xequil)
        print(x0)
        my_system.set_state("plant", x0)
        # 2. simulate
        trajs = my_system.simulate_tspan(tspan, show_status=show_status)
        # 3. analyze (overshoot, setting time, etc)
        # TODO: get the right pieces from the simulated trajectory (x_ref and x)
        # times component
        t = trajs['plant'].times
        # variable to plot
        # TODO: this would ideally be encoded in the TOML file
        x = np.array(getattr(trajs['plant'], 'states'))[:, 0]
        x_ref = np.array(getattr(trajs['maneuver'], 'outputs'))[:, 0]

        if max_overshoot is not None:
            if not analyze_overshoot(t, x_ref, x, max_overshoot):
                print("Max overshoot failed")
                should_continue = False
            else:
                print("Max overshoot OK")

        if rise_time is not None:
            if not analyze_rise_time(t, x_ref, x, rise_time):
                print("Rise time failed")
                should_continue = False
            else:
                print("Rise time OK")

        if steady_state_error is not None:
            if not analyze_steady_state_error(t, x_ref, x, steady_state_error):
                print("Steady state error failed")
                should_continue = False
            else:
                print("Steady state error OK")

        if settling_time is not None:
            if not analyze_settling_time(t, x_ref, x, settling_time):
                print("Settling time failed")
                should_continue = False
            else:
                print("Settling time OK")

        # TODO: how to specify different values for different variables? Better TOML structure?

        # 4. plot results
        if do_plot:
            plot_results(config_filename, trajs)
        
        # 5. destroy the system
        # TODO: how to make this not necessary?
        my_system.unbind()

        if not should_continue:
            print("terminating")
            break
