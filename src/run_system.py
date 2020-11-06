""" Demo System Plot

Ethan Lew
08/03/2020

Runs demo systems and plots the results to matplotlib axes

Supports the Stanley Bak's f16 model, GCAS shield f16 and
cart inverted pendulum (CIP)
"""
import os
import sys
import toml


import csaf.system as csys
import csaf.config as cconf
import csaf.trace as ctc
from csaf import csaf_logger


import plot
import run_parallel


def get_attribute(conf, attribute):
    val = conf.get(attribute, None)
    if val is None:
        csaf_logger.error(f"Missing attribute: {attribute} is {val}")
        exit(-1)
    else:
        csaf_logger.info(f"Retrieved attribute: {attribute} = {val}")
        return val

## build and simulate system
csaf_dir=sys.argv[1]
csaf_config=sys.argv[2]
config_filename = os.path.join(csaf_dir, csaf_config)

# NOTE: appending path to have access to example specific files
# Hacky way to do this?
sys.path.append(csaf_dir)

model_conf = cconf.SystemConfig.from_toml(config_filename)

# save plot of pub/sub diagram
model_conf.plot_config()

# Optional job configuration
if len(sys.argv) > 3:
    job_filename = os.path.join(csaf_dir, sys.argv[3])
    with open(job_filename, 'r') as f:
        csaf_logger.info(f"Loading job config: {job_filename}")
        job_conf = toml.load(f)
else:
    job_conf = {}

# Get (default) simulation settings
show_status = job_conf.get('show_status', True)
tspan = job_conf.get('tspan', [0, 35.0])

# Get (default) plot settings
do_plot = job_conf.get('plot', True)
filename = job_conf.get('plot_filename', None)

if job_conf.get('parallel', False):
    # Parallel run
    csaf_logger.info(f"Running parallel simulation.")
    x0 = get_attribute(job_conf, 'x0')

    if x0 == "random":
        csaf_logger.info(f"Generating random states.")
        iterations = get_attribute(job_conf, 'iterations')
        bounds = get_attribute(job_conf, 'bounds')

        # define states of component to run
        # format [{"plant": <list>}, ..., {"plant" : <list>, "controller": <list>}]
        states = run_parallel.gen_random_states(bounds, "plant", iterations)
    elif x0 == "fixed":
        csaf_logger.info(f"Generating states using fixed step.")
        bounds = get_attribute(job_conf, 'bounds')
        # define states of component to run
        states = run_parallel.gen_fixed_states(bounds, "plant")
        csaf_logger.info(f"Generated {len(states)} initial states.")
        iterations = len(states)
    elif x0 == "from_file":
        x0_path = get_attribute(job_conf, 'x0_path')
        csaf_logger.info(f"Loading states from a file: {x0_path}")
        states = run_parallel.load_states_from_file(x0_path, "plant")
        csaf_logger.info(f"Loaded {len(states)} initial states.")
        iterations = len(states)
    else:
        csaf_logger.error(f"Unknown value x0 = {x0}. Valid values are [random, fixed, from_file]")
        raise NotImplementedError

    csaf_logger.debug(f"Initial states are: {states}")

    # get terminating condition
    termcond = job_conf.get('terminating_conditions', None)
    if termcond:
        # file 'terminating_conditions' is example specific
        from terminating_conditions import *
        termcond = eval(termcond)
    csaf_logger.info(f"Terminating condition is: {termcond}")

    # run tasks in a workgroup
    runs = run_parallel.run_workgroup(iterations, model_conf, states, tspan, terminating_conditions=termcond)

    passed_termcond = [val for val,_,_ in runs].count(True)
    csaf_logger.info(f"Out of {iterations}, {len(runs)} finished, and {passed_termcond} passed the terminating conditions")

    # save initial conditions to a file
    if job_conf.get('x0_save_to_file', False):
        filename = os.path.join(model_conf.output_directory, f"{model_conf.name}-x0.csv")
        csaf_logger.info(f"Saving inital conditions to {filename}")
        run_parallel.save_states_to_file(filename, states)

else:
    # Regular run
    csaf_logger.info(f"Running a single simulation.")
    my_system = csys.System.from_config(model_conf)
    # Initial conditions
    x0 = job_conf.get('initial_conditions', None)
    if x0:
        my_system.set_state("plant", x0)
    trajs = my_system.simulate_tspan(tspan, show_status=show_status)
    my_system.unbind()
    filename = os.path.join(model_conf.output_directory, f"{model_conf.name}-run.png")
    if do_plot:
        plot.plot_results(config_filename, trajs, filename)

csaf_logger.info(f"CSAF done.")
