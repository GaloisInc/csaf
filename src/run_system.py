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

import plot

## build and simulate system
csaf_dir=sys.argv[1]
csaf_config=sys.argv[2]
config_filename = os.path.join(csaf_dir, csaf_config)

# NOTE: appending path to have access to example specific files
# Hacky way to do this?
sys.path.append(csaf_dir)

model_conf = cconf.SystemConfig.from_toml(config_filename)

# Optional job configuration
if len(sys.argv) > 3:
    job_filename = os.path.join(csaf_dir, sys.argv[3])
    with open(job_filename, 'r') as f:
        job_conf = toml.load(f)
else:
    job_conf = {}

# Get (default) simulation settings
show_status = job_conf.get('show_status', True)
tspan = job_conf.get('tspan', [0, 35.0])

# Get (default) plot settings
do_plot = job_conf.get('plot', True)
filename = job_conf.get('plot_filename', None)

# Get (deafult) analysis settings
do_analyze = job_conf.get('analyze', False)

if do_analyze:
    print("Running controller analytics")
    from analyze import controller_analyzer
    controller_analyzer(model_conf, job_conf, config_filename)
else:
    # Regular run
    print("Running simulation!")
    my_system = csys.System.from_config(model_conf)
    trajs = my_system.simulate_tspan(tspan, show_status=show_status)
    my_system.unbind()
    filename = os.path.join(model_conf.output_directory, f"{model_conf.name}-run.png")
    if do_plot:
        plot.plot_results(config_filename, trajs, filename)

# save plot of pub/sub diagram
model_conf.plot_config()
