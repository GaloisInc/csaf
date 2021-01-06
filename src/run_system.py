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
import pathlib
import collections

import csaf.config as cconf
import csaf.parser_test as ctp
from csaf import csaf_logger
from tests_static import RunSystemTest, RunSystemParallelTest, StaticRunTest

import plot
import run_parallel


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
    csaf_logger.info(f"Loading job config: {job_filename}")
    bdir = str(pathlib.Path(job_filename).parent.resolve())
    parser = ctp.ParallelParser(bdir, context_str=job_filename)
    job_conf = toml.load(job_filename)
    job_conf = parser.parse(job_conf)
else:
    job_conf = {}

# Get (default) plot settings
do_plot = job_conf.get('plot', True)
filename = job_conf.get('plot_filename', None)

if job_conf.get('parallel', False):
    # Parallel run
    csaf_logger.info(f"Running parallel simulation.")
    csaf_logger.debug(f"Number of initial states are: {job_conf['x0']}")

    # get terminating condition
    termcond = job_conf.get('terminating_conditions', None)
    csaf_logger.info(f"Terminating condition is: {termcond.__name__ if isinstance(termcond, collections.Callable) else termcond}")

    # Run static tests if desired
    static_tests = job_conf.get('tests', None)
    if static_tests:
        csaf_logger.info(f"Running static tests")
        from tests_static import *
        for t in static_tests:
            csaf_logger.info(f"Evaluating {t}")
            test = static_tests[t]
            #test = StaticRunTest(bdir)
            tester = test["_test_object"]
            tester.execute(model_conf)
    else:
        # Run only once
        # run tasks in a workgroup
        test = RunSystemParallelTest(job_conf, model_conf)
        runs, _ = test.execute(model_conf)

    # save initial conditions to a file
    states = job_conf['x0']
    if job_conf.get('x0_save_to_file', False) and states:
        filename = os.path.join(model_conf.output_directory, f"{model_conf.name}-x0.csv")
        csaf_logger.info(f"Saving inital conditions to {filename}")
        run_parallel.save_states_to_file(filename, states)

else:
    # Regular run
    bdir = str(pathlib.Path(config_filename).parent.resolve())
    csaf_logger.info(f"Running a single simulation.")
    test = RunSystemTest(bdir)
    test.parse({})
    trajs = test.execute(model_conf)
    filename = os.path.join(model_conf.output_directory, f"{model_conf.name}-run.png")
    if do_plot:
        plot.plot_results(config_filename, trajs, filename)

csaf_logger.info(f"CSAF done.")
