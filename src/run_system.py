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


import csaf.system as csys
import csaf.config as cconf
import csaf.trace as ctc
import csaf.test_parser as ctp
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
    csaf_logger.info(f"Loading job config: {job_filename}")
    bdir = str(pathlib.Path(job_filename).parent.resolve())
    parser = ctp.ParallelParser(bdir, context_str=job_filename)
    job_conf = toml.load(job_filename)
    job_conf = parser.parse(job_conf)
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
    states = None

    iterations = get_attribute(job_conf, 'iterations')
    bounds = job_conf['bounds']
    states = x0

    csaf_logger.debug(f"Initial states are: {states}")

    # get terminating condition
    termcond = job_conf.get('terminating_conditions', None)
    csaf_logger.info(f"Terminating condition is: {termcond}")

    # Run static tests if desired
    static_tests = job_conf.get('tests', None)
    if static_tests:
        csaf_logger.info(f"Running static tests")
        from tests_static import *
        for t in static_tests:
            csaf_logger.info(f"Evaluating {t}")
            # TODO: sanity check the values
            test = static_tests[t]

            # configure generator (require some generator config?)
            generator_config = test.get('generator_config',None)
            if generator_config:
                for param_name in generator_config:
                    #from IPython import embed; embed()
                    model_conf.config_dict['components']['autopilot']['config']['parameters'][param_name] = generator_config[param_name]

            # run tasks in a workgroup
            runs = run_parallel.run_workgroup(iterations, model_conf, states, tspan, terminating_conditions=termcond)

            # Filter out terminated runs
            passed_termcond = [val for val,_,_ in runs].count(True)
            success_rate = float(passed_termcond)/float(iterations)
            failed_runs = iterations - len(runs)

            csaf_logger.info(f"Out of {iterations}, {passed_termcond} passed the terminating conditions. {success_rate*100:1.2f} [%] success.")
            csaf_logger.info(f"{failed_runs} simulations failed with an exception.")

            # Evaluate tests
            fcn = test['fcn_name']
            ref_cmp = test['reference'][0]
            ref_idx = int(test['reference'][1])
            res_cmp = test['response'][0]
            res_idx = int(test['response'][1])

            z = [fcn(trajs,ref_cmp, ref_idx, res_cmp, res_idx) if passed else None for passed,trajs,_ in runs]
            test_passed = z.count(True)
            test_success_rate = float(test_passed)/float(passed_termcond) if passed_termcond > 0 else 0.0
            csaf_logger.info(f"{t} evaluated. {test_passed}/{passed_termcond} passed, {test_success_rate*100:1.2f} [%] success.")
    else:
        # Run only once
        # run tasks in a workgroup
        runs = run_parallel.run_workgroup(iterations, model_conf, states, tspan, terminating_conditions=termcond)

        passed_termcond = [val for val,_,_ in runs].count(True)
        success_rate = float(passed_termcond)/float(iterations)
        failed_runs = iterations - len(runs)

        csaf_logger.info(f"Out of {iterations}, {passed_termcond} passed the terminating conditions. {success_rate*100:1.2f} [%] success.")
        csaf_logger.info(f"{failed_runs} simulations failed with an exception.")

    # save initial conditions to a file
    if job_conf.get('x0_save_to_file', False) and states:
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
