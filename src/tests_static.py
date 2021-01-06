import numpy as np
import csaf.test as cst
import csaf.system as csys
import run_parallel


def get_variables(trajs, reference_component, reference_index,
             response_component, response_index,
             reference_subtopic="outputs", response_subtopic="outputs"):
    ref_time = trajs[reference_component].times
    reference = np.array(getattr(trajs[reference_component], reference_subtopic))[:, reference_index]

    res_time = trajs[response_component].times
    response = np.array(getattr(trajs[response_component], response_subtopic))[:, response_index]
    return ref_time, reference, res_time, response


def max_norm_deviation(trajs, reference_component, reference_index,
                       response_component, response_index,
                       reference_subtopic="outputs", response_subtopic="outputs",
                       start_time=1.0, threshold = 0.1):
    """Note that the reference and the response are likely being sampled
    at different rates.
    Returns true if the max difference between the normalized reference and
    the normalized response is below threshold.
    The default threshold is 0.1 (10%)
    """
    # Fetch variables
    ref_time, reference, res_time, response = get_variables(trajs,
                                                            reference_component, reference_index, response_component,
                                                            response_index, reference_subtopic, response_subtopic)

    # Get only values *after* start time
    _, ref_idx = min((val, idx) for (idx, val) in enumerate(ref_time) if val > start_time)
    _, res_idx = min((val, idx) for (idx, val) in enumerate(res_time) if val > start_time)
    # Select relevant datapoints
    ref = reference[ref_idx:]
    res = response[res_idx:]
    # Normalize
    norm = max(max(res),max(ref))
    ref = ref/norm
    res = res/norm
    # Calculate max deviation
    max_dev = abs(max(res) - max(ref))
    # Pass/fail?
    return max_dev <= threshold


class RunSystemTest(cst.SystemTest):
    """test that ones one system simulation"""
    valid_fields = ["initial_conditions",
                    "show_status",
                    "tspan"]

    required_fields = []

    defaults_fields = {"initial_conditions": None,
                       "show_status": True,
                       "tspan": (0.0, 10.0)}

    def execute(self, system_conf):
        system = csys.System.from_config(system_conf)
        x0 = self.initial_conditions
        if x0:
            # TODO: bad hard coding
            system.set_state("plant", x0)
        trajs = system.simulate_tspan(self.tspan, show_status=self.show_status)
        system.unbind()
        return trajs


class RunSystemParallelTest(cst.SystemTest):
    """test that runs one or more system configurations in parallel"""
    valid_fields = ["iterations",
                       "fcn_name",
                       "tspan",
                       "x0",
                       "terminating_conditions"]

    required_fields = ["iterations",
                       "fcn_name",
                       "tspan",
                       "x0",
                       "terminating_conditions"]

    defaults_fields = {"tspan" : (0.0, 10.0),
                       "terminating_conditions": None}

    def execute(self, system_conf):
        runs = run_parallel.run_workgroup(self.iterations, system_conf, self.x0, self.tspan, self.terminating_conditions)
        # Filter out terminated runs
        passed_termcond = [val for val,_,_ in runs].count(True)
        success_rate = float(passed_termcond)/float(self.iterations)
        failed_runs = self.iterations - len(runs)
        self.logger("info", f"Out of {self.iterations}, {passed_termcond} passed the terminating conditions. {success_rate*100:1.2f} [%] success.")
        self.logger("info", f"{failed_runs} simulations failed with an exception.")
        return runs, passed_termcond


def attack(func,
           space,
           acq_func="EI",
           n_calls=5,
           n_random_starts=2):
    """
    Attack with BO
    :param func: evaluated function with a collector callback
    :param space: lower and upper boundary
    :param acq_func: BO acquisition function
    :param n_calls: the number of evaluations of f
    :param n_random_starts: the number of random initialization points
    :return: unsafe initial states
    """
    from skopt import gp_minimize
    dimensions = np.array(space).T
    dimensions[:, 0] -= 1e-5

    gp_minimize(func, dimensions,
                acq_func=acq_func,
                n_calls=n_calls,
                n_random_starts=n_random_starts,
                noise=0,
                n_jobs=1)

    return func.get_collector()


class StaticRunTest(RunSystemParallelTest):
    """test that evaluates traces from system simulations"""
    valid_fields = ["iterations",
                       "tspan",
                       "x0",
                       "terminating_conditions",
                       "generator_config",
                       "reference",
                       "response",
                       "fcn_name"]

    required_fields = ["iterations",
                       "tspan",
                       "x0",
                       "terminating_conditions",
                       "generator_config",
                       "reference",
                       "response",
                       "fcn_name"]

    def execute(self, system_conf):
        # configure generator (require some generator config?)
        generator_config = self.generator_config
        if generator_config:
            for param_name in generator_config:
                system_conf.config_dict['components']['autopilot']\
                    ['config']['parameters'][param_name]\
                    = generator_config[param_name]

        # run the parent test
        runs, passed_termcond = super().execute(system_conf)

        # Evaluate tests
        fcn = self.fcn_name
        ref_cmp = self.reference[0]
        ref_idx = int(self.reference[1])
        res_cmp = self.response[0]
        res_idx = int(self.response[1])

        z = [fcn(trajs, ref_cmp, ref_idx, res_cmp, res_idx)
             if passed else None for passed, trajs, _ in runs]
        test_passed = z.count(True)
        test_success_rate = float(test_passed) / \
            float(passed_termcond) if \
            passed_termcond > 0 else 0.0
        self.logger("info", f"evaluated. "
                         f"{test_passed}/{passed_termcond} passed, "
                         f"{test_success_rate*100:1.2f} [%] success.")
        return test_success_rate


class BayesianFalsifierTest(RunSystemTest):
    """test attempts to falsify a condition using bayesian optimization"""
    valid_fields = ["n_calls",
                    "initial_conditions",
                    "bounds",
                    "terminating_conditions",
                    "reward_fcn",
                    "show_status",
                    "tspan"]

    required_fields = ["n_calls",
                       "initial_conditions",
                       "reward_fcn",
                       "bounds",
                       "terminating_conditions",
                       "show_status",
                       "tspan"]

    @_("reward_fcn")
    def _(self, fcn_name):
        import pathlib, sys
        if fcn_name is None:
            return None
        #FIXME: hard-coded
        pypath = str(pathlib.Path(self.base_dir) / "terminating_conditions.py")
        mod_path = str(pathlib.Path(pypath).parent.resolve())
        if mod_path not in sys.path:
            sys.path.insert(0, mod_path)
        # TODO: this is hard-coded
        return getattr(__import__("terminating_conditions"), fcn_name)

    def execute(self, system_conf):
        def simulate(initial_conditions):
            system = csys.System.from_config(system_conf)
            x0 = initial_conditions
            if x0:
                # TODO: bad hard coding
                system.set_state("plant", x0)
            trajs, passed = system.simulate_tspan(self.tspan, show_status=self.show_status, return_passed = True, terminating_conditions=self.terminating_conditions)
            system.unbind()
            return self.reward_fcn(trajs)
        ret = attack(simulate, list(zip(*[b[:-1] for b in self.bounds])))



