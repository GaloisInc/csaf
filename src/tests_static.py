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

    return gp_minimize(func, dimensions,
                acq_func=acq_func,
                n_calls=n_calls,
                n_random_starts=n_random_starts,
                noise=0,
                n_jobs=1)


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
            for cname, ic in x0:
                system.set_state(cname, ic)
        trajs = system.simulate_tspan(self.tspan, show_status=self.show_status)
        system.unbind()
        return trajs


class RunSystemParallelTest(cst.SystemTest):
    """test that runs one or more system configurations in parallel"""
    valid_fields = ["iterations",
                    "tspan",
                    "x0",
                    "terminating_conditions"]

    required_fields = ["iterations",
                       "tspan",
                       "x0",
                       "terminating_conditions"]

    defaults_fields = {"tspan" : (0.0, 10.0),
                       "terminating_conditions": None}

    def execute(self, system_conf):
        iterations = self.iterations if self.iterations else len(self.x0)
        runs = run_parallel.run_workgroup(iterations,
                                          system_conf,
                                          self.x0,
                                          self.tspan,
                                          terminating_conditions=self.terminating_conditions)
        # Filter out terminated runs
        passed_termcond = [val for val,_,_ in runs].count(True)
        success_rate = float(passed_termcond)/float(iterations)
        failed_runs = iterations - len(runs)
        self.logger("info", f"Out of {iterations}, {passed_termcond} passed the terminating conditions. {success_rate*100:1.2f} [%] success.")
        self.logger("info", f"{failed_runs} simulations failed with an exception.")
        return runs, passed_termcond


class StaticRunTest(RunSystemParallelTest):
    """test that evaluates traces from system simulations"""
    valid_fields = ["iterations",
                    "tspan",
                    "x0",
                    "test_methods_file",
                    "terminating_conditions",
                    "generator_config",
                    "reference",
                    "response",
                    "fcn_name"]

    required_fields = ["tspan",
                       "x0",
                       "terminating_conditions",
                       "generator_config",
                       "reference",
                       "response",
                       "fcn_name"]

    defaults_fields = {"test_methods_file": None,
                       "iterations": None}

    @_("fcn_name")
    def _(self, fcn_name: str):
        """function to analyze signals exists in this file.
        FIXME?"""
        return globals()[fcn_name]

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

        # FIXME: this is arbitrary
        if test_success_rate > 0.8:
            return True
        else:
            return False


class BayesianFalsifierTest(RunSystemTest):
    """test attempts to falsify a condition using bayesian optimization"""
    valid_fields = ["n_calls",
                    "region",
                    "property",
                    "reward",
                    "show_status",
                    "tspan"]

    required_fields = ["reward",
                       "region",
                       "property"]

    defaults_fields = {"n_calls": 10,
                       "show_status": True,
                       "tspan": (0.0, 20.0)}

    def execute(self, system_conf):
        x_false = []
        assert len(self.region) == 1, f"For now Bayesian falsifier can only test one component"
        component = list(self.region.keys())[0]

        def simulate(initial_conditions):
            system = csys.System.from_config(system_conf)
            system.set_state(component, initial_conditions)
            trajs, passed = system.simulate_tspan(self.tspan,
                                                  show_status=self.show_status,
                                                  return_passed = True,
                                                  terminating_conditions=self.property)
            system.unbind()
            if not passed:
                x_false.append(trajs[component])
                return -1.0
            return self.reward(trajs)

        ret = attack(simulate, list(zip(*[b[:-1] for b in self.region[component]])),
                     n_calls=self.n_calls)
        if len(x_false) > 0:
            self.logger("info", f"Falsified! Number of false states: {len(x_false)}")
            self.logger("info", f"Solver Minimimum: {component}<{ret['x']}>")
        else:
            self.logger("info", f"Could not falsify the terminating conditions!")
