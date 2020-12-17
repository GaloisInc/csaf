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


class RunSystemTest(cst.SystemTest):
    required_fields = ["initial_conditions",
                       "show_status",
                       "tspan"]

    defaults_fields = {"initial_conditions": None,
                       "show_status": True,
                       "tspan": (0.0, 10.0)}

    def execute(self):
        system = csys.System.from_config(self.system_config)
        x0 = self.initial_conditions
        if x0:
            # TODO: bad hard coding
            system.set_state("plant", x0)
        trajs = system.simulate_tspan(self.tspan, show_status=self.show_status)
        system.unbind()
        return trajs


class RunSystemParallelTest(cst.SystemTest):
    required_fields = ["iterations",
                       "tspan",
                       "x0",
                       "terminating_conditions"]

    defaults_fields = {"tspan" : (0.0, 10.0),
                       "terminating_conditions": None}

    def execute(self):
        runs = run_parallel.run_workgroup(self.iterations, self.system_config, self.x0, self.tspan, self.terminating_conditions)
        # Filter out terminated runs
        passed_termcond = [val for val,_,_ in runs].count(True)
        success_rate = float(passed_termcond)/float(self.iterations)
        failed_runs = self.iterations - len(runs)
        self.logger.info(f"Out of {self.iterations}, {passed_termcond} passed the terminating conditions. {success_rate*100:1.2f} [%] success.")
        self.logger.info(f"{failed_runs} simulations failed with an exception.")
        return runs, passed_termcond


class StaticRunTest(RunSystemParallelTest):
    required_fields = ["iterations",
                       "tspan",
                       "x0",
                       "terminating_conditions",
                       "generator_config",
                       "reference",
                       "response",
                       "fcn_name"]

    def execute(self):
        # configure generator (require some generator config?)
        generator_config = self.generator_config
        if generator_config:
            for param_name in generator_config:
                self.system_config.config_dict['components']['autopilot']\
                    ['config']['parameters'][param_name]\
                    = generator_config[param_name]

        # run the parent test
        runs, passed_termcond = super().execute()

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
        self.logger.info(f"evaluated. "
                         f"{test_passed}/{passed_termcond} passed, "
                         f"{test_success_rate*100:1.2f} [%] success.")
        return test_success_rate


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
