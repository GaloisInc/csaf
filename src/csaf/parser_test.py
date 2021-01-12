from .parser import ConfigParser

import numpy as np
import pathlib
import sys
import typing as typ
import collections


def save_states_to_file(filename, states):
    np.savetxt(filename, [val['plant'] for val in states], delimiter=",")


def load_states_from_file(filename, component_name):
    x0s = np.loadtxt(filename, delimiter=",")
    return [{component_name: initial_state} for initial_state in x0s]


def gen_fixed_states(bounds, component_name):
    def sanity_check(bounds):
        # sanity check
        for b in bounds:
            assert (len(b) == 1 or len(b) == 3)
            if len(b) == 3:
                # lower bound is always first
                lower = b[0]
                upper = b[1]
                step = b[2]
                assert (lower <= upper)
                # the step is smaller than the bounds interval
                assert (upper - lower >= step), f"{upper} - {lower} >= {step}"

    def interpolate_bounds(lower, upper, step) -> np.ndarray:
        step = 1.0 if np.isclose(step, 0.0) else step
        iters = int((upper - lower) / step)
        return np.linspace(lower, upper, iters)

    bounds = bounds[component_name]

    sanity_check(bounds)

    # create initial vector
    x0 = np.array([b[0] for b in bounds])
    x0s = [x0]
    # iterate over bounds
    for idx, b in enumerate(bounds):
        # ignore static values
        if len(b) == 1:
            continue
        vals = interpolate_bounds(b[0], b[1], b[2])
        new_x0s = []
        for x in x0s:
            for val in vals:
                new_x0 = x.copy()
                # ignore the value that already exists
                if new_x0[idx] == val:
                    continue
                new_x0[idx] = val
                new_x0s.append(new_x0)
        x0s += new_x0s

    return [{component_name: initial_state} for initial_state in x0s]


def gen_random_states(bounds, component_name, iterations):
    bounds = bounds[component_name]
    def generate_single_random_state(bounds):
        sample = np.random.rand(len(bounds))
        ranges = np.array(
            [b[1] - b[0] if len(b) == 2 else b[0] for b in bounds])
        offset = np.array([-b[0] for b in bounds])
        return sample * ranges - offset

    return [{
        component_name: generate_single_random_state(bounds)
    } for _ in range(iterations)]


class ParallelParser(ConfigParser):
    valid_fields = [
        "tspan", "show_status", "plot", "initial_conditions", "parallel", "x0",
        "x0_path", "bounds", "terminating_conditions", "x0_save_to_file",
        "tests", "plot_filename", "iterations", "x0_component_name", "component_under_test",
        "test_methods_file"
    ]

    required_fields = ["initial_conditions"]

    defaults_fields = {
        "tspan": (0.0, 35.0),
        "show_status": True,
        "plot": True,
        "parallel": True,
        "x0": "fixed",
        "x0_component_name": "plant",
        "x0_save_to_file": True,
        "x0_path": "x0.csv",
        "test_methods_file": "terminating_conditions.py",
        "component_under_test": "plant",
        "bounds": None,
        "plot_filename": None,
        "iterations": 100,
        "tests": {},
        "terminating_conditions": None
    }

    @_("iterations")
    def _(self, iterations: int) -> int:
        return iterations

    @_("tspan")
    def _(self, tspan: typ.List[float]) -> typ.Tuple:
        if len(tspan) != 2:
            self.logger("error", f"tspan provided must be length 2 (got {tspan} instead)",
                        error=ValueError)
        if tspan[1] <= tspan[0]:
            self.logger("error",
                        f"tspan provided must be in the form (tmin, tmax), where tmax > tmin (got {tspan} instead)",
                        error=ValueError)
        return tuple(tspan)

    @_("test_methods_file")
    def _(self, py_fname):
        if isinstance(py_fname, str):
            pypath = str(pathlib.Path(self.base_dir) / py_fname)
            # update python path to include module directory
            mod_path = str(pathlib.Path(pypath).parent.resolve())
            if mod_path not in sys.path:
                sys.path.insert(0, mod_path)
            return __import__(pathlib.Path(py_fname).stem)
        else:
            return py_fname

    @_("terminating_conditions", depends_on = ("test_methods_file",))
    def _(self, fcn_name: typ.Union[str, None, typ.Callable]) -> typ.Callable:
        if fcn_name is None:
            return None
        return getattr(self.test_methods_file, fcn_name)

    @_("x0_component_name")
    def _(self, name: str) -> str:
        # TODO: check?
        return name

    @_("x0_save_to_file")
    def _(self, xstf: bool) -> str:
        return xstf

    @_("x0",
       depends_on=("bounds",
                   "iterations",
                   "bounds",
                   "x0_path",
                   "x0_component_name",
                   "component_under_test"))
    def _(self, x0: str) -> typ.Sequence[typ.Dict]:
        acceptable_x = {"fixed", "random", "from_file"}
        if x0 not in acceptable_x:
            self.logger("error",
                        f"x0 {x0} not an acceptable value {acceptable_x}",
                        error=AssertionError)
        if x0 == "random":
            self.logger("info", "Generating random states")
            return gen_random_states(self.bounds, self.component_under_test, self.iterations)
        elif x0 == "fixed":
            self.logger("info", "Generating states using fixed step.")
            states = gen_fixed_states(self.bounds, self.component_under_test)
            self._config["iterations"] = len(states)
            self.logger("info", f"Generated {len(states)} fixed initial states.")
            return states
        elif x0 == "from_file":
            pathx0 = (pathlib.Path(self.base_dir) / self.x0_path).resolve()
            self.logger("info", "Loading states from a file: {pathx0}")
            states = load_states_from_file(str(pathx0), self.component_under_test)
            self._config["iterations"] = len(states)
            self.logger("info", f"Loaded {len(states)} initial states.")
            return states
        else:
            self.logger("error",
                        f"Unknown value x0 = {x0}. Valid values are {acceptable_x}",
                        error=ValueError)

    @_("bounds")
    def _(self, bounds):
        b_new = {}
        if bounds is None:
            return None
        for pname, pbound in bounds.items():
            bn = []
            for b in pbound:
                if len(b) not in [1, 2, 3]:
                    self.logger("error",
                                f"bound {b} must be length 1, 2, or 3",
                                error=AssertionError)
                minv = b[0]
                if len(b) > 1:
                    if b[0] > b[1]:
                        self.logger("error",
                                    f"lower bound is greater than upper bound {b}",
                                    error=AssertionError)
                    maxv = b[1]
                else:
                    maxv = minv
                if len(b) == 3:
                    step = b[2]
                else:
                    step = (maxv - minv) / 10.0
                bn.append((minv, maxv, step))
            b_new[pname] = tuple(bn)
        return b_new

    @_("tests", depends_on=("x0",
                            "bounds",
                            "tspan",
                            "iterations",
                            "terminating_conditions"))
    def _(self, tests: dict) -> dict:
        for tname, tconf in tests.items():
            if not isinstance(tconf, dict):
                self.logger("error",
                            f"{tname} field is not a component map",
                            error=ValueError)
            tcls = getattr(__import__("tests_static"), tconf["test_type"])
            tpr = tcls(self.base_dir,
                             context_str=self.context_str + f"<{tname}>")
            preconf = {k: v for k, v in self._config.items() if k in tpr.valid_fields}
            methods = {f: self._field_name_methods[f] for f in tconf.get('parameters', {}) if f in self._field_name_methods}
            tpr._field_name_methods = {**methods, **tpr._field_name_methods}
            tests[tname] = tpr.parse({**preconf, **tconf.get("parameters", {})})
            tests[tname]["_test_object"] = tpr
        return tests


class TestParser(ConfigParser):
    valid_fields = ["test_type", "parameters"]

    required_fields = ["test_type"]

    fields_defaults = {"parameters": {}}
