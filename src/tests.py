
# signal_gen: seed, time, params -> signal

# component: signal -> signal

# test: signal ->

#Extend to component types.
import collections

import tests
import component_lib as lib

import numpy as np

random = np.random.default_rng(seed=None)

def impulse(t): return 1 if t == 0 else 0
def step(t): return 1
def ramp(t): return t
def sin(t): return np.sin(t)

def gaussian_noise(mu, sigma, x0):
    yield x0
    while True:
        yield x0 + random.normal(mu, sigma)

def overshoot(x0, xref, xt):
    if xref > x0:
        y = np.max(xt - xref)
    elif xref < x0:
        y = np.max(xref - xt)
    else:
        y = np.abs(np.max(xt - xref))
    return y


def settling_time(tf, delta, xt):
    """
    min. t s.t. FG[ti,tf=tmax] abs(x) <= \delta
    """
    assert xt.shape == 1
    ctr = 0
    # check if the signal has settled
    for i in (xt[-1:] <= delta):
        if i:
            ctr += 1
        else:
            break
    if ctr != 0:
        return ctr
    else:
        return np.inf

def single_step_robust_io(delta, y):
    """
    Bounded change of controller outputs w.r.t. controller inputs
    """
    assert delta >= 0
    bounds = [np.min(y - yref), np.max(y - yref)]
    return np.max(np.abs(y-yref)) <= delta

def io_stable():
    """
    robustness: decay of transients
    the output of the plant should be stable w.r.t. the perturbations at the in
    input of controller
    """
    raise NotImplementedError

def robust_control_plant_changes():
    """
    Tests the controller robustness against change in plant
    [Mutation Testing]
    """
    raise NotImplementedError


# def open_loop(component, input_gen, output_property, n=10):
#     """
#     Returns an open loop test
#     """
#     component.stimulate()
#     output = collections.defaultdict(list)
#     for i in range(n):
#         y = component.collect_data()
#         res = output_property(y)
#         output[res].append(x, y)

def test(system, input_gen, output_property, n=10):
    """
    Returns a test
    """
    system.controller.stimulate()
    output = collections.defaultdict(list)
    for i in range(n):
        yt = system.plant.collect_data()
        res = output_property(yt)
        output[res].append(xt, yt)




