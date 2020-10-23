
# signal_gen: seed, time, params -> signal

# component: signal -> signal

# test: signal ->

#Extend to component types.

import tests
import component_lib as lib

import numpy as np

random = np.random.default_rng(seed=None)

def ramp(t):
    return t
def ramp(t):
    return t
def sin(t):
    return np.sin(t)

def gaussian_noise(mu, sigma, x0):
    return x0 + random.normal(mu, sigma)

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

# robustness of magnitude
def bibo():
    raise NotImplementedError
# robustness: decay of transients
def io_stabile():
    raise NotImplementedError

def test(component_name, input_gen, output_property):
    """
    Returns a test
    """
    raise NotImplementedError



