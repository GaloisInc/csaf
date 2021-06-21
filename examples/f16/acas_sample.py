"""
ACAS Sampling Interfaces
"""
import functools
import numpy as np
import run_parallel as rp  # for now, batch simulations is not in CSAF package


def generate_air_collision(radius):
    def air_collision_condition(ctraces):
        """ground collision premature termnation condition
        TODO: I had to alter the terminating conditions to support this expression
        """
        # get the aircraft states
        sa, sb = ctraces['planta']['states'], ctraces['plantb']['states']
        if sa and sb:
            # look at distance between last state
            return (np.linalg.norm(np.array(sa[-1][9:11]) - np.array(sb[-1][9:11]))) < radius
    return air_collision_condition


class AirCollisionSphere:
    def __init__(self, radius):
        self.radius = radius

    def __call__(self, ctraces):
        """ground collision premature termnation condition
        TODO: I had to alter the terminating conditions to support this expression
        """
        # get the aircraft states
        sa, sb = ctraces['planta']['states'], ctraces['plantb']['states']
        if sa and sb:
            # look at distance between last state
            return (np.linalg.norm(np.array(sa[-1][9:11]) - np.array(sb[-1][9:11]))) < self.radius


class AcasSampler(object):
    """Interface Object to Direct ACAS Simulations"""
    def __init__(self, conf, tspan=(0.0, 20.0), radius=20.0):
        """
        :param conf: CSAF system config
        :param tspan: time span for simulations
        :param radius: well clear radius for collision condition
        """
        self.config = conf
        self.radius = radius
        self.tspan = tspan
        self.condition = AirCollisionSphere(self.radius)

    def run_samples(self, x_samp):
        """given an array of plant initial states,
        run batches of simulations to see if they meet the termination criteria
        :param x_samp: initial conditions of samples
        """
        # assign samples to the system plant
        init_states = [{"planta": xai, "plantb": xbi} for xai, xbi in x_samp]
        res = rp.run_workgroup(len(init_states), self.config, init_states, self.tspan,
                               terminating_conditions_all=self.condition)
        return res

    def fit(self, X, y):
        """this model cannot be trained
        :param X: initial conditions
        :param y: collision conditions
        """
        pass

    def predict(self, X):
        """classification prediction
        :param X: initial conditions
        """
        ret = self.run_samples(X)
        return np.array([b for b,_, _ in ret])


def transform_indices(func):
    """transform reduced states toi full states"""
    def transform_index(xi, center, ri):
        xt = list(center)
        for idx, ridx in enumerate(ri):
            xt[ridx] = xi[idx]
        return xt

    @functools.wraps(func)
    def inner(self, *args, **kwargs):
        xc, *rargs = args
        ri = self.reduced_indices
        center = self.center
        for xai, xbi in xc:
            assert len(xai) == len(self.reduced_indices), f"initial state {xai} doesn't have reduced state length"
            assert len(xbi) == len(self.reduced_indices), f"initial state {xbi} doesn't have reduced state length"
        xc = [(transform_index(xia, center[0], ri), transform_index(xib, center[1], ri)) for xia, xib in xc]
        return func(*(self, xc, *rargs), **kwargs)
    return inner


class AcasSamplerReduced(AcasSampler):
    """Sample ACAS with a reduced number of initial states"""
    def __init__(self, conf, center, reduced_indices, **kwargs):
        super().__init__(conf, **kwargs)
        assert len(center) == 2
        self.center = center
        self.reduced_indices = reduced_indices

    @transform_indices
    def fit(self, X, y):
        return super().fit(X, y)

    @transform_indices
    def predict(self, X):
        return super().predict(X)
