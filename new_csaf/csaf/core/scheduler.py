"""
CSAF Component Scheduler
"""
from __future__ import annotations

import sys
import functools
import numpy as np

import typing

if typing.TYPE_CHECKING:
    # cyclic imports issue
    from csaf.core.component import Component

__all__ = ['Scheduler']


def coroutine(func: typing.Callable):
    """prime coroutine by advancing to first yield"""

    @functools.wraps(func)
    def primer(*args, **kwargs):
        gen = func(*args, **kwargs)
        next(gen)
        return gen

    return primer


class Scheduler:
    """ Scheduler takes an array of components and produces a schedule of when they
    should send input/output to one another
    """

    @staticmethod
    def get_uniform_events(ts: float, tp: float, tspan):
        """for a uniform time event component, get component events"""
        t0, tf = float(tspan[0]), float(tspan[1])
        # t0p, tfp = 0.0, float(tf - t0)
        n0 = np.ceil((t0 - tp) / ts)
        nf = np.floor((tf - tp) / ts)
        return list(np.arange(n0, nf + 1) * ts + tp)

    @staticmethod
    def get_next_event(ts: float, tp: float, t0):
        """for uniform time event component, get next event time from time t0"""
        t0p = 0.0
        n0 = np.ceil((t0p - tp) / (ts * t0p)) if not np.abs(ts * t0p) < sys.float_info.epsilon else 0.0
        return n0 * ts + tp + t0

    def __init__(self, components: typing.Dict[str, Component], component_priority: typing.Sequence[str]):
        self._components: typing.Dict[str, Component] = components
        self._priority: typing.Sequence[str] = component_priority

    @coroutine
    def get_scheduler(self, t0=0.0):
        """starting from t0, yield next events"""
        # priority sort components before iteration
        components_p = [self._components[ident] for ident in self._priority]
        ns = [np.ceil((t0 - c.sampling_phase) * c.sampling_frequency) for c in components_p]
        ctimes = [n / c.sampling_frequency + c.sampling_phase for n, c in zip(ns, components_p)]
        yield None  # for primer
        while True:
            current_time = min(ctimes)
            for cidx, ctime in enumerate(ctimes):
                c = components_p[cidx]
                if abs(ctime - current_time) < sys.float_info.epsilon:
                    yield self._priority[cidx], current_time
                    ctimes[cidx] += 1 / c.sampling_frequency

    def get_schedule_tspan(self, tspan):
        """over a given timespan tspan, determine which components will be active
        :param tspan: (t0, tf) tuple of times to schedule over
        :return list of tuples (component name, times) in time sorted order
        """
        # assert that times are valid
        assert tspan[0] <= tspan[1], f"timespan '{tspan}' is not larger at index 1"
        sched = self.get_scheduler(tspan[0])
        ret = []
        for e, t in sched:
            if t >= tspan[1]:
                return ret
            else:
                ret.append((e, t))
