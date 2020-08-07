""" Component Scheduler
"""
import sys
import functools
import numpy as np


class Scheduler:
    """ Scheduler takes an array of components and produces a schedule of when they
    should send input/output to one another
    """
    @staticmethod
    def get_uniform_events(ts: float, tp: float, tspan):
        """for a uniform time event component, get component events"""
        t0, tf = float(tspan[0]), float(tspan[1])
        t0p, tfp = 0.0, float(tf - t0)
        n0 = np.ceil((t0p - tp) / ts)
        nf = np.floor((tfp - tp) / ts)
        return list(np.arange(n0, nf + 1) * ts + tp + t0)

    @staticmethod
    def get_next_event(ts: float, tp: float, t0):
        """for uniform time event component, get next event time from time t0"""
        t0p = 0.0
        n0 = np.ceil((t0p - tp) / (ts * t0p)) if not np.abs(ts * t0p) < sys.float_info.epsilon else 0.0
        return n0 * ts + tp + t0

    @staticmethod
    def cmp_sort_priority(s0, s1, priority):
        if np.abs(s0[1] - s1[1]) < sys.float_info.epsilon:
            # prefer by eval order
            if priority.index(s0[0]) < priority.index(s1[0]):
                return -1
            else:
                return 1
        else:
            if s0[1] < s1[1]:
                return -1
            else:
                return 1

    def __init__(self, components, component_priority):
        self._components = components
        self._priority = component_priority

    def get_schedule_next(self, ct: float):
        """from current time, get next event
        TODO: switch to co-routine
        """
        sort_struct = [(c.name, self.get_next_event(1 / c.sampling_frequency, c.sampling_phase, ct))
                       for c in self._components]
        sort_struct.sort(key=functools.cmp_to_key(lambda x, y: self.cmp_sort_priority(x, y, self._priority)))
        return sort_struct[0]

    def get_schedule_tspan(self, tspan):
        """over a given timespan tspan, determine which components will be active
        :param tspan: (t0, tf) tuple of times to schedule over
        :return list of tuples (component name, times) in time sorted order
        """
        # assert that times are valid
        assert tspan[0] < tspan[1], f"timespan '{tspan}' is not larger at index 1"
        # collect times for each component
        times = [(self.get_uniform_events(1 / c.sampling_frequency, c.sampling_phase, tspan)) for c in self._components]
        # create parallel arrays between component names and times
        names, times = list(zip(*[((c.name,) * len(times[idx]), times[idx]) for idx, c in enumerate(self._components)]))
        sort_struct = list(zip(*[np.concatenate(names), np.concatenate(times)]))
        # sort the parallel array structure and return it
        sort_struct.sort(key=functools.cmp_to_key(lambda x, y: self.cmp_sort_priority(x, y, self._priority)))
        return sort_struct
