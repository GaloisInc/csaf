"""
CSAF Solver
"""
from __future__ import annotations

import csaf.core.base as cbase
import csaf.core.scheduler as sched

import scipy.integrate  # type: ignore

import typing

if typing.TYPE_CHECKING:
    # cyclic imports issue
    from csaf.core.component import Component

__all__ = ['LSODASolver', 'DiscreteSolver', 'SystemSolver']


class SystemSolver(cbase.CsafBase):
    """
    CSAF System Solver Base Class
    """

    def __init__(self, component: Component):
        self.component = component

    def solve(self,
              inputs: typing.Sequence,
              states: typing.Sequence,
              teval: typing.Sequence) -> typing.Sequence:
        raise NotImplementedError

    def __call__(self,
                 inputs: typing.Sequence,
                 states: typing.Sequence,
                 teval: typing.Sequence) -> typing.Sequence:
        return self.solve(inputs, states, teval)


class LSODASolver(SystemSolver):
    """
    Wrap LSODA into a CSAF Solver (via scipy.integrate.odeint)
    """

    def solve(self,
              inputs: typing.Sequence,
              states: typing.Sequence,
              teval: typing.Sequence) -> typing.Sequence:
        def _state_diff_fcn(y: typing.Sequence, t: float) -> typing.Sequence:
            return self.component.flows['states'](self.component, t, y, inputs)

        states = scipy.integrate.odeint(_state_diff_fcn, states,
                                        teval)
        return list(states)


class DiscreteSolver(SystemSolver):
    """
    Discrete System Solver

    TODO: Check this
    """

    def __init__(self, component: 'Component'):
        super().__init__(component)
        self._scheduler = sched.Scheduler({"comp": self.component}, ["comp"])

    def solve(self,
              inputs: typing.Sequence,
              states: typing.Sequence,
              teval: typing.Sequence) -> typing.Sequence:
        assert len(teval) == 2
        tspan = min(teval), max(teval)
        tnext = self._scheduler.get_uniform_events(1.0 / self.component.sampling_frequency,
                                                   self.component.sampling_phase, tspan)
        if len(tnext) > 0:
            ns = self.component.flows['states'](self.component, tnext[0], states, inputs)
            return [states, ns]
        else:
            return [states, states]
