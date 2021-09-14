"""
CSAF System Environment (implement component flows externally)
"""
import csaf.base as cbase
from csaf.system import ComponentComposition
from csaf.scheduler import Scheduler
from csaf.trace import TimeTrace
import typing

__all__ = ['SystemEnv']


class SystemEnv(cbase.CsafBase):
    """
    ```
    class ExampleEnv(SystemEnv):
        system = ExampleSystem
        agents = ("controllera", "controllerb")

    my_env =
    ```
    """
    system_type: typing.Type[ComponentComposition]

    agents: typing.Sequence[str]

    def __init__(self, terminating_conditions=None, terminating_conditions_all=None):
        self.system = self.system_type()
        self.termconds, self.termconds_all = terminating_conditions, terminating_conditions_all
        self._iter: typing.Optional[typing.Coroutine] = None

    def _set_coroutine(self):
        """ populate _iter member with coroutine object"""
        self._iter = self.make_system_coroutine(terminating_conditions=self.termconds,
                                                terminating_conditions_all=self.termconds_all)
        next(self._iter)

    def reset(self):
        self.system.reset()
        self._iter = self.make_system_coroutine(terminating_conditions=self.termconds,
                                                terminating_conditions_all=self.termconds_all)
        next(self._iter)

    def step(self, component_output):
        """step through the simulation generator"""
        if self._iter is None:
            self._set_coroutine()
        return self._iter.send(component_output)

    def make_system_coroutine(self,
                              tstart=0.0,
                              terminating_conditions=None,
                              terminating_conditions_all=None) -> typing.Generator:
        """make an iterator that can step through a simulation and accept input from external agents

        :param tstart: time to start (now only supports 0.0)
        :param terminating_conditions: system terminating conditions
        :param terminating_conditions_all: system terminating_all conditions
        :return:
        """
        # TODO: FIXME: use tstart
        self.system.initialize_buffer()

        sched = Scheduler(self.system._components, list(self.system._components.keys()) \
            if self.system.priority is None else self.system.priority)
        evts_it = sched.get_scheduler()

        # get time trace fields
        # NOTE: we need dtraces for the terminating_conditions_all
        dnames = self.system._components.keys()
        dtraces = {}
        for dname in dnames:
            fields = ['times'] + list(self.system._components[dname].flow_names)
            dtraces[dname] = TimeTrace(fields)

        yield None

        try:
            for cname, ctime in evts_it:
                if cname in self.agents:
                    out: typing.Dict[str, typing.Dict[str, typing.Sequence]] = yield (ctime,
                                                                                      self.system.build_input_vec(
                                                                                          cname)) # typing: ignore
                    # update the context
                    r = {(cname, k): v for k, v in out.items()}
                    self.system._signals_buffer.update(r)
                    self.system._update_times[cname] = ctime
                else:
                    out = self.system.update_component(cname, ctime) # typing: ignore
                out["times"] = ctime  # typing: ignore
                dtraces[cname].append(**out)
                if terminating_conditions is not None and terminating_conditions(cname, out):
                    return
                if terminating_conditions_all is not None and terminating_conditions_all(dtraces):
                    return
        except Exception as exc:
            # FIXME: TODO
            print(exc)
            raise exc
            pass

    def set_component_iv(self, component_name: str, iv_name: str, state: typing.Sequence):
        self.system.set_component_iv(component_name, iv_name, state)

    def set_state(self, component_name: str, state: typing.Sequence):
        self.system.set_component_iv(component_name, "states", state)

    def validate(self) -> None:
        """
        validate the env semantics
        """
        # for now...
        assert len(self.agents) == 1, "CSAF only supports 1 agent currently"
        try:
            # validate the system itself
            self.system.validate()

            # assert that agents referenced are actually defined in the system object
            assert set(self.agents).issubset(self.system_type.components.keys()), f"{self.agents} are not in " \
                                                                                  f"system {self.system_type.__class__.__name__}"
        except Exception as exc:
            raise exc.__class__(f"|SystemEnv <{self.__class__.__name__}>| {exc}")
