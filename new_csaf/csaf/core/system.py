"""
CSAF System
"""
from csaf.core.component import Component
from csaf.core.scheduler import Scheduler
from csaf.core.trace import TimeTrace
import csaf.core.base as cbase

import typing
import tqdm  # type: ignore

__all__ = ['ComponentComposition', 'System']


class ComponentComposition(cbase.CsafBase):
    """ create compositions of components that can also be simulated

    TODO: share the same simulation interface as a component
    """
    components: typing.Dict[str, typing.Type[Component]]

    connections: typing.Dict[typing.Tuple[str, str], typing.Tuple[str, str]]

    priority: typing.Optional[typing.Sequence[str]] = None

    def __init__(self):
        # create instances of all components mentioned in the composition description
        self._components: typing.Dict[str, Component] = {k: v() for k, v in self.components.items()}

        # signal buffer is where signals are stored during simulation
        self._signals_buffer: typing.Dict[typing.Tuple[str, str], typing.Sequence] = {}
        self._update_times: typing.Dict[str, float] = {}
        self._iv_changes = []
        self._param_changes = []

        self._initialized = False

    def initialize_buffer(self) -> None:
        """
        initialize the signals for a simulation
        """
        # for now, initialized the components with the smallest number of inputs
        # TODO: FIXME: this is a heuristic! The user should be made aware of this
        components_least_input = sorted([(k, ci) for k, ci in self._components.items()],
                                        key=lambda args: len(args[1].inputs))
        for namei, componenti in components_least_input:
            r = componenti.solve_default()
            self._signals_buffer.update({(namei, k): v for k, v in componenti.initial_values.items()})
            self._signals_buffer.update({(namei, k): v for k, v in r.items()})
            self._update_times[namei] = 0.0

    def build_input_vec(self, component_name: str) -> typing.Sequence:
        """ extract the input vector described in the component

        a component describes its inputs as a sequence of messages, which can
        translate into a vector.

        TODO: is this too slow?
        TODO: should some of this be in the csaf.core.Component class?

        :param component_name: name of component to build an input vector for
        :return: input vector
        """
        # TODO: FIXME
        component = self._components[component_name]
        conns = [(component_name, inname) for inname in component.inputs_names]
        inputs = []
        for conn in conns:
            lu = self.connections[conn]
            n = list(self._signals_buffer[lu])
            inputs += n
        return list(inputs)

    def update_component(self, component_name: str, ctime: float) -> typing.Dict[str, typing.Sequence]:
        """
        update the signal buffer affected by the component with component name

        TODO: is this too slow?
        """
        assert component_name in self._components
        component = self._components[component_name]

        # build out the inputs to solve the component
        states = self._signals_buffer[(component_name, "states")]
        inputs = self.build_input_vec(component_name)

        # solve and update keys for signal buffer
        ret = component.solve_sys_tspan(inputs, states, [self._update_times[component_name], ctime])
        r = {(component_name, k): (list(v[-1]) if len(v) > 0 else list(v)) for k, v in ret.items()}

        # update the context
        self._signals_buffer.update(r)
        self._update_times[component_name] = ctime

        return {k: (list(v[-1]) if len(v) > 0 else list(v)) for k, v in ret.items()}

    def simulate_tspan(self, tspan,
                       show_status: bool = False,
                       terminating_conditions: typing.Optional[typing.Callable] = None,
                       terminating_conditions_all: typing.Optional[typing.Callable] = None,
                       return_passed: bool = False) -> typing.Union[typing.Dict[str, TimeTrace],
                                                                    typing.Tuple[typing.Dict[str, TimeTrace], bool]]:
        """ simulate the composed system over a given time span
        :param tspan: time span (tmin, tmax)
        :param show_status: show progress bar in stdout
        :param terminating_conditions: callable that accepts the current system state, when returning true,
                                        will stop the simulation
        :param terminating_conditions_all: callable that accepts the current system state AND all past system states,
                                            when returning true, will stop the simulation
        :param return_passed: whether to return a boolean value that if false means that the simulation met the
                                terminating conditions
        """
        self.reset()
        self.initialize_buffer()

        sched = Scheduler(self._components, list(self._components.keys()) if self.priority is None else self.priority)
        evts = sched.get_schedule_tspan(tspan)
        evts_it = evts if not show_status else tqdm.tqdm(evts)

        # get time trace fields
        dnames = self._components.keys()
        dtraces = {}
        for dname in dnames:
            fields = ['times'] + list(self._components[dname].flow_names)
            dtraces[dname] = TimeTrace(fields)

        try:
            for cname, ctime in evts_it:
                out = self.update_component(cname, ctime)
                out["times"] = ctime
                dtraces[cname].append(**out)

                if terminating_conditions is not None and terminating_conditions(cname, out):
                    return dtraces if not return_passed else (dtraces, False)

                if terminating_conditions_all is not None and terminating_conditions_all(dtraces):
                    return dtraces if not return_passed else (dtraces, False)
        except Exception as exc:
            # FIXME: TODO
            raise exc
            pass

        return dtraces if not return_passed else (dtraces, True)

    def validate_tspan(self, tspan: typing.Tuple[float, float],
                       show_status: bool = False,
                       terminating_conditions: typing.Optional[typing.Callable] = None,
                       terminating_conditions_all: typing.Optional[typing.Callable] = None) -> bool:
        """ determine whether a simulation will complete over a time span
        :param tspan: time span (tmin, tmax)
        :param show_status: show progress bar in stdout
        :param terminating_conditions: callable that accepts the current system state, when returning true,
                                        will stop the simulation
        :param terminating_conditions_all: callable that accepts the current system state AND all past system states,
                                            when returning true, will stop the simulation
        """
        ret = self.simulate_tspan(tspan,
                                  show_status=show_status,
                                  terminating_conditions=terminating_conditions,
                                  terminating_conditions_all=terminating_conditions_all,
                                  return_passed=True)
        assert isinstance(ret, tuple)
        assert isinstance(ret[1], bool)
        return ret[1]

    def _set_component_iv(self, component_name: str, iv_name: str, state: typing.Sequence):
        assert component_name in self._components
        component = self.component_instances[component_name]
        assert iv_name in component.initial_values
        component.initial_values[iv_name] = state
        for cin, cout in self.connections.items():
            if cout == (component_name, iv_name):
                self._components[cin[0]].initial_values[cin[1]] = state

    def set_component_iv(self, component_name: str, iv_name: str, state: typing.Sequence):
        self._set_component_iv(component_name, iv_name, state)
        self._iv_changes.append((component_name, iv_name, state))

    def _set_component_param(self, component_name: str, param_name: str, param: typing.Sequence):
        assert component_name in self._components
        component = self.component_instances[component_name]
        assert param_name in component.parameters, f"component '{component_name}' has no parameter '{param_name}'"
        component.parameters[param_name] = param
        if component.initialize is not None:
            component.initialize()

    def set_component_param(self, component_name: str, param_name: str, param: typing.Sequence):
        self._set_component_param(component_name, param_name, param)
        self._param_changes.append((component_name, param_name, param))

    def reset(self):
        # create instances of all components mentioned in the composition description
        self._components: typing.Dict[str, Component] = {k: v() for k, v in self.components.items()}
        for change in self._iv_changes:
            self._set_component_iv(*change)
        for param in self._param_changes:
            self._set_component_param(*param)

    def set_state(self, component_name: str, state: typing.Sequence):
        self.set_component_iv(component_name, "states", state)

    @property
    def component_instances(self) -> typing.Dict[str, Component]:
        return self._components

    def plot_config(self, fname=None, **kwargs):
        """visualize the configuration file"""
        import pydot  # type: ignore
        import pathlib
        import os

        def join_if_not_abs(*args, project_dir=None):
            """if last argument is an absolute path, don't join the path arguments together"""
            if os.path.isabs(args[-1]):
                return args[-1]
            else:
                if project_dir:
                    pathname = os.path.join(*args[:-1], project_dir, args[-1])
                else:
                    pathname = os.path.join(*args)
                return pathname

        # TODO: FIXME: this is a mess
        fname = fname if fname is not None else self.__class__.__name__ + "-config.pdf"

        graph = pydot.Dot(graph_type='digraph', prog='LR', color="white")
        graph.set_node_defaults(shape='box',
                                fontsize='8')

        verts = {}
        for nname, ncomp in self.components.items():
            devname = ncomp.name  # ninfo['config']["system_name"]
            if ncomp.is_discrete:
                verts[nname] = pydot.Node(f"'{nname}'\n{devname}\n{len(ncomp.default_parameters)} "
                                          f"Parameter(s)\n{ncomp.sampling_frequency} Hz", style="solid")
            else:
                verts[nname] = pydot.Node(f"'{nname}'\n{devname}\n{len(ncomp.default_parameters)} "
                                          f"Parameter(s)", style="bold")
            graph.add_node(verts[nname])

        pairs = []
        for inc, outc in self.connections.items():
            topics = []
            widths = []
            if (inc[0], outc[0]) in pairs:
                continue
            for i, o in self.connections.items():
                if (i[0], o[0]) == (inc[0], outc[0]) and o[-1] != outc[-1]:
                    topics.append(o[-1])
                    comp = self._components[o[0]]
                    widths.append(str(len({**dict(comp.outputs), "states": comp.states}[o[1]].__annotations__)))
            comp = self._components[outc[0]]
            widths.append(str(len({**dict(comp.outputs), "states": comp.states}[outc[1]].__annotations__)))
            topics.append(outc[-1])
            graph.add_edge(pydot.Edge(verts[outc[0]], verts[inc[0]], fontsize=8,
                                      label=f"{', '.join(topics)}\n({', '.join(widths)})", arrowsize=0.5))
            pairs.append((inc[0], outc[0]))

        graph_path = pathlib.Path(join_if_not_abs("./", fname))
        extension = graph_path.suffix[1:]
        graph.write(graph_path, format=extension, **kwargs)

    def validate(self) -> None:
        try:
            for component in self._components.values():
                component.validate()
            for (inc, inflow), (outc, outflow) in self.connections.items():
                cin, cout = self._components[inc], self._components[outc]
                try:
                    indict = dict(list(cin.inputs) + [("states", cin.states)])
                except Exception as exc:
                    raise ValueError(f"[component {inc}<{cin.__class__.__name__}>] invalid inputs {exc}")
                try:
                    outdict = dict(list(cout.outputs) + [("states", cout.states)])
                except Exception as exc:
                    raise ValueError(f"[component {outc}<{cout.__class__.__name__}>] invalid outputs {exc}")
                assert outflow in outdict, f"[edge {inc}<-{outc}][component {outc}<{cout.__class__.__name__}>] flow " \
                                           f"referenced '{outflow}' is not in component!"
                assert inflow in indict, f"[edge {inc}<-{outc}][component {inc}<{cin.__class__.__name__}>] flow " \
                                         f"referenced '{inflow}' is not in component!"
                insig = outdict[outflow].__annotations__.items()
                outsig = indict[inflow].__annotations__.items()
                assert len(insig) == len(outsig)
                for (inid, intype), (outid, outtype) in zip(insig, outsig):
                    assert issubclass(intype, outtype), f"subclass errors ({inid, intype})<-({outid, outtype})"
        except Exception as exc:
            raise exc.__class__(f"|System <{self.__class__.__name__}>|{exc}")


class System(ComponentComposition):
    """ alternative name for ComponentComposition for compatibility
    """
    pass
