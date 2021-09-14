"""
CSAF Component
"""
import csaf.base as cbase
from csaf.solver import LSODASolver, DiscreteSolver, SystemSolver

import abc
import typing
import enum

# custom types
RepresentationIdentifier = typing.Annotated[int, 'RepresentationIdentifier']
SolverIdentifier = typing.Annotated[int, 'SolverIdentifier']
FlowIdentifier = typing.Annotated[str, 'FlowIdentifier']
ParameterIdentifier = typing.Annotated[str, 'ParameterIdentifier']
Parameters = typing.Dict[str, typing.Any]


class SystemRepresentationEnum(enum.IntEnum):
    BLACK_BOX = 0


class Component(cbase.CsafBase, metaclass=abc.ABCMeta):
    """
    CSAF System Component
    """

    # component name
    name: str

    # system representation (e.g. black box)
    system_representation: SystemRepresentationEnum

    # how system can be solved
    system_solver: typing.Type['SystemSolver']

    # frequency to sample system
    sampling_frequency: float

    # time skew for the component
    sampling_phase: float

    # whether system operated in discrete time
    is_discrete: bool

    # time invariant system parameters
    parameters: Parameters

    # input messages
    inputs: typing.Sequence[typing.Tuple[FlowIdentifier, typing.Type[typing.Tuple]]]

    # output messages
    outputs: typing.Sequence[typing.Tuple[FlowIdentifier, typing.Type[typing.Tuple]]]

    # internal message used to evolve state
    states: typing.Type[typing.Tuple]

    # initial values to use when component isn't connected
    default_initial_values: typing.Dict[FlowIdentifier, typing.Sequence]

    # function to map input + state -> state evolution + outputs
    flows: typing.Dict[FlowIdentifier, typing.Callable]

    # initializer
    initialize: typing.Optional[typing.Callable] = None

    def __init__(self):
        if set(self.flows) != self.flow_names:
            raise RuntimeError(f"{self.__class__.__name__} flows are not correctly specified "
                               f"(must have names {self.flow_names})")

        if self.initialize:
            self.initialize()

        self.reset()
        self._solver = self.system_solver(self)

    def reset(self):
        """reset component members to their defaults"""
        self.initial_values = self.default_initial_values.copy()

    def solve_state(self,
                    inputs: typing.Sequence,
                    states: typing.Sequence,
                    teval: typing.Sequence) -> typing.Sequence:
        """
        solve the system state for each time specified in trange
        """
        return self._solver(inputs, states, teval)

    def solve_sys_tspan(self,
                        inputs: typing.Sequence,
                        states: typing.Sequence,
                        teval: typing.Sequence) -> typing.Dict[FlowIdentifier, typing.Sequence]:
        """
        solve all flows for each time in trange
        """
        sols = {}
        if len(self.states.__annotations__) > 0:
            states = self.solve_state(inputs, states, teval)
            sols['states'] = states
        else:
            states = [[], ] * len(teval)
        for k, flow_func in self.flows.items():
            if k != "states":
                ret = [flow_func(self, ti, si, inputs) for ti, si in zip(teval, states)]
                sols[k] = ret
        return sols

    def solve_default(self, epsilon: float = 1E-5) -> typing.Dict[FlowIdentifier, typing.Sequence]:
        states = self.initial_values['states']
        inputs: typing.List = []
        for ivname in self.initial_values:
            if ivname != "states":
                inputs += list(self.initial_values[ivname])
        ret = self.solve_sys_tspan(inputs, states, [0, epsilon])
        return {k: v[-1] if len(v) > 0 else [] for k, v in ret.items()}

    @property
    def flow_names(self) -> typing.Set[str]:
        return set(v for v, _ in self.flows.items())

    @property
    def state_names(self) -> typing.Set[str]:
        return {"states"}

    @property
    def outputs_names(self) -> typing.Tuple[str, ...]:
        return tuple([v for v, _ in self.outputs])

    @property
    def inputs_names(self) -> typing.Tuple[str, ...]:
        return tuple([v for v, _ in self.inputs])

    @property
    def is_continuous(self) -> bool:
        return not self.is_discrete

    @property
    def input_signature(self) -> typing.Tuple:
        return tuple([item[1] for sublist in self.inputs for item in sublist.__annotations__.items()])

    def __getattr__(self, item):
        """access the component parameters in the object dir"""
        if item == "parameters":
            raise RuntimeError
        if item in self.parameters:
            return self.parameters[item]
        else:
            raise RuntimeError(f"component {self.name} has no parameter {item}")

    def validate(self) -> None:
        def validate_signature(name, signature, value):
            assert len(signature.__annotations__) == len(
                value), f"value with name {name} doesn't match length of signature {signature}"

        try:
            assert self.sampling_frequency > 0.0, f"sampling frequency must be greater than 0.0"
            assert set(self.initial_values.keys()) == ({k for k, _ in self.inputs} | {
                "states"}), f"initial values must reference all inputs and states"
            assert set(self.flows.keys()) == ({k for k, _ in self.outputs} | ({"states"} if len(
                self.states.__annotations__) > 0 else set())), f"flows must references all states and outputs"

            # values must match the signatures
            # TODO: make this a property?
            sig_dict = dict(list(self.inputs) + [('states', self.states)])
            for vname, vval in self.initial_values.items():
                validate_signature(vname, sig_dict[vname], vval)
        except Exception as exc:
            raise exc.__class__(f"|Component <{self.__class__.__name__}>| {exc}")


class ContinuousComponent(Component):
    system_representation = SystemRepresentationEnum.BLACK_BOX
    system_solver = LSODASolver
    is_discrete = False
    sampling_phase = 1E-8


class DiscreteComponent(Component):
    system_representation = SystemRepresentationEnum.BLACK_BOX
    system_solver = DiscreteSolver
    is_discrete = True
    sampling_phase = 1E-8
