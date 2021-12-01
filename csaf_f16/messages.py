from typing import NamedTuple


class EmptyMessage(NamedTuple):
    pass


class F16PlantStateMessage(NamedTuple):
    vt: float
    alpha: float
    beta: float
    phi: float
    theta: float
    psi: float
    p: float
    q: float
    r: float
    pn: float
    pe: float
    h: float
    pow: float


class F16PlantOutputMessage(NamedTuple):
    Nz: float
    Ny: float
    az: float
    ay: float


class F16ControllerOutputMessage(NamedTuple):
    delta_e: float
    delta_a: float
    delta_r: float
    throttle: float


class F16LlcStateMessage(NamedTuple):
    int1: float
    int2: float
    int3: float


class F16AutopilotOutputMessage(NamedTuple):
    fda: str


class F16MonitorOutputMessage(NamedTuple):
    selection: str


class PredictorOutputMessage(NamedTuple):
    will_collide: bool
