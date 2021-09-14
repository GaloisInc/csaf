import enum
import numpy as np
import typing

states = ['vt', 'alpha', 'beta', 'phi', 'theta', 'psi', 'p', 'q', 'r', 'pn', 'pe', 'h', 'power']


# State: typing.Any = enum.IntEnum('State', states, start=0)

class State(enum.IntEnum):
    vt = 0
    alpha = 1
    beta = 2
    phi = 3
    theta = 4
    psi = 5
    p = 6
    q = 7
    r = 8
    pn = 9
    pe = 10
    h = 11
    power = 12


State.des = lambda s: state_description_short[s]  # type: ignore
State.desf = lambda s: state_description[s]  # type: ignore

state_description_short = {
    State.vt: 'airspeed(ft/s)',
    State.alpha: 'attack(rad)',
    State.beta: 'sideslip(rad)',
    State.phi: 'roll(rad)',
    State.theta: 'pitch(rad)',
    State.psi: 'yaw(rad)',
    State.p: 'roll r.(rad/s)',
    State.q: 'pitch r.(rad/s)',
    State.r: 'yaw r.(rad/s)',
    State.pn: 'north(ft)',
    State.pe: 'east(ft)',
    State.h: 'altitude(ft)',
    State.power: 'power'
}

state_description = {
    State.vt: 'air speed (ft/s)',
    State.alpha: 'angle of attack (rad)',
    State.beta: 'angle of sideslip (rad)',
    State.phi: 'roll angle (rad)',
    State.theta: 'pitch angle (rad)',
    State.psi: 'yaw angle (rad)',
    State.p: 'roll rate (rad/s)',
    State.q: 'pitch rate (rad/s)',
    State.r: 'yaw rate (rad/s)',
    State.pn: 'northward horizontal displacement (ft)',
    State.pe: 'eastward horizontal displacement (ft)',
    State.h: 'altitude (ft)',
    State.power: 'engine thrust dynamics lag state'
}

ctrl_inputs = ['throttle', 'elevator', 'aileron', 'rudder', 'Nz_ref', 'ps_ref', 'Ny_r_ref', 'throttle_ref']
Ctrlinput = enum.IntEnum('Ctrlinput', ctrl_inputs, start=0)  # type: ignore

ctrl_description = {
    Ctrlinput.throttle: 'throttle',  # type: ignore
    Ctrlinput.elevator: 'elevator',  # type: ignore
    Ctrlinput.aileron: 'aileron',  # type: ignore
    Ctrlinput.rudder: 'rudder',  # type: ignore
    Ctrlinput.Nz_ref: 'Acceleration in Z axis (reference set by autopilot)',  # type: ignore
    Ctrlinput.ps_ref: 'stability roll (reference set by autopilot)',  # type: ignore
    Ctrlinput.Ny_r_ref: 'Ny+r (reference set by autopilot)',  # type: ignore
    Ctrlinput.throttle_ref: 'throttle reference from autopilot',  # type: ignore
}

pilot_inputs = ['Nz_ref', 'ps_ref', 'Ny_r_ref', 'throttle_ref']
PilotInput = enum.IntEnum('PilotInput', pilot_inputs, start=0)  # type: ignore

outputs = ['Nz', 'Ny', 'az', 'ay']
Output = enum.IntEnum('Output', outputs, start=0)  # type: ignore


def state_vector(default=None, **states):
    x = []
    Number = (int, float, np.number)

    if default is not None: assert isinstance(default, Number)
    x = [states.get(s.name, default) for s in State]
    if any(i is None for i in x):
        raise ValueError(f'at least one value in the state vector is not a number: {x}')
    return np.array(x)


def x0deg2rad(x0_deg):
    angles_idx = set((1, 2, 3, 4, 5, 6, 7, 8))
    xp0, xc0 = x0_deg
    xp0_rad = [np.deg2rad(xi) if i in angles_idx else xi for (i, xi) in enumerate(xp0)]
    return xp0_rad, xc0
