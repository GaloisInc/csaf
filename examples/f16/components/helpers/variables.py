import enum
import numpy as np

states = ['vt', 'alpha', 'beta', 'phi', 'theta', 'psi', 'p', 'q', 'r', 'pn', 'pe', 'h', 'power']
State = enum.IntEnum('State', states, start=0)
State.des = lambda s: state_description_short[s]
State.desf = lambda s: state_description[s]

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

ctrl_inputs = ['throttle', 'elevator', 'aileron','rudder', 'Nz_ref', 'ps_ref', 'Ny_r_ref', 'throttle_ref']
Ctrlinput = enum.IntEnum('Ctrlinput', ctrl_inputs, start=0)

ctrl_description = {
        Ctrlinput.throttle: 'throttle',
        Ctrlinput.elevator: 'elevator',
        Ctrlinput.aileron: 'aileron',
        Ctrlinput.rudder: 'rudder',
        Ctrlinput.Nz_ref: 'Acceleration in Z axis (reference set by autopilot)',
        Ctrlinput.ps_ref: 'stability roll (reference set by autopilot)',
        Ctrlinput.Ny_r_ref: 'Ny+r (reference set by autopilot)',
        Ctrlinput.throttle_ref: 'throttle reference from autopilot',
        }

pilot_inputs = ['Nz_ref', 'ps_ref', 'Ny_r_ref', 'throttle_ref']
PilotInput = enum.IntEnum('PilotInput', pilot_inputs, start=0)

outputs = ['Nz', 'Ny', 'az', 'ay']
Output = enum.IntEnum('Output', outputs, start=0)

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
