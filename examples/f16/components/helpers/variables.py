import enum
import numpy as np
#from util import AutoNumber


def get_var_ref(trace, var_str):
    vs = var_str
    if vs in states:        idx, name = State[vs],     'states'
    elif vs in ctrl_inputs: idx, name = Ctrlinput[vs], 'u'
    elif vs in outputs:     idx, name = Output[vs],    'outputs'
    else: raise RuntimeError(f'invalid var_str: {var_str}')
    return name, idx

def get_var_trace(system_trace, var_str):
    '''
    example usage: get_var_trace(trajs[0], 'vt')
    '''
    name, vidx = get_var_ref(system_trace, var_str)
    return np.vstack(system_trace[name])[:, vidx]

def get_var_at_t(system_trace, var_str, t):
    '''
    Get the value of a var at a given time.
    Not imeplemented yet as csaf.trace does not support it
    '''
    raise NotImplementedError

def get_var_at_idx(system_trace, var_str, idx):
    '''
    example usage: get_var_trace(trajs[0], 'vt', vidx=0)
    '''
    name, vidx = get_var_ref(system_trace, var_str)
    return np.vstack(system_trace[name])[:, vidx][idx]

#TODO: Change h to more descriptive alt
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

#TODO: Add description

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

#TODO: plant outputs
outputs = ['Nz', 'Ny', 'az', 'ay']
Output = enum.IntEnum('Output', outputs, start=0)

#TODO: Whats the best way to have named references for list? Named tuple vs enum/list vs numpy struct?
def state_vector(default=None, **states):
    x = []
    Number = (int, float, np.number)

    if default is not None: assert isinstance(default, Number)
    x = [states.get(s.name, default) for s in State]
    if any(i is None for i in x):
            raise ValueError(f'at least one value in the state vector is not a number: {x}')
    return np.array(x)
#     for k, d in x.items():
#         l[getattr(State, k)] = d
#     return l

#          x[0] = air speed, VT    (ft/sec)
#          x[1] = angle of attack, alpha  (rad)
#          x[2] = angle of sideslip, beta (rad)
#          x[3] = roll angle, phi  (rad)
#          x[4] = pitch angle, theta  (rad)
#          x[5] = yaw angle, psi  (rad)
#          x[6] = roll rate, P  (rad/sec)
#          x[7] = pitch rate, Q  (rad/sec)
#          x[8] = yaw rate, R  (rad/sec)
#          x[9] = northward horizontal displacement, pn  (feet)
#          x[10] = eastward horizontal displacement, pe  (feet)
#          x[11] = altitude, h  (feet)
#          x[12] = engine thrust dynamics lag state, power

#          u[0] = throttle command  0.0 < u(1) < 1.0
#          u[1] = elevator command in degrees
#          u[2] = aileron command in degrees
#          u[3] = rudder command in degrees

def x0deg2rad(x0_deg):
    angles_idx = set((1, 2, 3, 4, 5, 6, 7, 8))
    xp0, xc0 = x0_deg
    xp0_rad = [np.deg2rad(xi) if i in angles_idx else xi for (i, xi) in enumerate(xp0)]
    return xp0_rad, xc0
