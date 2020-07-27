import os
import toml
import json

import numpy as np
from csaf import message


def llcoutput(t, cstate, u, parameters):
    """ get the reference commands for the control surfaces """

    x_f16, y, u_ref = u[:13], u[13:15], u[15:]

    x_ctrl = get_x_ctrl(np.concatenate([x_f16, cstate]), parameters)

    # Initialize control vectors
    u_deg = np.zeros((4,))  # throt, ele, ail, rud

    # Calculate control using LQR gains
    u_deg[1:4] = np.dot(-np.array(parameters['K_lqr']), x_ctrl)  # Full Control

    # Set throttle as directed from output of getOuterLoopCtrl(...)
    u_deg[0] = u_ref[3]

    u_deg = clip_u(u_deg, parameters)

    return u_deg


def llcdf(t, cstate, u, parameters):
    """ get the derivatives of the integrators in the low-level controller """
    x_f16, y, u_ref = u[:13], u[13:15], u[15:]
    Nz, Ny = y
    x_ctrl = get_x_ctrl(np.concatenate([x_f16, cstate]), parameters)

    # Nonlinear (Actual): ps = p * cos(alpha) + r * sin(alpha)
    ps = x_ctrl[4] * np.cos(x_ctrl[0]) + x_ctrl[5] * np.sin(x_ctrl[0])

    # Calculate (side force + yaw rate) term
    Ny_r = Ny + x_ctrl[5]

    return np.array([Nz - u_ref[0], ps - u_ref[1], Ny_r - u_ref[2]])


def get_x_ctrl(f16_state, parameters):
    """ transform f16_state to control input (slice array and apply setpoint)
    :param f16_state: f16 plant state + controller state
    :param parameters: parameters containing equilibrium state (xequil)
    :return: x_ctrl vector
    """
    # Calculate perturbation from trim state
    x_delta = f16_state.copy()
    x_delta[:len(parameters['xequil'])] -= parameters['xequil']

    ## Implement LQR Feedback Control
    # Reorder states to match controller:
    # [alpha, q, int_e_Nz, beta, p, r, int_e_ps, int_e_Ny_r]
    return np.array([x_delta[i] for i in [1, 7, 13, 2, 6, 8, 14, 15]], dtype=float)


def clip_u(u_deg, parameters):
    """ helper to clip controller output within defined control limits
    :param u_deg: controller output
    :param parameters: containing equilibrium state (uequil) and control limits (ctrlLimits)
    :return: saturated control output
    """
    uequil = parameters["uequil"]

    # Add in equilibrium control
    u_deg[0:4] += uequil

    # Limit throttle from 0 to 1
    u_deg[0] = max(min(u_deg[0], parameters["throttle_max"]), parameters["throttle_max"])

    # Limit elevator from -25 to 25 deg
    u_deg[1] = max(min(u_deg[1], parameters["elevator_min"]), parameters["elevator_max"])

    # Limit aileron from -21.5 to 21.5 deg
    u_deg[2] = max(min(u_deg[2], parameters["aileron_min"]), parameters["aileron_max"])

    # Limit rudder from -30 to 30 deg
    u_deg[3] = max(min(u_deg[3], parameters["rudder_min"]), parameters["rudder_max"])
    return u_deg


def main(time=0.0, state=[.1]*3, input=[0]*19, update=False, output=False):
    this_path = os.path.dirname(os.path.realpath(__file__))
    info_file = os.path.join(this_path, "f16llc.toml")
    with open(info_file, 'r') as ifp:
        info = toml.load(ifp)

    parameters = info["parameters"]

    n_states = 3
    n_outputs = 4

    xd = llcdf(time, state, input, parameters)
    xout = llcoutput(time, state, input, parameters)
    assert len(xd) == n_states
    assert len(xout) == n_outputs

    if update:
        return list(xd)
    elif output:
        return list(xout)
    else:
        return

if __name__ == "__main__":
    import fire
    fire.Fire(main)
