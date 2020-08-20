import os
import toml

import numpy as np

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import autopilot_helper as ah


parameters = {}


def main(time=0.0, state=(.1,)*3, input=(0,)*19, update=False, output=False):
    global parameters
    if len(parameters.keys()) == 0:
        this_path = os.path.dirname(os.path.realpath(__file__))
        info_file = os.path.join(this_path, "f16llc.toml")
        with open(info_file, 'r') as ifp:
            info = toml.load(ifp)
        parameters = info["parameters"]
        assert "computer_fcn" not in parameters and "trim_points" not in parameters

    compute_fcn, *trim_points = getattr(ah, parameters["lqr_name"])()

    parameters["compute_fcn"] = compute_fcn
    parameters["trim_points"] = trim_points

    xd = llcdf(time, state, input)
    xout = llcoutput(time, state, input)

    if update:
        return list(xd)
    elif output:
        return list(xout)
    else:
        return


def llcoutput(t, cstate, u):
    """ get the reference commands for the control surfaces """
    global parameters
    compute = parameters["compute_fcn"]
    xequil, uequil = parameters["trim_points"]
    assert len(u) == 21
    #TODO: hard coded indices!
    x_f16, y, u_ref = u[:13], u[13:17], u[17:]
    x_ctrl = get_x_ctrl(np.concatenate([x_f16, cstate]))

    # Initialize control vectors
    u_deg = np.zeros((4,))  # throt, ele, ail, rud
    u_deg[1:4] = compute(x_ctrl)

    # Set throttle as directed from output of getOuterLoopCtrl(...)
    u_deg[0] = u_ref[3]

    # Add in equilibrium control
    u_deg[0:4] += uequil
    u_deg = clip_u(u_deg)

    return u_deg


def llcdf(t, cstate, u):
    """ get the derivatives of the integrators in the low-level controller """
    x_f16, y, u_ref = u[:13], u[13:17], u[17:]
    Nz, Ny, az, ay = y
    x_ctrl = get_x_ctrl(np.concatenate([x_f16, cstate]))

    # Nonlinear (Actual): ps = p * cos(alpha) + r * sin(alpha)
    ps = x_ctrl[4] * np.cos(x_ctrl[0]) + x_ctrl[5] * np.sin(x_ctrl[0])

    # Calculate (side force + yaw rate) term
    Ny_r = Ny + x_ctrl[5]

    return [Nz - u_ref[0], ps - u_ref[1], Ny_r - u_ref[2]]


def get_x_ctrl(f16_state):
    """ transform f16_state to control input (slice array and apply setpoint)
    :param f16_state: f16 plant state + controller state
    :param parameters: parameters containing equilibrium state (xequil)
    :return: x_ctrl vector
    """
    global parameters
    xequil, _ = parameters["trim_points"]

    # Calculate perturbation from trim state
    x_delta = f16_state.copy()
    x_delta[:len(xequil)] -= xequil

    ## Implement Feedback Control
    # Reorder states to match controller:
    # [alpha, q, int_e_Nz, beta, p, r, int_e_ps, int_e_Ny_r]
    return np.array([x_delta[i] for i in [1, 7, 13, 2, 6, 8, 14, 15]])


def clip_u(u_deg):
    """ helper to clip controller output within defined control limits
    :param u_deg: controller output
    :param parameters: containing equilibrium state (uequil) and control limits (
    :return: saturated control output
    """
    global parameters
    ThrottleMin, ThrottleMax = parameters["throttle_min"], parameters["throttle_max"]
    ElevatorMinDeg, ElevatorMaxDeg = parameters["elevator_min"], parameters["elevator_max"]
    AileronMinDeg, AileronMaxDeg = parameters["aileron_min"], parameters["aileron_max"]
    RudderMinDeg, RudderMaxDeg = parameters["rudder_min"], parameters["rudder_max"]

    # Limit throttle from 0 to 1
    u_deg[0] = max(min(u_deg[0], ThrottleMax), ThrottleMin)

    # Limit elevator from -25 to 25 deg
    u_deg[1] = max(min(u_deg[1], ElevatorMaxDeg), ElevatorMinDeg)

    # Limit aileron from -21.5 to 21.5 deg
    u_deg[2] = max(min(u_deg[2], AileronMaxDeg), AileronMinDeg)

    # Limit rudder from -30 to 30 deg
    u_deg[3] = max(min(u_deg[3], RudderMaxDeg), RudderMinDeg)
    return u_deg


if __name__ == "__main__":
    import fire
    fire.Fire(main)
