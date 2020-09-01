import numpy as np
import autopilot_helper as ah


def model_output(model, time_t, state_controller, input_all):
    """ get the reference commands for the control surfaces """
    compute_fcn, *trim_points = getattr(ah, model.lqr_name)()

    assert len(input_all) == 21
    #TODO: hard coded indices!
    x_f16, y, u_ref = input_all[:13], input_all[13:17], input_all[17:]
    x_ctrl = get_x_ctrl(trim_points, np.concatenate([x_f16, state_controller]))

    # Initialize control vectors
    u_deg = np.zeros((4,))  # throt, ele, ail, rud
    u_deg[1:4] = compute_fcn(x_ctrl)

    # Set throttle as directed from output of getOuterLoopCtrl(...)
    u_deg[0] = u_ref[3]

    # Add in equilibrium control
    u_deg[0:4] += trim_points[1]
    u_deg = clip_u(model, u_deg)

    return u_deg


def model_state_update(model, time_t, state_controller, input_all):
    """ get the derivatives of the integrators in the low-level controller """
    compute_fcn, *trim_points = getattr(ah, model.lqr_name)()
    x_f16, y, u_ref = input_all[:13], input_all[13:17], input_all[17:]
    Nz, Ny, az, ay = y
    x_ctrl = get_x_ctrl(trim_points, np.concatenate([x_f16, state_controller]))

    # Nonlinear (Actual): ps = p * cos(alpha) + r * sin(alpha)
    ps = x_ctrl[4] * np.cos(x_ctrl[0]) + x_ctrl[5] * np.sin(x_ctrl[0])

    # Calculate (side force + yaw rate) term
    Ny_r = Ny + x_ctrl[5]

    return [Nz - u_ref[0], ps - u_ref[1], Ny_r - u_ref[2]]


def get_x_ctrl(trim_points, state_f16):
    """ transform f16_state to control input (slice array and apply setpoint)
    :param state_f16: f16 plant state + controller state
    :param parameters: parameters containing equilibrium state (xequil)
    :return: x_ctrl vector
    """
    xequil, _ = trim_points

    # Calculate perturbation from trim state
    x_delta = state_f16.copy()
    x_delta[:len(xequil)] -= xequil

    ## Implement Feedback Control
    # Reorder states to match controller:
    # [alpha, q, int_e_Nz, beta, p, r, int_e_ps, int_e_Ny_r]
    return np.array([x_delta[i] for i in [1, 7, 13, 2, 6, 8, 14, 15]])


def clip_u(model, u_deg):
    """ helper to clip controller output within defined control limits
    :param u_deg: controller output
    :param parameters: containing equilibrium state (uequil) and control limits (
    :return: saturated control output
    """
    parameters = model.parameters
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

