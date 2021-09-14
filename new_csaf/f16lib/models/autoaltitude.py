"""
CSAF F-16 Model

taken from https://github.com/stanleybak/AeroBenchVVPython
"""

import f16lib.models.helpers.autopilot_helper as ah


def model_output(model, time_t, state_x, input_f16):
    vt = input_f16[0]  # airspeed      (ft/sec)
    alpha = input_f16[1]  # AoA           (rad)
    theta = input_f16[4]  # Pitch angle   (rad)
    gamma = theta - alpha  # Path angle    (rad)
    h = input_f16[11]  # Altitude      (feet)

    # Proportional Control
    k_alt = 0.025
    h_error = model.setpoint - h
    Nz = k_alt * h_error  # Allows stacking of cmds

    # (Psuedo) Derivative control using path angle
    k_gamma = 25
    # k_gamma = self.p_gain
    Nz = Nz - k_gamma * gamma

    # try to maintain a fixed airspeed near trim point
    K_vt = 0.25
    airspeed_setpoint = 540
    vt_des = model.xequil[0]
    throttle = ah.p_cntrl(kp=K_vt, e=(vt_des - vt))

    return Nz, 0, 0, throttle
