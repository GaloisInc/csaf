"""
CSAF F-16 Model

taken from https://github.com/stanleybak/AeroBenchVVPython
"""

import f16lib.models.helpers.autopilot_helper as ah


def model_output(model, time_t, state_controller, input_f16):
    """airspeed controller output"""
    vt = input_f16[0]
    vt_des = model.xequil[0]

    # basic speed control
    throttle = ah.p_cntrl(kp=0.25, e=(vt_des - vt))
    Nz, ps, Ny_r = 0.0, 0.0, 0.0
    return Nz, ps, Ny_r, throttle
