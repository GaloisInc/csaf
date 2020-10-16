""" Fuzzy Logic Low Level Controller Model

TODO: implement the TODOs
"""
import numpy as np
import autopilot_helper as ah
from f16llc import get_x_ctrl, clip_u, model_state_update


def model_init(model):
    """function to load resources needed by the controller

    save them in the model parameters, and access them anywhere model
    is passed
    """
    # TODO: load inference table?
    inference_table = None
    # by using the key inference, you can access with model.inference or
    # model.parameters["inference"]
    model.parameters["inference"] = inference_table


def compute_fcn(model, x_ctrl):
    """compute 3-dim control signal from 7-dim x_ctrl signal"""
    #TODO: implement this
    #show that inference (added in model_init) can be accessed
    assert model.inference is None
    return [0, 0, 0]


def model_output(model, time_t, state_controller, input_all):
    """ get the reference commands for the control surfaces """
    _, *trim_points = getattr(ah, model.lqr_name)()

    x_f16, _y, u_ref = input_all[:13], input_all[13:17], input_all[17:]
    x_ctrl = get_x_ctrl(trim_points, np.concatenate([x_f16, state_controller]))

    # Initialize control vectors
    u_deg = np.zeros((4,))  # throt, ele, ail, rud
    u_deg[1:4] = compute_fcn(model, x_ctrl)

    # Set throttle as directed from output of getOuterLoopCtrl(...)
    u_deg[0] = u_ref[3]

    # Add in equilibrium control
    u_deg[0:4] += trim_points[1]
    u_deg = clip_u(model, u_deg)

    return u_deg
