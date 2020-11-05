""" Fuzzy Logic Low Level Controller Model

TODO: implement the TODOs
"""
import numpy as np
from fileops import prepend_curr_path
from helpers import lqr
from f16llc import get_x_ctrl, clip_u, model_state_update
from f16_fuzzy_mode import F16ModeController


def model_init(model):
    """function to load resources needed by the controller

    save them in the model parameters, and access them anywhere model
    is passed
    """
    long_path = prepend_curr_path(('../', 'CentersLong.npy'))
    lat_path = prepend_curr_path(('../', 'CentersLat.npy'))
    ail_path = prepend_curr_path(('../', 'GainsAileron.npy'))
    ele_path = prepend_curr_path(('../', 'GainsElevator.npy'))
    rud_path = prepend_curr_path(('../', 'GainsRudder.npy'))
    model.parameters["centers_long"] = np.load(long_path)
    model.parameters["centers_lat"] = np.load(lat_path)
    model.parameters["gains_aileron"] = np.load(ail_path)
    model.parameters["gains_elevator"] = np.load(ele_path)
    model.parameters["gains_rudder"] = np.load(rud_path)
    model.parameters["ctrlr"] = F16ModeController()


def compute_fcn(model, x_ctrl):
    """compute 3-dim control signal from 8-dim x_ctrl signal"""
    ctrlr = model.parameters["ctrlr"]
    return np.array(ctrlr.Controller(x_ctrl)).flatten()


def model_output(model, time_t, state_controller, input_all):
    """ get the reference commands for the control surfaces """
    _, *trim_points = getattr(lqr, model.lqr_name)()
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
