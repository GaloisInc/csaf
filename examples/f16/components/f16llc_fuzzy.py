""" Fuzzy Logic Low Level Controller Model

TODO: implement the TODOs
"""
import numpy as np
from fileops import prepend_curr_path
from helpers import lqr, llc_helper as lh
from f16llc import model_state_update
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
    #model.parameters["ctrlr"] = F16ModeController()
    ctrlr = F16ModeController()

    def compute_fcn(x_ctrl):
        """compute 3-dim control signal from 8-dim x_ctrl signal"""
        print('x')
        return np.array(ctrlr.Controller(x_ctrl)).flatten()

    _, xequil, uequil = getattr(lqr, model.lqr_name)()
    model.parameters['llc'] = lh.FeedbackController(lh.CtrlLimits(), model, compute_fcn, xequil, uequil)


def model_output(model, t, state_controller, input_all):
    assert len(input_all) == 21
    #TODO: hard coded indices!
    """ get the reference commands for the control surfaces """
    return model.parameters['llc'].output(t, np.array(state_controller), np.array(input_all))
