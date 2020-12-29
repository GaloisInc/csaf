import numpy as np

import helpers.llc_helper as lh
from helpers import lqr


def model_init(model):
    """load trained model"""
    ctrl_fn, xequil, uequil = getattr(lqr, model.lqr_name)()
    model.parameters['llc'] = lh.FeedbackController(lh.CtrlLimits(), model, ctrl_fn, xequil, uequil)


def model_output(model, t, state_controller, input_all):
    assert len(input_all) == 21
    #TODO: hard coded indices!
    """ get the reference commands for the control surfaces """
    return model.parameters['llc'].output(t, np.array(state_controller), np.array(input_all))


def model_state_update(model, t, state_controller, input_all):
    """ get the derivatives of the integrators in the low-level controller """
    #return llc.step(t, 1/model.sampling_frequency, state_controller, input_all)
    return model.parameters['llc']._der(t, np.array(state_controller), np.array(input_all))
