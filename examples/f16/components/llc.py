import helpers.llc_helper as lh
from helpers import lqr

#TODO: push the global value into run_system script
#TODO: is_discrete


def model_init(model):
    """load trained model"""
    model.parameters['llc'] = lh.FeedbackController(
            lh.CtrlLimits(),
            is_discrete=model.is_discrete,
            feedback_controller=lqr.get_lqr(model.lqr_name)
            )

def model_output(model, t, state_controller, input_all):
    """ get the reference commands for the control surfaces """
    return model.parameters['llc'].output(t, state_controller, input_all)


def model_state_update(model, t, state_controller, input_all):
    """ get the derivatives of the integrators in the low-level controller """
    #return llc.step(t, 1/model.sampling_frequency, state_controller, input_all)
    return model.parameters['llc']._der(t, state_controller, input_all)
