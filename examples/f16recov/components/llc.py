import llc_helper as lh
import lqr

#TODO: push the global value into run_system script
#TODO: is_discrete
llc = lh.FeedbackController(lh.CtrlLimits(), is_discrete=False, feedback_controller=lqr.get_lqr('lqr_original'))

def model_output(model, t, state_controller, input_all):
    """ get the reference commands for the control surfaces """
    return llc.output(t, state_controller, input_all)


def model_state_update(model, t, state_controller, input_all):
    """ get the derivatives of the integrators in the low-level controller """
    #return llc.step(t, 1/model.sampling_frequency, state_controller, input_all)
    return llc._der(t, state_controller, input_all)
