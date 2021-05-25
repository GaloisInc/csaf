import autopilot_helper as ah
from acasxu import *
import numpy as np


def model_init(model):
    """load trained model"""
    model.parameters['auto'] = None


def get_auto(model, f16_state):
    if model.auto is None:
        model.parameters['auto'] = AcasXuAutopilot(f16_state)
    return model.auto


def model_output(model, time_t, state_controller, input_f16):
    auto = get_auto(model, input_f16)
    return auto.get_u_ref(time_t, input_f16)[:4]


def model_state_update(model, time_t, state_controller, input_f16):
    auto = get_auto(model, input_f16)
    return [auto.advance_discrete_mode(time_t, input_f16)]
