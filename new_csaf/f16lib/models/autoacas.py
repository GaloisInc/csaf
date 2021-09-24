import f16lib.models.helpers.autopilot_helper as ah
from f16lib.models.acasxu import *
import numpy as np


def model_init(model):
    """load trained model"""
    model.parameters['auto'] = None


def get_auto(model, f16_state):
    if model.auto is None:
        model.parameters['auto'] = AcasXuAutopilot(f16_state, roll_rates=model.parameters['roll_rates'])
        if model.gains == "recovery":
            self = model.parameters['auto']

            # TODO: restructure this?
            # change the gains so the recovery controller is much more aggressive
            # control config
            # Gains for speed control
            self.cfg_k_vt = 0.7

            # Gains for altitude tracking
            self.cfg_k_alt = 0.005
            self.cfg_k_h_dot = 0.02

            # Gains for heading tracking
            self.cfg_k_prop_psi = 12.0
            self.cfg_k_der_psi = 2.0

            # Gains for roll tracking
            self.cfg_k_prop_phi = 10.0
            self.cfg_k_der_phi = 2.0
            self.cfg_max_bank_deg = 90.0  # maximum bank angle setpoint

    return model.auto


def model_output(model, time_t, state_controller, input_f16):
    # NOTE: not using llc
    expanded_states = [[*input_f16[i*13:(i+1)*13], 0.0, 0.0, 0.0] for i in range(len(input_f16)//13)]
    input_f16 = (np.concatenate(expanded_states))
    auto = get_auto(model, input_f16)
    return auto.get_u_ref(time_t, input_f16)[:4]


def model_state_update(model, time_t, state_controller, input_f16):
    # NOTE: not using llc
    expanded_states = [[*input_f16[i*13:(i+1)*13], 0.0, 0.0, 0.0] for i in range(len(input_f16)//13)]
    input_f16 = (np.concatenate(expanded_states))
    auto = get_auto(model, input_f16)
    return [auto.advance_discrete_mode(time_t, input_f16)]
