import autopilot_helper as ah
import numpy as np
from acasxu import *


def model_init(model):
    """load trained model"""
    #init = np.zeros(16*2)
    model.parameters['auto'] = None #AcasXuAutopilot(init)


def get_auto(model, f16_state):
    if model.auto is None:
        model.parameters['auto'] = AcasXuAutopilot(f16_state)
    return model.auto



def model_output(model, time_t, state_controller, input_f16):
    # TODO: address the extraction better
    input_f16 = input_f16[:13] + [0, 0, 0] + input_f16[13+4:(13+4+13)] + [0, 0, 0]
    print(input_f16)
    auto = get_auto(model, input_f16)
    return auto.get_u_ref(time_t, input_f16)[:4]


def model_state_update(model, time_t, state_controller, input_f16):
    input_f16 = input_f16[:13] + [0, 0, 0] + input_f16[13+4:(13+4+13)] + [0, 0, 0]
    auto = get_auto(model, input_f16)
    return [auto.advance_discrete_mode(time_t, input_f16)]

    #vt = input_f16[0]         # airspeed      (ft/sec)
    #alpha = input_f16[1]      # AoA           (rad)
    #theta = input_f16[4]      # Pitch angle   (rad)
    #gamma = theta - alpha # Path angle    (rad)
    #h = input_f16[11]         # Altitude      (feet)

    # Proportional Control
    #k_alt = 0.025
    #h_error = model.setpoint - h
    #Nz = k_alt * h_error # Allows stacking of cmds

    # (Psuedo) Derivative control using path angle
    #k_gamma = 25
    # k_gamma = self.p_gain
    #Nz = Nz - k_gamma*gamma

    # try to maintain a fixed airspeed near trim point
    #K_vt = 0.25
    #airspeed_setpoint = 400
    #vt_des = model.xequil[0]
    #throttle = ah.p_cntrl(kp=K_vt, e=(vt_des - vt))

    #return 0, 5, 0, throttle
