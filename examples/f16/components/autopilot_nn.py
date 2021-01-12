import tensorflow as tf
import numpy as np
from spinup.utils.test_policy import load_policy_and_env
import autopilot_helper as ah

class NNAutopilot():
    STATE_START= 'Waiting'
    STATE_ENGAGED = 'Engaged'
    STATE_DONE = 'Done'

def model_init(model):
    path = "/home/greg/Documents/csaf_architecture/models/ddpg-small-roll"
    _, model.parameters["model"] = load_policy_and_env(path)

def model_output(model, time_t, state_controller, input_f16):
    phi = input_f16[3]             # Roll angle    (rad)
    p = input_f16[6]               # Roll rate     (rad/sec)
    theta = input_f16[4]           # Pitch angle   (rad)
    alpha = input_f16[1]           # AoA           (rad)
    vt = input_f16[0]
    # Note: pathAngle = theta - alpha
    gamma = theta-alpha

    # Determine which angle is "level" (0, 180, 360, 720, etc)
    radsFromWingsLevel = round(phi/np.pi)
    phi_des = np.pi*radsFromWingsLevel
    p_des = 0

    # Determine "which" angle is level (0, 360, 720, etc)
    radsFromNoseLevel = round(gamma/np.pi)
    gamma_des = np.pi*radsFromNoseLevel
    if state_controller[0] == NNAutopilot.STATE_START:
        return 0, 0, 0, 0
    elif state_controller[0] == NNAutopilot.STATE_DONE:
        Nz, ps = state_done(gamma_des, phi_des, p_des, gamma, phi, p)
        throttle = ah.p_cntrl(kp=0.25, e=(model.vt_des-vt))
        return Nz, ps, 0, throttle
    state = np.empty_like(input_f16[:13])
    # Normalize state
    state[0] = input_f16[0] / 2500
    state[1] = input_f16[1] / 3.2
    state[2] = input_f16[2] / 3.2
    state[3] = input_f16[3] / 3.2
    state[4] = input_f16[4] / 3.2
    state[5] = input_f16[5] / 3.2
    state[6] = input_f16[6] / 3.2
    state[7] = input_f16[7] / 3.2
    state[8] = input_f16[8] / 3.2
    state[9] = input_f16[9] / 10000
    state[10] = input_f16[10] / 10000
    state[11] = input_f16[11] / 45000
    state[12] = input_f16[12] / 20
    action = model.model(state)
    action_low = np.array([-2 / 9, -1, 0])
    action_high = np.array([1, 1, 1])
    action = np.clip(action, action_low, action_high)
    # Unnormalize action
    return action[0] * 5, action[1] * 3, 0, action[2] * 30

def model_state_update(model, time_t, state_controller, input_f16):
    state = state_controller[0]

    phi = input_f16[3]             # Roll angle    (rad)
    p = input_f16[6]               # Roll rate     (rad/sec)
    theta = input_f16[4]           # Pitch angle   (rad)
    alpha = input_f16[1]           # AoA           (rad)
    h = input_f16[11]

    if time_t >= 2:
        state = NNAutopilot.STATE_ENGAGED

    if state == NNAutopilot.STATE_ENGAGED:
        if h >= 800 and theta >= 0 and theta <= 30 and abs(theta - alpha) <= 0.01 and abs(input_f16[7]) <= 1 and abs(phi) <= 0.01:
            state = NNAutopilot.STATE_DONE

    return [state]

def model_info(model, time_t, state_controller, input_f16):
    return state_controller

def state_done(gamma_des, phi_des, p_des, gamma, phi, p):
    # steady-level hold
    # Set Proportional-Derivative control gains for roll
    K_prop = 1
    K_der = K_prop*0.3
    e_ps, ed_ps = phi_des - phi, p_des-p
    # PD Control on phi using roll rate
    ps = ah.pd_cntrl(K_prop, K_der, e_ps, ed_ps)

    # Set Proportional-Derivative control gains for pitch

    # Unstability counterexample
    #./rundemo.py --test-id=dc --endtime=19.4 --animate
    #K_prop2 = 78

    K_prop2 = 2
    K_der2 = K_prop*0.3
    #XXX: Why is roll rate (p) being used here? Should be q, the pitch rate
    e_nz, ed_nz = gamma_des-gamma, p_des-p
    # PD Control on theta using Nz
    #Nz = -(gamma - gamma_des) * K_prop2 - p*K_der2
    Nz = ah.pd_cntrl(K_prop2, K_der2, e_nz, ed_nz)
    return Nz, ps
