import math


def model_state_update(model, time_t, state_pendulum, input_controller):
    return [state_pendulum[1],
            - model.g/model.l * math.sin(state_pendulum[0])
            - model.b / (model.m * model.l) * state_pendulum[1]
            + input_controller[0]]


def model_output(model, time_t, state_pendulum, input_controller):
    return [state_pendulum[0]]
