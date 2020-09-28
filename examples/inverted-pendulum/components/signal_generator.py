"""
Generate signal for testing the inverted pendulum controller

The only available output is the desired cart position, we are controlling pendulum position to zero.

Taken from :http://ctms.engin.umich.edu/CTMS/index.php?example=InvertedPendulum&section=ControlStateSpace
"""
from numpy import sin

def model_output(model, time_t, _state_controller, _input):
    # Position
    output = 0
    if model.response_type == 'step':
        # Don't generate step signal at the first time step
        if time_t > model.response_time:
            output = model.response_amplitude
    elif model.response_type == 'impulse':
        if time_t >= model.response_time-model.impulse_epsilon and time_t <= 1.0+model.impulse_epsilon:
            output = model.response_amplitude
    elif model.response_type == 'sin':
        output = sin(time_t) * model.response_amplitude
    else:
        raise ValueError('Unsupported model response type: ' + model.response_type)

    return [output]
