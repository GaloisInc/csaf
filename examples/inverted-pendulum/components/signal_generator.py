"""
Generate signal for testing the inverted pendulum controller

The only available output is the desired cart position, we are controlling pendulum position to zero.

Taken from :http://ctms.engin.umich.edu/CTMS/index.php?example=InvertedPendulum&section=ControlStateSpace
"""
def model_output(model, time_t, _state_controller, _input):
    # Position
    output = 0
    if model.response_type == 'step':
        # Don't generate step signal at the first time step
        if time_t > 1.0:
            output = model.step_value
    elif model.response_type == 'impulse':
        if time_t >= 1.0-model.impulse_epsilon and time_t <= 1.0+model.impulse_epsilon:
            output = model.step_value
    else:
        raise ValueError('Unsupported model response type: ' + model.response_type)

    return [output]
