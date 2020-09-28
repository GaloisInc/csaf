"""
Generate signal for a component
"""
from numpy import sin

def model_output(model, time_t, _state_controller, _input_f16):
    epsilon = 0.01

    outputs = [0] * model.output_size
    if model.response_type == 'step':
        # Don't generate step signal at the first time step
        if time_t > model.response_time:
            outputs[model.output_idx] = model.response_amplitude
    elif model.response_type == 'impulse':
        if time_t >= model.response_time-epsilon and time_t <= model.response_time+epsilon:
            outputs[model.output_idx] = model.response_amplitude
    elif model.response_type == 'sin':
        outputs[model.output_idx] = sin(time_t) * model.response_amplitude
    else:
        raise ValueError('Unsupported model response type: ' + model.response_type)

    return outputs
