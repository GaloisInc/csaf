def model_output(model, time_t, _state_controller, _input_f16):
    epsilon = 0.01

    if model.output_to_excite == 'Nz':
        output_idx = 0
    elif model.output_to_excite == 'ps':
        output_idx = 1
    else:
        raise ValueError('Unsupported output to excite: ' + model.output_to_excite)

    # Nz, ps, Ny_r, throttle
    outputs = [0, 0, 0, 0]
    if model.response_type == 'step':
        # Don't generate step signal at the first time step
        if time_t > 1.0:
            outputs[output_idx] = 1
    elif model.response_type == 'impulse':
        if time_t >= 1.0-epsilon or time_t <= 1.0+epsilon:
            outputs[output_idx] = 1
    else:
        raise ValueError('Unsupported model response type: ' + model.response_type)

    return outputs[0],outputs[1],outputs[2],outputs[3]
