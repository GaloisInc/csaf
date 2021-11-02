def model_output(model, time_t, state_switch, input_autopilot):
    assert input_autopilot[-1] in model.mapper, f"{input_autopilot[-1]} not in {model.mapper}"
    controller = input_autopilot[-1]
    if controller == 0.0:
        sidx = int(0.0)
    else:
        sidx = model.mapper.index(controller)
    assert len(input_autopilot) == 13
    return input_autopilot[4 * sidx:4 * sidx + 4]
