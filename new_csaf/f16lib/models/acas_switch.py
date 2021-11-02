def model_output(model, time_t, state_switch, input_autopilot):
    #controller, state, controllerr, stater, select = input_autopilot[:4], input_autopilot[5] input_autopilot[4:8], input_autopilot[-1]
    controller, state = input_autopilot[:4], input_autopilot[4]
    controllerr, stater = input_autopilot[5:5+4], input_autopilot[9]
    select = input_autopilot[-1]
    return controller if not select else controllerr


def model_output_state(model, time_t, state_switch, input_autopilot):
    controller, state = input_autopilot[:4], input_autopilot[4]
    controllerr, stater = input_autopilot[5:5+4], input_autopilot[9]
    select = input_autopilot[-1]
    return [state] if not select else [stater]
