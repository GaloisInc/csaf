def model_output(model, time_t, state_switch, input_autopilot):
    controller, controllerr, select = input_autopilot[:4], input_autopilot[4:8], input_autopilot[-1]
    return controller if not select else controllerr
