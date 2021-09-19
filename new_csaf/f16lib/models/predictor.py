def model_output(model, time_t, state_switch, input_autopilot):
    ownstate, ownout, intstate, inout = input_autopilot[:13], \
                                        input_autopilot[13:13 + 4], \
                                        input_autopilot[17:17 + 13], \
                                        input_autopilot[30:34]
    return [True]
