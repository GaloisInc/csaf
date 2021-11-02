from f16lib.predictor import CollisionPredictor


def model_output(model, time_t, state_switch, input_autopilot):
    p: CollisionPredictor = model.predictor
    p.step(time_t, input_autopilot)
    return [p.make_prediction()]


def model_init(model):
    model.parameters['predictor'] = CollisionPredictor(
        model.parameters["intruder_waypoints"],
        model.parameters["own_waypoints"]
    )
