"""
Generates a half-elipse reference, from t=[0,2*xm]
"""
def model_output(model, time_t, state_controller, input_pendulum):
    if model.maneuver_name == "step":
        m = 0.05
        xm = 9.0
        if time_t > 0.0 and time_t < xm*2:
            return [-m*(time_t - xm)**2 + m* xm**2]
        else:
            return [0.0]
    elif model.maneuver_name == "const":
        return [0.0]
