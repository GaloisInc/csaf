import numpy as np


def model_output(model, time_t, state_noise, input_f16_plant):
    return (np.array(input_f16_plant) + np.array(model.scales) * np.random.randn(13)).tolist()
