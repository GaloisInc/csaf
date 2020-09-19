"""
This is a flight gear visualisation tool for faster-than-realtime simulation
In displays the simulated trajectory *during* the simulation, so the user can get
a quick visual feedback about what is happening

To use this component, add these into your model config:

  [components.flightgear]
    # environment to run under
    run_command = "python3"

    # path to model executable
    process = "flightgear.py"

    # whether to print debug diagnostics
    debug = false

    # subscribe to topic of a component (component name, topic name)
    sub = [["plant", "states"], ["controller", "outputs"]]

"""

from fgnetfdm import FGNetFDM

fdm = FGNetFDM()

def model_update(model, time_t, state_fg, input_f16):
    if not fdm.running:
        fdm.init_from_params(model.parameters)
    fdm.update_and_send(input_f16)
