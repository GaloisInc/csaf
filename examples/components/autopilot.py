import os
import toml
import json

import numpy as np
from csaf import message

def main():

    msg_writer = message.Message()
    epoch = 0
    fs = 100

    this_path = os.path.dirname(os.path.realpath(__file__))
    info_file = os.path.join(this_path, "f16plant.toml")
    with open(info_file, 'r') as ifp:
        info = toml.load(ifp)

    parameters = info["parameters"]

    state_names = info["states"]["names"]
    input_names = info["inputs"]["names"]
    output_names = info["outputs"]["names"]

    n_states = len(state_names)
    n_outputs = len(output_names)
    n_inputs = len(input_names)

    while True:
        ins = input(f"msg0 at [t={epoch/fs}]>")
        try:
            msg = json.loads(ins)
        except json.decoder.JSONDecodeError as exc:
            raise Exception(f"input <{ins}> couldn't be interpreted as json! {exc}")

        in_epoch = msg["epoch"]
        epoch = in_epoch
        #assert in_epoch == epoch

        output = np.zeros((4,))
        msg = msg_writer.write_message(epoch/fs, output=output)

        print(msg)

if __name__ == '__main__':
    main()