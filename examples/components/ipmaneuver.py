import os
import toml

parameters = {}


def main(time=0.0, state=None, input=None, update=False, output=False):
    global parameters
    if len(parameters.keys()) == 0:
        this_path = os.path.dirname(os.path.realpath(__file__))
        info_file = os.path.join(this_path, "ipmaneuver.toml")
        with open(info_file, 'r') as ifp:
            info = toml.load(ifp)
        parameters = info["parameters"]

    if output:
        if parameters["maneuver_name"] == "step":
            if time > 0.0 and time < 6.0:
                return [-0.1*(time - 3.1)**2 + 0.9]
            else:
                return [0.0]
        elif parameters["maneuver_name"] == "const":
            return [0.0]
    else:
        return []
