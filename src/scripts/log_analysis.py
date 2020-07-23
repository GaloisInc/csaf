import re
import json
import pathlib
import os

from csaf.config import attempt_parse_toml

import matplotlib.pyplot as plt
import numpy as np

def parse_log(log_file, config_file):
    """parse a log file into a dictionary of sent and received messages"""

    config = attempt_parse_toml(config_file)
    devices = config["device"]

    # get all referenced components
    component_pattern = re.compile(r"Component \'(\w+)\'")
    with open(log_file, "r") as fobj:
        log_text = fobj.read()
    components = list(set(re.findall(component_pattern, log_text)))

    for c in components:
        dconf = devices[c]

        process_path = pathlib.Path(dconf["process"])
        assert os.path.exists(process_path), f"component executable {process_path} could not be found for component"

        if 'config' not in dconf:
            config_path = process_path.with_suffix('.toml')
        else:
            config_path = dconf["config"]

        dtoml = attempt_parse_toml(config_path)
        snames = dtoml["states"]["names"]
        onames = dtoml["outputs"]["names"]
        inames = dtoml["inputs"]["names"]

    # collect sent and received messages
    msgs = {}
    for c in components:
        send_pattern = re.compile(fr"Component \'{c}\'(.*)Sending(.*)<(.*)>")
        recv_pattern = re.compile(fr"Component \'{c}\'(.*)Received(.*)<(.*)>")
        sends_s  = re.findall(send_pattern, log_text)
        recvs_s = re.findall(recv_pattern, log_text)
        sends = [json.loads(s[2].replace("'", '"')) for s in sends_s]
        recvs = [json.loads(s[2].replace("'", '"')) for s in recvs_s]

        dconf = devices[c]

        process_path = pathlib.Path(dconf["process"])
        assert os.path.exists(process_path), f"component executable {process_path} could not be found for component"

        if 'config' not in dconf:
            config_path = process_path.with_suffix('.toml')
        else:
            config_path = dconf["config"]

        dtoml = attempt_parse_toml(config_path)
        snames = dtoml["states"]["names"]
        onames = dtoml["outputs"]["names"]
        inames = dtoml["inputs"]["names"]
        names = {"State": snames, "Input": inames, "Output": onames}

        msgs[c] = {"Sent": sends, "Received": recvs, "Names": names}

    return msgs


def plot_field(msg, component, field, indices):
    """given a parsed log file, plot a component's field at specific indices"""
    data = msg[component]["Sent"]
    names  = msg[component]['Names'][field]

    def get_field(data_elem):
        array  = np.array(data_elem[field])
        return array[np.array(indices)]

    t = [d["time"] for d in data]
    data = np.array([get_field(d) for d in data])

    fig, ax = plt.subplots(figsize=(6, len(indices)*2), nrows=len(indices), sharex=True)
    if len(indices) > 1:
        for iidx, ind in enumerate(indices):
            ax[iidx].plot(t, data[:, iidx])
            ax[iidx].set_title(names[ind])
            ax[iidx].grid()
        ax[-1].set_xlabel("Time (s)")
        ax[-1].set_xlim([min(t), max(t)])
    else:
        ax.plot(t, data)
        ax.set_title(names[indices[0]])
    return fig, ax


if __name__ == '__main__':
    msg = parse_log("./f16run.log", './config.toml')
    fig, ax = plot_field(msg, "plant", "State", (0, 1, 6, 7, 8, 11))
    plt.show()

