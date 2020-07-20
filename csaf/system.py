""" Component Based System

Ethan Lew
07/13/20
"""

import pathlib
import os

import toml
from .config import *
from .dynamics import DynamicalSystem



class System:
    @classmethod
    def from_toml(cls, config_file):

        check_config(config_file)
        setup_logging(config_file)
        config = attempt_parse_toml(config_file)

        # TODO: sanitize the top level
        device_configs = config["device"]
        eval_order = config["evaluation_order"]
        devices = []
        ports = []
        names = []
        for dname, dconfig in device_configs.items():
            # TODO: sanitize the config
            pub_ports, sub_ports = get_from_base_string("pub", dconfig), get_from_base_string("sub", dconfig)

            process_path = pathlib.Path(dconfig["process"])
            assert os.path.exists(process_path), f"component executable {process_path} could not be found for component"

            if 'config' not in dconfig:
                config_path = process_path.with_suffix('.toml')
            else:
                config_path = dconfig["config"]

            assert os.path.exists(config_path), f"config file {config_path} could not be found for component"

            dev_config = attempt_parse_toml(config_path)

            n_inputs =  dev_config["inputs"]["names"]
            n_outputs = dev_config["outputs"]["names"]
            n_states = dev_config["states"]["names"]

            command = dconfig["run_command"] + " " + dconfig["process"]

            comp = DynamicalSystem(command, n_inputs, n_outputs, n_states, name=dname)
            comp.bind(sub_ports, pub_ports)
            if dconfig["debug"]:
                comp.debug_node = True
            devices.append(comp)
            names.append(dname)
            ports += [*pub_ports, *sub_ports]

        system = cls()
        system.components = devices
        system.ports = list(set(ports))
        system.names = names

        system.eval_order = [names.index(e) for e in eval_order]
        return system

    def __init__(self):
        self.components = []
        self.names = []
        self.ports = []
        self.eval_order = []

    def activate_system(self):
        for eidx in self.eval_order:
            self.components[eidx].activate_subscribers()
