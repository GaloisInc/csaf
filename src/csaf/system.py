""" Component Based System

Ethan Lew
07/13/20
"""
from .config import SystemConfig
from .dynamics import DynamicalSystem


class System:
    @classmethod
    def from_toml(cls, config_file):
        """produce a system from a toml file"""
        config = SystemConfig.from_toml(config_file)
        return cls.from_config(config)

    @classmethod
    def from_config(cls, config):
        """produce system from SystemConfig object"""
        eval_order = config.config_dict["evaluation_order"]
        devices = []
        ports = []
        names = []
        for dname, dconfig in config.config_dict["devices"].items():
            command = ' '.join([dconfig["run_command"], dconfig["process"]])

            name_inputs = dconfig["config"]["inputs"]["names"]

            if config.has_topic(dname, "outputs"):
                name_outputs = config.get_msg_setting(dname, "outputs", "msg").fields_no_header
            else:
                name_outputs = []

            if config.has_topic(dname, "states"):
                name_states = config.get_msg_setting(dname, "states", "msg").fields_no_header
            else:
                name_states = []

            sub_ports = [[str(config.config_dict["devices"][l]["pub"]), l+"-"+t] for l, t in dconfig["sub"]]
            pub_ports = [str(dconfig["pub"])]
            comp = DynamicalSystem(command, name_inputs, name_outputs, name_states, name=dname)
            if dconfig["debug"]:
                comp.debug_node = True
            comp.bind(sub_ports, pub_ports)
            devices.append(comp)
            names.append(dname)
            ports += pub_ports

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
