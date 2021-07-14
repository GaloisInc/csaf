"""Component Based System

Ethan Lew
07/13/20
"""
from .config import SystemConfig
from .dynamics import DynamicComponent
from .messenger import SerialMessenger
from .scheduler import Scheduler
from .model import ModelNative
from .trace import TimeTrace


import socket
import socketserver


class System:
    """ System accepts a component configuration, and then configures and composes them into a controlled system

    Has a scheduler to permit time simulations of the component system
    """
    @staticmethod
    def check_port(port):
        # TODO: add to config processor
        location = ("127.0.0.1", port)
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as a_socket:
            result_of_check = a_socket.connect_ex(location)
        return result_of_check == 0

    @staticmethod
    def reassign_ports(config):
        # TODO: add to config processor
        for dname, dconfig in config.config_dict["components"].items():
            if "pub" in dconfig:
                port_num = dconfig["pub"]
                if System.check_port(port_num):
                    with socketserver.TCPServer(("localhost", 0), None) as s:
                        free_port = s.server_address[1]
                else:
                    free_port = port_num
                config.config_dict["components"][dname]["pub"] = free_port
        return config

    @classmethod
    def from_toml(cls, config_file: str, **kwargs):
        """produce a system from a toml file"""
        config = SystemConfig.from_toml(config_file)
        return cls.from_config(config, **kwargs)

    @classmethod
    def from_config(cls, config: SystemConfig, recover_port_conflicts=True):
        """produce system from SystemConfig object
        TODO: decompose long classmethod into functions (?)
        """
        eval_order = config.config_dict["evaluation_order"]
        components = []
        ports = []
        names = []

        if recover_port_conflicts:
            config = cls.reassign_ports(config)

        for dname, dconfig in config.config_dict["components"].items():
            # dynamic model
            # TODO: better Model class selection here
            is_discrete = dconfig["config"]["is_discrete"]
            #model = ModelNative.from_filename(dconfig["process"], is_discrete=is_discrete)
            model = ModelNative.from_config(dconfig["process"], dconfig["config"])

            # pub/sub parameters
            sub_ports = [[str(config.config_dict["components"][l]["pub"]), l+"-"+t] for l, t in dconfig["sub"]]
            if "pub" in dconfig:
                pub_ports = [str(dconfig["pub"])]
            else:
                pub_ports = []
            topics_in = [s[1] for s in sub_ports]

            # produce serial messengers
            mss_out = dconfig['config']['topics']
            mss_out = {f"{dname}-{t}": v['serializer'] for t, v in mss_out.items()}
            mss_in ={}
            for sname, stopic in dconfig['sub']:
                k = f"{sname}-{stopic}"
                mss_in[k] = config.get_msg_setting(sname, stopic, 'serializer')
            mss_out = SerialMessenger(mss_out)
            mss_in = SerialMessenger(mss_in)

            def_buff = {}
            for tname in mss_out.topics:
                if "initial" in config.get_component_settings(dname)["config"]["topics"][tname.split("-")[1]]:
                    def_buff[tname] = config.get_msg_setting(dname, tname.split("-")[1], "initial")

            # sampling frequency
            sampling_frequency = dconfig['config']['sampling_frequency']
            comp = DynamicComponent(model, topics_in, mss_out, mss_in, sampling_frequency, name=dname, default_output=def_buff)

            # set properties
            if dconfig["debug"]:
                comp.debug_node = True

            # bind and update structures
            comp.init_net(sub_ports, pub_ports)
            comp.bind_output()
            components.append(comp)
            names.append(dname)
            ports += pub_ports

        # make connections now that components are up
        for c in components:
            c.connect_input()

        system = cls(components, eval_order, config)
        return system

    def __init__(self, components, eval_order, config):
        self.components = components
        self.eval_order = eval_order
        self.config = config

    def unbind(self):
        """unbind components from ports, teardown system"""
        for c in self.components:
            c.unbind()
        self.components = []
        self.eval_order = []
        self.config = None

    def reset(self):
        for c in self.components:
            c.reset()

    def validate_tspan(self, tspan, terminating_conditions, show_status=False):
        """over a given timespan tspan, determine if simulation fully runs"""
        sched = Scheduler(self.components, self.eval_order)
        s = sched.get_schedule_tspan(tspan)

        # produce stimulus
        input_for_first = list(set([p for p, _ in self.config._config["components"][self.eval_order[0]]["sub"]]))
        for dname in input_for_first:
            idx = self.names.index(dname)
            self.components[idx].send_stimulus(float(tspan[0]))

        if show_status:
            import tqdm
            s = tqdm.tqdm(s)

        for cidx, _ in s:
            idx = self.names.index(cidx)
            # bugfix: receive input must fail here
            self.components[idx].receive_input()
            out = self.components[idx].send_output()
            if terminating_conditions is not None and terminating_conditions(cidx, out):
                return False
        return True

    def simulate_tspan(self, tspan, show_status=False, terminating_conditions=None, return_passed=False):
        """over a given timespan tspan, simulate the system"""
        sched = Scheduler(self.components, self.eval_order)
        s = sched.get_schedule_tspan(tspan)

        # produce stimulus
        input_for_first = list(set([p for p, _ in self.config._config["components"][self.eval_order[0]]["sub"]]))

        for dname in input_for_first:
            idx = self.names.index(dname)
            self.components[idx].send_stimulus(float(tspan[0]))

        # get time trace fields
        dnames = self.config.get_name_components
        dtraces = {}
        for dname in dnames:
            fields = (['times'] + [f"{topic}" for topic in self.config.get_topics(dname)])
            dtraces[dname] = TimeTrace(fields)

        if show_status:
            import tqdm
            s = tqdm.tqdm(s)

        # TODO collect updated topics only
        for cidx, t in s:
            idx = self.names.index(cidx)
            self.components[idx].receive_input()
            out = self.components[idx].send_output()
            out["times"] = t
            if terminating_conditions is not None and terminating_conditions(cidx, out):
                return dtraces if not return_passed else (dtraces, False)
            dtraces[cidx].append(**out)

        return dtraces if not return_passed else (dtraces, True)

    def set_state(self, component_name, state):
        component = self.components[self.names.index(component_name)]
        component.state = state

    @property
    def names(self):
        """names of the components used in the system"""
        return [c.name for c in self.components]

    @property
    def ports(self):
        """zmq ports being used by the system"""
        p = []
        for c in self.components:
            p += c.output_socks
        return p


class SystemEnv:
    def __init__(self, cname, sys, terminating_conditions=None):
        """ SystemEnv exposes one component to the user during simulation, allowing step and reset
        :param cname: component to expose
        :param sys: CSAF system
        """
        self.sys: System = sys
        self._cname = cname
        self._iter = self.make_system_iterator(terminating_conditions=terminating_conditions)
        next(self._iter)

    def step(self, component_output):
        """step through the simulation generator"""
        return self._iter.send(component_output)

    def reset(self):
        """reset components and create a new sim"""
        for c in self.sys.components:
            c.reset()
        self._iter = self.make_system_iterator()
        next(self._iter)

    def set_state(self, component_name, state):
        """allow component state to be configured"""
        self.sys.set_state(component_name, state)

    def make_system_iterator(self, tstart=0.0, terminating_conditions=None):
        """make a generator for the step implementation"""
        sched = Scheduler(self.sys.components, self.sys.eval_order)
        s = sched.get_scheduler()
        input_for_first = list(set([p for p, _ in self.sys.config._config["components"][self.sys.eval_order[0]]["sub"]]))

        # produce stimulus
        for dname in input_for_first:
            idx = self.sys.names.index(dname)
            self.sys.components[idx].send_stimulus(tstart)

        # needed to use send
        yield None

        # Loop over an infinite generator
        for cidx, _ in s:
            idx = self.sys.names.index(cidx)
            self.sys.components[idx].receive_input()

            # if cname, get yield for input, simulate otherwise
            if cidx == self._cname:
                in_buffer = yield self.sys.components[idx]._input_buffer
                out = self.sys.components[idx].send_output(overwrite_buffer = in_buffer)
            else:
                out = self.sys.components[idx].send_output()

            # evaluate terminating conditions
            if terminating_conditions is not None and terminating_conditions(cidx, out):
                return
