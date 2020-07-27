"""Component Based System

Ethan Lew
07/13/20
"""
from .config import SystemConfig
from .dynamics import DynamicComponent
from .messenger import SerialMessenger
from .scheduler import Scheduler
from .model import ModelNative
from.trace import TimeTrace



class System:
    """ System accepts a component configuration, and then configures and composes them into a controlled system

    Has a scheduler to permit time simulations of the component system
    """
    @classmethod
    def from_toml(cls, config_file: str):
        """produce a system from a toml file"""
        config = SystemConfig.from_toml(config_file)
        return cls.from_config(config)

    @classmethod
    def from_config(cls, config: SystemConfig):
        """produce system from SystemConfig object
        TODO: decompose long classmethod into functions (?)
        """
        eval_order = config.config_dict["evaluation_order"]
        devices = []
        ports = []
        names = []
        for dname, dconfig in config.config_dict["devices"].items():
            # dynamic model
            # TODO: better Model class selection here
            is_discrete = dconfig["config"]["is_discrete"]
            model = ModelNative.from_filename(dconfig["process"], is_discrete=is_discrete)

            # pub/sub parameters
            sub_ports = [[str(config.config_dict["devices"][l]["pub"]), l+"-"+t] for l, t in dconfig["sub"]]
            pub_ports = [str(dconfig["pub"])]
            topics_in = [s[1] for s in sub_ports]

            # produce serial messengers
            mss_out = dconfig['config']['topics']
            mss_out = {f"{dname}-{t}": v['serializer'] for t, v in mss_out.items()}
            mss_in ={}
            for sname, stopic in dconfig['sub']:
                k = f"{sname}-{stopic}"
                sconfig = config.get_device_settings(sname)
                mss_in[k] = sconfig['config']['topics'][stopic]['serializer']
            mss_out = SerialMessenger(mss_out)
            mss_in = SerialMessenger(mss_in)

            # sampling frequency
            sampling_frequency = dconfig['config']['sampling_frequency']
            comp = DynamicComponent(model, topics_in, mss_out, mss_in, sampling_frequency, name=dname)

            # set properties
            if dconfig["debug"]:
                comp.debug_node = True
            if config.has_topic(dname, 'states'):
                comp.state = config.get_msg_setting(dname, 'states', 'initial')

            # bind and update structures
            comp.bind(sub_ports, pub_ports)
            devices.append(comp)
            names.append(dname)
            ports += pub_ports

        system = cls(devices, eval_order, config)
        return system

    def __init__(self, components, eval_order, config):
        self.components = components
        self.eval_order = eval_order
        self.config = config

    def simulate_tspan(self, tspan, show_status=False):
        """over a given timespan tspan, simulate the system"""
        sched = Scheduler(self.components, self.eval_order)
        s = sched.get_schedule_tspan(tspan)

        # produce stimulus
        input_for_first = list(set([p for p, _ in self.config._config["devices"]["controller"]["sub"]]))
        for dname in input_for_first:
            idx = self.names.index(dname)
            self.components[idx].send_stimulus(tspan[0])

        # get time trace fields
        dnames = self.config.get_name_devices
        dtraces = {}
        for dname in dnames:
            fields = (['times'] + [f"{topic}" for topic in self.config.get_topics(dname)])
            dtraces[dname] = TimeTrace(fields)

        if show_status:
            import tqdm
            s = tqdm.tqdm(s)

        # TODO collect updated topics only
        for cidx, time in s:
            idx = self.names.index(cidx)
            self.components[idx].receive_input()
            out = self.components[idx].send_output()
            dtraces[cidx].append(**out)

        return dtraces

    @property
    def names(self):
        return [c.name for c in self.components]

    @property
    def ports(self):
        p = []
        for c in self.components:
            p += c.output_socks
        return p
