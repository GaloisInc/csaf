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

import numpy as np
from gym import spaces
import socket
import socketserver
import importlib


class System:
    """ System accepts a component configuration, and then configures and composes them into a controlled system Has a scheduler to permit time simulations of the component system
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
            comp.bind(sub_ports, pub_ports)
            components.append(comp)
            names.append(dname)
            ports += pub_ports

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
            self.components[idx].receive_input()
            out = self.components[idx].send_output()
            if self.components[idx].internal_error or (terminating_conditions is not None and terminating_conditions(cidx, out)):
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
            if self.components[idx].internal_error or (terminating_conditions is not None and terminating_conditions(cidx, out)):
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
    def __init__(self, cname, sys, terminating_conditions=None, corerl=False, blend=0.5):
        """ SystemEnv exposes one component to the user during simulation, allowing step and reset
        :param cname: component to expose
        :param sys: CSAF system
        """
        self.sys: System = sys
        self._cname = cname
        #self.action_space = spaces.Box(-1., 1., shape=(4,), dtype='float32')
        #self.action_space = spaces.Box(np.array([-2 / 9, -1, 0, 0]), np.array([1, 1, 0, 1]))
        self.action_space = spaces.Box(np.array([-2 / 9, -1, 0]), np.array([1, 1, 1]))
        #self.action_space = spaces.Box(np.array([-2 / 5, 0]), np.array([1, 1]))
        #self.observation_space = spaces.Box(-1., 1., shape=(13,), dtype='float32')  #spaces.Box(low=0, high=255, shape=(screen_height, screen_width, 3), dtype=np.uint8)
        self.observation_space = spaces.Box(
                np.array([300 / 2500, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0, 0]),
                np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]))
        self._iter = self.make_system_iterator(terminating_conditions=terminating_conditions)
        self.corerl = corerl
        self.blend = blend
        if corerl:
            self.autopilot_model = ModelNative.from_filename("/home/greg/Documents/csaf_architecture/examples/f16/components/autopilot.py", "/home/greg/Documents/csaf_architecture/examples/f16/components/autopilot.toml")
            self.last_state = None
        next(self._iter)

    def step(self, component_output):
        """step through the simulation generator"""

        # Unnormalize actions
        #component_output[0] *= 9    # Nz \in [-2, 9]
        ## From the plots of running f16-simple, it looks like ps_ref stays roughly in [-2.5, 0.5].
        ## We'll start by letting it vary in [-3, 3]
        #component_output[1] *= 3
        #component_output[2] = 0     # Ny = 0
        # Throttle is already in [0, 1] so we don't need to normalize it.
        action = []
        action.append(component_output[0] * 5)
        action.append(component_output[1] * 3)
        #action.append(0)
        action.append(0)
        action.append(component_output[2])
        #action.append(component_output[1])

        if self.corerl and self.last_state is not None:
            self.autopilot_state = self.autopilot_model.get_state_update(2.0, [self.autopilot_state], self.last_state)[0]
            symb_action = self.autopilot_model.get_output(2.0, [self.autopilot_state], self.last_state)

            for i in range(len(action)):
                action[i] = self.blend * action[i] + (1 - self.blend) * symb_action[i]

        # states: [vt, alpha, beta, phi, theta, psi, p, q, r, pn, pe, h, power]
        # actions: [Nz_ref, ps_ref, Ny_r_ref, throttle_ref]

        try:
            stuff = self._iter.send({"autopilot-states": ["Waiting"],
                                  "autopilot-fdas": ["Waiting"],
                                  "autopilot-outputs": np.asarray(action)})
            #print("after step:", stuff)
            ob = stuff['plant-states']
            #reward = -stuff['time']
            # 1st attempt: Reward = - c1 * (altitude deviation) - c2 * (pitch deviation) - 1
            # The constant negative term helps penalize extra time taken
            # c2 should be small so that the policy prioritizes reaching a good altitude
            #   before leveling off.
            reward = -5 + 0.001 * (ob[11] - 1000) - abs(ob[3])
            pouts = stuff['plant-outputs']
            done = False
            # Find a good stopping condition
            # 1st attempt: altitude is within [800, 1500] ft and pitch angle is
            # within [0, 30] deg
            if 800 <= ob[11] and 0 <= ob[4] and ob[4] <= 30 and abs(ob[4] - ob[1]) <= 0.01 and abs(ob[7]) <= 1:
                done = True
            # TODO: Check pitch/roll/yaw rate.

        except StopIteration:
            #print("StopIteration")
            done = True
            ob = [0 for _ in range(13)]
            # Penalty for crashing
            reward = -1000
            reward = 0
            pouts = [0 for _ in range(3)]

        except Exception:
            done = True
            ob = [0 for _ in range(13)]
            # Penalty for crashing
            reward = -1000
            pouts = [0 for _ in range(3)]

        if ob[11] <= 0:
            print("crash:", ob[11])
            done = True
            ob = [0 for _ in range(13)]
            # Penalty for crashing
            reward = -10000
            pouts = [0 for _ in range(3)]

        if self.corerl:
            self.last_state = ob

        # Normalize observations - All limits are linear scalings with no offset
        #ob[0] /= 2500    # vt \in [300, 2500]
        #ob[1] /= 45      # alpha \in [-10, 45]
        #ob[2] /= 30      # beta \in [-30, 30]
        #ob[3] /= 180     # For the roll, pitch, and yaw angles (phi, theta,
        #ob[4] /= 180     #   psi) we will just assume they fit in [-180, 180].
        #ob[5] /= 180     #   (0 seems to be the stable position)
        #ob[6] /= 180     # Not sure about angular rates, but I'll assume it's
        #ob[7] /= 180     #   probably less than 180 deg / s. May be refined.
        #ob[8] /= 180
        #ob[9] /= 10000   # North/east displacement should be irrelevant, so
        #ob[10] /= 10000  #   I'll just put some limit.
        #ob[11] /= 45000  # h \in [0, 45000]
        #ob[12] /= 20     # power \in [0, 20] (based on f16-simple)
        res = []
        res.append(ob[0] / 2500)
        res.append(ob[1] / 3.2)
        res.append(ob[2] / 3.2)
        res.append(ob[3] / 3.2)
        res.append(ob[4] / 3.2)
        res.append(ob[5] / 3.2)
        res.append(ob[6] / 3.2)
        res.append(ob[7] / 3.2)
        res.append(ob[8] / 3.2)
        res.append(ob[9] / 10000)
        res.append(ob[10] / 10000)
        res.append(ob[11] / 45000)
        res.append(ob[12] / 20)

        # Check whether something is diverging. Sometimes we trigger some issue in
        # the ODE solver which causes vt to grow unchecked.
        #print("res: ", res)
        #if not self.observation_space.contains(res):
        #    #print("res is outside box")
        #    done = True
        #    ob = [0 for _ in range(13)]
        #    # Penalty for crashing
        #    reward = -10000
        #    pouts = [0 for _ in range(4)]

        return np.asarray(res), reward, done, {'plant-outputs': pouts}

    def reset(self):
        """reset components and create a new sim"""
        # TODO: Try simpler initial conditions/action space (Nz only, zero roll angle)
        for c in self.sys.components:
            c.reset()
        self._iter = self.make_system_iterator()
        stuff = next(self._iter)
        # ob = stuff['plant-states']
        # reward = -stuff['time']
        # pouts = stuff['plant-outputs']
        # done = False
        # next(self._iter)

        if self.corerl:
            self.autopilot_state = "Waiting"
            self.last_state = None

        return np.asarray([0 for _ in range(13)])

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
            if self.sys.components[idx].internal_error:
                print("system iterator: internal error in component", self.sys.components[idx]._name)
            if self.sys.components[idx].internal_error or (terminating_conditions is not None and terminating_conditions(cidx, out)):
                return
