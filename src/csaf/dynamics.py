import numpy as np
import time
import zmq

import scipy.integrate

from .component import Component
from .messenger import SerialMessenger
from .model import Model


class DynamicComponent(Component):
    """  components that broadcast dynamic model information at a given sampling frequency

    component broadcasts and accepts ROSmsg's, as genpy.Message objects

    component is pub/sub architecture, so temporal model information is messaged unto a states and outputs topic. These
    topics follow the naming convention "{component name}-{states|outputs}"

    component has input and state buffers, so it must be run in order to be valid (Model object is stateless)
    """
    def __init__(self,
                 model: Model,
                 topics_input: list,
                 messenger_out: SerialMessenger,
                 messenger_in: SerialMessenger,
                 sampling_frequency: float,
                 name: str,
                 default_output = None):
        # components can have only one output (but many topics)
        super().__init__(len(messenger_in.topics), 1 if len(messenger_out.topics) > 0 else 0)

        # dynamic model
        self._model: Model = model

        # input/output serializers
        self._messenger_out: SerialMessenger = messenger_out
        self._messenger_in: SerialMessenger = messenger_in

        # component attributes
        self._sampling_frequency: float = sampling_frequency
        self._name: str = name

        # topics order to input vector
        self._topics_input: list = topics_input

        # input buffering for mixed rate components
        self._input_buffer: dict = {}
        self.current_time = 0

        # initial state and consequent state buffer
        self._default_output = default_output if default_output is not None else {}
        self._output_buffer = {}
        self.init_state_buffer()

    def init_state_buffer(self):
        # TODO: move to Component class
        for tname in self._messenger_out.topics:
            if tname in self._default_output:
                self._output_buffer[tname] = self._default_output[tname]
            else:
                num_key = self._num_topic(tname, self._messenger_out)
                self._output_buffer[tname] = ([0.0, ] * num_key if num_key > 0 else None)

    def reset(self):
        """reset the time, state and input buffers of component"""
        # input buffering for mixed rate components
        self._input_buffer: dict = {}
        self.current_time = 0
        self.init_state_buffer()

    def receive_input(self):
        """receive all necessary topics for a DynamicComponent"""
        for sidx, sn in enumerate(self.input_socks):
            # message from publishers may or may not be ready (depends on frequency of components)
            topics = []
            recvs = []

            # poll zmq socket and see if message is available
            time.sleep(1e-5)
            while sn.poll(0) == zmq.POLLIN:
                topics.append(sn.recv_string(zmq.DONTWAIT))
                recvs.append(sn.recv_pyobj(zmq.DONTWAIT))

            # is message received, update the input buffer
            if len(topics) > 0:
                topic = topics[-1]
                recv = recvs[-1]
                t, self._input_buffer[topic] = self._messenger_in.deserialize_message(recv, topic, 0.0)
                self._input_buffer['time'] = t

        # avoid infinite recursion by checking whether _input_buffer was initialized
        if len(self._input_buffer.keys()) == 0 and len(self.input_socks) > 0:
            time.sleep(1e-4)
            self.receive_input()
        self.current_time += 1.0 / self.sampling_frequency
        return self._input_buffer

    def send_output(self):
        """send {states, outputs} for DynamicComponent"""
        update_increment = 1.0 / self.sampling_frequency
        current_time = self.current_time + update_increment
        return_buffer = {}

        # obtain the input vector by concatenating the messages together
        input_vector = []
        if len(self._topics_input) > 0:
            for f in self._topics_input:
                print(self.name, self._input_buffer)
                input_vector += self._input_buffer[f]

        # obtain state vector
        state_vector = self._output_buffer[f"{self.name}-states"] if f"{self.name}-states" in self._output_buffer else []

        # iterate through topics and send output
        for tname in self._messenger_out.topics:
            # TODO: getter is a mess
            getter = tname.split('-')[1][:-1]
            getter = getter if getter != "state" else "update"

            # continuous state is a special case
            if not (self._model.is_continuous and getter == "update"):
               return_value = self._model.get(current_time, state_vector, input_vector, getter)
            else:
                state_diff_fcn = lambda y, t: self._model.get_state_update(t, y, input_vector)
                ivp_solution = scipy.integrate.odeint(state_diff_fcn, state_vector, [current_time-update_increment, current_time])
                return_value = list(ivp_solution[-1])
            self._output_buffer[tname] = return_value

            # serialize, send message, and update return buffer
            msg = self._messenger_out.serialize_message(return_value, tname, current_time)
            self.send_message(0, msg, topic=tname)
            return_buffer[tname.split('-')[1]] = return_value

        # default caller -- model state but not component state
        self._model.update_model(current_time, state_vector, input_vector)

        # add time to return buffer
        return_buffer['times'] = current_time

        return return_buffer

    def send_stimulus(self, t: float):
        """send message of components at its current state at time t"""
        state_buffer = self._output_buffer[f"{self.name}-states"] if f"{self.name}-states" in self._output_buffer else []
        u = list(np.zeros((self.num_inputs)))
        for tname in self._messenger_out.topics:
            if tname in self._default_output:
                msg = self._messenger_out.serialize_message(self._output_buffer[tname], tname, t)
                self.send_message(0, msg, topic=tname)
            else:
                getter = tname.split("-")[1][:-1]
                getter = getter if getter != "state" else "update"
                out = self._model.get(t, state_buffer, u, getter)
                msg = self._messenger_out.serialize_message(out, tname, t)
                self.send_message(0, msg, topic=tname)

    def _names_topic(self, topic: str, messenger: SerialMessenger) -> list:
        """generic names getter for messenger"""
        if topic in messenger.topics:
            return messenger.names_topic(topic)
        else:
            return []

    def _num_topic(self, topic: str, messenger: SerialMessenger) -> int:
        """generic lengths getter for messenger"""
        return len(self._names_topic(topic, messenger))

    @property
    def names_states(self):
        return self._names_topic(f'{self.name}-states', self._messenger_out)

    @property
    def names_input(self):
        n = []
        for t in self._topics_input:
            n += self._messenger_in.names_topic(t)
        return n

    @property
    def names_outputs(self):
        return self._names_topic(f'{self.name}-outputs', self._messenger_out)

    @property
    def num_inputs(self):
        return len(self.names_input)

    @property
    def num_states(self):
        return self._num_topic(f'{self.name}-states', self._messenger_out)

    @property
    def num_outputs(self):
        return self._num_topic(f'{self.name}-outputs', self._messenger_out)

    @property
    def topics(self):
        """topics that the component subscribes to"""
        return self._messenger_in.topics

    @property
    def publish_topics(self):
        """topics that the component will publish"""
        return self._messenger_out.topics

    @property
    def sampling_frequency(self):
        """component sampling and update rate"""
        return self._sampling_frequency

    @property
    def sampling_phase(self):
        """time skew of component"""
        return 0.0

    @property
    def name(self):
        return self._name

    @property
    def state(self):
        return self._output_buffer[f"{self.name}-states"]

    @state.setter
    def state(self, state: list):
        assert len(state) == self.num_states, f"state must be array with length {self.num_states} " \
                                              f"(got length {len(state)} instead)"
        self._output_buffer[f"{self.name}-states"] = state
