import numpy as np
import time
import typing
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
                 name: str):
        # components can have only one output (but many topics)
        super().__init__(len(messenger_in.topics), 1)
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
        self._state_buffer: typing.Union[None, list] = ([0.0, ] * self.num_states if self.num_states > 0 else None)

    def reset(self):
        """reset the time, state and input buffers of component"""
        # input buffering for mixed rate components
        self._input_buffer: dict = {}
        self.current_time = 0
        # initial state and consequent state buffer
        self._state_buffer: typing.Union[None, list] = ([0.0, ] * self.num_states if self.num_states > 0 else None)

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
        t = self.current_time
        dt = 1.0 / self.sampling_frequency
        dout = {}

        # obtain the input vector by concatenating the messages together
        if len(self._topics_input) > 0:
            u = []
            for f in self._topics_input:
                u += self._input_buffer[f]
        else:
            u = []

        if self._state_buffer is not None:
            if self._model.is_discrete:
                xp = self._model.get_state_update(t, self._state_buffer, u)
                assert len(xp) == self.num_states
            else:
                df = lambda y, t: self._model.get_state_update(t, y, u)
                sol = scipy.integrate.odeint(df, self._state_buffer, [t, t+dt])
                xp = list(sol[-1])
                #df = lambda t, y: self._model.get_state_update(t, y, u)
                #sol = scipy.integrate.solve_ivp(df, [t, t+dt+1E-3], np.array(self._state_buffer), method='RK45', t_eval=[t+dt])
                #xp = list(sol.y.flatten())
            self._state_buffer = xp
            msg = self._messenger_out.serialize_message(xp, f'{self.name}-states', t)
            self.send_message(0, msg, topic=f"{self.name}-states")
            dout['states'] = xp
        else:
            xp = []

        for ptopic in self._messenger_out.topics:
            if ptopic != f"{self.name}-states":
                _, field = ptopic.split('-')
                out = self._model.get(t, xp, u, field[:-1])
                msg = self._messenger_out.serialize_message(out, ptopic, t)
                self.send_message(0, msg, topic=ptopic)
                dout[field] = out
        dout['times'] = t
        return dout

    def send_stimulus(self, t: float):
        """send message of components at its current state at time t"""
        u = list(np.zeros((self.num_inputs)))

        if self._state_buffer is not None:
            msg = self._messenger_out.serialize_message(self._state_buffer, f'{self.name}-states', t)
            self.send_message(0, msg, topic=f"{self.name}-states")

        for ptopic in self._messenger_out.topics:
            if ptopic != f"{self.name}-states":
                _, field = ptopic.split('-')
                print(self.name, field)
                out = self._model.get(t, self._state_buffer, u, field[:-1])
                msg = self._messenger_out.serialize_message(out, ptopic, t)
                self.send_message(0, msg, topic=ptopic)

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
        return self._state_buffer

    @state.setter
    def state(self, state: list):
        assert len(state) == self.num_states, f"state must be array with length {self.num_states} " \
                                              f"(got length {len(state)} instead)"
        self._state_buffer = state
