import numpy as np
import typing

from .component import Component
from .messenger import SerialMessenger
from .model import Model


class DynamicComponent(Component):
    def __init__(self,
                 model: Model,
                 topics_input: list,
                 messenger_out: SerialMessenger,
                 messenger_in: SerialMessenger,
                 sampling_frequency: float,
                 name: str):
        super().__init__(len(messenger_in.topics), 1)
        self._model: Model = model
        self._messenger_out: SerialMessenger = messenger_out
        self._messenger_in: SerialMessenger = messenger_in
        self._sampling_frequency: float = sampling_frequency
        self._name: str = name
        self._topics_input: list = topics_input
        self._state: typing.Union[None, list] = ([0.0,] * self.num_states if self.num_states > 0 else None)

    def receive_input(self):
        """receive all necessary topics for a DynamicComponent"""
        col = {}
        topics_collect = self.topics.copy()
        while len(topics_collect) > 0:
            for sidx, sn in enumerate(self.input_socks):
                topic = sn.recv_string()
                recv = sn.recv_pyobj()
                t, col[topic] = self._messenger_in.receive_message(recv, topic, 0.0)
                assert topic in topics_collect
                topics_collect.remove(topic)
        col['time'] = t
        return col

    def send_output(self, input):
        """send {states, outputs} for DynamicComponent"""
        t = input['time']

        if len(self._topics_input) > 0:
            u = list(np.concatenate([input[f] for f in self._topics_input]))
        else:
            u = []

        if self._state is not None:
            xp = self._model.get_state_update(t, self._state, u)
            assert len(xp) == self.num_states
            if self._model.is_continuous:
                xp = list(np.array(self._state) + (1/self._sampling_frequency) * np.array(xp))
            self._state = xp
            msg = self._messenger_out.send_message(xp, f'{self.name}-states', t)
            self.send_message(0, msg, topic=f"{self.name}-states")
        else:
            xp = []

        if f'{self.name}-outputs' in self._messenger_out.topics:
            out = self._model.get_output(t, xp, u)
            msg = self._messenger_out.send_message(out, f'{self.name}-outputs', t)
            self.send_message(0, msg, topic=f"{self.name}-outputs")

    def send_stimulus(self, t):
        u = list(np.zeros((self.num_inputs)))

        if self._state is not None:
            msg = self._messenger_out.send_message(self._state, f'{self.name}-states', t)
            self.send_message(0, msg, topic=f"{self.name}-states")

        if f'{self.name}-outputs' in self._messenger_out.topics:
            out = self._model.get_output(t, self._state, u)
            msg = self._messenger_out.send_message(out, f'{self.name}-outputs', t)
            self.send_message(0, msg, topic=f"{self.name}-outputs")

    def _names_topic(self, topic, messenger):
        """generic names getter for messenger"""
        if topic in messenger.topics:
            return messenger.names_topic(topic)
        else:
            return []

    def _num_topic(self, topic, messenger):
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
        return self._messenger_in.topics

    @property
    def sampling_frequency(self):
        return self._sampling_frequency

    @property
    def sampling_phase(self):
        return 0.0

    @property
    def name(self):
        return self._name

    @property
    def state(self):
        return self._state

    @state.setter
    def state(self, state):
        assert len(state) == self.num_states
        self._state = state

