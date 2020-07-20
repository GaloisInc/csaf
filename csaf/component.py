""" Simple Pub/Sub Component

Ethan Lew
06/17/2020

component models a simple class that contains multiple subscribers/publishers to represent a component
"""

import time
import json
import sys
import threading
import logging

import zmq

class Component:
    """ Represent a Pub/Sub Component

    Can have publishers and subscribers, with serialization/deserialization to encode/decode
    information.
    """
    @staticmethod
    def serialize(data_dict):
        """component serialization implementation"""
        return str.encode(json.dumps(data_dict))

    @staticmethod
    def deserialize(data_string):
        """component deserialization implementation"""
        return json.loads(data_string.decode())

    def __init__(self, num_inputs, num_outputs, name=None):
        # zeroMQ members
        self.zmq_context = None
        self.input_socks = None
        self.output_socks = None

        # threads to manage subscribers over
        self.subscriber_threads = None
        self.stop_threads = None

        # pub/sub count parameters
        self._num_inputs = num_inputs
        self._num_outputs = num_outputs

        # debug flag
        self._debug = False
        self._n_publish = [0] * num_outputs
        self._n_subscribe = [0] * num_inputs

        self.name = "system" if name is None else name

    def bind(self, in_ports, out_ports):
        """bind subscribers/publishers to their respective ports"""

        self.stop_threads = threading.Event()

        if self.zmq_context is not None:
            self.unbind()

        self.zmq_context = zmq.Context()

        assert len(out_ports) == self.num_output_socks
        assert len(in_ports) == self.num_input_socks

        # setup publishers over TCP network
        # TODO: uses TCP -- maybe make configurable for zeroMQ's various transport options
        self.output_socks = []
        for idx in range(self.num_output_socks):
            sock = self.zmq_context.socket(zmq.PUB)
            sock.bind(f"tcp://*:{out_ports[idx]}")
            self.output_socks.append(sock)

        # setup subscribers over TCP network
        self.input_socks = []
        for idx in range(self.num_input_socks):
            sock = self.zmq_context.socket(zmq.SUB)
            sock.connect(f"tcp://127.0.0.1:{in_ports[idx]}")
            sock.subscribe("")
            self.input_socks.append(sock)

        # TODO: needed for some reason -- better way to make sure subscribers are ready?
        time.sleep(0.2)

    def unbind(self):
        """unbind from ports and destroy zmq context"""

        if self.stop_threads is not None:
            self.stop_threads.clear()

        def close_socks(socks):
            for sock in socks:
                sock.close()

        if self.input_socks is not None:
            close_socks(self.input_socks)

        if self.output_socks is not None:
            close_socks(self.output_socks)

        if self.zmq_context is not None:
            self.zmq_context.term()

    def send_message(self, output_idx, message):
        """send a message over output number output_idx """
        if self.debug_node:
            logging.debug(f"Component '{self.name}' {self.__class__.__name__} Socket {output_idx} Sending "
                  f"{self._n_publish[output_idx]} Message <{message}>")
        self.output_socks[output_idx].send(self.serialize(message))
        self._n_publish[output_idx] += 1

    def subscriber_iteration(self, recv, sock_num):
        pass

    def subscriber_thread(self, stop_event=None, sock_num=None):
        def debug_start():
            return f"Component '{self.name}' {self.__class__.__name__} Socket {sock_num}"
        def print_if_debug(s):
            if self.debug_node:
                logging.debug(s)
        print_if_debug(f"{debug_start()} Initialized")
        while (not stop_event.is_set()):
            recv = self.input_socks[sock_num].recv()
            print_if_debug(f"{debug_start()} Received {self._n_subscribe[sock_num]} Message <{self.deserialize(recv)}>")
            self.subscriber_iteration(recv, sock_num)
            self._n_subscribe[sock_num] += 1
        print_if_debug(f"{debug_start()} received kill event. Exiting...")
        sys.exit()

    def activate_subscribers(self):
        """launch all subscribers as threads"""

        # populate receivers as threads
        threads = []
        for tidx in range(self.num_input_socks):
            threads.append(threading.Thread(target=self.subscriber_thread, kwargs={'stop_event': self.stop_threads, 'sock_num': tidx}))
            threads[tidx].daemon = True
            threads[tidx].start()

    @property
    def num_input_socks(self):
        return self._num_inputs

    @property
    def num_output_socks(self):
        return self._num_outputs

    @property
    def debug_node(self):
        return self._debug

    @debug_node.setter
    def debug_node(self, s):
        assert type(s) is bool, f"setting debug_node must be boolean type (got {type(s)})"
        self._debug = s


if __name__ == "__main__":
    # simple echo loopback test
    compa = Component(0, 1)
    compb = Component(2, 0)
    compa.bind([], ['1234'])
    compb.bind(['1234', '1234'], [])
    compb.activate_subscribers()
    while True:
        mesg = input("Message to Send:")
        compa.send_message(0, mesg)
        # TODO ugh
        time.sleep(0.2)
    compa.unbind()
    compb.unbind()

