"""
Simple System Components

Ethan Lew
06/17/2020
"""

import sys
import pexpect
import numpy as np
import logging

from csaf.component import Component


def dict_to_vec(dict, names):
    """convenience to infer vector from dictionary structure"""
    return np.array([dict[n] for n in names])


def vec_to_dict(vec, names):
    """convenience to infer dictionary from vector structure"""
    assert len(vec) == len(names)
    return {n:v for n,v in zip(names, vec)}


class DynamicalSystem(Component):
    def __init__(self, command, names_inputs, names_outputs, names_states, name=None):
        def get_io_socket(s):
            """get number of necessary sockets """
            if len(s) > 0:
                if type(s[0]) is not str:
                    num_socks = len(names_inputs)
                else:
                    num_socks = 1
            else:
                num_socks = 0
            return num_socks

        num_in_socks, num_out_socks = get_io_socket(names_inputs), get_io_socket(names_outputs)
        super().__init__(num_in_socks, num_out_socks, name=name)
        self._names_inputs = names_inputs
        self._names_outputs = names_outputs
        self._names_states = names_states

        self.command = command
        self.comp_process = None
        self.prompt = r'msg(.*)>'

        # for the ROSmsg serializer
        self.serializer = None

    def subscriber_thread(self, stop_event=None, sock_num=None):
        """DynamicalSystem expects that component interfaces as an interactive prompt
        """
        def debug_start(sdx):
            return f"Component '{self.name}' {self.__class__.__name__} Socket {sdx}"

        def print_if_debug(s):
            if self.debug_node:
                logging.debug(s)
        print_if_debug(f"{debug_start(str([i for i in range(self.num_input_socks)]))} Initialized", )

        # spawn a pexpect session to interact with middleware app
        self.comp_process = pexpect.spawn(self.command)
        self.comp_process.expect(self.prompt)

        while (not stop_event.is_set()):
            for sidx, sn in enumerate(self.input_socks):
                # expect to receive one message per socker per epoch
                topic = sn.recv().decode()
                recv = sn.recv()
                print_if_debug(f"{debug_start(sidx)} Received {self._n_subscribe[sock_num]} Topic '{topic}' Message <{self.deserialize(recv)}>")
                mesg = self.deserialize(recv)

                # check message contents
                assert 'Output' in mesg
                assert 'epoch' in mesg

                # send message and skip over output
                self.comp_process.sendline(self.serialize(mesg))
                self.comp_process.expect('\r\n')

                # update subscribe receives
                self._n_subscribe[sidx] += 1

            # now expect message output, and send it
            self.comp_process.expect(self.prompt)
            recv = self.comp_process.before
            recv_msg = self.deserialize(recv)

            if 'State' in recv_msg:
                recv_msg_state = recv_msg.copy()
                recv_msg_state['Output'] = recv_msg_state['State']
                del recv_msg_state['State']
                self.send_message(0, recv_msg_state, topic=self.name + "-" +'states')
                del recv_msg['State']
            if 'Output' in recv_msg:
                self.send_message(0, recv_msg, topic=self.name + "-" +'outputs')

        # quit
        print_if_debug(f"{debug_start()} received kill event. Exiting...")
        sys.exit()

    @property
    def num_inputs(self):
        return len(self._names_inputs)

    @property
    def num_states(self):
        return len(self._names_states)

    @property
    def num_outputs(self):
        return len(self._names_outputs)
