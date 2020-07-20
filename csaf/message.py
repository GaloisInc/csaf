""" Message Writing

Ethan Lew
07/13/20
"""
import json

class Message:
    """ produces message
    """
    def __init__(self):
        self._version_major = 0
        self._version_minor = 1
        self._epoch = 0

    def write_message(self, current_time, output=None, state=None, differential=None):
        dictionary = {"version_major" : self._version_major,
                      "version_minor" : self._version_minor,
                      "epoch" : self._epoch, "time": current_time}
        if output is not None:
            dictionary["Output"] = list(output)

        if state is not None:
            dictionary["State"] = list(state)

        if differential is not None:
            dictionary["Differential"] = list(differential)

        self._epoch += 1

        return json.dumps(dictionary)

