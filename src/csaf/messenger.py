""" Component Serializer Interfaces
"""
import io


class SerialMessenger:
    """topic based serializer/deserializer pairs to be passed to components"""
    @staticmethod
    def header(topic, time=0.0):
        """csaf ROSmsg header"""
        return [('version_major', 0), ('version_minor', 1), ('topic', topic), ('time', time)]

    def __init__(self, serializers):
        self._serializers = serializers

    def receive_message(self, msg_str, topic, time):
        """receive a message of a certain topic related to a time"""
        assert topic in self._serializers
        nslice = len(self.header(topic, time))
        ser = self._serializers[topic]
        serializer_state = ser()
        serializer_state.deserialize(msg_str)
        sret = [getattr(serializer_state, name) for name in serializer_state.__slots__]
        return sret[nslice - 1], sret[nslice:]

    def send_message(self, msg, topic, time):
        """send a message of a certain topic at a specific time"""
        assert topic in self._serializers
        ser = self._serializers[topic]
        buf = io.BytesIO()
        _, hvalues = list(zip(*self.header(topic, time)))
        msg_total = list(hvalues) + list(msg)
        state = ser(*msg_total)
        assert (len(msg_total)) == len(state.__slots__)
        state.serialize(buf)
        return buf.getvalue()

    def names_topic(self, topic):
        assert topic in self._serializers
        nslice = len(self.header(topic))
        return self._serializers[topic]().__slots__[nslice:]

    def num_topics(self, topic):
        return len(self.names_topic(topic))

    @property
    def topics(self):
        return list(self._serializers.keys())

