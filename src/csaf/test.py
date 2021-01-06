import abc
from .parser import ConfigParser
from .system import SystemConfig


class Test(ConfigParser):
    """Abstract Test Interface

    A test is an object that accepts a static configuration and
    a CSAF system config. The test implements some execution
    action on the system with respect to parameters defined in
    the config.
    """
    def execute(self, system_config: SystemConfig):
        pass


class SystemTest(Test):
    pass


class StaticTest(Test):
    pass
