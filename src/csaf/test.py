import abc
from .system import System
from . import csaf_logger


class Test(abc.ABC):
    """Abstract Test Interface

    A test is an object that accepts a static configuration and
    a CSAF system config. The test implements some execution
    action on the system with respect to parameters defined in
    the config.
    """
    required_fields = []

    defaults_fields = {}

    def __init__(self, test_config: dict, trajs=None):
        self._test_params = {}
        self._trajs = trajs
        self.logger = csaf_logger
        for rf in self.required_fields:
            if rf not in test_config:
                if rf in self.defaults_fields:
                    self._test_params[rf] = self.defaults_fields[rf]
                else:
                    raise ValueError(f"<{self.__class__.__name__}>: required field {rf} not supplied!")
            else:
                self._test_params[rf] = test_config[rf]

    @abc.abstractmethod
    def execute(self):
        pass

    def __getattr__(self, name):
        if name in self._test_params:
            return self._test_params[name]
        else:
            raise AttributeError(f"{self.__class__.__name__} has no attribute {name}")


class SystemTest(Test):
    def __init__(self, test_config, system_config, **kwargs):
        super().__init__(test_config, **kwargs)
        self.system_config = system_config


class StaticTest(Test):
    def __init__(self, test_config, trajs, **kwargs):
        super().__init__(test_config, **kwargs)
        self.traces = trajs
