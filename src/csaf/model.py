""" Model Interfaces and Wrappers
"""
import collections.abc as cabc
import abc
import functools
import typing as typ
import keyword

import os, sys
import importlib
import pathlib
import subprocess as subproc
import json
import toml
from inspect import signature
from . import csaf_logger as logging


def dynamical_input(func: typ.Callable):
    """asserts that the input passed to a callable matches the signature of a dynamical system, being a 4-tuple
    (m, t: float, x: typ.Sized, u: typ.Sized), being
    m: (Model) model with parameter attributes
    t: (float) current time
    x: (vector like) state vector
    u: (vector like) input vector
    """
    @functools.wraps(func)
    def check_input(*args, **kwargs):
        assert len(args) == 4, f"got length {len(args)}"
        assert isinstance(args[0], Model), f"argument 0 must be a model in dynamical function {func.__name__}"
        assert isinstance(args[1], float), f"argument 1 must be a time value in dynamical function {func.__name__}"
        assert isinstance(args[2], cabc.Sized), f"argument 2 must be sized in dynamical function {func.__name__}"
        assert isinstance(args[3], cabc.Sized), f"argument 3 must be sized in dynamical function {func.__name__}"
        assert kwargs == {}, f"no keyword arguments are permitted in dynamical function {func.__name__}"
        return func(*args, **kwargs)
    return check_input


class Model(abc.ABC):
    """CSAF Model Interface

    A CSAF Model encapsulates models seen in dynamical system theory. It is initialized with parameters, a
    representation, and whether the system time is continuous or discrete. Users implemented methods starting with an
    underscore. Their corresponding interface callables have additional input/output assertions for safety.
        1. If the state vector dimension is greater than zero, the method _get_state_update must be implemented,
        2. If the output message is non-empty, the method _get_output must be implemented
        3. _get_info is available for representation specific calls
        4. _update_model is available for updating entities that are NOT relevant to the dynamic model
        5. parameters are initialized by default but also settable
    """
    dynamic_callables: typ.Sequence[str] = ["get_output", "get_state_update", "get_info", "update_model"]

    def __init__(self, parameters, representation, is_discrete):
        self._parameters: typ.Mapping =  parameters
        self._representation: str = representation
        self._is_discrete: bool = is_discrete

    @dynamical_input
    def get_output(self, t: float, x: typ.Sized, u: typ.Sized) -> typ.Sized:
        """returns system output"""
        return self._get_output(t, x, u)

    @dynamical_input
    def get_state_update(self, t: float, x: typ.Sized, u: typ.Sized) -> typ.Sized:
        """returns system state update"""
        xp = self._get_state_update(t, x, u)
        assert len(xp) == len(x), f"state update dimension must equal the dimension of the state (update is {len(xp)}, " \
                        f"but state is {len(x)})"
        return xp

    @dynamical_input
    def get_info(self, t: float, x: typ.Sized, u: typ.Sized) -> typ.Any:
        """returns representation specific information"""
        return self._get_info(t, x, u)

    @dynamical_input
    def update_model(self, t: float, x: typ.Sized, u: typ.Sized) -> None:
        """update attributes in the Model, but not the state x"""
        return self._update_model(t, x, u)

    @abc.abstractmethod
    def _get_output(self, t: float, x: typ.Sized, u: typ.Sized) -> typ.Sized:
        """user implemented"""
        raise NotImplementedError

    @abc.abstractmethod
    def _get_state_update(self, t: float, x: typ.Sized, u: typ.Sized) -> typ.Sized:
        """user implemented"""
        raise NotImplementedError

    @abc.abstractmethod
    def _get_info(self, t: float, x: typ.Sized, u: typ.Sized) -> typ.Any:
        """user implemented"""
        raise NotImplementedError

    @abc.abstractmethod
    def _update_model(self, t: float, x: typ.Sized, u: typ.Sized) -> None:
        """user implemented"""
        raise NotImplementedError

    @property
    def representation(self) -> str:
        return self._representation

    @property
    def is_discrete(self) -> bool:
        return self._is_discrete

    @property
    def is_continuous(self) -> bool:
        return not self._is_discrete

    @property
    def parameters(self) -> typ.Mapping:
        return self._parameters

    @parameters.setter
    def parameters(self, parameters):
        assert set(parameters.keys()) == set(self._parameters.keys())
        self._parameters = parameters

    def get(self, t, x, u, getter):
        mp = {"update" : "get_state_update", "output": "get_output", None: "get_info"}
        f = getattr(self, mp.setdefault(getter, "get_info"))
        return f(t, x, u)

    def __getattr__(self, item) -> typ.Any:
        """allow parameters to be accessed at object level"""
        # error case - item is a keyword
        item = item[:-1] if (item[-1] == '_' and (keyword.iskeyword(item[:-1]) or hasattr(self, item[:-1]))) else item
        if item in self.parameters:
            return self.parameters[item]
        else:
            raise AttributeError(f"{item} not found as a parameter")


class ModelNative(Model):
    """ ModelNative

    For models written in an environment compatible python, interact with their main function
    natively.
    """
    user_functions: typ.Sequence[str] = ["model_output", "model_state_update", "model_info", "model_update"]

    @classmethod
    def from_filename(cls, mname:str, cname: str):
        """given a configuration file, load in a Model object with correct members"""
        assert os.path.exists(cname)
        with open(cname, 'r') as fp:
            info = toml.load(fp)
        return cls.from_config(mname, info)

    @classmethod
    def from_config(cls, mname:str, config):
        parameters = config["parameters"]
        is_discrete = config["is_discrete"]
        representation = config["system_representation"]
        return cls(mname, parameters, representation, is_discrete)

    def __init__(self, mname: str, *args, **kwargs):
        super().__init__(*args, **kwargs)

        assert os.path.exists(mname)
        # update python path to include module directory
        mod_path = str(pathlib.Path(mname).parent.resolve())
        if mod_path not in sys.path:
            sys.path.insert(0, mod_path)

        spec = importlib.util.spec_from_file_location(pathlib.Path(mname).stem, mname)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        # relate user defined functions to the model interface
        self.funcs = {}
        for iface, uface in zip(self.dynamic_callables, self.user_functions):
            if hasattr(module, uface):
                ufunc = getattr(module, uface)
                # check the input length
                sig = signature(ufunc)
                assert len(sig.parameters.keys()) == 4, f"user defined function {uface} must have 4 arguments"
                self.funcs[iface] = ufunc
            else:
                self.funcs[iface] = lambda m, t, x, u: m.null_dynamics(t, x, u)

    def _get_output(self, *args) -> typ.Sized:
        return self.funcs["get_output"](self, *args)

    def _get_state_update(self, *args) -> typ.Sized:
        return self.funcs["get_state_update"](self, *args)

    def _get_info(self, *args) -> typ.Any:
        return self.funcs["get_info"](self, *args)

    def _update_model(self, *args) -> None:
        return self.funcs["update_model"](self, *args)

    @dynamical_input
    def null_dynamics(self, t: float, x: typ.Sized, u: typ.Sized) -> list:
        return []








class ModelJsonRpc(Model):
    """ ModelJsonRpc

    For external models that communicate via a simple JSON RPC over stdin/stdout:
    send:
        {"function" := "model_output" OR "model_state_update" OR "model_info" OR "model_update",
        "model" := parameter dictionary,
        "time" := float,
        "state" := list or object with `tolist()` method,
        "input" := list}
    receive:
        JSON

    N.B., the component spec must indicate where the executable/script is located (i.e., the `process`
    field in the TOML config) and what command to run it with if necessary (i.e., the optional `run_command`
    field in the components TOML config)
    """
    user_functions: typ.Sequence[str] = ["model_output", "model_state_update", "model_info", "model_update"]

    @classmethod
    def from_filename(cls, mname:str, cname: str):
        """given a configuration file, load in a Model object with correct members"""
        assert os.path.exists(cname)
        with open(cname, 'r') as fp:
            info = toml.load(fp)
        return cls.from_config(info.get('run_command'), mname, info)

    @classmethod
    def from_config(cls, opt_run_cmd:typ.Union[str, None], mname:str, config):
        parameters = config["parameters"]
        is_discrete = config["is_discrete"]
        representation = config["system_representation"]
        return cls(opt_run_cmd, mname, parameters, representation, is_discrete)

    def __init__(self, opt_run_cmd:typ.Union[str, None], mname: str, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._model_name = mname
        assert os.path.exists(mname)
        # initialize the external model process
        logging.info("Starting external JsonRpc model " + mname + "...")

        if opt_run_cmd == None:
            cmd = [str(pathlib.Path(mname).absolute())]
        else:
            cmd = [str(pathlib.Path(opt_run_cmd).absolute()),str(pathlib.Path(mname).absolute())]

        self._model_proc = subproc.Popen(\
            cmd,\
            stdin=subproc.PIPE,\
            stdout=subproc.PIPE,\
            stderr=subproc.STDOUT,\
            universal_newlines=True) # opens the pipes in text instead of binary mode


    def _build_model(self) -> typ.Dict:
        """Extract the model's explicit model parameters from the object into a dictionary."""
        model = {}
        for key, val in self.parameters.items():
            model[key] = val
        return model

    def _build_state(self, state) -> typ.Dict:
        if isinstance(state, list):
            return state
        else:
            return state.tolist()

    def _do_rpc(self, fn_name, time, state, input) -> typ.Any:
        """Send a JSON RPC to the external process."""
        model = self._build_model()
        state = self._build_state(state)
        rpc = {                 \
            'function':fn_name, \
            'model':model,      \
            'time':time,        \
            'state':state, \
            'input':input}
        rpc = json.dumps(rpc).replace('\n', ' ').replace('\r', '')
        # logging.debug("Sending RPC to " + self._model_name + ":\n" + rpc)
        self._model_proc.stdin.write(rpc)
        self._model_proc.stdin.write('\n')
        self._model_proc.stdin.flush()
        # logging.debug("Waiting on RPC response..." )
        rawResponse = self._model_proc.stdout.readline().strip()
        # logging.debug("RPC response: " + rawResponse)
        return json.loads(rawResponse)

    def _get_output(self, time, state, input) -> typ.Sized:
        return self._do_rpc("model_output", time, state, input)

    def _get_state_update(self, time, state, input) -> typ.Sized:
        return self._do_rpc("model_state_update", time, state, input)

    def _get_info(self, time, state, input) -> typ.Any:
        return self._do_rpc("model_info", time, state, input)

    def _update_model(self, time, state, input) -> None:
        self._do_rpc("model_update", time, state, input)
        ## FIXME this method should probably update this objects state appropriately
        return None

    @dynamical_input
    def null_dynamics(self, t: float, x: typ.Sized, u: typ.Sized) -> list:
        return []
