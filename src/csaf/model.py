""" Model Interfaces and Wrappers
"""
import os
import abc
import subprocess
import pathlib
import importlib


def subprocess_check_output(command=()):
    """convenience to run command and return its output
    TODO: this is slow, reimplement to be fast and with better error handling
    """
    proc = subprocess.Popen(command, stdout=subprocess.PIPE)
    out = proc.stdout.read()
    return out


class Model(abc.ABC):
    """Model Abstract Base Class

    Introduces an interface to get output, get state update output,
    and whether the model operates in discrete or continuous time
    """
    def __init__(self, is_discrete: bool = True):
        self._is_discrete = is_discrete

    @abc.abstractmethod
    def get_output(self, t: float, x, u):
        raise NotImplementedError

    @abc.abstractmethod
    def get_state_update(self, t: float, x, u):
        raise NotImplementedError

    @property
    def is_discrete(self):
        return self._is_discrete

    @property
    def is_continuous(self):
        return not self._is_discrete


class ModelNative(Model):
    """ ModelNative

    For models written in an environment compatible python, interact with their main function
    natively.

    Makes assumptions about the function signature:

    output: main(time=t, state=x, input=u, output=True)
    state update: main(time=t, state=x, input=u, update=True)
    """
    @classmethod
    def from_filename(cls, mname: str, fname: str = "main", **kwargs):
        """given a executable filename and run environment, produce a ModelExecutable"""
        assert os.path.exists(mname)
        spec = importlib.util.spec_from_file_location(pathlib.Path(mname).stem, mname)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        func = getattr(module, fname)
        return cls(func, **kwargs)

    def __init__(self, func, **kwargs):
        super().__init__(**kwargs)
        self._func = func

    def get_output(self, t: float, x, u):
        """executable output implementation"""
        return self._func(time=t, state=x, input=u, output=True)

    def get_state_update(self, t: float, x, u):
        """state update implementation"""
        return self._func(time=t, state=x, input=u, update=True)


class ModelExecutable(Model):
    """ relate Model interface to an executable file

    makes assumptions about the executable interface

    output: env executable --time <t> --state <state> --input <in> --output
    state update: env executable --time <t> --state <state> --input <in> --update
    """
    @classmethod
    def from_filename(cls, fname: str, environment=None, **kwargs):
        """given a executable filename and run environment, produce a ModelExecutable"""
        os.path.exists(fname)
        command = ([environment] if environment else []) + [fname]
        return cls(command, **kwargs)

    def __init__(self, command: list, **kwargs):
        super().__init__(**kwargs)
        self._command = command

    def _run_command(self, t: float, x, u, trigger):
        """run model command with a given output trigger"""
        x = [] if x is None else x
        arguments = ["--time", str(t)] + \
                    (["--state", '[' + ','.join([str(xi) for xi in x]) + ']'] if len(x) > 0 else []) + \
                    (["--input", '[' +','.join([str(ui) for ui in u])+']'] if len(u) > 0 else [])
        arguments += [trigger]
        run_command = self._command + arguments
        out = subprocess_check_output(run_command)
        return [float(s) for s in out.decode().splitlines()]

    def get_output(self, t: float, x, u):
        """executable output implementation"""
        return self._run_command(t, x, u, "--output")

    def get_state_update(self, t: float, x, u):
        """state update implementation"""
        return self._run_command(t, x, u, "--update")
