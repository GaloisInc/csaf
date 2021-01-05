import os
import toml
import pathlib
import typing as typ
from . import csaf_logger
from toposort import toposort_flatten, CircularDependencyError


def mkdir_if_not_exist(dirname):
    """if path doesn't exist, make an empty directory with its name"""
    if os.path.exists(dirname):
        return False
    else:
        pathlib.Path(dirname).mkdir(parents=True, exist_ok=True)
        return True


def join_if_not_abs(*args, project_dir=None, exist=True):
    """if last argument is an absolute path, don't join the path arguments together"""
    if os.path.isabs(args[-1]):
        return args[-1]
    else:
        if project_dir:
            pathname = os.path.join(*args[:-1], project_dir, args[-1])
        else:
            pathname = os.path.join(*args)
        #assert os.path.exists(pathname) or not exist, f"path name {pathname} is required to exist!"
        return pathname


def attempt_parse_toml(fname):
    """ try to parse a TOML file
    TODO: remove print and restructure how toml's are read from disk
    :param fname: filename
    :return: dict or None
    """
    try:
        with open(fname, 'r') as fc:
            return toml.load(fc)
    except Exception as e:
        print(
            f"ERROR! Failed to parse {fname} as TOML configuration file <{e}>")


class ConfigParserMetaDict(dict):
    """resolve overloaded attributes"""
    def __setitem__(self, key, value):
        if hasattr(value, 'field_name_spec'):
            if "_field_name_methods" not in self:
                self["_field_name_methods"] = {}
            self["_field_name_methods"][(getattr(
                value, 'field_name_spec'))[0]] = value, getattr(
                    value, 'field_name_spec')[1]
        else:
            super().__setitem__(key, value)

    def _getitem__(self, key):
        if key not in self and '_' and key.isupper():
            return key.upper()
        else:
            return super().__getitem__(key)


class ConfigParserMeta(type):
    """recognize special decorators, and build out the implied attributes"""
    @classmethod
    def __prepare__(metacls, name, bases):
        def _register(*args, depends_on=()):
            field_name_spec = (*args, depends_on)

            def decorate(func):
                func.field_name_spec = field_name_spec
                return func

            return decorate

        d = ConfigParserMetaDict()
        d["_"] = _register
        return d

    @classmethod
    def _build(cls, attributes):
        pass

    def __new__(meta, clsname, bases, attributes):
        del attributes["_"]
        cls = super().__new__(meta, clsname, bases, attributes)
        return cls


class ConfigParser(metaclass=ConfigParserMeta):
    """parse CSAF config descriptions

    A CSAF config exists as a dictionary,
    {
        field_name0: field_value,
        field_name1: field_value
    }

    Config processing can be implemented by using the "_" decorator

    @_("field_name1", depends_on="field_name0")
    def _(self, field_value):
        assert field_value < MAX_VAL
        return FieldInstance(field_value)

    @_("field_name0")
    def _(self, field_value):
        return field_value
    """
    valid_fields = None

    required_fields = None

    defaults_fields = None

    def __init__(self, base_dir, context_str='', parent_conf = {}):
        self.base_dir = str(pathlib.Path(base_dir).resolve())
        self.context_str = context_str
        self.parent_conf = parent_conf
        if not hasattr(self, "_field_name_methods"):
            self._field_name_methods = {}
        self._config = None

    def logger(self, log_type_name: str, msg: str, error: typ.Type[Exception] = None):
        msg_full = f"{self.context_str + (': ' if self.context_str else '')}{msg}"
        if hasattr(self, "sys_logger"):
            getattr(self.sys_logger, log_type_name)(msg_full)
        else:
            getattr(csaf_logger, log_type_name)(msg_full)
        if error is not None:
            raise error(msg_full)

    def parse(self, cconf: dict) -> dict:
        """parse a dict from a TOML parse"""
        if self.valid_fields is not None:
            for k in cconf.keys():
                if k not in self.valid_fields:
                    self.logger("error",
                                f"found invalid field {k}",
                                error=ValueError)
        if self.required_fields is not None:
            for r in self.required_fields:
                if r not in cconf.keys():
                    self.logger("error",
                                f"unable to find required field {r}",
                                error=ValueError)
        if self.defaults_fields is not None:
            for d, dval in self.defaults_fields.items():
                if d not in cconf:
                    cconf[d] = dval
        self._config = cconf
        return self._process_node(self._config)

    def _process_node(self, conf: typ.Any) -> typ.Any:
        """apply field name methods to nodes"""
        if not isinstance(conf, dict):
            return conf
        for e in self.eval_order:
            if e in conf:
                if e in self._field_name_methods:
                    conf[e] = self._process_node(
                        self._field_name_methods[e][0](self, conf[e]))
                else:
                    conf[e] = self._process_node(conf[e])
        return conf

    def __getattr__(self, name: str):
        if name in self.valid_fields:
            return self._config[name]
        return self.__getattribute__(name)

    @property
    def eval_order(self) -> list:
        dep_graph = {}
        for k, v in self._field_name_methods.items():
            assert isinstance(v[1], tuple)
            dep_graph[k] = set(v[1])
        try:
            return toposort_flatten(dep_graph)
        except CircularDependencyError as exc:
            self.logger(
                "error",
                f"DEV ERROR: configuration parser has circular dependency <{exc}>",
                error=CircularDependencyError)

