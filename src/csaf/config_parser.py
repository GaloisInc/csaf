import os
import toml
import pathlib
import os
import logging
import toml
import typing as typ
from . import csaf_logger, has_global_config
from toposort import toposort_flatten, CircularDependencyError

from .rosmsg import CsafMsg, generate_serializer
from . import csaf_logger, has_global_config


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

    def __init__(self, base_dir, context_str='', parent_conf={}):
        self.base_dir = str(pathlib.Path(base_dir).resolve())
        self.context_str = context_str
        self.parent_conf = parent_conf

    def logger(self, log_type_name: str, msg: str, error: Exception = None):
        msg_full = f"{self.context_str + (': ' if self.context_str else '')}{msg}"
        if hasattr(self, "sys_logger"):
            getattr(self.sys_logger, log_type_name)(msg_full)
        else:
            getattr(csaf_logger, log_type_name)(msg_full)
        if error is not None:
            raise error(msg_full)

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


class SystemParser(ConfigParser):
    """Parser for CSAF System Descriptions"""
    valid_fields = [
        "codec_dir", "output_dir", "name", "log_level", "log",
        "evaluation_order", "components", "inputs", "topics",
        "sampling_frequency", "sampling_phase"
    ]

    required_fields = ["evaluation_order", "components", "name", "log"]

    defaults_fields = {
        "codec_dir": "codec",
        "output_dir": "output",
        "log_level": "info",
        "inputs": {},
        "topics": {},
        "sampling_frequency": 100.0,
        "sampling_phase": 0.0,
    }

    @_("log", depends_on=("output_dir", "log_level", "name"))
    def _(self, log_path):
        global has_global_config
        log_level = self._config["log_level"]
        logpath = join_if_not_abs(self._config["output_dir"],
                                       log_path,
                                       exist=False)
        self.sys_logger = logging.getLogger('log-test')

        formatter = logging.Formatter(
            f'%(asctime)s: (%(levelname)s)  %(message)s {self._config["name"]}',
            datefmt='%I:%M:%S %p')

        # reflect user specified log level as logger level
        self.sys_logger.setLevel(log_level)

        # setup file logging -- accept log level
        open(logpath, 'w').close()
        fh = logging.FileHandler(logpath)
        fh.setFormatter(formatter)
        fh.setLevel(log_level)

        # add the two handlers to csaf logging
        self.sys_logger.addHandler(fh)

        # print config paths
        self.sys_logger.info(f"setting up CSAF System")
        self.sys_logger.info(f"Output Dir: {self._config['output_dir']}")
        self.sys_logger.info(f"Codec Dir: {self._config['codec_dir']}")
        self.sys_logger.info(f"Log Level: {self._config['log_level']}")
        return self.sys_logger

    @_("log_level")
    def _(self, level: str) -> int:
        llevel = {
            "critical": logging.CRITICAL,
            "error": logging.ERROR,
            "warning": logging.WARNING,
            "info": logging.INFO,
            "debug": logging.DEBUG,
            "notset": logging.NOTSET
        }
        if level in llevel:
            return llevel[level]
        else:
            raise AssertionError(
                f"log level '{level}' found in config is not valid!")

    @_("codec_dir")
    def _(self, codec_path: str) -> str:
        dpath = join_if_not_abs(self.base_dir, codec_path)
        path = str(pathlib.Path(dpath).resolve())
        outcome = mkdir_if_not_exist(dpath)
        if outcome:
            self.logger = self.logger(
                "info",
                f"created codec directory {dpath} because it did not exist")
        return path

    @_("output_dir")
    def _(self, output_path: str) -> str:
        dpath = join_if_not_abs(self.base_dir, output_path)
        path = str(pathlib.Path(dpath).resolve())
        outcome = mkdir_if_not_exist(dpath)
        if outcome and self.logger:
            self.logger = self.logger(
                "info",
                f"created output directory {dpath} because it did not exist")
        return path

    @_("components", depends_on=("codec_dir", "output_dir"))
    def _(self, components: dict):
        """TODO: implement this with a component parser"""
        for cname, cconf in components.items():
            if not isinstance(cconf, dict):
                self.logger("error",
                            f"{cname} field is not a component map",
                            error=ValueError)
            cpr = ComponentParser(self.base_dir,
                                  context_str=self.context_str + f"<{cname}>")
            components[cname] = cpr.parse(cconf)
        return components

    @_("evaluation_order", depends_on=("components", "inputs"))
    def _(self, eval_order: list) -> list:
        return eval_order

    @_("inputs")
    def _(self, iconf: dict) -> dict:
        msgs = iconf.get("msgs", [])
        iconf["msgs"] = [self.load_msg(m) for m in msgs]
        if "identifiers" not in iconf:
            iconf["identifiers"] = []
        return iconf

    @_("topics")
    def _(self, tconf: dict) -> dict:
        for tval, tc in tconf.items():
            msg = tc.get("msg", None)
            dbase_dir = os.path.join(self.base_dir, "components")
            msg_path = join_if_not_abs(dbase_dir, msg, project_dir="msg")
            tconf[tval]["msg"] = self.load_msg(
                msg) if msg is not None else None
            tconf[tval]["_msg_path"] = msg_path
        return tconf

    def load_msg(self, msg: str) -> CsafMsg:
        dbase_dir = os.path.join(self.base_dir, "components")
        msg_path = join_if_not_abs(dbase_dir, msg, project_dir="msg")
        return CsafMsg.from_msg_file(msg_path)


class ComponentParser(ConfigParser):
    valid_fields = ["type", "run_command", "process", "debug", "sub", "pub", "config"]

    required_fields = None

    defaults_fields = {"type": "model", "config": None, "sub": [], "debug": False}

    @_("type")
    def _(self, type_name: str) -> str:
        return type_name

    @_("process")
    def _(self, process_name: str) -> str:
        dpath = os.path.join(self.base_dir, "components", process_name)
        return str(pathlib.Path(dpath).resolve())

    @_("config", depends_on=("type", "process"))
    def _(self, config_path: str) -> dict:
        if config_path is None:
            if self._config["type"] == "model":
                config_path = os.path.splitext(
                    self._config["process"])[0] + '.toml'
            else:
                self.logger("error",
                            f"component of type system needs a config field",
                            error=AttributeError)
        mconf = attempt_parse_toml(join_if_not_abs(self.base_dir, config_path))
        if self._config["type"] == "model":
            ccpr = ComponentConfigParser(
                os.path.join(self.base_dir, "components"),
                context_str=self.context_str + "<config>")
            return ccpr.parse(mconf)
        elif self._config["type"] == "system":
            spr = SystemParser(self.base_dir,
                               context_str=self.context_str + "<system>")
            return spr.parse(mconf)
        return config_path


class ComponentConfigParser(ConfigParser):
    valid_fields = [
        "system_name", "system_representation", "system_solver",
        "sampling_frequency", "is_discrete", "is_hybrid", "parameters",
        "inputs", "topics", "sampling_phase"
    ]

    required_fields = ["system_name", "sampling_frequency", "is_discrete"]

    defaults_fields = {
        "system_representation": "black box",
        "system_solver": "Euler",
        "is_hybrid": False,
        "sampling_phase": 0.0
    }

    @_("is_hybrid", depends_on=("is_discrete", ))
    def _(self, is_hybrid: bool) -> bool:
        return is_hybrid

    @_("inputs")
    def _(self, iconf: dict) -> dict:
        msgs = iconf.get("msgs", [])
        iconf["msgs"] = [self.load_msg(m) for m in msgs]
        return iconf

    @_("topics")
    def _(self, tconf: dict) -> dict:
        for tval, tc in tconf.items():
            msg = tc.get("msg", None)
            dbase_dir = os.path.join(self.base_dir)
            msg_path = join_if_not_abs(dbase_dir, msg, project_dir="msg")
            tconf[tval]["msg"] = self.load_msg(
                msg) if msg is not None else None
            tconf[tval]["_msg_path"] = msg_path
        return tconf

    def load_msg(self, msg: str) -> CsafMsg:
        dbase_dir = os.path.join(self.base_dir)
        msg_path = join_if_not_abs(dbase_dir, msg, project_dir="msg")
        return CsafMsg.from_msg_file(msg_path)
