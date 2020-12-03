import os
import toml
import pathlib
import os
import logging
import toml
import typing as typ
from . import csaf_logger
from toposort import toposort_flatten, CircularDependencyError

from .rosmsg import CsafMsg, generate_serializer
from . import csaf_logger


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
        print(f"ERROR! Failed to parse {fname} as TOML configuration file <{e}>")


class ConfigParserMetaDict(dict):
    """resolve overloaded attributes"""
    def __setitem__(self, key, value):
        if hasattr(value, 'field_name_spec'):
            if "_field_name_methods" not in self:
                self["_field_name_methods"] = {}
            self["_field_name_methods"][(getattr(value, 'field_name_spec'))[0]] = value, getattr(value, 'field_name_spec')[1]
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

    def __init__(self, base_dir, context_str=''):
        self.base_dir = str(pathlib.Path(base_dir).resolve())
        self.context_str = context_str

    def logger(self, log_type_name, msg):
        getattr(csaf_logger, log_type_name)(f"{self.context_str + (': ' if self.context_str else '')}{msg}")

    @property
    def eval_order(self) -> list:
        dep_graph = {}
        for k, v in self._field_name_methods.items():
            assert isinstance(v[1], tuple)
            dep_graph[k] = set(v[1])
        try:
            return toposort_flatten(dep_graph)
        except CircularDependencyError as exc:
            self.logger("error", f"DEV ERROR: configuration parser has circular dependency <{exc}>")

    def parse(self, cconf: dict) -> dict:
        """parse a dict from a TOML parse"""
        if self.valid_fields is not None:
            for k in cconf.keys():
                if k not in self.valid_fields:
                    self.logger("error", f"found invalid field {k}")
        if self.required_fields is not None:
            for r in self.required_fields:
                if r not in cconf.keys():
                    self.logger("error", f"unable to find required field {r}")
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
    valid_fields = ["codec_dir", "output_dir", "name", "global_config", "log_level", "log_file", "evaluation_order", "components"]

    component_valid_fields =  ["run_command", "process", "config", "debug", "sub", "pub"]

    @_("log_file", depends_on=("output_dir", "log_level"))
    def _(self, log_path):
        log_level = self._config["log_level"]
        log_filepath  = join_if_not_abs(self._config["output_dir"],
                                        self._config["log_file"], exist=False)
        formatter = logging.Formatter('%(asctime)s: (%(levelname)s)  %(message)s', datefmt='%I:%M:%S %p')

        # reflect user specified log level as logger level
        csaf_logger.setLevel(log_level)

        # setup file logging -- accept log level
        fh = logging.FileHandler(log_filepath)
        fh.setFormatter(formatter)
        fh.setLevel(log_level)

        # setup console logging -- set to info log level
        sh = logging.StreamHandler()
        sh.setFormatter(formatter)
        sh.setLevel(logging.INFO)

        # add the two handlers to csaf logging
        csaf_logger.addHandler(fh)
        csaf_logger.addHandler(sh)

        # print config paths
        csaf_logger.info(f"setting up CSAF System")
        csaf_logger.info(f"Output Dir: {self._config['output_dir']}")
        csaf_logger.info(f"Codec Dir: {self._config['codec_dir']}")
        csaf_logger.info(f"Log Level: {self._config['log_level']}")

        return log_filepath

    @_("log_level")
    def _(self, level):
        llevel = {"critical" : logging.CRITICAL, "error" : logging.ERROR,
                    "warning" : logging.WARNING, "info" : logging.INFO,
                    "debug" : logging.DEBUG, "notset" : logging.NOTSET}
        if level in llevel:
            return llevel[level]
        else:
            raise AssertionError(f"log level '{level}' found in config is not valid!")

    @_("codec_dir")
    def _(self, codec_path: str) -> str:
        dpath = join_if_not_abs(self.base_dir, codec_path)
        path = str(pathlib.Path(dpath).resolve())
        outcome = mkdir_if_not_exist(dpath)
        if outcome:
            self.logger = self.logger("info", f"created codec directory {dpath} because it did not exist")
        return path

    @_("output_dir")
    def _(self, output_path: str) -> str:
        dpath = join_if_not_abs(self.base_dir, output_path)
        path = str(pathlib.Path(dpath).resolve())
        outcome = mkdir_if_not_exist(dpath)
        if outcome and self.logger:
            self.logger = self.logger("info", f"created output directory {dpath} because it did not exist")
        return path

    @_("components", depends_on=("codec_dir", "output_dir"))
    def _(self, components: dict):
        """TODO: implement this with a component parser"""
        for dname, dconfig in components.items():
            if "process" in dconfig:
                process_path = pathlib.Path(join_if_not_abs(self.base_dir, dconfig["process"], project_dir="components"))
                assert os.path.exists(process_path), f"process path '{process_path}' for component '{dname}' must exist!"
                components[dname]["process"] = str(process_path.resolve())

            # load in config file per component
            if 'config' in dconfig:
                dcconfig_path = join_if_not_abs(self.base_dir, dconfig['config'])
            else:
                dcconfig_path = pathlib.Path(join_if_not_abs(self.base_dir, dconfig['process'])).with_suffix('.toml')
            assert os.path.exists(dcconfig_path), f"config file '{dcconfig_path}' for component '{dname}' must exist"
            dcconfig = attempt_parse_toml(dcconfig_path)
            components[dname]['config'] = dcconfig
            dbase_dir = pathlib.Path(dcconfig_path).parent.resolve()

            # load special field inputs
            if 'inputs' in dcconfig:
                assert 'msgs' in dcconfig['inputs'], f"inputs field in {dcconfig_path} needs to have field msgs"
                msg_paths = [join_if_not_abs(dbase_dir, m, project_dir="msg") for m in dcconfig['inputs']['msgs']]
                components[dname]["config"]["inputs"]['msgs'] = [CsafMsg.from_msg_file(msg_path) for msg_path in msg_paths]

            # make all path to msg files absolute
            if 'topics' in dcconfig:
                for tname, tconf in dcconfig['topics'].items():
                    if 'msg' in tconf:
                        msg_path = join_if_not_abs(dbase_dir, tconf["msg"], project_dir="msg")
                        assert os.path.exists(msg_path), f"message file '{msg_path}' in topic '{tname}' for " \
                                                         f"component '{dname}' must exist!"
                        components[dname]["config"]["topics"][tname]['msg'] = CsafMsg.from_msg_file(msg_path)
                        components[dname]["config"]["topics"][tname]['serializer'] = generate_serializer(msg_path, self._config["codec_dir"])

        return components

    @_("evaluation_order", depends_on=("components",))
    def _(self, eval_order: list) -> list:
        return eval_order

