import pathlib
import os
import logging
import toml
from .parser import ConfigParser, join_if_not_abs, mkdir_if_not_exist
from .rosmsg import CsafMsg, generate_serializer


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
