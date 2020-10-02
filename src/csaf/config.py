""" System Configuration Functions

Ethan Lew
07/13/20
"""
import pathlib
import os
import logging
import toml

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


class SystemConfig:
    """SystemConfig accepts a system description file"""
    @staticmethod
    def get_valid_fields():
        return ["codec_dir", "output_dir", "name", "global_config", "log_level",
                "log_file", "evaluation_order", "components"]

    @staticmethod
    def get_component_valid_fields():
        return ["run_command", "process", "config", "debug", "sub", "pub"]

    @classmethod
    def from_toml(cls, toml_file: str):
        """from a toml file, ingest a system configuration and apply checks
        TODO: restructure into smaller functions -- long
        """
        assert os.path.exists(toml_file),  f"TOML file '{toml_file}' doesn't exist!"
        config = attempt_parse_toml(toml_file)
        base_dir = pathlib.Path(toml_file).parent

        # load in global config
        if "global_config" in config:
            gconfig_filepath = join_if_not_abs(base_dir, config['global_config'])
            config["global_config"] = str(pathlib.Path(gconfig_filepath).resolve())
            assert os.path.exists(gconfig_filepath),  f"TOML file '{gconfig_filepath}' doesn't exist!"
            global_config = attempt_parse_toml(gconfig_filepath)
            config = {**global_config, **config}  # favor the local config

        # check that all config keys are valid
        for k in config.keys():
            assert k in SystemConfig.get_valid_fields(), f"config field '{k}' not a valid config field!"

        # make directories
        dpath = join_if_not_abs(base_dir, config['codec_dir'])
        config["codec_dir"] = str(pathlib.Path(dpath).resolve())
        outcome = mkdir_if_not_exist(dpath)
        if outcome:
            csaf_logger.info(f"created codec directory {dpath} because it did not exist")
        dpath = join_if_not_abs(base_dir, config['output_dir'])
        config["output_dir"] = str(pathlib.Path(dpath).resolve())
        outcome = mkdir_if_not_exist(dpath)

        # setup logging
        log_filepath  = join_if_not_abs(config["output_dir"], config["log_file"], exist=False)
        config["log_file"] = str(pathlib.Path(log_filepath).resolve())

        # log is permissive -- will use filepath specified
        if os.path.exists(log_filepath):
            print(f"WARNING! log will output to file that already exists: '{log_filepath}'")
            open(log_filepath, 'w').close()
        if "log_level" in config:
            level = config["log_level"]
            llevel = {"critical" : logging.CRITICAL, "error" : logging.ERROR,
                      "warning" : logging.WARNING, "info" : logging.INFO,
                      "debug" : logging.DEBUG, "notset" : logging.NOTSET}
            if level in llevel:
                log_level = llevel[level]
            else:
                AssertionError(f"log level '{level}' found in config file {toml_file} is not valid!")
        else:
            log_level = logging.INFO
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
        csaf_logger.info(f"setting up CSAF System from TOML file '{toml_file}'")
        if outcome:
            csaf_logger.info(f"created output directory {dpath} because it did not exist")
        csaf_logger.info(f"Output Dir: {config['output_dir']}")
        csaf_logger.info(f"Codec Dir: {config['codec_dir']}")
        csaf_logger.info(f"Log Level: {config['log_level']}")

        # load component level config file into this config dict
        for dname, dconfig in config["components"].items():
            # make process absolute path
            process_path = pathlib.Path(join_if_not_abs(base_dir, dconfig["process"], project_dir="components"))
            assert os.path.exists(process_path), f"process path '{process_path}' for component '{dname}' must exist!"
            config["components"][dname]["process"] = str(process_path.resolve())

            # load in config file per component
            if 'config' in dconfig:
                dcconfig_path = join_if_not_abs(base_dir, dconfig['config'])
            else:
                dcconfig_path = pathlib.Path(join_if_not_abs(base_dir, dconfig['process'])).with_suffix('.toml')
            assert os.path.exists(dcconfig_path), f"config file '{dcconfig_path}' for component '{dname}' must exist"
            dcconfig = attempt_parse_toml(dcconfig_path)
            config["components"][dname]['config'] = dcconfig
            dbase_dir = pathlib.Path(dcconfig_path).parent.resolve()

            # load special field inputs
            if 'inputs' in dcconfig:
                assert 'msgs' in dcconfig['inputs'], f"inputs field in {dcconfig_path} needs to have field msgs"
                msg_paths = [join_if_not_abs(dbase_dir, m, project_dir="msg") for m in dcconfig['inputs']['msgs']]
                config["components"][dname]["config"]["inputs"]['msgs'] = [CsafMsg.from_msg_file(msg_path) for msg_path in msg_paths]

            # make all path to msg files absolute
            if 'topics' in dcconfig:
                for tname, tconf in dcconfig['topics'].items():
                    if 'msg' in tconf:
                        msg_path = join_if_not_abs(dbase_dir, tconf["msg"], project_dir="msg")
                        assert os.path.exists(msg_path), f"message file '{msg_path}' in topic '{tname}' for " \
                                                         f"component '{dname}' must exist!"
                        config["components"][dname]["config"]["topics"][tname]['msg'] = CsafMsg.from_msg_file(msg_path)
                        config["components"][dname]["config"]["topics"][tname]['serializer'] = generate_serializer(msg_path, config["codec_dir"])

        return cls(config)

    def __init__(self, config: dict):
        self._config = config
        self.assert_io_widths()

    def build_component_graph(self):
        """build a graph representation of the system from the config"""
        # populate nodes
        csaf_logger.debug('Building component graph.')
        nodes = {}
        for dname, dconfig in self._config["components"].items():
            csaf_logger.debug(f'processing component:: {dname}')
            nodes[dname] = {**dconfig, "dname" : dname}

        # populate edges and edge labels
        edges = []
        edge_labels = {'topic' : [], 'width' : [], 'name' : []}
        for dname, dconfig in self._config["components"].items():
            if "sub" in dconfig:
                subs = dconfig["sub"]
                targets = subs

            for tidx, t in enumerate(targets):
                name = t[0]

                # update the edges
                edges.append((name, dname))

                # update the edge labels
                edge_labels['topic'].append(t[1])
                edge_labels['width'].append(self.get_msg_width(*t))
                ddconfig = self.get_component_settings(dname)['config']
                if 'inputs' in ddconfig:
                    msgs = self.get_component_settings(dname)['config']['inputs']['msgs']
                    names = [m.fields_no_header for m in msgs]
                    edge_labels['name'].append(names[tidx])

        return nodes, edges, edge_labels

    def get_component_settings(self, dname: str):
        """get information about a component by its component name (dname)"""
        assert dname in self._config["components"]
        return self._config["components"][dname]

    def get_msg_width(self, dname: str, tname: str):
        """given component name and topic name, return the number of fields in a message
        """
        cmsg = self.get_msg_setting(dname, tname, "msg")
        return len(cmsg.fields_no_header)

    def has_topic(self, dname, tname):
        """whether a component with dname has topic name tname"""
        assert dname in self._config["components"]
        return tname in self._config["components"][dname]['config']['topics']

    def get_topics(self, dname):
        """given a component with component name dname, """
        assert dname in self._config["components"]
        return list(self._config["components"][dname]['config']['topics'].keys())

    def get_msg_setting(self, dname, tname, prop):
        """safer method to get topic property"""
        assert dname in self._config["components"], f"Failed to get property {prop} for topic {tname} component {dname}"
        assert tname in self._config["components"][dname]['config']['topics'], f"Failed to get property {prop} for topic {tname} component {dname}"
        assert prop in self._config["components"][dname]['config']['topics'][tname], f"Failed to get property {prop} for topic {tname} component {dname}"
        return self._config["components"][dname]['config']['topics'][tname][prop]

    def assert_io_widths(self):
        """check that the input/output size between topics are valid"""
        nodes, edges, el = self.build_component_graph()
        width, name = el['width'], el['name']
        for e, w, n in zip(edges, width, name):
            dout = nodes[e[0]]["dname"]
            din = nodes[e[1]]["dname"]
            csaf_logger.debug(f'{n}, {dout} -> {din}')
            assert len(n) == w, f"edge between publishing component '{dout}' and subscribing component '{din}' have width " \
                                f"disagreement (publishing {w} values but naming {len(n)})"

    def plot_config(self, fname=None, **kwargs):
        """visualize the configuration file"""
        import pydot
        fname = fname if fname is not None else self.config_dict["name"] + "-config.pdf"

        nodes, edges, edge_labels = self.build_component_graph()
        eorder = self.config_dict["evaluation_order"]

        graph = pydot.Dot(graph_type='digraph', prog='UD', concentrate=True, color="white")
        graph.set_node_defaults(shape='box',
                                fontsize='10')

        verts = {}
        for nname, ninfo in nodes.items():
            devname = ninfo['config']["system_name"]
            dname = ninfo["dname"]
            if ninfo["config"]["is_discrete"]:
                verts[nname] = pydot.Node(devname+f"\n({eorder.index(dname)})", style="solid")
            else:
                verts[nname] = pydot.Node(devname+f"\n({eorder.index(dname)})", style="bold")
            graph.add_node(verts[nname])

        for eidx, e in enumerate(edges):
            topic = edge_labels["topic"][eidx]
            width = edge_labels["width"][eidx]
            port = nodes[e[0]]["pub"] if "pub" in nodes[e[0]] else "NONE"
            graph.add_edge(pydot.Edge(verts[e[0]], verts[e[1]], fontsize=10,
                                      label=topic + f" ({str(width)})\nport {port}"))

        graph_path = pathlib.Path(join_if_not_abs(self.config_dict['output_dir'], fname, exist=False))
        extension = graph_path.suffix[1:]
        graph.write(graph_path, format=extension, **kwargs)

    @property
    def config_dict(self):
        """exposed internal _config for read/copy
        preferably, use the interface accessors instead to safely handle the config
        """
        return self._config

    @property
    def output_directory(self):
        """configuration output directory"""
        return self._config["output_dir"]

    @property
    def name(self):
        """configuration name"""
        return self._config["name"]

    @property
    def get_name_components(self):
        """names of components in the configuration (not the component name)"""
        return list(self._config["components"].keys())

    @property
    def get_num_components(self):
        """number of component in a configuration"""
        return len(self.get_name_components)
