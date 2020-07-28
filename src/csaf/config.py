""" System Configuration Functions

Ethan Lew
07/13/20
"""
import pathlib
import os
import logging

import toml
import pydot

from .rosmsg import CsafMsg, generate_serializer


def mkdir_if_not_exist(dirname):
    """if path doesn't exist, make an empty directory with its name"""
    if os.path.exists(dirname):
        return False
    else:
        pathlib.Path(dirname).mkdir(parents=True, exist_ok=True)
        return True


def join_if_not_abs(*args):
    """if last argument is an absolute path, don't join the path arguments together"""
    if os.path.isabs(args[-1]):
        return args[-1]
    else:
        return os.path.join(*args)


def attempt_parse_toml(fname):
    """ try to parse a TOML file
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
                "log_file", "evaluation_order", "devices"]

    @staticmethod
    def get_device_valid_fields():
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

        # setup logging
        log_filepath  = join_if_not_abs(base_dir, config["log_file"])
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
        logging.basicConfig(format='%(asctime)s: (%(levelname)s)  %(message)s', datefmt='%I:%M:%S %p', level=log_level,
                            handlers=[logging.FileHandler(log_filepath), logging.StreamHandler()])
        logging.info(f"setting up CSAF System from TOML file '{toml_file}'")

        # make directories
        dpath = join_if_not_abs(base_dir, config['codec_dir'])
        config["codec_dir"] = str(pathlib.Path(dpath).resolve())
        outcome = mkdir_if_not_exist(dpath)
        if outcome:
            logging.info(f"created codec directory {dpath} because it did not exist")
        dpath = join_if_not_abs(base_dir, config['output_dir'])
        config["output_dir"] = str(pathlib.Path(dpath).resolve())
        outcome = mkdir_if_not_exist(dpath)
        if outcome:
            logging.info(f"created output directory {dpath} because it did not exist")
        logging.info(f"Output Dir: {config['output_dir']}")
        logging.info(f"Codec Dir: {config['codec_dir']}")
        logging.info(f"Log Level: {config['log_level']}")

        # load component level config file into this config dict
        for dname, dconfig in config["devices"].items():
            # make process absolute path
            process_path = pathlib.Path(join_if_not_abs(base_dir, dconfig["process"]))
            assert os.path.exists(process_path), f"process path '{process_path}' for device '{dname}' must exist!"
            config["devices"][dname]["process"] = str(process_path.resolve())

            # load in config file per device
            if 'config' in dconfig:
                dcconfig_path = join_if_not_abs(base_dir, dconfig['config'])
            else:
                dcconfig_path = pathlib.Path(join_if_not_abs(base_dir, dconfig['process'])).with_suffix('.toml')
            assert os.path.exists(dcconfig_path), f"config file '{dcconfig_path}' for device '{dname}' must exist"
            dcconfig = attempt_parse_toml(dcconfig_path)
            config["devices"][dname]['config'] = dcconfig

            # make all path to msg files absolute
            if 'topics' in dcconfig:
                dbase_dir = pathlib.Path(dcconfig_path).parent.resolve()
                for tname, tconf in dcconfig['topics'].items():
                    if 'msg' in tconf:
                        msg_path = join_if_not_abs(dbase_dir, tconf["msg"])
                        assert os.path.exists(msg_path), f"message file '{msg_path}' in topic '{tname}' for " \
                                                         f"device '{dname}' must exist!"
                        config["devices"][dname]["config"]["topics"][tname]['msg'] = CsafMsg.from_msg_file(msg_path)
                        config["devices"][dname]["config"]["topics"][tname]['serializer'] = generate_serializer(msg_path, config["codec_dir"])

        return cls(config)

    def __init__(self, config: dict):
        self._config = config

    def get_device_settings(self, dname: str):
        """get information about a device by its device name (dname)"""
        assert dname in self._config["devices"]
        return self._config["devices"][dname]

    def build_device_graph(self):
        """build a graph representation of the system from the config"""
        # populate nodes
        nodes = {}
        for dname, dconfig in self._config["devices"].items():
            pub = dconfig["pub"]
            nodes[pub] = {**dconfig, "dname" : dname}

        # populate edges and edge labels
        edges = []
        edge_labels = {'topic' : [], 'width' : [], 'name' : []}
        for dname, dconfig in self._config["devices"].items():
            subs = dconfig["sub"]
            pub = dconfig["pub"]
            targets = subs
            source = pub

            for tidx, t in enumerate(targets):
                name = t[0]
                sub_port = self.get_device_settings(name)["pub"]

                # update the edges
                edges.append((sub_port, source))

                # update the edge labels
                edge_labels['topic'].append(t[1])
                edge_labels['width'].append(self.get_msg_width(*t))
                ddconfig = self.get_device_settings(dname)['config']
                if 'inputs' in ddconfig:
                    edge_labels['name'].append(self.get_device_settings(dname)['config']['inputs']['names'][tidx])

        return nodes, edges, edge_labels

    def get_msg_width(self, dname: str, tname: str):
        """given device name and topic name, return the number of fields in a message
        """
        cmsg = self.get_msg_setting(dname, tname, "msg")
        return len(cmsg.fields_no_header)

    def has_topic(self, dname, tname):
        """whether a device with dname has topic name tname"""
        assert dname in self._config['devices']
        return tname in self._config['devices'][dname]['config']['topics']

    def get_topics(self, dname):
        assert dname in self._config['devices']
        return list(self._config['devices'][dname]['config']['topics'].keys())

    def get_msg_setting(self, dname, tname, prop):
        """safer method to get topic property"""
        assert dname in self._config['devices']
        assert tname in self._config['devices'][dname]['config']['topics']
        assert prop in self._config['devices'][dname]['config']['topics'][tname]
        return self._config['devices'][dname]['config']['topics'][tname][prop]

    def assert_io_widths(self):
        """check that the input/output size between topics are valid"""
        nodes, edges, el = self.build_device_graph()
        width, name = el['width'], el['name']
        for e, w, n in zip(edges, width, name):
            dout = nodes[e[0]]["dname"]
            din = nodes[e[1]]["dname"]
            assert len(n) == w, f"edge between publishing device '{dout}' and subscribing device '{din}' have width " \
                                f"disagreement (publishing {w} values but naming {len(n)})"

    def plot_config(self, fname=None):
        """visualize the configuration file"""
        fname = fname if fname is not None else self.config_dict["name"] + "-config.png"

        nodes, edges, edge_labels = self.build_device_graph()
        eorder = self.config_dict["evaluation_order"]

        graph = pydot.Dot(graph_type='digraph', prog='LR')
        graph.set_node_defaults(shape='box',
                                fontsize='10')

        verts = {}
        for nname, ninfo in nodes.items():
            devname = ninfo['config']["system_name"]
            dname = ninfo["dname"]
            verts[nname] = pydot.Node(devname+f"\n({eorder.index(dname)})")
            graph.add_node(verts[nname])


        for eidx, e in enumerate(edges):
            topic = edge_labels["topic"][eidx]
            width = edge_labels["width"][eidx]
            graph.add_edge(pydot.Edge(verts[e[0]], verts[e[1]], fontsize=10,
                                      label=topic + f" ({str(width)})\nport " + str(e[0])))

        graph_path = pathlib.Path(join_if_not_abs(self.config_dict['output_dir'], fname))
        graph.write_png(graph_path)

    @property
    def config_dict(self):
        return self._config

    @property
    def get_name_devices(self):
        return list(self._config["devices"].keys())

    @property
    def get_num_devices(self):
        return len(self.get_name_devices)
