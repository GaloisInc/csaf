""" System Configuration Functions

Ethan Lew
07/13/20
"""
import pathlib
import os
import logging

import toml
import pydot


def setup_logging(config_file):
    """enable logging with settings to be used by components"""

    config = attempt_parse_toml(config_file)
    log_filepath = config["log_file"]

    # log is permissive -- will use filepath specified
    if os.path.exists(log_filepath):
        print(f"WARNING! log will output to file that already exists: '{log_filepath}'")
        open(log_filepath, 'w').close()

    # determine log level
    if "log_level" in config:
        level = config["log_level"]
        log_level = logging.DEBUG if level == "debug" else logging.INFO
    else:
        log_level = logging.INFO

    # setup with basic config
    logging.basicConfig(format='%(asctime)s: (%(levelname)s)  %(message)s', datefmt='%I:%M:%S %p', level=log_level,
                        handlers=[logging.FileHandler(log_filepath), logging.StreamHandler()])


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


def get_from_base_string(key_base, d):
    """get all values with a key that starts with a key_base (sorted)
    :param key_base: string that keys start with
    :param d: dictionary to get values from
    :return: list of values
    """
    keys = list(d.keys())
    base_keys = [i for i in keys if i.startswith(key_base)]
    base_keys.sort()
    return [str(d[b]) for b in base_keys]


def build_device_graph(config_file):
    """build graph data structure from configuration toml"""
    config = attempt_parse_toml(config_file)
    devices = config["device"]

    # produce nodes
    nodes = {}
    for dname, dconfig in devices.items():
        pub = get_from_base_string("pub", dconfig)
        process_path = pathlib.Path(dconfig["process"])
        if 'config' not in dconfig:
            config_path = process_path.with_suffix('.toml')
        else:
            config_path = dconfig["config"]

        assert os.path.exists(config_path), f"config file {config_path} could not be found for component"
        assert pub[0] not in nodes, f"port {pub[0]} is published to twice!"

        dev_config = attempt_parse_toml(config_path)

        # add dname -- name referenced in the config
        nodes[pub[0]] = {**dev_config, "dname" : dname}

    # produce edges
    edges = []
    for dname, dconfig in devices.items():
        subs = get_from_base_string("sub", dconfig)
        pub = get_from_base_string("pub", dconfig)

        targets = subs
        source = pub[0]
        for tidx, t in enumerate(targets):
            edges.append((t, source))

    return nodes, edges


def check_config(config_file):
    """run all config_file checks"""
    check_fields(config_file)
    check_system_io(config_file)


def check_system_io(config_file):
    """check that io relationship of edges makes sense"""
    nodes, edges = build_device_graph(config_file)
    for e in edges:
        # nodes
        s0, s1 = [nodes[t] for t in e]
        # names
        n0, n1 = s0["outputs"]["names"], s1["inputs"]["names"]
        # device names
        d0, d1 = s0["dname"], s1["dname"]
        if len(n1) > 0:
            if type(n1[0]) is not str:
                assert len(n0) in [len(n) for n in n1], f"input lengths of {d1} was found to be incorrect against {d0} " \
                                                        f"(expected {len(n0)} to be included in {[len(n) for n in n1]})"
            else:
                assert len(n0) == len(n1), f"input length of {d1} was found to be incorrect against {d0} (expected {len(n0)} but found {len(n1)})"
        else:
            assert len(n0) == 0, f"device {d1} was found to have no input, but {d0} has length {len(n0)}"


def check_fields(config_file):
    """check that config TOML field has the required fields"""
    config = attempt_parse_toml(config_file)
    required_fields = ("device", "log_file", "evaluation_order", "global_config")
    required_device = ("run_command", "process")

    for rf in required_fields:
        assert rf in config, f"config file {config_file} must have required field '{rf}'"

    devices = config["device"]
    for dname, dconfig in devices.items():
        for rf in required_device:
            assert rf in dconfig, f"device '{dname}' in config file {config_file} must have required field '{rf}'"


def plot_config(config_file):
    """plot system configuration as a graph"""
    def get_length(names):
        if names is not None:
            if len(names) > 0:
                if type(names[0]) is not str:
                    return str([len(n) for n in names])
                else:
                    return str(len(names))
        return '0'

    nodes, edges = build_device_graph(config_file)
    config = attempt_parse_toml(config_file)
    eorder = config["evaluation_order"]


    graph = pydot.Dot(graph_type='digraph', prog='LR')
    graph.set_node_defaults(shape='box',
                            fontsize='10')

    vertices = {}
    for nname, ninfo in nodes.items():
        devname = ninfo["system_name"]
        dname = ninfo["dname"]
        vertices[nname] = pydot.Node(devname+f"\n({eorder.index(dname)})")
        graph.add_node(vertices[nname])

    for e in edges:
        ninfo = nodes[e[0]]
        onames = ninfo["outputs"]["names"]
        snames = ninfo["states"]["names"]
        lstr = f"{get_length(onames)}/{get_length(snames)}"
        graph.add_edge(pydot.Edge(vertices[e[0]], vertices[e[1]], fontsize=10, label=lstr + "\nport " +str(e[0])))

    config_path = pathlib.Path(config_file)
    graph_path  = config_path.with_suffix(".png")
    graph.write_png(graph_path)
