""" System Configuration Functions

Ethan Lew
07/13/20
"""
import pathlib
import os
import logging
import toml

from .rosmsg import CsafMsg, generate_serializer
from .config_parser import SystemParser, attempt_parse_toml, join_if_not_abs
from . import csaf_logger


class SystemConfig:
    """SystemConfig accepts a system description file"""
    @classmethod
    def from_toml(cls, toml_file: str):
        """from a toml file, ingest a system configuration and apply checks
        TODO: restructure into smaller functions -- long
        """
        toml_path = str(pathlib.Path(toml_file).parent.resolve())
        toml_conf = attempt_parse_toml(toml_file)
        cfp = SystemParser(toml_path, context_str=toml_path)
        config = cfp.parse(toml_conf)
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
