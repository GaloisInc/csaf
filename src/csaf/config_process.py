from . import csaf_logger
from .rosmsg import generate_serializer
from .messenger import SerialMessenger

def get_field_iterator(ddict, fname, paths=()):
    """iterator to search for field entries"""
    for k, v in ddict.items():
        if k == fname:
            yield (*paths, fname), v
        if isinstance(v, dict):
            yield from get_field_iterator(v, fname, paths=(*paths, k))


def search_field(ddict, field_name):
    """given a nested dict find all values that have a key name"""
    return [i for i in get_field_iterator(ddict, field_name)][::-1]


def get_dict_path(ddict: dict, path: tuple):
    cdict = ddict
    for key in path[:-1]:
        cdict = cdict[key]
    return cdict


def set_dict_path(ddict: dict, path: tuple, val):
    """given a tuple of keys, set nested dict ddict to val"""
    cdict = ddict
    for key in path[:-1]:
        cdict = cdict[key]
    cdict[path[-1]] = val


def resolve_port_conflicts(dconf, do_resolve=True):
    """given a CSAF config, look for port conflicts and resolve them"""
    # reverse as prefer replacing deepest fields
    paths, ports = list(zip(*search_field(dconf, "pub")))
    ports = list(ports)
    for idx, p in enumerate(ports):
        if p in ports[:idx]:
            conflicts = [
                pathmp for pathmp, portmp in zip(paths, ports) if portmp == p
            ]
            csaf_logger.warning(
                f"found port conflict between {['->'.join(c) for c in conflicts]}, with port {p}"
            )
            if do_resolve:
                new_port = max(ports) + 1
                ports[idx] = new_port
                set_dict_path(dconf, paths[idx], new_port)
                csaf_logger.warning(
                    f"replacing {'->'.join(paths[idx])} from port {p} to port {new_port}"
                )


def prepare_serializers(dconf):
    def _dist(t0, t1):
        if t0 == t1:
            return 1
        if len(t0) > len(t1):
            t0, t1 = t1, t0
        for i in range(len(t0)):
            if t0[i] != t1[i]:
                return i / len(t1)
        return len(t0) / len(t1)

    cdpaths, codec_dirs = list(zip(*search_field(dconf, "codec_dir")))
    tinfo = search_field(dconf, "topics")
    for tpath, topics in tinfo:
        for tname, tconfig in topics.items():
            dists = [_dist(tpath, cdp[:-1]) for cdp in cdpaths]
            idist = dists.index(max(dists))
            msg_path = tconfig["_msg_path"]
            serializer = generate_serializer(msg_path, codec_dirs[idist])
            tconfig["_codec_path"] = codec_dirs[idist]
            tconfig["serializer"] = serializer


def identifier_map(dconf):
    def id_from_pub_path(path):
        return "-".join([c for c in path if c not in ("components", "pub", "config")][::-1])
    paths, ports = list(zip(*search_field(dconf, "pub")))
    ids = [id_from_pub_path(p) for p in paths]
    return {iden: get_dict_path(dconf, path) for iden, path in zip(ids, paths) }


def prepare_messengers(dconf, append_str="", idm_above={}):
    idm = identifier_map(dconf)
    def _prepare(dconf, append_str=append_str, idm_above=idm_above):
        scope_identifiers = dconf["inputs"]["identifiers"]
        for iden, idconf in dconf["components"].items():
            mss_in = {}
            sub_ports = []
            mss_out = {f"{iden+append_str}-{t}": v['_serializer'] for t, v in idconf['config']['topics'].items()}
            ibuffer = {f"{iden+append_str}-{t}": v['initial'] for t, v in idconf['config']['topics'].items() if 'initial' in v}
            pub_ports = idm[iden+append_str].get("pub", None)
            pub_ports = [pub_ports] if pub_ports is not None else []
            for sname, stopic in idconf["sub"]:
                skey = sname + append_str
                if skey not in idm:
                    mss_in[sname] = idm_above[scope_identifiers.index(sname)]
                else:
                    mss_in[skey] = idm[skey]["config"]["topics"][stopic]["_serializer"]
                    sub_ports += [(idm[skey]["pub"], f"{skey}-{stopic}")]
                if idconf["type"] == "system":
                    _prepare(idconf["config"],
                             append_str=append_str+f"-{iden}",
                             idm_above=[idm[s[0]+append_str]["config"]["topics"][s[1]]["_serializer"] for s in idconf['sub']])
            idconf["_messenger_in"] = SerialMessenger(mss_in)
            idconf["_messenger_out"] = SerialMessenger(mss_out)
            idconf["_sub_ports"] = sub_ports
            idconf["_pub_ports"] = pub_ports
            idconf["_initial_buffer"] = ibuffer
            idconf["_hier_name"] = iden + append_str
    return _prepare(dconf, append_str=append_str, idm_above=idm_above)
