""" Dataset Creator

@author: Ethan Lew <elew@galois.com>
08/03/2020

Simulates systems based on user supplied initial conditions
"""
import argparse
import os
import pathlib
import json
import typing as typ

import numpy as np

from run_parallel import run_workgroup
import csaf.config as cconf
import csaf.system as csys
import csaf.trace as ctr

tspan: list = [0.0, 5.0]
RunsType = typ.List[typ.Union[typ.Tuple[bool, ctr.TimeTrace], Exception]]


def format_openai(runs: RunsType,
                  model_conf: cconf.SystemConfig,
                  component_name="autopilot") -> typ.List[dict]:
    """processes workgroup runs into a data structure serializable into json

    follows an openai gym like format:

    state, next_state, action, time, did_terminate, is_exception
    """
    out = []
    for r in runs:
        if isinstance(r, Exception):
            out.append({"is_exception": True})
        else:
            outd = {"is_exception": False}
            trajs = r[1]
            cio = ctr.get_component_io(component_name, trajs, model_conf)
            outd["did_terminate"] = not r[0]
            outd["time"] = cio["times"].tolist()
            outd["state"] = cio["inputs"].tolist()
            outd["actions"] = cio["outputs"].tolist()
            outd["next_state"] = np.vstack(
                (cio["inputs"][1:], cio["inputs"][-1])).tolist()
            out.append(outd)
    return out


def format_default(runs: RunsType,
                   model_conf: cconf.SystemConfig,
                   component_name="autopilot") -> typ.List[dict]:
    """processes workgroup runs into a data structure serializable into json

    packs output from get_component_io as well as is_exception
    """
    out = []
    for r in runs:
        if isinstance(r, Exception):
            out.append({"is_exception": True})
        else:
            outd = {"is_exception": False}
            trajs = r[1]
            cio = ctr.get_component_io(component_name, trajs, model_conf)
            for k, v in cio.items():
                if isinstance(v, np.ndarray):
                    cio[k] = v.tolist()
            outd = {**outd, **cio}
            out.append(outd)
    return out


def fmt_parser(v: str) -> str:
    """ ensures that a correct formate is supplied by a parse string"""
    assert v in fmts, f"format value {v} not in {fmts}"
    return v


# parse user specified options
fmts = ["default", "openai"]

parser = argparse.ArgumentParser(
    description="Create a dataset from a CSAF system")
parser.add_argument("--ic-file",
                    type=str,
                    help="Path to Initial Conditions JSON",
                    default="./ic.json")
parser.add_argument("--data-format",
                    type=str,
                    help=f"Dataset Formats ({'|'.join(fmts)})",
                    default="default")
parser.add_argument("--config-file",
                    type=str,
                    help=f"Path to CSAF System TOML")
parser.add_argument("--output-file",
                    type=str,
                    help=f"Name of Output JSON File to Save",
                    default="./output.json")
parser.add_argument("--component",
                    type=str,
                    help=f"Name of Component to Collect",
                    default="autopilot")
args = parser.parse_args()

# resolve filepaths and ensure that they exist
ic_file: str = str(pathlib.Path(args.ic_file).resolve())
config_file: str = str(pathlib.Path(args.config_file).resolve())
output_file: str = str(pathlib.Path(args.output_file).resolve())

assert os.path.exists(
    ic_file), f"Initial Conditions JSON File {ic_file} does not exist!"
assert os.path.exists(
    config_file), f"CSAF System Config File {config_file} does not exist"

# run simulations
model_conf = cconf.SystemConfig.from_toml(config_file)

with open(ic_file, 'r') as fp:
    ic_states: list = json.load(fp)

n_tasks: int = len(ic_states)
runs: RunsType = run_workgroup(n_tasks, model_conf, ic_states, tspan)

# output to JSON file
if args.data_format == "default":
    out: typ.List[dict] = format_default(runs,
                                         model_conf,
                                         component_name=args.component)
elif args.data_format == "openai":
    out: typ.List[dict] = format_openai(runs,
                                        model_conf,
                                        component_name=args.component)
else:
    raise RuntimeError(f"invalid format {args.data_format}")
with open(output_file, "w") as fp:
    json.dump(out, fp)
