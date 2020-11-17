#!/usr/bin/python3
""" Dataset Creator

@author: Ethan Lew <elew@galois.com>
08/03/2020

Simulates systems based on user supplied initial conditions

usage: generate_dataset.py [-h] [--ic-file IC_FILE]
                           [--data-format DATA_FORMAT]
                           [--config-file CONFIG_FILE]
                           [--output-file OUTPUT_FILE] [--component COMPONENT]
                           [--t-max T_MAX]

TODO: the fixed terminating conditions makes this work for the f16 systems only.
Make this configurable.

optional arguments:
  -h, --help            show this help message and exit
  --ic-file IC_FILE     Path to Initial Conditions JSON
  --data-format DATA_FORMAT
                        Dataset Formats (default|openai)
  --config-file CONFIG_FILE
                        Path to CSAF System TOML
  --output-file OUTPUT_FILE
                        Name of Output JSON File to Save
  --component COMPONENT
                        Name of Component to Collect
  --t-max T_MAX         Maximum time to simulate each run

Example Usage:
    # in /src
    python generate_dataset.py --ic-file=ic.json --config-file=../examples/f16/f16_simple_config.toml

Specify Initial Conditions to Simulate:
    IC (Initial Conditions) JSON format:
        // list of tables, mapping component identifiers to initial condition arrays
        [
            {"componentname": [...]},
            ...
            {"componentname": [...]},
            {"componentname" :[...]}
        ]

    Example  F16 <ic.json>:
    [
        {"plant": [532.2267341599213, 0.037027160081059704,
                    0.0, 0.7853981633974483, 3.0113293881657786,
                    -0.7427364757763312, 0.0, -0.436285343076811,
                    0.0, 0.0, 0.0,
                    4324.831502507831, 9.0]
        }
    ]
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


def ground_collision_condition(cname, outs):
    """ground collision premature termnation condition"""
    return cname == "plant" and outs["states"][11] <= 0.0


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
parser.add_argument("--t-max",
                    type=int,
                    help=f"Maximum time to simulate each run",
                    default=35)
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

tspan: list = [0.0, args.t_max]
n_tasks: int = len(ic_states)
runs: RunsType = run_workgroup(
    n_tasks,
    model_conf,
    ic_states,
    tspan,
    terminating_conditions=ground_collision_condition)

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
