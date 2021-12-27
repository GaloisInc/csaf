"""generate_f16_ic.py

@author: Ethan Lew <elew@pdx.edu>
11-20-2020

usage: generate_f16_ic.py [-h] [--ic-file IC_FILE] [--n-samples N_SAMPLES]

Generate Random Initial Conditions for the F16 Model, with respect to defined
bounds (see script for them)

optional arguments:
  -h, --help            show this help message and exit
  --ic-file IC_FILE     Path to Output Initial Conditions JSON
  --n-samples N_SAMPLES
                        Number of Samples to Produce
"""
import json
import argparse
import pathlib
import os
import numpy as np

# bounds to sample -- CHANGE THIS
bounds = [(200, 1000), (np.deg2rad(2.1215), np.deg2rad(2.1215)), (0.0, 0.0),
          ((np.pi / 2) * 0.5, (np.pi / 2) * 0.5), (-np.pi, np.pi),
          (-np.pi / 4, np.pi / 4), (0.0, 0.0), (-0.5, 0.5), (0.0, 0.0),
          (0.0, 0.0), (0.0, 0.0), (500, 8000), (9, 9)]


def gen_random_state(bounds) -> np.ndarray:
    """create a uniformly distributed initial condition"""
    sample = np.random.rand(len(bounds))
    ranges = np.array([b[1] - b[0] for b in bounds])
    offset = np.array([-b[0] for b in bounds])
    return sample * ranges - offset


def filestr(fname: str) -> str:
    path = pathlib.Path(fname)
    assert path.suffix.lower() == ".json", f"{fname} must have json extension"
    return str(path)


# parse user input
parser = argparse.ArgumentParser(
    description=
    "Generate Random Initial Conditions for the F16 Model, with respect to"
    "defined bounds (see script for them)"
)
parser.add_argument("--ic-file",
                    type=filestr,
                    help="Path to Output Initial Conditions JSON",
                    default="./ic.json")
parser.add_argument("--n-samples",
                    type=int,
                    default=50,
                    help="Number of Samples to Produce")
args = parser.parse_args()

x_init = [gen_random_state(bounds) for _ in range(args.n_samples)]

x_init_l = [{"plant": list(x)} for x in x_init]
with open(args.ic_file, "w") as fp:
    json.dump(x_init_l, fp)
