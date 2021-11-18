# Control Systems Analysis Framework (CSAF)

![CI](https://github.com/GaloisInc/csaf/actions/workflows/main.yml/badge.svg)


- [Quick Start](#quick-start)
- [Examples](#examples)
- [Jupyter notebooks](#jupyter-notebooks)
- [Job configuration](#job-configuration)
- [Development](#development)
- [Licensing](#licensing)
- [Acknowledgment](#acknowledgment)

CSAF is a framework to minimize the effort required to evaluate, implement, and **verify** controller design (classical and learning enabled) with respect to the system dynamics. Its key features are:

* Component based controller design
* Native support for python and C language executables
* Compatibility with external hardware and software processes
* Ease of deployment

[//]: # (![csaf_importing_components](docs/srs/img/csaf_importing_controllers.png))

Controllers, subsystems and plants are implemented as a collection of components.
Components communicate via a 0MQ pub/sub configuration and serialize/deserialize ROS messages. Below is an example of a topology graph of F16 system with GCAS autopilot.

[//]: # (![f16_with_gcas](docs/srs/img/csaf_system_diagram.png))

## Quick Start

### Installation 

CSAF runs inside a [Docker container](https://www.docker.com/), and in order to use CSAF, you first need to [install docker](https://docs.docker.com/engine/install/). CSAF has been tested on Linux (Ubuntu 18.04 and 20.04) and OS X, but should run on any nix-like system that runs docker. CSAF can be also run natively on your host machine, but this option is recommended only for the developers and isn't officially supported.

[//]: #  ![csaf_quickstart](docs/srs/img/csaf_quickstart.png)

### Running CSAF

## Tests

## Examples
CSAF currently contains a number of examples, including the F-16 shown below.
The examples are located in the `examples` directory and include licensing and attribution information.
Please read the [examples README.md](./examples/README.md) for a detailed list.

### F-16 Control System

## Jupyter notebooks

CSAF can be used from within a [jupyter notebook](https://jupyter-notebook.readthedocs.io/en/stable/examples/Notebook/What%20is%20the%20Jupyter%20Notebook.html#Introduction). To start CSAF in the notebook mode, run `./run-csaf.sh -j -e f16-simple` - the `-j` flag specified *notebook mode*, and using the `-e f16-simple` as an example will set the paths necessary for using the F16 model.

**NOTE:** notebooks are run inside docker, and because of that the directory paths are different than if they were run natively. Keep this in mind when writing new notebooks, and have a look at the provided examples in `docs/notebooks` directory.


## Goal configuration


## Development
`CONTRIBUTING.md` contains CSAF development guildelines, please familiarize yourself with the guidelines before opening a pull request. The best way to contact the dev team is via GitHub issues.

## Licensing

The code in this repository is licensed under two different licenses. The core of CSAF (`src` and `docs` directories) and the majority of
examples is licensed under [BSD license](LICENSE.txt). The [f16 examples](f16lib) in the `f16lib` module is licensed under [GPL license](f16lib/LICENSE.txt).

## Acknowledgment
This material is based upon work supported by the DARPA Assured Autonomy program under the United States Air Force under Contract No. FA8750-19-C-0092. Any opinions, findings and conclusions or recommendations expressed in this material are those of the author(s) and do not necessarily reflect the views of DARPA or the United States Air Force.
