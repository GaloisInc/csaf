# Control Systems Analysis Framework (CSAF)

CSAF is a framework to minimize the effort required to evaluate, implement, and **verify** controller design (classical and learning enabled) with respect to the system dynamics. Its key features are:

* Component based controller design
* Native support for python and C language executables
* Compatibility with external hardware and software processes
* Ease of deployment 

![csaf_importing_components](/uploads/c8ba6291daf48f2ab49270f577576b31/csaf_importing_controllers.png)

Controllers, subsystems and plants are implemented as a collection of components.
Components communicate via a 0MQ pub/sub configuration and serialize/deserialize ROSmsgs. Below is an example of a topology graph of F16 system with GCAS autopilot.


![f16_with_gcas](/uploads/27e47ebbb19aa11d144db1b01435afb0/image.png)

CSAF currently contains two examples, one is F16 with low level LQR controller and GCAS autopilot, and the second one is a classic [Inverted pendulum model](http://ctms.engin.umich.edu/CTMS/index.php?example=InvertedPendulum&section=ControlDigital#4). Both examples are in `examples` directory. 


## Quick Start
CSAF runs inside a [Docker container](https://www.docker.com/), and in order to use CSAF you first need to install docker. CSAF has been tested on Linux (Ubuntu 18.04 and 20.04) and OS X. CSAF can be also run natively on your host machine, but this option is recommended only for the developers.

![csaf_quickstart](/uploads/3fd963be2ef6929d63ceb02ad1a2bcf8/csaf_quickstart.png)

Once you clone the main repository, `run-csaf.sh` is the entry point to the CSAF framework. For a simple start use `-e` flag and select one of the provided examples to run `f16-shield, f16-simple, f16-llc, inv-pendulum`

To get help, type `run-csaf.sh`:

```
./run-csaf.sh -h
CSAF
    Control System Analysis Framework (CSAF) is a middleware framework that
    makes creating and evaluating control systems as simple as possible. Control
    loop topologies and component implementations are specified independently of
    the middleware.

USAGE
   -e      the name of the example { f16-shield, f16-simple, f16-llc, inv-pendulum }
   -c      the name of the model config file (must be in the same directory as your system)
   -d      fully qualified path to the directory defining the model system
   -f      name of the job config file (must be in the same directory as your system)
   -j      launch a jupyter notebook
   -l      build the image locally
   -n      run CSAF natively
   -t      the tag of the image { stable, edge, latest }
   -h      prints the help menu
   -x      clear the output for a particular example/config

EXAMPLES
Run f16-simple example:
    ./run-csaf.sh -e f16-simple
Run f16-simple example natively (not in a docker container):
    ./run-csaf.sh -e f16-simple -n
Start a jupyter notebook with f16 example:
    ./run-csaf.sh -e f16-simple -j
    ./run-csaf.sh -e f16-shield -j
    ./run-csaf.sh -d ${PWD}/examples/f16 -j
Start jupyter notebook with your own example:
    ./run-csaf.sh -j -d ${PWD}/examples/inverted-pendulum
Run f16-shield with your own job config:
    ./run-csaf.sh -e f16-shield -f f16_job_conf.toml
Clear generated outputs for f16 example:
    ./run-csaf.sh -e f16-simple -x
```

## Running examples

To see the F16 model with GCAS autopilot in action, run the following command:

`./run-csaf.sh -e f16-shield`


Once the simulation completes, navigate to `examples/f16/output` to view the 
generated run:

![f16-shield-run](/uploads/7a1f3d000298b9f55baa6adbc712bb6f/f16-shield-run.png)

Here's a quick glance at what's going on behind the scenes:
* The overall system and loop topology is defined in `examples/f16/f16_shield_config.toml`. This file
dictates which components are in the system and how they are connected together.
* Individual components are defined in `examples/f16/components`, where a component
consists of a configuration file and an implementation.
  * For example, the F16 plant
is defined by `examples/f16/components/f16plant.py` and 
`examples/f16/components/f16plant.toml`.
* The middleware message formats that
each component speaks are defined in the ROS message format. The F16 messages 
can be found in `examples/f16/components/msg`.

## Jupyter notebooks

CSAF can be used from within a [jupyter notebook](https://jupyter-notebook.readthedocs.io/en/stable/examples/Notebook/What%20is%20the%20Jupyter%20Notebook.html#Introduction). To start CSAF in the notebook mode, run `./run-csaf.sh -j -e f16-simple` - the `-j` flag specified *notebook mode*, and using the `-e f16-simple` as an example will set the paths necessary for using the F16 model.

CSAF Notebook examples are in `docs/notebooks` directory.

## Development
`CONTRIBUTING.md` contains CSAF development guildelines, please familiarize yourself with the guidelines before opening a pull request. The best way to contact the dev team is via gitlab issues.