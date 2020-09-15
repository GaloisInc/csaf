# C/C++ Components in CSAF

This example demonstrates how SWIG can be setup to interface with 
programs compiled from C/C++ source code. The sources define interfaces and
implement a simple proportional-integral-derivative (PID) controller. The `Makefile`
creates a shared object (`.so` extension), as well as a thin python wrapper that allows
the model python programs to interact with them.

## Instructions

Build the shared object and python wrapper. Based on the python environment, the variables
`PYTHON_INCLUDE` and `PYTHON_LIB` will need to be set. For example, if a conda enviroment is used,
the setup may look like
```bash
export PYTHON_INCLUDE=/Users/username/opt/anaconda3/envs/env_name/include/python3.6m
export PYTHON_LIB=/Users/username/opt/anaconda3/envs/env_name/lib
```
Also, if building on MacOS, the linker will need different flags. So, run using the flags
```
LDFLAGS = -bundle -flat_namespace -undefined suppress
```
Once the variables are properly set, run
```
(cd src; make)
```
After this has finished running, the file `pidc.py` should be in the current working directory. Next, install
with
```
(cd src; make install)
```
to add the wrapper to the `./components` path.

Now, the system defined in `ip_pid_config.toml` can be run successfully.