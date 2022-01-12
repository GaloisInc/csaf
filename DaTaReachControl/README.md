# On-The-Fly Control of Unknown Smooth Systems from Limited Data

The library depends on ```numba```, ```numpy```, and if available Gurobi in order to work.

If Gurobi is not available, only the idealistic control formulation can be used. In this case,
the parameter ```gurobiSolver=False``` should be set when constructing an instance of ```DaTaControl```.
Furthermore,  only the IDEALISTIC_APG param can be used (see ```control.py```) for more documentation.

To install the package
```
cd DaTaReachControl

python -m pip install -e .
```
