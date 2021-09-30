# CSAF (No ZMQ Interpreter)

## Usage

Install a python environment using [requirements.txt](./requirements.txt) 
(its been tested using python 3.9.6, so the package versions may need to be
updated). In this directory, run
```
jupyter notebook
```
Navigate to the notebooks directory and run through the examples.

## Tests

In this directory, run
```
PYTHONPATH=$PWD pytest --mypy -s tests
```
To test the notebooks, run
```
PYTHONPATH=$PWD pytest --nbmake "./notebooks"
```
