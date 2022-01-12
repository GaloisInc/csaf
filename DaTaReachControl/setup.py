from setuptools import setup

with open("README.md", 'r') as f:
    long_description = f.read()

setup(
   name='DaTaReachControl',
   version='0.1',
   description='A module for the over-approximation of the reachable set'+\
                    ' and control of unknown dynamical systems using data from a'+\
                    ' single trajectory',
   license="GNU 3.0",
   long_description=long_description,
   author='Franck Djeumou',
   author_email='fdjeumou@utexas.edu',
   url="https://github.com/wuwushrek/DaTaReachControl.git",
   packages=['DaTaReachControl'],
   package_dir={'DaTaReachControl': 'DaTaReachControl/'},
   install_requires=['numpy', 'scipy'],
   tests_require=['pytest', 'pytest-cov'],
)
