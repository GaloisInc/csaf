from os.path import join, dirname, realpath
from setuptools import setup
import sys

assert sys.version_info.major == 3 and sys.version_info.minor >= 6, \
    "The Spinning Up repo is designed to work with Python 3.6 and greater." \
    + "Please install it before proceeding."

with open(join("spinup", "version.py")) as version_file:
    exec(version_file.read())

setup(
    name='spinup',
    #py_modules=['spinup'],
    packages=['spinup',
              'spinup.algos',
              'spinup.utils',
              'spinup.algos.tf1',
              'spinup.algos.tf1.ddpg',
              'spinup.algos.tf1.ppo',
              'spinup.algos.tf1.sac',
              'spinup.algos.tf1.td3',
              'spinup.algos.tf1.trpo',
              'spinup.algos.tf1.vpg',
              'spinup.algos.pytorch',
              'spinup.algos.pytorch.ddpg',
              'spinup.algos.pytorch.ppo',
              'spinup.algos.pytorch.sac',
              'spinup.algos.pytorch.td3',
              'spinup.algos.pytorch.trpo',
              'spinup.algos.pytorch.vpg'],
    version=__version__,#'0.1',
    install_requires=[
        'cloudpickle==1.2.1',
        'gym[atari,box2d,classic_control]~=0.15.3',
        'ipython',
        'joblib',
        'matplotlib==3.1.1',
        'mpi4py',
        'numpy',
        'pandas',
        'pytest',
        'psutil',
        'scipy',
        'seaborn==0.8.1',
        'torch==1.3.1',
        'tqdm'
    ],
    description="Teaching tools for introducing people to deep RL.",
    author="Joshua Achiam",
)
