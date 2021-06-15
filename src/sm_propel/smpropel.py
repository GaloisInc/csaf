import numpy as np
import random
import argparse
from keras.models import model_from_json, Model
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.optimizers import Adam
import tensorflow as tf
import json
import logging
import sys
import datetime
import time
import functools
import copy
from bayes_opt import BayesianOptimization
from scipy import spatial
from neural_update import NeuralAgent
from controllers import Controller
from utils import *


def learn_policy():

    all_observations = []
    all_actions = []
    for i_iter in range(6):
        logging.info("Iteration {}".format(i_iter))
        # Learn/Update Neural Policy
        if i_iter == 0:
            nn_agent.update_neural([steer_prog, accel_prog, brake_prog], episode_count=200)
        else:
            nn_agent.update_neural([steer_prog, accel_prog, brake_prog], episode_count=100)

        # Collect Trajectories
        observation_list, action_list = nn_agent.collect_data([steer_prog, accel_prog, brake_prog])
        all_observations += observation_list
        # Relabel Observations
        all_actions = nn_agent.label_data([steer_prog, accel_prog, brake_prog], all_observations)

        # Learn new programmatic policy
        param_finder = ParameterFinder(all_observations, all_actions, steer_prog, accel_prog, brake_prog)

        pid_ranges = [steer_ranges, accel_ranges, brake_ranges]
        new_paras = param_finder.pid_parameters(pid_ranges)

        prog.update_parameters([new_paras['max_params'][i] for i in ['sp0', 'sp1', 'sp2']], new_paras['max_params']['spt'])
        
    return None



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--trackfile', default='practgt2.xml')
    parser.add_argument('--seed', default=None)
    parser.add_argument('--logname', default='AdaptiveProgramIPPG_')
    args = parser.parse_args()

    random.seed(args.seed)
    logPath = 'logs'
    logFileName = args.logname + args.trackfile[:-4]
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s",
        handlers=[
            logging.FileHandler("{0}/{1}.log".format(logPath, logFileName)),
            logging.StreamHandler(sys.stdout)
        ])
    logging.info("Logging started with level: INFO")
    learn_policy(track_name=args.trackfile)
