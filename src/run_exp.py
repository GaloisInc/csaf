from spinup.utils.run_utils import ExperimentGrid
from spinup import ddpg_tf1
import gym
import argparse
import csaf.config as cconf
import csaf.system as csys
import csaf


def run_experiment(args):
    def env_fn():

        # create a csaf configuration out of toml
        my_conf = cconf.SystemConfig.from_toml(
            "/home/averma/csaf_architecture/examples/f16/f16_simple_config.toml")  # "/csaf-system/f16_simple_config.toml")

        # termination condition
        def ground_collision_condition(cname, outs):
            """ground collision premature termnation condition"""
            return cname == "plant" and outs["states"][11] <= 0.0

        # create pub/sub components out of the configuration
        my_system = csys.System.from_config(my_conf)

        # create an environment from the system, allowing us to act as the controller
        my_env = csys.SystemEnv("autopilot", my_system, terminating_conditions=ground_collision_condition)
        # import safety_gym  # registers custom envs to gym env registry
        return my_env #gym.make(args.env_name)

    eg = ExperimentGrid("env_fn()")
    eg.add('env_fn', env_fn)
    eg.add('seed', [0])
    eg.add('epochs', 500)
    eg.add('gamma', 0.9)
    eg.add('steps_per_epoch', 100)
    eg.add('save_freq', 50)
    eg.add('max_ep_len', 10)
    eg.add('update_after', 10)
    #eg.add('ac_kwargs:activation', tf.tanh, '')
    eg.add('ac_kwargs:hidden_sizes', [(256, 256)], 'hid')
    # eg.run(ddpg_tf1)
    eg.run(ddpg_tf1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--cpu", type=int, default=1)
    parser.add_argument('--num_runs', type=int, default=0)
    parser.add_argument('--env_name', type=str, default="Safexp-PointGoal1-v0")
    parser.add_argument('--exp_name', type=str, default='ddpg-9gamma-rawruntil')
    args = parser.parse_args()
    run_experiment(args)
