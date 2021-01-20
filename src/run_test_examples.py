""" CSAF Test Infrastructure Demo

Demonstrates how to run the test objects in CSAF

Approach:
    1. Define properties as python functions that map a state to a boolean
    2. Define a reward function for the optimizer
    3. Define a region of state space to run the test (boundaries)
    4. Create systems to test
    5. Specify the tests
"""
from tests_static import BayesianFalsifierTest, StaticRunTest
from csaf.config import SystemConfig
import csaf.test as cst


## Goal properties
def ground_collision(cname, outs) -> bool:
    """ground collision"""
    return cname == "plant" and outs["states"][11] <= 0.0


def low_airspeed(cname, outs) -> bool:
    return cname == "plant" and outs["states"][0] <= 200.0


def high_pitch(cname, outs) -> bool:
    """pitch"""
    return cname == "plant" and abs(outs["states"][4]) >= 0.8


# Bopt Reward Function
def reward_func(trajs):
    import numpy as np

    altitude_min = 0  # ft AGL
    altitude_max = 4000  # ft AGL 45000
    nz_max = 15  # G's original is 9
    nz_min = -3  # G's original is -2
    # ps_max_accel_deg = 500  # /s/s

    v_min = 300  # ft/s
    v_max = 2500  # ft/s
    alpha_min_deg = -10  # deg
    alpha_max_deg = 45  # deg
    beta_max_deg = 30  # deg

    # did not consider the change rate of ps here
    constraints_dim = [0, 1, 2, 11, 13]
    constraints_box = np.array([[v_min, alpha_min_deg, -beta_max_deg, altitude_min, nz_min]
                                   , [v_max, alpha_max_deg, beta_max_deg, altitude_max, nz_max]])

    states = np.hstack((np.array(trajs["plant"].states), np.array(trajs["plant"].outputs)))
    dist_to_lb = np.abs(states[:, constraints_dim] - constraints_box[0])
    dist_to_ub = np.abs(states[:, constraints_dim] - constraints_box[1])

    min_dist = np.min(np.array([dist_to_ub, dist_to_lb]), axis=0)
    norm_min_dist = min_dist / (constraints_box[1] - constraints_box[0])

    return np.mean(norm_min_dist)


## Region of the state space to tests
bounds = {"plant": [[200.0, 1000.0, 100.0], # vt [ft/s]
                    [0.0, 0.0, 0.0], # alpha [rad]
                    [0.0, 0.0, 0.0], # beta [rad]
                    [0.0, 0.0, 0.0], # phi [rad]
                    [-0.5, 0.5, 0.2], # theta [rad]
                    [0.0, 0.0, 0.0], # psi [rad]
                    [0.0, 0.0, 0.0], # p [rad/s]
                    [0.0, 0.0, 0.0], # q [rad/s]
                    [0.0, 0.0, 0.0], # r [rad/s]
                    [0.0, 0.0, 0.0], # pn [m]
                    [0.0, 0.0, 0.0], # pe [m]
                    [2000.0, 4000.0, 500.0], # h [ft]
                    [9.0, 9.0, 0.0]]}


## Systems to test
f16_gcas = SystemConfig.from_toml("../examples/f16/f16_simple_config.toml")
f16_signal_Ny = SystemConfig.from_toml("../examples/f16/f16_llc_analyze_config.toml")
f16_signal_Nz = SystemConfig.from_toml("../examples/f16/f16_llc_analyze_config.toml")
initial_conditions = [{"plant": [b[0] for b in bounds["plant"]]},]*4


# Example custom test
class F16PlantTest(cst.SystemTest):
    """example of an architectural static test -- look at system config"""
    def execute(self, system_conf: SystemConfig):
        plant_params = system_conf.get_component_settings("plant")["config"]["parameters"]
        try:
            assert plant_params['g'] > 0.0
            assert plant_params["model"] in {"morelli", "morrison"}
        except Exception as exc:
            self.logger("warn", f"failed f16 plant params test {exc}")
            return False
        return True


## Test description
tests = {
    "test_f16_plant" : {
      "type": "F16PlantTest",
      "system": f16_gcas
    },
    "test_falsify_airspeed":
        {
            "type" : "BayesianFalsifierTest",
            "system": f16_gcas,
            "property": low_airspeed,
            "reward": reward_func,
            "region" : bounds
        },
    "test_falsify_gcas":
        {
            "type" : "BayesianFalsifierTest",
            "system": f16_gcas,
            "property": ground_collision,
            "reward": reward_func,
            "region" : bounds
         },
    "test_falsify_pitch":
        {
            "type" : "BayesianFalsifierTest",
            "system": f16_gcas,
            "property": high_pitch,
            "reward": reward_func,
            "region" : bounds
        },
    "test_overshoot_nz": {
        "type" : "StaticRunTest",
        "tspan": [0, 35.0],
        "system": f16_gcas,
        "x0": initial_conditions,
        "terminating_conditions": ground_collision,
        "generator_config": {"output_idx": 0},
        "reference": ["autopilot", 0],
        "response": ["plant", 0],
        "fcn_name": "max_norm_deviation"
    },
    "test_overshoot_ny": {
        "type" : "StaticRunTest",
        "tspan": [0, 35.0],
        "system": f16_gcas,
        "x0": initial_conditions,
        "terminating_conditions": ground_collision,
        "generator_config": {"output_idx": 1},
        "reference": ["autopilot", 0],
        "response": ["plant", 0],
        "fcn_name": "max_norm_deviation"
    }
}


# Run the tests
example_path = "../examples/f16/"
for tname, tconf in tests.items():
    print(f"Running Test {tname}")
    test_type, system_conf = tconf["type"], tconf["system"]
    del tconf["type"]
    del tconf["system"]
    bft = globals()[test_type](example_path)
    bft.parse(tconf)
    ret = bft.execute(system_conf)
    print(f"Finished Running Tests {tname}")
    if isinstance(ret, bool):
        if ret:
            print(f"SUCCESS -- Test {tname} Passed Successfully")
        else:
            print(f"FAILED -- Test {tname} Failed")
