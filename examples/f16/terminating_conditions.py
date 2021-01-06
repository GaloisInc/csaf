# NOTE: the terminating conditions are static and state-less
# In other words, it is essentially a check on state bounds
def ground_collision(cname, outs) -> bool:
        """ground collision"""
        return cname == "plant" and outs["states"][11] <= 0.0


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
