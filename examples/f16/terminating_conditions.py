# NOTE: the terminating conditions are static and state-less
# In other words, it is essentially a check on state bounds
def ground_collision(cname, outs) -> bool:
        """ground collision"""
        return cname == "plant" and outs["states"][11] <= 0.0