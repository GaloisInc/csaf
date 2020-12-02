def ground_collision(cname, outs) -> bool:
        """ground collision"""
        return cname == "plant" and outs["states"][11] <= 0.0