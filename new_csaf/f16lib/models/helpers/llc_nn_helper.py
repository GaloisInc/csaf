import numpy as np
import f16lib.models.helpers.llc_helper as lh


class LowLevelControllerNN(lh.FeedbackController):
    def __init__(self, ddpg_model, model):
        self.ddpg_model = ddpg_model

        def ctrl_fn(x):
            return self.ddpg_model.predict(x)

        # self.tf_model = load_ddpg_model()

        # These are the same as of the LQR
        # TODO: They are needed for autopilots (GCAS and FixedAltitudeAutopilot)
        # to mantain speed near the trim point. Remove them.
        self.xequil = np.array([502.0, 0.03887505597600522, 0.0, 0.0, \
                                0.03887505597600522, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1000.0, \
                                9.05666543872074])
        self.uequil = np.array([0.13946204864060271, -0.7495784725828754, 0.0, 0.0])
        super().__init__(lh.CtrlLimits(), model, ctrl_fn, self.xequil, self.uequil)
