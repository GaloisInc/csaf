from csaf.utils.app import CsafApp
from csaf_examples.rejoin import generate_dubins_system, plot_aircrafts
from csaf_examples.cansat import generate_cansat_system, plot_sats
import numpy as np


if __name__ == '__main__':
    descr = f"CSAF F16 Systems Viewer"
    # chaser initial states (pos + vel)
    sat_states = [[10, -10, -2.0, 2.1],
                [10, -7, 0.7, 0.0],
                [-12, -7, -0.3, 1.0],
                [10, 0, -0.2, .1],
                 [5, 5, .4, -0.2],
                 [-5, 1, 0.0, 0.0]]

    CanSat = generate_cansat_system(sat_states)
    j_states = [[0, 0, np.deg2rad(45)],
              [-5, -10, np.deg2rad(-30)],
            [-3, -15, np.deg2rad(90)],
            [0, -20, np.deg2rad(0)]]

    DubinsRejoin = generate_dubins_system(j_states)
    plotters = {CanSat.__class__: plot_sats, DubinsRejoin.__class__: plot_aircrafts}
    example_systems = ([CanSat, DubinsRejoin])
    capp = CsafApp("F16 Components", description=descr, systems=example_systems, plotters=plotters)
    capp.main()
