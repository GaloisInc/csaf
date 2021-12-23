from csaf.utils.app import CsafApp
import csaf_f16.systems as f16sys
from csaf_f16.plot import plot_simple, plot_shield


if __name__ == '__main__':
    descr = f"CSAF F16 Systems Viewer"
    plotters = {f16sys.F16Simple: plot_simple, f16sys.F16Shield: plot_shield}
    example_systems = ([c() for n, c in f16sys.__dict__.items() if n.startswith('F16')])
    capp = CsafApp("F16 Components", description=descr, systems=example_systems, plotters=plotters)
    capp.main()