import argparse
import typing
import csaf
import pathlib
from csaf.utils import open_image
import matplotlib.pyplot as plt
import csaf_f16.systems as f16sys
from csaf_f16.plot import plot_simple


descr = f"CSAF F16 Systems Viewer"


def parse_args():
    ap = argparse.ArgumentParser(description=descr)
    example_names = ([n for n, c in f16sys.__dict__.items() if n.startswith('F16')])
    ap.add_argument('-s', '--system', type=str, default='F16Simple', help=f"CSAF System to Simulate (examples are {example_names})")
    ap.add_argument('-o', '--output-dir', type=str, default="./", help="Directory to Store Artifacts")
    ap.add_argument('-t', '--time-max', type=float, default=10.0, help="Timespan to Simulate")
    ap.add_argument('-d', '--do-open', type=str, default='true', help="Open Simulation Result")
    return ap.parse_args()


def main(args):
    print("Running F16 Viewer...")
    assert hasattr(f16sys, args.system), f"system {args.system} could not be found in the examples"
    system: typing.Type[csaf.System] = getattr(f16sys, args.system)
    print(f"Simulating {system.__class__.__name__} over [{0.0},{args.time_max}]...")
    trajs = system().simulate_tspan((0.0, args.time_max), show_status=True)
    plot_simple(trajs)
    out_dir = pathlib.Path(args.output_dir)
    fname = f"{system.__class__.__name__}_simulation.pdf"
    print(f"Saving file {out_dir / fname}...")
    plt.savefig(str(out_dir / fname))
    if args.do_open == 'true':
        open_image(str(out_dir / fname))
    print("Finished!")


if __name__ == '__main__':
    main(parse_args())