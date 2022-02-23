"""
Utilities to add simple app functionality to libraries using CSAF components
"""
import json
import os.path
import typing
from typing import Sequence, Optional, Type, NamedTuple, Dict, Callable
from csaf.utils import view_block_diagram, open_image
import abc
import matplotlib.pyplot as plt
import numpy as np
import pathlib
import argparse
import sys
import csaf


class CsafApp(abc.ABC):
    """base class for CSAF cli utility classes"""
    @abc.abstractmethod
    def parse_args(self) -> argparse.Namespace:
        """method to collect user input"""
        pass

    @abc.abstractmethod
    def main(self) -> None:
        """main app method"""
        pass

    @property
    def app_name(self) -> str:
        return "CSAF App"


class ScenarioCsafApp(CsafApp):
    """
    Creates a CLI tool to configure and simulate CSAF scenarios

    planned usage:

        #__main__.py
        if __name__ == "__main__":
            ca = ScenarioCsafApp(MyScenarioType)
            ca.main()
    """

    @staticmethod
    def step_done():
        print("DONE")

    def __init__(self, scenario: Type[csaf.Scenario]):
        self._scenario_type = scenario

    def parse_args(self) -> argparse.Namespace:
        """
        tool to parse cli args
        TODO: refine this and generate a more meaningful description
        """
        descr = f"""Input Format: {[f'{k}:{v}' for k, v in self._scenario_type.configuration_space.__annotations__.items()]}.
        Output Fields: {list(self._scenario_type.system_type.components.keys())}.
        """
        ap = argparse.ArgumentParser(description=f"{self.app_name} (Scenario App). " + f"{descr}")
        ap.add_argument('-o', '--output-fname', type=str, default="./traces.out", help="Output Filename")
        ap.add_argument('-t', '--time-max', type=float, default=10.0, help="Timespan to Simulate")
        ap.add_argument('-i', '--input-fname', type=str, default="./scenario.json", help="Input Filename")
        return ap.parse_args()

    def collect_conf(self, in_fname: str) -> Sequence[typing.Any]:
        """given scenario config file in_fname, collect the scenario input vector"""
        assert os.path.isfile(in_fname), f"{in_fname} must be a file!"
        fields = self._scenario_type.configuration_space._fields
        with open(in_fname, "r") as f:
            conf = json.load(f)
        if isinstance(conf, list):
            assert len(fields) == len(conf), f"JSON Input mismatches configuration length " \
                                             f"(Expected {len(fields)}, received {len(conf)})"
            return conf
        elif isinstance(conf, dict):
            req = set(fields) - set(conf.keys())
            if len(req) > 0:
                raise ValueError(f"JSON Input is missing field(s) {req}")
            return [conf[fv] for fv in fields]
        else:
            raise ValueError("JSON Input needs to be a list or dictionary!")

    def dump_output(self, out_fname: str, traces: typing.Dict[str, csaf.TimeTrace]) -> None:
        """given a scenario simulation output file out_fname, write the output"""
        import json
        out_obj = {}
        if os.path.exists(out_fname):
            print(f"\n\tWarning: {out_fname} exists!")
        for n, v in traces.items():
            dat = {vi : np.array(getattr(v, vi)).tolist() for vi in v.NT._fields}
            out_obj[n] = dat
        with open(out_fname, "w") as fp:
            json.dump(out_obj, fp, indent=2)

    def main(self) -> None:
        print(f"Start {self.app_name}")
        ap = self.parse_args()
        print(f"Collecting Configuration '{ap.input_fname}'...", end='')
        conf = self.collect_conf(ap.input_fname)
        self.step_done()
        print(f"Generating System '{self._scenario_type.__name__}'...", end='')
        tspan = (0.0, ap.time_max)
        scenario = self._scenario_type()
        system  = scenario.generate_system(conf)
        self.step_done()
        print(f"Simulating System '{system.__class__.__name__}'...", end='')
        ret: typing.Dict[str, csaf.TimeTrace] = system.simulate_tspan(tspan, show_status=True)
        self.step_done()
        print(f"Writing Output '{ap.output_fname}'...", end='')
        self.dump_output(ap.output_fname, ret)
        self.step_done()
        print("Finished!")

    @property
    def app_name(self) -> str:
        return f"{self._scenario_type.__name__} Scenario App"


class SystemCsafApp(CsafApp):
    """
    for libraries that create CSAF components, create an app layer that allows for CLI access for

    * component search
    * system visualization
    * system simulation

    planned usage:

        #__main__.py
        if __name__ == "__main__":
            ca = SystemCsafApp("My Component Library", my_components, my_systems, my_messages)
            ca.main()
    """
    def __init__(self,
                 app_name: str,
                 description: str = "",
                 components: Optional[Sequence[Type[csaf.Component]]] = None,
                 systems: Optional[Sequence[csaf.System]] = None,
                 messages: Optional[Sequence[Type[NamedTuple]]] = None,
                 plotters: Dict[Type[csaf.System], Callable] = {}):
        self.components = components if components is not None else []
        self.systems = systems if systems is not None else []
        self.system_names = [si.__class__.__name__ for si in self.systems]
        self.messages = messages if messages is not None else []
        self._app_name = app_name
        self.description = description
        self.plotters = plotters

    def parse_args(self):
        """
        tool to parse cli args
        """
        ap = argparse.ArgumentParser(description=f"{self.app_name} (CSAF Library)\n" + self.description)
        example_names = self.system_names
        ap.add_argument('-s', '--system',
                        type=str,
                        default=self.system_names[0],
                        help=f"CSAF System to Simulate (examples are {example_names})")
        ap.add_argument('-o', '--output-dir', type=str, default="./", help="Directory to Store Artifacts")
        ap.add_argument('-t', '--time-max', type=float, default=10.0, help="Timespan to Simulate")
        ap.add_argument('-d', '--do-open', type=str, default='true', help="Open Simulation Result")
        return ap.parse_args()

    def plot_system(self, system, trajs):
        if system.__class__ in self.plotters:
            self.plotters[system.__class__](trajs)
        else:
            # default plotter
            ncols = len(trajs)
            # figure out how many rows there should be
            dims = []
            for n, tr in trajs.items():
                dim = 0
                for field in tr.names:
                    arr = getattr(tr, field)
                    if len(arr) > 0 and field != 'times':
                        dim += len(arr[0])
                dims.append(dim)
            nrows = max(dims)

            # start the plotting
            fig, ax = plt.subplots(figsize=(6*ncols, 3*nrows), ncols=ncols, nrows=nrows, sharex=True)
            for xidx, (n, tr) in enumerate(trajs.items()):
                count = 0
                ax[count][xidx].set_title(n)
                for field in tr.names:
                    if field != 'times':
                        arr = np.array(getattr(tr, field))
                        for i in range(arr.shape[1]):
                            ax[count][xidx].plot(tr.times, arr[:, i])
                            count += 1
                # label the last nonempty row with x-axis
                if count > 0:
                    ax[count-1][xidx].set_xlabel("Time")

                # clear out the unused axes
                for rem in range(count, nrows):
                    fig.delaxes(ax[rem][xidx])
            plt.tight_layout()

    def simulate_app(self, args):
        """cli option to simulate a system"""
        system: csaf.System = self.systems[self.system_names.index(args.system)]
        print(f"Simulating {system.__class__.__name__} over [{0.0},{args.time_max}]...")
        trajs = system.simulate_tspan((0.0, args.time_max), show_status=True)
        self.plot_system(system, trajs)
        out_dir = pathlib.Path(args.output_dir)
        fname = f"{system.__class__.__name__}_simulation.pdf"
        print(f"Saving file {out_dir / fname}...")
        plt.savefig(str(out_dir / fname))
        if args.do_open == 'true':
            open_image(str(out_dir / fname))

    def main(self):
        print(f"{self.app_name} (CSAF Library)")
        args = self.parse_args()
        assert args.system in self.system_names, f"system {args.system} could not be found in the examples"
        # for now, the only implemented feature is to simulate and plot the system
        self.simulate_app(args)
        print("Finished!")

    @property
    def app_name(self) -> str:
        return self._app_name