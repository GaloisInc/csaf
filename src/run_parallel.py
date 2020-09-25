import sys, os
from multiprocessing import Process, Event, Queue, JoinableQueue
import multiprocessing

import tqdm

import csaf.system as csys
import csaf.config as cconf
import csaf.trace as ctc


class Worker(Process):
    def __init__(self, evt, config, task_queue, result_queue):
        super().__init__()
        self._evt = evt
        self._config = config
        self.system = None
        self._task_queue = task_queue
        self._result_queue = result_queue

    def run(self):
        proc_name = self.name
        self.system = csys.System.from_config(self._config)
        self._evt.set()
        while True:
            next_task = self._task_queue.get()
            if next_task is None:
                self._task_queue.task_done()
                break
            answer = next_task(self.system)
            self._task_queue.task_done()
            self._result_queue.put(answer)
            self.system.reset()
        return


class Task(object):
    def __init__(self, idx, system_attr, initial_states, *args, **kwargs):
        self.idx = idx
        self.system_attr = system_attr
        self.args = args
        self.kwargs = kwargs
        self.states = initial_states

    def __call__(self, system: csys.System):
        for cname, cstate in self.states.items():
            system.set_state(cname, cstate)
        assert hasattr(system, self.system_attr)
        try:
            ret = getattr(system, self.system_attr)(*self.args, **self.kwargs)
            answer = [self.idx, ret]
        except Exception as exc:
            answer = [self.idx, exc]
        # some ugliness to get around the unpicklable named tuple
        if self.system_attr == "simulate_tspan" and isinstance(answer[1], dict):
            answer_picklable = {}
            for k, v in answer[1].items():
                if isinstance(v, ctc.TimeTrace):
                    fields = [a for a in dir(v.NT) if not a.startswith('_') and a not in ["count", "index"]]
                    tt_dict = {fieldn: getattr(v, fieldn) for fieldn in fields}
                    answer_picklable[k] = tt_dict
            answer[1] = answer_picklable
        return tuple(answer)

    def __str__(self):
        return f"id {self.idx} -- {self.system_attr}(args={self.args}, kwargs={self.kwargs})"


def run_workgroup(n_tasks, config, initial_states, *args, fname="simulate_tspan", **kwargs):
    # Establish communication queues
    tasks = JoinableQueue()
    results = Queue()

    # Start workers
    n_workers = min(n_tasks, multiprocessing.cpu_count() * 2)
    workers = []
    for idx in range(n_workers):
        evt = Event()
        w = Worker(evt, config, tasks, results)
        w.start()
        evt.wait()
        workers.append(w)

    # Enqueue jobs
    for idx in tqdm.tqdm(range(n_tasks)):
        t = Task(idx, fname, initial_states[idx], *args, **kwargs, show_status=False)
        tasks.put(t)

    # Stop all workers
    for _ in range(n_workers):
        tasks.put(None)

    # Wait for all of the tasks to finish
    tasks.join()

    # Start printing results
    ret = [None] * n_tasks
    while n_tasks:
        result = results.get()
        ret[result[0]] = result[1]
        n_tasks -= 1
    return ret


if __name__ == '__main__':
    import numpy as np
    import matplotlib.pyplot as plt

    def term_condition(cname, outs):
        """ground collision"""
        return cname == "plant" and outs["states"][11] <= 0.0

    def gen_random_state(bounds):
        sample = np.random.rand(len(bounds))
        ranges = np.array([b[1] - b[0] for b in bounds])
        offset = np.array([- b[0] for b in bounds])
        return sample * ranges - offset

    bounds = [(200, 1500),
              (np.deg2rad(2.1215-0.6), np.deg2rad(2.1215+0.6)),
              (0.0, 0.0),
              ((np.pi/2)*0.5, (np.pi/2)*0.5),
              (-np.pi, np.pi),
              (-np.pi/4, -np.pi/4 ),
              (0.0, 0.0),
              (-0.1, 0.1),
              (0.0, 0.0),
              (0.0, 0.0),
              (0.0, 0.0),
              (500, 8000),
              (9, 9)]

    ## build and simulate system
    csaf_dir=sys.argv[1]
    csaf_config=sys.argv[2]
    state_index = int(sys.argv[3])

    ## system to run in parallel
    config_filename = os.path.join(csaf_dir, csaf_config)
    model_conf = cconf.SystemConfig.from_toml(config_filename)

    # number of jobs to run
    n_tasks = 16

    # define states of component to run
    # format [{"plant": <list>}, ..., {"plant" : <list>, "controller": <list>}]
    states = [{"plant" : gen_random_state(bounds)} for _ in range(n_tasks)]

    # run tasks in a workgroup
    runs = run_workgroup(n_tasks, model_conf, states, (0.0, 35.0), terminating_conditions=term_condition)
    altitudes = [np.array(r["plant"]["states"])[:, state_index] for r in runs if not isinstance(r, Exception)]
    times = [r["plant"]["times"] for r in runs if not isinstance(r, Exception)]
    fig, ax = plt.subplots(figsize=(12, 3 * len(altitudes)), nrows=len(altitudes), sharex=True)
    for idx, traces in enumerate(zip(times, altitudes)):
        ax[idx].plot(*traces)
        ax[idx].set_ylabel(f"Run {idx}")
    ax[-1].set_xlabel("Time (s)")
    ax[0].set_title("Simulation Workgroup Runs")
    plt.show()
