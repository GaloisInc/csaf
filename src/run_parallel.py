import sys, os
from multiprocessing import Process, Event, Queue, JoinableQueue
import multiprocessing
import tqdm
import dill
import numpy as np

import csaf.system as csys
import csaf.config as cconf
import csaf.trace as ctc
from csaf import csaf_logger

def save_states_to_file(filename, states):
    np.savetxt(filename, [val['plant'] for val in states], delimiter=",")

def load_states_from_file(filename, component_name):
    x0s = np.loadtxt(filename, delimiter=",")
    return [{component_name : initial_state} for initial_state in x0s]

def gen_fixed_states(bounds, component_name):
    def sanity_check(bounds):
        # sanity check
        for b in bounds:
            assert(len(b) == 1 or len(b) == 3)
            if len(b) == 3:
                # lower bound is always first
                lower = b[0]
                upper = b[1] 
                step = b[2]
                assert(lower <= upper)
                # the step is smaller than the bounds interval
                assert(upper - lower > step)

    def interpolate_bounds(lower, upper, step) -> np.ndarray:
        iters = int((upper - lower)/step)
        return np.linspace(lower, upper, iters)

    sanity_check(bounds)

    # create initial vector
    x0 = np.array([b[0] for b in bounds])
    x0s = [x0]
    # iterate over bounds    
    for idx, b in enumerate(bounds):
        # ignore static values
        if len(b) == 1:
            continue
        vals = interpolate_bounds(b[0],b[1],b[2])
        new_x0s = []
        for x in x0s:
            for val in vals:
                new_x0 = x.copy()
                # ignore the value that already exists
                if new_x0[idx] == val:
                    continue
                new_x0[idx] = val
                new_x0s.append(new_x0)
        x0s += new_x0s

    return [{component_name : initial_state} for initial_state in x0s]

def gen_random_states(bounds, component_name, iterations):
    def generate_single_random_state(bounds):
        sample = np.random.rand(len(bounds))
        ranges = np.array([b[1] - b[0] if len(b) == 2 else b[0] for b in bounds])
        offset = np.array([- b[0] for b in bounds])
        return sample * ranges - offset

    return [{component_name : generate_single_random_state(bounds)} for _ in range(iterations)]

class Worker(Process):
    def __init__(self, evt, config, task_queue, result_queue, progress_queue=None):
        super().__init__()
        self._evt = evt
        self._config = config
        self.system = None
        self._task_queue = task_queue
        self._result_queue = result_queue
        self._progress_queue = progress_queue

    def run(self, on_finish=None):
        self.system = csys.System.from_config(self._config)
        self._evt.set()
        while True:
            next_task = self._task_queue.get()
            if next_task is None:
                self._task_queue.task_done()
                break
            answer = next_task(self.system)
            if self._progress_queue is not None:
                self._progress_queue.put(True)
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
            answer = [self.idx, dill.dumps(ret), self.states]
        except Exception as exc:
            csaf_logger.warning(f"running {self.system_attr} failed for states {self.states}\n<{exc}>")
            answer = [self.idx, exc, self.states]
        return tuple(answer)

    def __str__(self):
        return f"id {self.idx} -- {self.system_attr}(args={self.args}, kwargs={self.kwargs})"


def run_workgroup(n_tasks, config, initial_states, *args, fname="simulate_tspan", show_status=True, **kwargs):
    def progress_listener(q):
        pbar = tqdm.tqdm(total = n_tasks)
        for _ in iter(q.get, None):
            pbar.update()

    csaf_logger.info(f"starting a parallel run of {n_tasks} tasks over the method {fname}")

    # Establish communication queues
    tasks = JoinableQueue()
    results = Queue()
    progress = Queue()

    # Start the progress bar
    if show_status:
        proc = Process(target=progress_listener, args=(progress,))
        proc.start()

    # Start workers
    n_workers = min(n_tasks, multiprocessing.cpu_count() * 2)
    workers = []
    for idx in range(n_workers):
        evt = Event()
        w = Worker(evt, config, tasks, results, progress)
        w.start()
        evt.wait()
        workers.append(w)

    # Enqueue jobs
    for idx in range(n_tasks):
        t = Task(idx, fname, initial_states[idx], *args, **kwargs, show_status=False, return_passed=True)
        tasks.put(t)

    # Stop all workers
    for _ in range(n_workers):
        tasks.put(None)

    # Wait for all of the tasks to finish
    tasks.join()
    if show_status:
        progress.put(None)
        proc.join()


    # Start printing results
    ret = [None] * n_tasks
    while n_tasks:
        result = results.get()
        if not isinstance(result[1], Exception):
            res = dill.loads(result[1])
            ret[result[0]] = tuple([res[1], res[0], result[2]])
        n_tasks -= 1

    csaf_logger.info("parallel run finished")
    ret = [val for val in ret if val != None]
    return ret
