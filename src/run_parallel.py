import sys, os
from multiprocessing import Process, Event, Queue, JoinableQueue
import multiprocessing

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
        print(f"{proc_name}: Starting")
        while True:
            next_task = self._task_queue.get()
            if next_task is None:
                print(f'{proc_name}: Exiting')
                self._task_queue.task_done()
                break
            answer = next_task(self.system)
            self._task_queue.task_done()
            self._result_queue.put(answer)
            self.system.reset()
        return


class Task(object):
    def __init__(self, idx, system_attr, states, *args, **kwargs):
        self.idx = idx
        self.system_attr = system_attr
        self.args = args
        self.kwargs = kwargs
        self.states = states

    def __call__(self, system: csys.System):
        for cname, cstate in self.states:
            system.set_state(cname, cstate)
        assert hasattr(system, self.system_attr)
        try:
            ret = getattr(system, self.system_attr)(*self.args, **self.kwargs)
            answer = [self.idx, ret]
        except Exception as exc:
            answer = [self.idx, exc]
        # some ugliness to get around the unpicklable named tuple
        if self.system_attr == "simulate_tspan":
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


def run_workgroup(n_tasks, config, states, *args):
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
    for idx in range(n_tasks):
        t = Task(idx, "simulate_tspan", states[idx], *args, show_status=False)
        print(t)
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
    ## build and simulate system
    csaf_dir=sys.argv[1]
    csaf_config=sys.argv[2]

    ## system to run in parallel
    config_filename = os.path.join(csaf_dir, csaf_config)
    model_conf = cconf.SystemConfig.from_toml(config_filename)

    # number of jobs to run
    n_tasks = 4

    # define states of component to run
    # format [{"plant": <list>}, ..., {"plant" : <list>, "controller": <list>}]
    states = [{},] * n_tasks

    # run 128 tasks in a workgroup
    print(run_workgroup(n_tasks, model_conf, states, (0.0, 35.0)))
