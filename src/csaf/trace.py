import collections
import csv
import datetime
import warnings

import numpy as np


class Trace(collections.abc.Sequence):
    """
    A sequence of named vars which can be iterated.
    """
    def __init__(self, elements_str, init_trace=None):
        self.NT = collections.namedtuple('Variables', elements_str)
        if init_trace is None:
            self.data = self.NT(*[[] for i in range(len(self.NT._fields))])
        else:
            #TODO: clean this up with better error handling and more informational error msg
            # check if all traces are of the same length
            assert(len(set(len(i) for i in init_trace)) == 1)
            self.data = self.NT(*init_trace)

        self.names = self.NT._fields

        for name in self.names:
            setattr(self, name, getattr(self.data, name))

    def append(self, **kwargs):
        if set(self.names) != kwargs.keys():
            raise TypeError(f'missing one of required positional arguments: {self.names}')

        for name, val in kwargs.items():
            getattr(self.data, name).append(val)

    def __eq__(self, trace2):
        # check types
        assert(set(self.names) == set(trace2.names))

        if len(self) != len(trace2):
            return False

        #TODO: Check exact values at each index

        raise NotImplementedError

    def get_element(self, elem_str):
        """
        Gets the element of the trace using its string.
        """
        warnings.warn('depricated. Use __getitem__/[] instead.')
        return getattr(self, elem_str)

    def __getitem__(self, i):
        if isinstance(i, str):
            return getattr(self, i)
        #TODO: index based accessor
        elif isinstance(i, (int)):
            return [getattr(self, n)[i] for n in self.names]
        else:
            raise TypeError('Expected int or str.')

    def __len__(self):
        return len(self.data._asdict[self.names[0]])

    def save(self, filename):
        raise NotImplementedError


#TODO: Improve this by further splitting the outputs and states into named var names
class TimeTrace(Trace):
    """
    A time trace/trajectory object which can be
        - plotted
        - replayed
        - genrelized
            - searched for duplication/cached
            - investigate symbolic representations?
    At the fundamental level, this is just a time series, but can be specialized
    towards a trajectory of a dynamical system and the enviroment's evolution in
    time.
    """
    def __init__(self, elements_str, init_trace=None, time_str='times'):
        super().__init__(elements_str, init_trace)
        self.time_str = time_str
        if(time_str not in self.names):
            raise AssertionError(f'time variable ({time_str}) missing from the time trace: {self.names}')


    def __eq__(self, trace2, eps=1e-2):
        data1 = self.data
        data2 = trace2.data

        # check types
        assert(set(data1._fields) == set(data2._fields))

        raise NotImplementedError

    # TODO: Implement an interpolator which computes the value of the trace
    # at the given time: t -> x
    def get_trace_at_t(self):
        raise NotImplementedError

    def time_length(self):
        """
        Return the last time stamp.
        Use len() to find the length of the sequence.
        """
        return self.times[:-1]

    #TODO: Implement a pretty printer
    #def __str__(self):
        #return str(self.data.times)

    def csv(self, header=None, filename=None):
        """
        Save simulation trajectory to a csv file
        times [n * 1]
        modes [n * 1]
        states [n * 16]
        ps [n * 1]
        Nz [n * 1]
        u [n * 7]
        """
        if filename is None:
            filename = datetime.now().strftime('%Y-%m-%d_%H:%M') + '_log.csv'

        print('Saving to ' + filename)

        # TODO: Remove this hard coding
        if header is None:
            warnings.warn('hardcoded headers for csv. Will be removed.')
            header = ['time',
                'VT', 'alpha', 'beta', 'phi', 'theta', 'psi', 'P', 'Q', 'R', 'pn', 'pe', 'h', 'pow',
                'ps', 'Nz',
                'thrust', 'elevator','aileron','rudder',
                'mode']

        with open(filename, 'w') as csvfile:
            wrt = csv.writer(csvfile, delimiter=',')
            wrt.writerow(header)
            rows = []
            for d in self:
                rows.append(np.hstack(d))
            wrt.writerows(rows)

    def np2csv(self, header):
        """
        Saves assuming numpy arrays as fields.
        Not yet implemented.
        """
        raise NotImplementedError
        data = np.vstack((
            np.array(self.times).reshape(len(self), 1),
            np.vstack(self.states),
            np.vstack(self.outputs),
            np.vstack(self.u),
            np.array(self.modes).reshape(len(self), 1)))
        np.savetxt('ss.csv', data, delimiter=',', header=','.join(header))

    @classmethod
    def fromListofArrays(cls, **kwargs):
        trace = cls(kwargs.keys(), kwargs.values())
        return trace

    @classmethod
    def from_named_tuple(cls, named_tuple):
        raise NotImplementedError
