import os, sys
import json
import typing as types


def model_output(model, time_t, state_controller, input_pendulum):
    if model['maneuver_name'] == "step":
        m = 0.05
        xm = 9.0
        if time_t > 0.0 and time_t < xm*2:
            return [-m*(time_t - xm)**2 + m* xm**2]
        else:
            return [0.0]
    elif model['maneuver_name'] == "const":
        return [0.0]


if __name__ == "__main__":
    while True:
        rawIn = sys.stdin.readline()
        js = json.loads(rawIn)
        if not isinstance(js, types.Mapping):
            print('ERROR: expected a JSON object describing the RPC, but got ' + str(rawIn))
            sys.exit()
        elif not ('function' in js and 'model' in js and 'time' in js and 'state' in js and 'input' in js):
            print('ERROR: expected JSON RPC object to have keys [function, model, time, state, input], but got ' + str(rawIn))
            sys.exit()
        elif js['function'] == 'model_output':
            result = model_output(js['model'], js['time'], js['state'], js['input'])
            sys.stdout.write(json.dumps(result).replace('\n', ' ').replace('\r', ''))
            sys.stdout.write('\n')
            sys.stdout.flush()
        else:
            sys.stdout.write(json.dumps([]).replace('\n', ' ').replace('\r', ''))
            sys.stdout.write('\n')
            sys.stdout.flush()
