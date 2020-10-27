""" Fuzzy Logic Low Level Controller Model

TODO: implement the TODOs
"""
import numpy as np
from fileops import prepend_curr_path
from helpers import lqr
from f16llc import get_x_ctrl, clip_u, model_state_update


def Index(x, i, d, centers):
    if i == 0 and x < centers[0]:
        idx = 0
    elif i == d and x > centers[d]:
        idx = i-1
    elif x >= centers[i] and x <= centers[i + 1]:
        idx = i
    else:
        # TODO: what happened here?
        idx = Index(x, i + 1, d, centers)
    return idx


def Multwolist(list1,list2):
    res_list = []
    for j in range(0,len(list2)):
        t = [list1[i] * list2[j] for i in range(len(list1))]
        res_list.extend(t)
    return res_list


def ModeInput(ni, x, centers):
    input_m = [1,x[0]]
    for i in range(1,ni):
        b = [1,x[i]]
        input_m = Multwolist(input_m,b)
    return input_m


def Mode(nC, x, centers):
    m = 0
    count = 1
    for j in range(0,nC):
        index = Index(x[j], 0, len(centers[j]) - 1, centers[j])
        m = m + index*count
        count = count*(len(centers[j]) - 1)
    return m


def Inference(nC, input_m, centers, gains):
    ni = len(input_m)
    res_list = ModeInput(ni, input_m, centers)
    res_input = np.reshape(res_list, (1,2**ni))
    m = Mode(nC, input_m, centers)
    K = np.reshape(gains[:, m], (2 ** ni, 1))
    u1 = np.dot(res_input,K)
    u = [j for sub in u1 for j in sub]
    return u


def model_init(model):
    """function to load resources needed by the controller

    save them in the model parameters, and access them anywhere model
    is passed
    """
    long_path = prepend_curr_path(('../', 'CentersLong.npy'))
    lat_path = prepend_curr_path(('../', 'CentersLat.npy'))
    ail_path = prepend_curr_path(('../', 'GainsAileron.npy'))
    ele_path = prepend_curr_path(('../', 'GainsElevator.npy'))
    rud_path = prepend_curr_path(('../', 'GainsRudder.npy'))
    model.parameters["centers_long"] = np.load(long_path)
    model.parameters["centers_lat"] = np.load(lat_path)
    model.parameters["gains_aileron"] = np.load(ail_path)
    model.parameters["gains_elevator"] = np.load(ele_path)
    model.parameters["gains_rudder"] = np.load(rud_path)


def compute_fcn(model, x_ctrl):
    """compute 3-dim control signal from 7-dim x_ctrl signal"""
    #TODO: implement this
    #show that inference (added in model_init) can be accessed
    Elevator =  Inference(len(model.centers_long), x_ctrl[0:3], model.centers_long, model.gains_elevator)
    Aileron = Inference(len(model.centers_lat), x_ctrl[3:8], model.centers_lat, model.gains_aileron)
    Rudder = Inference(len(model.centers_lat), x_ctrl[3:8], model.centers_lat, model.gains_rudder)
    return np.array([Elevator,Aileron,Rudder]).flatten()


def model_output(model, time_t, state_controller, input_all):
    """ get the reference commands for the control surfaces """
    _, *trim_points = getattr(lqr, model.lqr_name)()

    x_f16, _y, u_ref = input_all[:13], input_all[13:17], input_all[17:]
    x_ctrl = get_x_ctrl(trim_points, np.concatenate([x_f16, state_controller]))

    # Initialize control vectors
    u_deg = np.zeros((4,))  # throt, ele, ail, rud
    u_deg[1:4] = compute_fcn(model, x_ctrl)

    # Set throttle as directed from output of getOuterLoopCtrl(...)
    u_deg[0] = u_ref[3]

    # Add in equilibrium control
    u_deg[0:4] += trim_points[1]
    u_deg = clip_u(model, u_deg)

    return u_deg
