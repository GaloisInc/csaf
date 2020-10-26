""" Fuzzy Logic Low Level Controller Model

TODO: implement the TODOs
"""
import numpy as np
from fileops import prepend_curr_path
from helpers import lqr
from f16llc import get_x_ctrl, clip_u, model_state_update


def Index(x,i,d,Centers):
    if i == 0 and x < Centers[0]:
        Index = 1
    elif i == d and x > Centers[d]:
        Index = i-1
    elif x >= Centers[i] and x <= Centers[i+1]:
        Index = i
    else:
        # TODO: what happened here?
        Index = Modeno(x,i+1,d,Centers)
    return Index


def Multwolist(list1,list2):
    ResList = []
    for j in range(0,len(list2)):
        t = [list1[i] * list2[j] for i in range(len(list1))]
        ResList.extend(t)
    return ResList


def ModeInput(ni,x,Centers):
    InputM = [1,x[0]]
    for i in range(1,ni):
        b = [1,x[i]]
        InputM = Multwolist(InputM,b)
    return InputM


def Mode(nC,x,Centers):
    m = 0
    count = 1
    for j in range(0,nC):
        index = Index(x[j],0,len(Centers[j])-1,Centers[j])
        m = m + index*count
        count = count*(len(Centers[j])-1)
    return m


def Inference(nC,Input,Centers,Gains):
    ni = len(Input)
    ResList = ModeInput(ni,Input,Centers)
    ResInput = np.reshape(ResList, (1,2**ni))
    m = Mode(nC,Input,Centers)
    K = np.reshape(Gains[:,m], (2**ni,1))
    u1 = np.dot(ResInput,K)
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
