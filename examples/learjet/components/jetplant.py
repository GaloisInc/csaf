import numpy as np
from collections import namedtuple
import typing as typ
from learjet_data import *


def make_state_update(p: LearJetParams) -> typ.Callable:
    """create the state update function, given learjet parameters"""
    Alat = np.array([[p.xu, p.xw, p.xq-p.w0, -p.g*np.cos(np.deg2rad(p.theta0))],
                     [p.zu, p.zw, p.zq+p.u0, -p.g*np.sin(np.deg2rad(p.theta0))],
                     [p.mu, p.mw, p.mq,      0.0],
                     [0.0,  0.0,  1.0,       0.0]])
    Blat = np.array([[p.xde, p.xdt],
                     [p.zde, p.zdt],
                     [p.mde, p.mdt],
                     [0.0,     0.0]])
    
    Alon = np.array([[p.yv, p.yp+p.w0, p.yr-p.u0,                    p.g*np.cos(np.deg2rad(p.theta0))],
                     [p.lv, p.lp,      p.lr,                         0.0],
                     [p.nv, p.np,      p.nr,                         0.0],
                     [0.0,  1.0,       np.tan(np.deg2rad(p.theta0)), 0.0]])
    Blon= np.array([[p.yda, p.ydr],
                    [p.lda, p.ldr],
                    [p.nda, p.ndr],
                    [0.0,   0.0]])
    def inner(t: float, x: typ.Tuple[float], u: typ.Tuple[float]) -> typ.Tuple[float]:
        x = np.array(x)[permute_states_idxs]
        u = np.array(u)[permute_inputs_idxs]
        xlat, xlon = x[:4], x[4:]
        ulat, ulon = u[:2], u[2:]
        xlatdot = Alat @ np.array(xlat)[:, np.newaxis] + Blat @ np.array(ulat)[:, np.newaxis]
        xlondot = Alon @ np.array(xlon)[:, np.newaxis] + Blon @ np.array(ulon)[:, np.newaxis]
        xdot = np.concatenate((xlatdot.flatten(), xlondot.flatten()))[permute_states_idxs]
        return tuple(xdot.tolist())
    return inner


def make_output(p: LearJetParams) -> typ.Callable:
    """create the output function, given learjet parameters"""
    Clat = np.array([[0.0, 0.0,    1.0,       0.0],
                     [0.0, 1/p.u0, 0.0,       0.0],
                     [p.xu, p.xw,  p.xq,      0.0],
                     [p.zu, p.zw,  p.zq,      0.0],
                     [p.xu, p.xw,  p.xq-p.w0, -p.g*np.cos(np.deg2rad(p.theta0))],
                     [p.zu, p.zw,  p.zq+p.u0, -p.g*np.sin(np.deg2rad(p.theta0))]])
    Dlat = np.array([[0.0,   0.0],
                     [0.0,   0.0],
                     [p.xde, p.xdt],
                     [p.zde, p.zdt],
                     [p.xde, p.xdt],
                     [p.zde, p.zdt]])
    Clon = np.array([[0.0,      1.0,       0.0,       0.0],
                     [0.0,      0.0,       1.0,       0.0],
                     [p.yv,     p.yp,      p.yr,      0.0],
                     [1/p.vtot, 0.0,       0.0,       0.0],
                     [p.yv,     p.yp+p.w0, p.yr-p.u0, p.g * np.cos(np.deg2rad(p.theta0))]])
    Dlon = np.array([[0.0,    0.0],
                     [0.0,    0.0],
                     [p.yda, p.ydr],
                     [0.0,    0.0],
                     [p.yda, p.ydr]])
    def inner(t: float, x: typ.Tuple[float], u: typ.Tuple[float]) -> typ.Tuple[float]:
        x = np.array(x)[permute_states_idxs]
        u = np.array(u)[permute_inputs_idxs]
        xlat, xlon = x[:4], x[4:]
        ulat, ulon = u[:2], u[2:]
        ylat = Clat @ np.array(xlat)[:, np.newaxis] + Dlat @ np.array(ulat)[:, np.newaxis]
        ylon = Clon @ np.array(xlon)[:, np.newaxis] + Dlon @ np.array(ulon)[:, np.newaxis]
        y = np.concatenate((ylat.flatten(), ylon.flatten()))[permute_outputs_idxs]
        return tuple(y.tolist())
    return inner


def model_init(model):
    """function to load resources needed by the controller
    """
    if model.load_type == "light":
        params = ljparam0
    elif model.load_type == "heavy":
        params = ljparams1
    model.parameters["update_fcn"] = make_state_update(params)
    model.parameters["out_fcn"] = make_output(params)


def model_output(model, time_t, state_jet, input_v):
    return model.out_fcn(model, time_t, state_jet, input_v)


def model_state_update(model, time_t, state_jet, input_v):
    return model.update_fcn(model, time_t, state_jet, input_v)
