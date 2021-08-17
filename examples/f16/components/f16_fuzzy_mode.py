#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed October 21 16:30:30 2020
@author: Kuldip Rattan, Modified by Ethan Lew
Class for the Fuzzy Mode controller inner loop for the F16 simulation
"""
import numpy as np
import sys
import itertools
import math
from numpy import savetxt
import os
import inspect


def construct_path(path):
    assert(len(path) >= 1)
    return os.path.join(path[0], *path[1:])


def prepend_curr_path(path):
    callers_path = inspect.stack()[1].filename
    return os.path.join(os.path.dirname(callers_path), construct_path(path))


class F16ModeController():
    def __init__(self):
        super().__init__()
        self.CentersLong = np.load(prepend_curr_path(('../', "CentersLong.npy")))
        self.CentersLat = np.load(prepend_curr_path(('../', "CentersLat.npy")))
        self.GainsAileron = np.load(prepend_curr_path(('../', "GainsAileron.npy")))
        self.GainsElevator = np.load(prepend_curr_path(('../', "GainsElevator.npy")))
        self.GainsRudder = np.load(prepend_curr_path(('../', "GainsRudder.npy")))

    def Controller(self,Input):
        # Assumes an eight member vector of inputs in the following order
        # Elevator Input states [0,1,2] Longitudal Controller
        # Aileron and Rudder Inputs states [3,4,5,6,7] Equivalent Lateral Controller
        Elevator =  self.Inference(len(self.CentersLong),Input[0:3],self.CentersLong,self.GainsElevator)
        Aileron = self.Inference(len(self.CentersLat),Input[3:8],self.CentersLat,self.GainsAileron)
        Rudder = self.Inference(len(self.CentersLat),Input[3:8],self.CentersLat,self.GainsRudder)
        return ([Elevator,Aileron,Rudder])

    def Index(self,x,i,d,Centers):
        if i == 0 and x < Centers[0]:
            Ind = i
        elif i == d and x > Centers[d]:
            Ind = i-1
        elif x >= Centers[i] and x <= Centers[i+1]:
            Ind = i
        else:
            Ind = self.Index(x,i+1,d,Centers)
        return Ind

    def Multwolist(self,list1,list2):
        ResList = []
        for j in range(0,len(list2)):
            t = [list1[i] * list2[j] for i in range(len(list1))]
            ResList.extend(t)
        return ResList

    def ModeInput(self,ni,x,Centers):
        InputM = [1,x[0]]
        for i in range(1,ni):
            b = [1,x[i]]
            InputM = self.Multwolist(InputM,b)
        return InputM

    def Mode(self,nC,x,Centers):
        m = 0
        count = 1
        for j in range(0,nC):
            Ind = self.Index(x[j],0,len(Centers[j])-1,Centers[j])
            m = m + Ind*count
            count = count*(len(Centers[j])-1)
        return m

    def Inference(self,nC,Input,Centers,Gains):
        ni = len(Input)
        ResList = self.ModeInput(ni,Input,Centers)
        ResInput = np.reshape(ResList, (1,2**ni))
        m = self.Mode(nC,Input,Centers)
        K = np.reshape(Gains[:,m], (2**ni,1))
        u1 = np.dot(ResInput,K)
        u = [j for sub in u1 for j in sub]
        return u


