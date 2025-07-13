#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 12 13:53:44 2025

@author: joshua
"""

import numpy as np
import matplotlib.pyplot as plt

class Perceptron(object):
    
    def __init__(self, learningRate=0.1, epoch=50, randomState=1):
        self.learningRate = learningRate
        self.epoch        = epoch
        self.randomState  = randomState
        self.error        = []
        self.weights      = []
        
    def Train(self,X,y):
        
        regGen       = np.random.RandomState(self.randomState)
        self.weights = regGen.normal(loc=0.0,scale=0.01,size=1+X.shape[1])
        
        for _ in range(self.epoch):
            errors = 0.0
            for xi, target in zip(X,y):
                update = self.learningRate * (target - self.Predict(xi))
                self.weights[1:] += update * xi
                self.weights[0]  += update
                errors += int(update != 0.0)
            self.error.append(errors)    
        return self;
                
    def NetInput(self,xi):
        return np.dot(self.weights[1:],xi) + self.weights[0] 
    
    def Predict(self,xi):
        return np.where(self.NetInput(xi) >= 0.0, 1.0, -1.0)

        
        

    
        