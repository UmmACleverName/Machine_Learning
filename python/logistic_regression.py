#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 13 18:56:49 2025

@author: joshua
"""

import numpy as np

class LogisticRegression(object):
    
    def __init__(self, learningRate=0.1, epoch=50, randomState=1):
        self.learningRate = learningRate
        self.epoch        = epoch
        self.randomState  = randomState
        self.error        = []
        self.weights      = []
        
    def Train(self,X,y):
        
        regGen       = np.random.RandomState(self.randomState)
        self.weights = regGen.normal(loc=0.0,scale=0.01,size=1+X.shape[1])
        self.error   = []
        
        for _ in range(self.epoch):
            netInput = self.NetInput(X)
            outPut   = self.Activation(netInput)
            #Error Update
            sigma  = y - outPut
            update = X.T.dot(sigma)
            self.weights[1:] += self.learningRate * update
            self.weights[0] += self.learningRate * sigma.sum()
            errors = (sigma**2).sum() / 2.0
            self.error.append(errors)
            
        return self;
    
    def Activation(self,z):
        return (1.0 / (1.0 + np.exp(-np.clip(z,-250,250))))
                
    def NetInput(self,xi):
        return np.dot(xi,self.weights[1:]) + self.weights[0] 
    
    def Predict(self,xi):
        return np.where(self.Activation(self.NetInput(xi)) >= 0.5, 1.0, 0.0)