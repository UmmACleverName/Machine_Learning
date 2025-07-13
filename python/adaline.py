#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 13 09:38:27 2025

@author: joshua

Note: ADAptive LInear NEuron (Adaline)
Improvement on the perceptron algorithm.

"""



import numpy as np

class Adaline(object):
    
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
    
    def Activation(self,x):
        return x 
                
    def NetInput(self,xi):
        return np.dot(xi,self.weights[1:]) + self.weights[0] 
    
    def Predict(self,xi):
        return np.where(self.Activation(self.NetInput(xi)) >= 0.0, 1.0, -1.0)