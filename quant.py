from __future__ import absolute_import, division, print_function
import torch
import torch.nn as nn
from math import *
import numpy as np


class quant():

    def __init__ (self, target_weights, b, moving_aver=0.9, needbias=False):
        super(quant, self).__init__()
        
        #self.model = model
        self.b = b
        self.moving_aver=moving_aver

        self.W = target_weights
        self.preW = []
        self.B = []
        self.v = []
        self.Wmean = []
        self.needbias = needbias

        for (i,weights) in enumerate(self.W):
            self.v.append(torch.max(weights.data)/ (2**self.b[i]-1))
            self.preW.append(weights.data.clone())
            self.B.append(weights.data.clone().zero_())
            # layer-wise mean of target weight
            if self.needbias:
                self.Wmean.append(torch.mean(weights.data))
            else:
                self.Wmean.append(0)

    def update(self, test=False):
        for i in range(len(self.W)):
            # compute Bi with v
            self.B[i] = (self.W[i].data - self.Wmean[i]) / self.v[i]
            self.B[i] = torch.round((self.B[i]-1)/2)*2+1
            self.B[i] = torch.clamp(self.B[i], -(pow(2,self.b[i])-1), pow(2,self.b[i])-1)

            # compute v[i] with B[i]
            vi = torch.sum(torch.mul(self.B[i],(self.W[i].data - self.Wmean[i])) / torch.sum(torch.mul(self.B[i],self.B[i])))
            # update v[i] with moving average
            if not test:
                self.v[i] = self.v[i] * self.moving_aver + vi * (1-self.moving_aver)

            # apply B[i] to W[i]
            self.W[i].data.copy_(self.B[i]*self.v[i] + self.Wmean[i])
            # update Wmean
            if self.needbias:
                self.Wmean[i] = torch.mean(self.W[i].data)

    def restoreW(self):
        for i in range(len(self.W)):
            self.W[i].data.copy_(self.preW[i])

    
    def storeW(self):
        for i in range(len(self.W)):
            self.preW[i].copy_(self.W[i].data)


    def apply(self, test=False):
        self.storeW()
        self.update(test=test)
        return
