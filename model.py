#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import Transformer,MolDataset
import os
os.chdir('F:\\DTI\\CPIScore\\model')

np.random.seed(1995)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.embedding = nn.Linear(37, 128)
        self.W = nn.ModuleList([nn.Linear(128, 128) for layer in range(2)])
        self.conv = nn.Conv1d(60, 60, 5, padding=2)
        self.bn = nn.BatchNorm1d(5)
        self.pool = nn.MaxPool1d(3, stride=1, padding=1)

        self.fc1 = nn.Linear(1152, 128)
        self.fc2 = nn.Linear(128, 10)
        self.fc3 = nn.Linear(10, 1)
        self.drop = nn.Dropout(0.2)

        self.dnn1 = nn.Linear(512,256)
        self.dnn2 = nn.Linear(256,128)
        self.dnn3 = nn.Linear(128,256)
        self.dnn4 = nn.Linear(256,512)
        self.dnn5 = nn.Linear(256,128)
        self.dnn6 = nn.Linear(128,256)

    def layer_vecl(self,vec):
        vec = F.relu(self.dnn5(vec))
        vec = self.drop(vec)
        vec = F.relu(self.dnn6(vec))
        return vec

    def layer_vecp(self,vec):
        vec = F.relu(self.dnn1(vec))
        vec = self.drop(vec)
        vec = F.relu(self.dnn2(vec))
        vec = self.drop(vec)
        vec = F.relu(self.dnn3(vec))
        vec = self.drop(vec)
        vec = F.relu(self.dnn4(vec))
        return vec

    def forward(self, x, A, l, p, vecl, vecp):
        x = self.embedding(x)
        for layernum in self.W:
            x = layernum(x)
            x = A @ x
            residue = x
            x = F.relu(self.conv(x))
            x = self.pool(x)
            x = self.conv(x)
            x = torch.add(x, residue)
            x = F.relu(x)
        x = x.mean(1)

        vecl = self.layer_vecl(vecl)
        vecp = self.layer_vecp(vecp)
        
        embed_size, num_layers, heads, forward_expansion = 256, 4, 4, 0.5
        tfm = Transformer(device, embed_size, num_layers, heads, forward_expansion).to(device)
      #  tfm_out = tfm(p,l)
        tfm_out = tfm(l,p)

        out = torch.cat((x, vecl, vecp, tfm_out), 1)
        try:
            output = self.fc1(out)
        except:
            print('out_shape:', out.shape)
            out = a + b
        output = self.fc2(output)
        output = self.fc3(output)
        return output

