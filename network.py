#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 27 23:41:16 2021

@author: tsukiyamashou
"""
import math
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import torch.nn.utils.rnn as rnn
import copy

class PS_denseNet(nn.Module):
    def __init__(self, window_size = 7, filter_num = 256, feature = 128, seq_len = 41):
        super(PS_denseNet, self).__init__()
        self.filter_num = filter_num
        self.feature = feature
        self.window_size = window_size
        self.seq_len = seq_len
        self.pad_len = int(self.window_size/2)

        self.dense_layer_net = []
        for i in range(self.seq_len):
            self.dense_layer_net.append(
                nn.Sequential(
                nn.Linear(self.window_size * self.feature, filter_num),
                nn.ReLU(),
                nn.Dropout(0.2)
                )
            )
        self.dense_layer_net = nn.ModuleList(self.dense_layer_net)

    def forward(self, input_mat):
        h_d = torch.transpose(input_mat, 1, 2)
        h_d = F.pad(h_d, (self.pad_len, self.pad_len), "constant", 0)
        h_d = torch.transpose(h_d, 1, 2)

        h_d_temp = []
        for i in range(self.pad_len, self.seq_len + self.pad_len):
            h_d_temp.append(self.dense_layer_net[i - self.pad_len](h_d[:, i - self.pad_len: i + self.pad_len + 1, :].reshape(-1, self.window_size * self.feature)))

        h_d = torch.stack(h_d_temp)
        h_d = torch.transpose(h_d, 0, 1)

        return h_d

class Deepnet(nn.Module):
    def __init__(self, filter_num = 128, feature = 64, dropout = 0.2, seq_len = 41):
        super(Deepnet, self).__init__()
        feature_num = 2

        self.embeddings = nn.Embedding(seq_len * 4, feature)


        self.dense_net_3 = PS_denseNet(window_size = 3, filter_num = filter_num, feature = feature, seq_len = seq_len)
        self.dense_net_5 = PS_denseNet(window_size = 5, filter_num = filter_num, feature = feature, seq_len = seq_len)
        
        self.HiddenDense = nn.Linear(filter_num * feature_num, filter_num * 2)

        self.LastDense = nn.Sequential(
            nn.Linear(filter_num * 2, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
            nn.Sigmoid()
            )

    def forward(self, inds):
        h_d = self.embeddings(inds)

        hd_3 = self.dense_net_3(h_d)
        hd_5 = self.dense_net_5(h_d)

        h_d = torch.cat((hd_3, hd_5), axis = 2)

        h_d = h_d.unsqueeze(2).expand(-1, -1, h_d.size(1), -1) + h_d.unsqueeze(1).expand(-1, h_d.size(1), -1, -1)
        h_d = self.HiddenDense(h_d)
        h_d, _ = torch.max(h_d, dim = 1)

        self.h_d = self.LastDense(h_d)

        score, _ = torch.max(self.h_d, dim = 1)

        return score
    
        
"""
a = torch.randn(32, 41, 64).float()
b = torch.randn(32, 41, 64).float()

model = Deepnet(feature = 64, seq_len = 41)
temp = model(a)

print(model.state_dict())
#temp = temp.detach().clone().numpy()
"""



    
