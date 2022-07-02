#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 14 16:08:34 2021

@author: kurata
"""

import os
import pandas as pd
import torch
import torch.utils.data as data
from torch.utils.data import DataLoader
import numpy as np
from network import Deepnet
import warnings
warnings.simplefilter('ignore')
import argparse
import joblib
from metrics import cofusion_matrix, sensitivity, specificity, auc, mcc, accuracy, precision, recall, f1, cutoff, AUPRC
metrics_dict = {"sensitivity":sensitivity, "specificity":specificity, "accuracy":accuracy,"mcc":mcc,"auc":auc,"precision":precision,"recall":recall,"f1":f1,"AUPRC":AUPRC}

amino_asids_vector = np.eye(20)
n_list = ["A", "C", "G", "T"]

def file_input_csv(filename,index_col = None):
    data = pd.read_csv(filename, index_col = index_col)
    return data

def save_joblib(filename, data):
    with open(filename, "wb") as f:
        joblib.dump(data, f, compress = 3)
        
def output_csv(filename, data, index=False):
    data.to_csv(filename, index = index)
    

class pv_data_sets(data.Dataset):
    def __init__(self, data_sets, enc_model, device):
        super().__init__()
        self.seq = data_sets["seq"].values.tolist()
        self.enc_model = enc_model
        self.device = device

    def __len__(self):
        return len(self.seq)

    def __getitem__(self, idx):
        inds = self.enc_model[self.seq[idx]]
        
        return self.seq[idx], inds.to(self.device).long()

class DeepNet():
    def __init__(self, out_path, deep_path, enc_dict, model_params, pred_params, encoding_params, vec_ind, target_pos, device):
        self.out_path = out_path
        self.deep_path = deep_path
        self.enc_model = enc_dict
        self.model_params = model_params
        self.vec_ind = vec_ind
        self.target_pos = target_pos
        
        self.batch_size = pred_params["batch_size"]
        self.thresh = pred_params["thresh"]
        self.seq_len = encoding_params["seq_len"]
        self.device = device
        
    def prediction(self, data_sets):
        os.makedirs(self.out_path, exist_ok=True)
       
        data_all = pv_data_sets(data_sets, enc_model = self.enc_model, device = self.device)
        loader = DataLoader(dataset = data_all, batch_size = self.batch_size)
        
        self.model = Deepnet(filter_num = self.model_params["filter_num"], feature = self.model_params["feature"], dropout = self.model_params["dropout"], seq_len = self.seq_len).to(self.device)
        self.model.load_state_dict(torch.load(self.deep_path, map_location = self.device)) 

        probs_all = []
        
        if(self.vec_ind == True):
            score_vec_list = []
        
        self.model.eval()
        for i, (seq, inds) in enumerate(loader):
            with torch.no_grad():
                probs = self.model(inds)
                
                probs_list = probs.cpu().detach().squeeze(1).numpy().flatten().tolist()
                probs_all.extend(list(zip(seq, probs_list)))
     
                if(self.vec_ind == True):
                    score_vec_list.extend(self.model.h_d.cpu().clone().detach().numpy())

        probs_all = pd.DataFrame(probs_all, columns = ["seqs", "scores"])
        output_csv(self.out_path + "/prediction_results.csv", probs_all)
        
        if(self.vec_ind == True):
            score_vec_list = np.array(score_vec_list)
            score_vec_list = score_vec_list.reshape([score_vec_list.shape[0], score_vec_list.shape[1]])
            output_csv(self.out_path + "/score_vec_list.csv", pd.DataFrame(score_vec_list, index = ["sample_{}".format(i + 1) for i in range(score_vec_list.shape[0])], columns = ["position_{}".format(i + 1) if i + 1 < self.target_pos else "position_{}".format(i + 2) for i in range(score_vec_list.shape[1])]), index = True)

def numbering(seq):
    return [n_list.index(seq[i]) for i in range(len(seq))]

def embed_main(seqs, target_pos):
    seqs = list(set(seqs))
    embed_dict = {}
    for i in range(len(seqs)):
        seq_temp = seqs[i][0:target_pos - 1] + seqs[i][target_pos:]
        embed_dict[seqs[i]] = torch.tensor(numbering(seq_temp))
    return embed_dict

def pred_main(in_path, out_path, deep_model_path, vec_ind = False, thr = 0.5, batch_size = 64, seq_len = 41, target_pos = 21, device = "cuda:0"):
    print("Setting parameters", flush = True)
    model_params = {"filter_num": 128, "feature": 64, "dropout": 0.2}
    pred_params = {"batch_size": batch_size, "thresh": thr}
    encoding_params = {"seq_len": seq_len - 1}
    
    print("Loading datasets", flush = True)
    data = file_input_csv(in_path)
    
    print("Encoding sequences", flush = True)
    mat_dict = embed_main(data["seq"].values.tolist(), target_pos)
    
    print("Start prediction", flush = True)
    net = DeepNet(out_path, deep_model_path, mat_dict, model_params, pred_params, encoding_params, vec_ind, target_pos, device)
    _ = net.prediction(data)
    
    print("Finish processing", flush = True)
    




































