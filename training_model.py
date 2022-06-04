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
from torch import optim
import numpy as np
from network import Deepnet
import argparse
import warnings
warnings.simplefilter('ignore')
from metrics import cofusion_matrix, sensitivity, specificity, auc, mcc, accuracy, precision, recall, f1, cutoff, AUPRC
metrics_dict = {"sensitivity":sensitivity, "specificity":specificity, "accuracy":accuracy,"mcc":mcc,"auc":auc,"precision":precision,"recall":recall,"f1":f1,"AUPRC":AUPRC}

amino_asids_vector = np.eye(20)
n_list = ["A", "C", "G", "T"]

def file_input_csv(filename,index_col = None):
    data = pd.read_csv(filename, index_col = index_col)
    return data

class pv_data_sets(data.Dataset):
    def __init__(self, data_sets, enc_model, device):
        super().__init__()
        self.seq = data_sets["seq"].values.tolist()
        self.y = np.array(data_sets["label"].values.tolist()).reshape([len(data_sets["label"]),1])
        self.enc_model = enc_model
        self.device = device

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        inds = self.enc_model[self.seq[idx]]
        
        return inds.to(self.device).long(), torch.tensor(self.y[idx], device = self.device, dtype=torch.float)

class DeepNet():
    def __init__(self, out_path, enc_dict, model_params, training_params, encoding_params, device):
        self.out_path = out_path
        self.enc_model = enc_dict
        self.model_params = model_params
        
        self.tra_batch_size = training_params["training_batch_size"]
        self.val_batch_size = training_params["validation_batch_size"]
        self.lr = training_params["lr"]
        self.max_epoch = training_params["max_epoch"]
        self.early_stop = training_params["early_stopping"]
        self.thresh = training_params["thresh"]
        self.seq_len = encoding_params["seq_len"]
        self.stopping_met = training_params["stopping_met"]
        self.device = device
        
    def model_training(self, train_data_sets, val_data_sets):
        os.makedirs(self.out_path, exist_ok=True)
       
        tra_data_all = pv_data_sets(train_data_sets, enc_model = self.enc_model, device = self.device)
        train_loader = DataLoader(dataset = tra_data_all, batch_size = self.tra_batch_size, shuffle=True)

        val_data_all = pv_data_sets(val_data_sets, enc_model = self.enc_model, device = self.device)
        val_loader = DataLoader(dataset = val_data_all, batch_size = self.val_batch_size, shuffle=True)
        
        self.model = Deepnet(filter_num = self.model_params["filter_num"], feature = self.model_params["feature"], dropout = self.model_params["dropout"], seq_len = self.seq_len).to(self.device)

        self.opt = optim.Adam(params = self.model.parameters(), lr = self.lr)
        self.criterion = torch.nn.BCELoss()
       
        max_met = 100 
        early_stop_count = 0
        
        for epoch in range(self.max_epoch):
            training_losses, validation_losses, train_probs, val_probs, train_labels, val_labels = [], [], [], [], [], []
            self.model.train()
            for i, (inds, labels) in enumerate(train_loader):
                self.opt.zero_grad()
                probs = self.model(inds)
               
                loss = self.criterion(probs, labels)

                loss.backward()
                self.opt.step()
                training_losses.append(loss)
                train_probs.extend(probs.cpu().clone().detach().squeeze(1).numpy().flatten().tolist())
                train_labels.extend(labels.cpu().clone().detach().squeeze(1).numpy().astype('int32').flatten().tolist())
               
            self.criterion = torch.nn.BCELoss()
            loss_epoch = self.criterion(torch.tensor(train_probs).float(), torch.tensor(train_labels).float())

            print("=============================", flush = True)
            print("training loss:: " + str(loss_epoch), flush = True)
            for key in metrics_dict.keys():
                if(key != "auc" and key != "AUPRC"):
                    metrics = metrics_dict[key](train_labels, train_probs, thresh = self.thresh)
                else:
                    metrics = metrics_dict[key](train_labels, train_probs)
                print("train_" + key + ": " + str(metrics), flush=True) 
            
            tn_t, fp_t, fn_t, tp_t = cofusion_matrix(train_labels, train_probs, thresh = self.thresh)
            print("train_true_negative:: value: %f, epoch: %d" % (tn_t, epoch + 1), flush=True)
            print("train_false_positive:: value: %f, epoch: %d" % (fp_t, epoch + 1), flush=True)
            print("train_false_negative:: value: %f, epoch: %d" % (fn_t, epoch + 1), flush=True)
            print("train_true_positive:: value: %f, epoch: %d" % (tp_t, epoch + 1), flush=True)

            print("-----------------------------", flush = True)
            self.model.eval()
            for i, (inds,  labels) in enumerate(val_loader):
                with torch.no_grad():
                    probs = self.model(inds)
                    
                    loss = self.criterion(probs, labels)

                    validation_losses.append(loss)
                    val_probs.extend(probs.cpu().detach().squeeze(1).numpy().flatten().tolist())
                    val_labels.extend(labels.cpu().detach().squeeze(1).numpy().astype('int32').flatten().tolist())
      
            self.criterion = torch.nn.BCELoss()
            loss_epoch = self.criterion(torch.tensor(val_probs).float(), torch.tensor(val_labels).float())
            
            print("validation loss:: "+ str(loss_epoch), flush = True)
            for key in metrics_dict.keys():
                if(key != "auc" and key != "AUPRC"):
                    metrics = metrics_dict[key](val_labels, val_probs, thresh = self.thresh)
                else:
                    metrics = metrics_dict[key](val_labels, val_probs)
                print("validation_" + key + ": " + str(metrics), flush=True)

            if(self.stopping_met == "loss"):
                epoch_met = loss_epoch
            else:
                epoch_met = 1 - metrics_dict[self.stopping_met](val_labels, val_probs)

            tn_t, fp_t, fn_t, tp_t = cofusion_matrix(val_labels, val_probs, thresh = self.thresh)
            print("validation_true_negative:: value: %f, epoch: %d" % (tn_t, epoch + 1), flush=True)
            print("validation_false_positive:: value: %f, epoch: %d" % (fp_t, epoch + 1), flush=True)
            print("validation_false_negative:: value: %f, epoch: %d" % (fn_t, epoch + 1), flush=True)
            print("validation_true_positive:: value: %f, epoch: %d" % (tp_t, epoch + 1), flush=True)
                
            if epoch_met < max_met:
                early_stop_count = 0
                max_met = epoch_met
                os.makedirs(self.out_path, exist_ok=True)
                os.chdir(self.out_path)
                torch.save(self.model.state_dict(), "deep_model")
                final_val_probs = val_probs
                final_val_labels = val_labels
                final_train_probs = train_probs
                final_train_labels = train_labels
                    
            else:
                early_stop_count += 1
                if early_stop_count >= self.early_stop:
                    print('Traning can not improve from epoch {}. Best {}: {}'.format(epoch + 1 - self.early_stop, self.stopping_met, 1 - max_met), flush=True)
                    break

        print(self.thresh, flush=True)
        for key in metrics_dict.keys():
            if(key != "auc" and key != "AUPRC"):
                train_metrics = metrics_dict[key](final_train_labels, final_train_probs, thresh = self.thresh)
                val_metrics = metrics_dict[key](final_val_labels, final_val_probs, thresh = self.thresh)
            else:
                train_metrics = metrics_dict[key](final_train_labels, final_train_probs)
                val_metrics = metrics_dict[key](final_val_labels, final_val_probs)
            print("train_" + key + ": " + str(train_metrics), flush=True)
            print("test_" + key + ": " + str(val_metrics), flush=True)
        
        tn_t, fp_t, fn_t, tp_t = cofusion_matrix(final_val_labels, final_val_probs, thresh = self.thresh)
        print("true_negative:: value: %f" % (tn_t), flush=True)
        print("false_positive:: value: %f" % (fp_t), flush=True)
        print("false_negative:: value: %f" % (fn_t), flush=True)
        print("true_positive:: value: %f" % (tp_t), flush=True)

        threshold_1, threshold_2 = cutoff(final_val_labels, final_val_probs)
        print("Best threshold (AUC) is " + str(threshold_1))
        print("Best threshold (PRC) is " + str(threshold_2))

        return ""

def ps_numbering(seq):
    return [n_list.index(seq[i]) + i * 4 for i in range(len(seq))]

def ps_embed_main(seqs, target_pos):
    seqs = list(set(seqs))
    embed_dict = {}
    for i in range(len(seqs)):
        seq_temp = seqs[i][0:target_pos - 1] + seqs[i][target_pos:]
        embed_dict[seqs[i]] = torch.tensor(ps_numbering(seq_temp))
    return embed_dict

def training_main(train_path, val_path, out_path, t_batch = 64, v_batch = 64, lr = 0.00001, max_epoch = 10000, stop_epoch = 30, thr = 0.5, seq_len = 41, target_pos = 21, device = "cuda:0"):
    print("Setting parameters", flush = True)
    model_params = {"filter_num": 128, "feature": 64, "dropout": 0.2}
    training_params = {"training_batch_size": t_batch, "validation_batch_size": v_batch, "lr": lr, "early_stopping": stop_epoch, "max_epoch": max_epoch, "thresh": thr, "stopping_met": "auc"}
    encoding_params = {"seq_len": seq_len - 1}
    
    print("Loading datasets", flush = True)
    training_data = file_input_csv(train_path)
    validation_data = file_input_csv(val_path)
    
    print("Encoding sequences", flush = True)
    mat_dict = ps_embed_main(training_data["seq"].values.tolist() + validation_data["seq"].values.tolist(), target_pos)
    
    print("Start training a deep neural network model", flush = True)
    net = DeepNet(out_path, mat_dict, model_params, training_params, encoding_params, device)
    _ = net.model_training(training_data, validation_data)
    
    print("Finish processing", flush = True)
    





































