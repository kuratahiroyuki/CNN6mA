
import sys
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
common_path = os.path.abspath("./")
import argparse

import warnings
warnings.simplefilter('ignore')

from training_model import training_main
from prediction import pred_main

def training_deep_model(args):
    train_data_path = args.training_file
    val_data_path = args.validation_file
    out_path = args.out_dir
    t_batch = args.training_batch_size
    v_batch = args.validation_batch_size
    lr = args.learning_rate
    max_epoch = args.max_epoch_num
    stop_epoch = args.early_stopping_epoch_num
    threshold = args.threshold
    seq_len = args.sequence_length
    target_pos = args.target_pos
    device = args.device
    
    training_main(train_data_path, val_data_path, out_path, t_batch = t_batch, v_batch = v_batch, lr = lr, max_epoch = max_epoch, stop_epoch = stop_epoch, thr = threshold, seq_len = seq_len, target_pos = target_pos, device = device)
    
def prediction(args):
    in_path = args.import_file
    out_path = args.out_dir
    deep_model_path = args.deep_model_file
    threshold = args.threshold
    batch_size = args.batch_size
    vec_ind = args.vec_index
    seq_len = args.sequence_length
    target_pos = args.target_pos
    device = args.device
    
    pred_main(in_path = in_path, out_path = out_path, deep_model_path = deep_model_path, vec_ind = vec_ind, thr = threshold, batch_size = batch_size, seq_len = seq_len, target_pos = target_pos, device = device)
    
    
def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="sub_command", help="inter6mA: this is a CLI to use Inter6mA")
    subparsers.required = True
    
    deep_train_parser = subparsers.add_parser("train", help = "sub-command <train> is used for training a deep learning model for 6mA sites")
    pred_parser = subparsers.add_parser("predict", help = "sub-command <predict> is used for prediction of 6mA sites")
    
    deep_train_parser.add_argument('-t', '--training_file', help = 'Path of training data file (.csv)', required = True)
    deep_train_parser.add_argument('-v', '--validation_file', help = 'Path of validation data file (.csv)', required = True)
    deep_train_parser.add_argument('-o', '--out_dir', help = 'Directory to output results', required = True)
    deep_train_parser.add_argument('-t_batch', '--training_batch_size', help = 'Training batch size', default = 64, type = int)
    deep_train_parser.add_argument('-v_batch', '--validation_batch_size', help = 'Validation batch size', default = 64, type = int)
    deep_train_parser.add_argument('-lr', '--learning_rate', help = 'Learning rate', default = 0.00001, type = float)
    deep_train_parser.add_argument('-max_epoch', '--max_epoch_num', help = 'Maximum epoch number', default = 10000, type = int)
    deep_train_parser.add_argument('-stop_epoch', '--early_stopping_epoch_num', help = 'Epoch number for early stopping', default = 30, type = int)
    deep_train_parser.add_argument('-thr', '--threshold', help = 'Threshold to determined whether interact or not', default = 0.5, type = float)
    deep_train_parser.add_argument('-seq_len', '--sequence_length', help = 'Sequence_length', default = 41, type = int)
    deep_train_parser.add_argument('-tp', '--target_pos', help = 'Modification site position', default = 21, type = int)
    deep_train_parser.add_argument('-device', '--device', help = 'Device to be used', default = "cuda:0")
    deep_train_parser.set_defaults(handler = training_deep_model)
    
    pred_parser.add_argument('-i', '--import_file', help = 'Path of data file (.csv)', required = True)
    pred_parser.add_argument('-o', '--out_dir', help = 'Directory to output results', required = True)
    pred_parser.add_argument('-d', '--deep_model_file', help = 'Path of a trained Inter6mA model', required = True)
    pred_parser.add_argument('-vec', '--vec_index', help = 'Flag whether features output', action='store_true', default = False)
    pred_parser.add_argument('-thr', '--threshold', help = 'Threshold to determined whether interact or not', default = 0.5, type = float)
    pred_parser.add_argument('-batch', '--batch_size', help = 'Batch size', default = 64, type = int)
    pred_parser.add_argument('-seq_len', '--sequence_length', help = 'Sequence_length', default = 41, type = int)
    pred_parser.add_argument('-tp', '--target_pos', help = 'Modification site position', default = 21, type = int)
    pred_parser.add_argument('-device', '--device', help = 'Device to be used', default = "cuda:0")
    pred_parser.set_defaults(handler = prediction)
    
    args = parser.parse_args()
    if hasattr(args, 'handler'):
        args.handler(args)
    
   



























