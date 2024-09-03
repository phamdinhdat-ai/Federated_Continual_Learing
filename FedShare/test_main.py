import numpy as np 
import torch
import pandas as pd 
import os 
import random
import copy

from torch.utils.data import DataLoader, Dataset
from dataset.SleepDataset import SleepDataset
from src.nets import LSTMModel 
from src.update import ModelTrainer, SleepSplit

from utils.options import args_parser
from utils.distribute import uniform_distribute, train_dg_split, train_dg_split_sleep, uniform_distribute_sleep
from utils.sampling import iid, noniid

dim = 3 
sq_len = 100
out_class = 12 


args   = args_parser()
dataset = SleepDataset(sequence_length=sq_len, overlap=0.4, n_classes=out_class)


model = LSTMModel(input_dim=3, hidden_dim=100, layer_dim=1, output_dim=12)
# print(model.eval)
# y = model(x)
# print(y.size())


dg  = copy.deepcopy(dataset)
dataset_train = copy.deepcopy(dataset)

dg_idx, dataset_train_idx = train_dg_split_sleep(dataset, args)
print(len(dg_idx))
print(len(dataset_train_idx))

dg.train_data, dataset_train.train_data = dataset.train_data[dg_idx], dataset.train_data[dataset_train_idx]
dg.train_label, dataset_train.train_label = dataset.train_label[dg_idx], dataset.train_label[dataset_train_idx]


model.train()
w_glob = model.state_dict()

print(w_glob)
initialization = ModelTrainer(args, dataset, dataset_train_idx)
