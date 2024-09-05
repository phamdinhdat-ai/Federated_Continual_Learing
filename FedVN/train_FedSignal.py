import torch 
import os 
import random 
import numpy as np 
import yaml
import copy 

from utils.util_parser import prepare_parser

from dataset.BreathDataset import BreathDataset
from models.time_model import TransAm
import matplotlib.pyplot as plt 
from methods.training_client import train_model
from methods.FedBT import train_Fed_common



device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
max_norm = 10
train_fedgen_feature = True










with open('config.yaml', 'r') as file:
    config = yaml.load(file, Loader=yaml.FullLoader)

root = './data_test'

breath_dataset = BreathDataset(root_dir=root, sequence_length=config['sq_len'], overlap=config['overlap'], n_classes=config['n_classes'])
# breath_dataset.make(mode=1, num_clients=config['n_clients'], show_plots=True, seed = config['seed'],dir_alpha = config['dir_alpha'], lognorm_std = config['lognorm_std'])
breath_dataset.load_data(path="data_test/breathing_10_1024_Dirichlet_0.300")
print("Train data Client: ", breath_dataset.clnt_x.shape)
print("Test data Client: ", breath_dataset.tst_x.shape)
print("Name value: ", breath_dataset.name)


print("="*30)
print("Load Model TimeTransformer...")
model  = TransAm(d_model=100, n_class=12, num_layers=2, dropout=0.2)
print(model.eval())

total_params = 0
for param in model.parameters():
    print(f"Name:{param.has_names()} | Size: {param.size()} | Parameters: {param.numel()}")
    total_params += param.numel()
print("Total Parameter of Transfomer Model: ", total_params)

print("="*30 + "Training with One CLient" + "="*30)
 
######## for 1 
clnt_x = breath_dataset.clnt_x
clnt_y = breath_dataset.clnt_y
tst_x = breath_dataset.tst_x
tst_y = breath_dataset.tst_y

###### normalize three axis to 1 by mean
trn_x = clnt_x[0].mean(axis = 2)
trn_y = clnt_y[0]
tst_x = torch.tensor(tst_x.mean(axis=2))

#### initalize model ### 
model.to(device=device, dtype=torch.float64)
init_model  = copy.deepcopy(model)
init_model.to(device=device, dtype=torch.float64)

######################### training model 
# client_model = train_model(model=init_model,
                        #    trn_x=trn_x, 
                        #    trn_y=trn_y, 
                        #    tst_x=tst_x,
                        #    tst_y=tst_y,
                        #    learning_rate=0.01, 
                        #    batch_size=32, 
                        #    epoch=2, 
                        #    print_per=1, 
                        #    weight_decay=1.0, 
                        #    dataset_name=breath_dataset.dataset_name, 
                        #    sch_gamma=1, 
                        #    sch_step=1)

print("Training FedAvg with Breathing Datasets...")


res_all_perform = train_Fed_common(data_obj=breath_dataset, 
                                   act_prob=config['active_frac'], 
                                   learning_rate=config['lr'], 
                                   batch_size=config['batch_size'], 
                                   epoch=config['epoch'],
                                   com_amount=config['comm_amount'], 
                                   print_per=config['print_seq'],
                                   weight_decay=config['weight_decay'],
                                   model_func=model, 
                                   init_model= init_model,
                                   sch_gamma=config['sch_gamma'],
                                   sch_step=config['sch_step'], 
                                   save_period=True, 
                                   rand_seed=config['seed'],
                                   lr_decay_per_round=2)
print("training done!")




