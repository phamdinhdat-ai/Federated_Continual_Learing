import torch 
import torch.nn as nn
import torch.nn.functional as F 
import random
import shutil
import numpy as np
from pathlib import Path
from tqdm import tqdm
from torch.distributions import Dirichlet, Categorical, LogNormal
import os 
import matplotlib.pyplot as plt 
from typing import List, Optional, Dict
from torch.utils.data import Dataset


class BreathDataset(Dataset):
    
    def __init__(self,root_dir, sequence_length= 20, overlap = 0.1, n_classes = 12, n_clients=10, seed = 1024, dir_alpha = 0.3, lognorm_std = None, rule="non_iid"):
        self.sequence_length = sequence_length
        self.root_dir = root_dir
        self.n_classes = n_classes
        self.n_clients = n_clients
        self.step  = int(sequence_length - overlap * sequence_length)
        self.dir_alpha = dir_alpha
        self.dataset_name = 'breathing'
        self.name = "%s_%d_%d_%s_%s" % (self.dataset_name, self.n_clients, seed, rule, self.sequence_length)
        self.name += '_%f' % self.dir_alpha if self.dir_alpha != 0 else ''
        self.seed = seed 
        self.lognorm_std = lognorm_std
        self.download_load()
        
        super(BreathDataset, self).__init__()
    def download_load(self):
        trainset = np.load(file="data/dataset/trainset_16.npy")[:, 1:]# for static dataset, there are 12 classes
        testset = np.load(file="data/dataset/testset_16.npy")[:, 1:]
        valset = np.load(file="data/dataset/val_3.npy")[:, 1:]
        self.train_data, self.train_label = self.slice_window(data= trainset[:, :3], labels=trainset[:, -1])
        self.test_data, self.test_label = self.slice_window(data= testset[:, :3], labels=testset[:, -1])
        self.val_data, self.val_label = self.slice_window(data= valset[:, :3], labels=valset[:, -1])
        
        self.train_data = torch.from_numpy(self.train_data)
        self.train_label = torch.from_numpy(self.train_label)
        self.val_data = torch.from_numpy(self.val_data)
        self.val_label = torch.from_numpy(self.val_label)
        self.test_data = torch.from_numpy(self.test_data)
        self.test_label = torch.from_numpy(self.test_label)
        print("Generate Slice Window data successfully")
        self.num_train_data = len(self.train_data)
        self.num_test_data = len(self.val_data)
        
    def make(self, 
             mode:int = 0, 
             num_clients:int = 10 ,
             show_plots:bool = False, 
             **kwargs):
        
        save_name = "%s_%d_%d_%s_%s" %  (self.dataset_name, num_clients, self.seed, 'Dirichlet', '%.3f' % self.dir_alpha)
        
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        random.seed(self.seed)

        if os.path.exists(f"{self.root_dir}/{save_name}"):
            shutil.rmtree(f"{self.root_dir}/{save_name}")
        client_data_path = Path(f"{self.root_dir}/{save_name}")
        client_data_path.mkdir()
        
        if mode == 0: #IID
            data_ids = torch.randperm(self.num_train_data, dtype=torch.int32)
            num_data_per_client = self.num_train_data // num_clients
            if not isinstance(self.train_label, torch.Tensor):
                self.train_label = torch.tensor(self.train_label)
            pbar = tqdm(range(num_clients), desc=f"{self.dataset_name}| IID: ")
            for i in pbar:
                client_path = Path(client_data_path / str(i))
                client_path.mkdir()
                
                train_data = [self.train_data[j] for j in data_ids[i * num_data_per_client: (i +1 ) * num_data_per_client]]
                pbar.set_postfix({"# Data / Client": num_data_per_client})
            if show_plots:
                self._plot(self.train_label, title=f"Client {i+1} Data Distribution") 
        elif mode == 1: # 
            num_data_per_client  = self.num_train_data // num_clients
            class_sampler = Dirichlet(torch.empty(self.n_classes).fill_(self.dir_alpha))
            if not isinstance(self.train_label, torch.Tensor):
                self.train_label = torch.tensor(self.train_label)

            assigned_ids = []
            pbar = tqdm(range(num_clients), desc = f"{self.dataset_name} Non-IID Balanced: ")
            clnt_x = []
            clnt_y = []
            
            for i in pbar:
                
                p_ij  = class_sampler.sample()
                weights = torch.zeros(self.num_train_data)
                for c_id in range(self.n_classes):
                    weights[self.train_label == c_id] = p_ij[c_id]
                weights[assigned_ids] = 0.0 
                
                
                data_ids = torch.multinomial(weights, num_data_per_client, replacement=False)
                train_data = [self.train_data[j] for j in data_ids]
                train_label = [self.train_label[j] for j in  data_ids]
                train_data_x = [data_.numpy() for data_ in train_data]
                train_data_y = [label_ for label_ in train_label]
                pbar.set_postfix({'# data / Client': len(train_data)})

                assigned_ids += data_ids.tolist()

                clnt_x.append(np.array(train_data_x).astype(np.float32))
                # print(clnt_x[-1].shape)
                clnt_y.append(np.array(train_data_y).astype(np.int64).reshape(-1, 1))

                if show_plots:
                    self._plot(train_label, title=f"Client {i+1} Data Distribution")

            clnt_x = np.asarray(clnt_x)
            clnt_y = np.asarray(clnt_y)
            np.save(client_data_path / 'clnt_x.npy', clnt_x)
            np.save(client_data_path / 'clnt_y.npy', clnt_y)
            np.save(client_data_path / 'tst_x.npy', self.test_data.numpy())
            np.save(client_data_path / 'tst_y.npy', self.test_label.numpy())    
             
        elif mode == 2:     # Non IID Unbalanced
            num_data_per_client = self.num_train_data // num_clients
            num_data_per_class = self.num_train_data / (self.n_classes * num_clients)
            classs_sampler = Dirichlet(torch.empty(self.n_classes).fill_(self.dir_alpha))

            assigned_ids = []
            pbar = tqdm(range(num_clients), desc = f"{self.dataset_name} Non-IID Unbalanced: ")

            if not isinstance(self.train_label, torch.Tensor):
                self.train_label = torch.tensor(self.train_label)

            for i in pbar:
                train_data = []
                train_label = []
                client_path = Path(client_data_path / str(i))
                client_path.mkdir()
                # Compute class prior probabilities for each client
                p_ij = classs_sampler.sample()  # Share of jth class for ith client (always sums to 1)
                c_sampler = Categorical(p_ij)
                data_sampler = LogNormal(torch.tensor(num_data_per_class).log(),
                                        self.lognorm_std)

                while(True):
                    num_data_left = num_data_per_client - len(train_data)
                    c = c_sampler.sample()
                    num_data_c = int(data_sampler.sample())
                    # print(c, num_data_c, len(train_data))
                    data_ids = torch.nonzero(self.train_label == c.item()).flatten()
                    # data_ids = [x for x in data_ids if x not in assigned_ids] # Remove duplicated ids
                    # print(data_ids.shape)
                    num_data_c = min(num_data_c, data_ids.shape[0])
                    if num_data_c >= num_data_left :
                        train_data += [self.train_data[j] for j in data_ids[:num_data_left]]
                        train_label += [self.train_label[j] for j in data_ids[:num_data_left]]
                        break
                    else:
                        train_data += [self.train_data[j] for j in data_ids[:num_data_c]]
                        train_label += [self.train_label[j] for j in data_ids[:num_data_c]]
                        
                        assigned_ids += data_ids[:num_data_c].tolist()

                pbar.set_postfix({'# data / Client': len(train_data)})
                torch.save(train_data,
                           client_data_path / str(i) / "train_data.pth")
                torch.save(train_label,
                           client_data_path / str(i) / "train_label.pth")
                if show_plots:
                    self._plot(train_label, title=f"Client {i+1} Data Distribution")

            else:
                raise ValueError("Unknown mode. Mode must be {0,1}")
            

    
    def load_data(self, path):
        self.clnt_x = np.load(f'{path}/clnt_x.npy')
        self.clnt_y = np.load(f'{path}/clnt_y.npy')
        self.tst_x = np.load(f'{path}/tst_x.npy')
        self.tst_y = np.load(f'{path}/tst_y.npy')
        
        
    def _plot(self, labels:List,  title: str = None) -> None:
        # labels = [int(d[1]) for d in data]
        # print(labels)
        
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.hist(labels, bins=np.arange(self.n_classes + 1) - 0.5)
        ax.set_xticks(range(self.n_classes))
        ax.set_xlim([-1, self.n_classes])
        ax.set_title(title, fontsize=15)
        ax.set_xlabel('Class ID', fontsize=13)
        ax.set_ylabel('# samples', fontsize=13)
        plt.tight_layout()
        plt.show()        
    
    
           
    def slice_window(self, data, labels):
        X_local = []
        y_local = []
        for start in range(0, data.shape[0] - self.sequence_length, self.step):
            end = start + self.sequence_length
            X_local.append(data[start:end])
            y_local.append(labels[end-1])
        return np.array(X_local), np.array(y_local)