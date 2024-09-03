import torch 
import torch.nn as nn 
import torch.nn.functional as F

import numpy as np
from torch.utils.data import DataLoader, Dataset


class SleepDataset(Dataset):
    def __init__(self, sequence_length= 100, overlap = 0.4, n_classes = 12):
        super(SleepDataset, self).__init__()
        self.sequence_length = sequence_length
        self.n_classes = n_classes
        self.step  = int(sequence_length - overlap * sequence_length)
        self.dataset_name = 'breathing'
        self.download_load()
 
    def download_load(self):
        trainset = np.load(file="./data/dataset/trainset_16.npy")[:, 1:]# for static dataset, there are 12 classes
        testset = np.load(file="./data/dataset/testset_16.npy")[:, 1:]
        self.train_data, self.train_label = self.slice_window(data= trainset[:, :3], labels=trainset[:, -1])
        self.test_data, self.test_label = self.slice_window(data= testset[:, :3], labels=testset[:, -1])
        print("Generate Slice Window data successfully")

    def clean_mem_test_set(self):
        self.test_set = None
        self.test_data = None
        self.test_label = None

    def slice_window(self, data, labels):
        X_local = []
        y_local = []
        for start in range(0, data.shape[0] - self.sequence_length, self.step):
            end = start + self.sequence_length
            X_local.append(data[start:end])
            y_local.append(labels[end-1])
        return torch.tensor(X_local), torch.tensor(y_local)
    

    def __len__(self):
        return len(self.train_label)


# dataset = SleepDataset(sequence_length=100, overlap=0.4, n_classes=12)
# print(f"Length of dataset: {len(dataset)}")
# print(f"Number of classes: {dataset.n_classes}")
# print(f"Sequence length: {dataset.sequence_length}")
# print(f"Step: {dataset.step}")
# print(f"Dataset name: {dataset.dataset_name}")
# print(f"Train data shape: {dataset.train_data.shape}")
# print(f"Train label shape: {dataset.train_label.shape}")
# print(f"Test data shape: {dataset.test_data.shape}")
# print(f"Test label shape: {dataset.test_label.shape}")