import os, sys
import numpy as np
import pandas as pd
import pickle as pk
import torch
from torch_geometric.loader import DataLoader

train_data = pd.read_pickle('data/processed/traindata.pkl')
test_data = pd.read_pickle('data/processed/testdata.pkl')
# torch.save(train_data, 'data/processed/train_data.pt')
# torch.save(test_data, 'data/processed/test_data.pt')
# print(train_data)
# batch_size = 16
# train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
# val_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)


for data in train_data:
    for node in data[0].nodes():
        print(node[1]['feature'])
    # print(data[0].nodes())
    # print(data[0].edges())
    # print(data[1])
    # print(data[2])
    break

# print(train_data[0][0].nodes())