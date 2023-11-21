import os
import numpy as np
import pandas as pd


from data_process.graphUtil import (getProteinGraph)
from util import *



train_ach_id = os.listdir('data/train/ach')
train_ca_id = os.listdir('data/train/ca')

train_k_id = os.listdir('data/train/k')
train_na_id = os.listdir('data/train/na')

train_ach_label = [0] * len(train_ach_id)
train_ca_label = [1] * len(train_ca_id)

train_k_label = [2] * len(train_k_id)
train_na_label = [3] * len(train_na_id)


train_id = train_ach_id + train_ca_id  + train_k_id + train_na_id
train_label = train_ach_label + train_ca_label + train_k_label + train_na_label

test_ach_id = os.listdir('data/test/ach')
test_ca_id = os.listdir('data/test/ca')

test_k_id = os.listdir('data/test/k')
test_na_id = os.listdir('data/test/na')

test_ach_label = [0] * len(test_ach_id)
test_ca_label = [1] * len(test_ca_id)

test_k_label = [2] * len(test_k_id)
test_na_label = [3] * len(test_na_id)


test_id = test_ach_id + test_ca_id  + test_k_id + test_na_id
test_label = test_ach_label + test_ca_label + test_k_label + test_na_label


train_drug_graph = []
test_drug_graph = []


new_cid = []
new_affinity = []
for i in range(len(train_id)):
    drug_info = {}
    g = getProteinGraph(train_id[i].split('.')[0])
    drug_info[train_id[i]] = g
    train_drug_graph.append(drug_info)
print('转换完成')



train_id, train_affinity =  np.asarray(train_id), np.asarray(train_label)

print('准备将数据转化为Pytorch格式')
protein_train_data = ProteinDataset(root='data', dataset='train_data1057',
                                    protein=train_id, protein_graph=train_drug_graph, affinity=train_affinity)

print('数据转化为Pytorch格式完成')
