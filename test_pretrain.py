import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from models.gcn import GraphConv
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import prettytable as pt
from util import ProteinDataset
from sklearn.model_selection import KFold
from torch_geometric.loader import DataLoader
from evaluate_metrics import *
from training import validation
device = torch.device('cuda:0')

batch_size =16
test_data = ProteinDataset(root='data', dataset='test_data')
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)
model = torch.load('best_model/best_model.pt')
val_labels, val_preds = validation(model, test_loader)
val_result = [accuracy_score(val_labels, val_preds), precision_score(val_labels, val_preds), recall_score(val_labels, val_preds), f1_score(val_labels, val_preds), mcc_score(val_labels, val_preds)]
tb = pt.PrettyTable()
tb.field_names = ['Epoch / Epochs', 'Test', 'accuracy', 'precision', 'recall', 'f1', 'mcc']
tb.add_row(['{} / {}'.format(1, 1), 'Validation', val_result[0], val_result[1], val_result[2], val_result[3], val_result[-1]])
print(tb)
