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
from torch_geometric.explain import GNNExplainer
from torch_geometric.explain import Explainer
from torch.utils.data.dataset import Subset
import matplotlib.pyplot as plt

device = torch.device('cuda:0')
# device = torch.device('cpu')
from sklearn.model_selection import train_test_split, cross_val_score

writer = SummaryWriter()
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
import optuna


def training(model, train_loader, optimizer, epoch, epochs):
    model.train()
    loop = tqdm(enumerate(train_loader), total=len(train_loader), colour='red')
    training_loss = 0.0
    # for batch, data in enumerate(compound_train_loader):
    for batch, data in loop:
        output = model(data.to(device))
        loss = criterion(output, data.y.to(torch.int64).to(device))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        training_loss += loss.item()
        loop.set_description(f'Training Epoch [{epoch} / {epochs}]')
        loop.set_postfix(loss=loss.item())
    writer.add_scalar('Training loss', training_loss, epoch)
    print('Training Epoch:[{} / {}], Mean Loss: {},data_len:{}'.format(epoch, epochs, training_loss / 1375,
                                                                       len(train_loader)))


def validation(model, loader, epoch=1, epochs=1):
    model.eval()
    total_preds = torch.Tensor()
    total_labels = torch.Tensor()
    with torch.no_grad():
        loop = tqdm(enumerate(loader), total=len(loader), colour='blue')
        # for batch, data in enumerate(loader):
        for batch, data in loop:
            output = model(data.to(device))
            output = output.argmax(dim=1)
            # loop.set_description(f'Testing Epoch [{epoch} / {epochs}]')
            total_preds = torch.cat((total_preds, output.detach().cpu()), 0)
            total_labels = torch.cat((total_labels, data.y.view(-1, 1).cpu()), 0)
    total_labels = total_labels.numpy().flatten()
    total_preds = total_preds.numpy().flatten()
    print(len(total_preds))
    print('the true value is:', total_labels)
    print('the pre value is:', total_preds)

    return total_labels, total_preds  # 定义数据迭代器


if __name__ == '__main__':

    train_data = ProteinDataset(root='data', dataset='train_data')
    y = train_data.data.y



    epochs = 500
    epoch = 1
    batch_size = 128
    f1_five = []
    acc_five = []
    pre_five = []
    rec_five = []
    mcc_five = []
    ultra_acc = 0
    ultra_pre = 0
    ultra_rec = 0
    ultra_f1 = 0
    ultra_mcc = 0

    stratifiedKFolds = StratifiedKFold(n_splits=5, shuffle=True)

    for i, (trn_idx, val_idx) in enumerate(stratifiedKFolds.split(train_data, y)):
        train_loader = DataLoader(Subset(train_data, trn_idx), batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(Subset(train_data, val_idx), batch_size=batch_size, shuffle=True)

        best_f1 = 0
        best_acc = 0
        best_pre = 0
        best_rec = 0
        best_mcc = 0

        model = GraphConv(1024, hidden_dim=128).to(device)
        print(model)
        learning_rate = 0.004521
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        criterion = nn.CrossEntropyLoss().to(device)
        for epoch in range(1, epochs + 1):
            training(model, train_loader, optimizer, epoch, epochs)
            # scheduler.step()

            val_labels, val_preds = validation(model, test_loader, epoch, epochs)
            print('val_labels:', len(val_labels))
            print('val_preds:', len(val_preds))
            # test_labels, test_preds = validation(model, compound_test_loader, epoch, epochs)
            val_result = [accuracy_score(val_labels, val_preds), precision_score(val_labels, val_preds),
                          recall_score(val_labels, val_preds), f1_score(val_labels, val_preds),
                          mcc_score(val_labels, val_preds)]

            tb = pt.PrettyTable()
            tb.field_names = ['Epoch / Epochs', 'Test', 'accuracy', 'precision', 'recall', 'f1', 'mcc']
            tb.add_row(['{} / {}'.format(epoch, epochs), 'Validation', val_result[0], val_result[1], val_result[2],
                        val_result[3], val_result[-1]])


            writer.add_scalar('RMSE/Test RMSE', val_result[1], epoch)

            if float(val_result[3]) > best_f1:
                best_acc = float(val_result[0])
                best_pre = float(val_result[1])
                best_rec = float(val_result[2])
                best_f1 = float(val_result[3])
                best_mcc = float(val_result[-1])
                torch.save(model, 'best_model/best_model.pt')
                torch.save(model.state_dict(), 'best_model/best_model_param.pt')
        acc_five.append(best_acc)
        pre_five.append(best_pre)
        rec_five.append(best_rec)
        f1_five.append(best_f1)
        mcc_five.append(best_mcc)

    for i in acc_five:
        ultra_acc = ultra_acc + i
    for i in pre_five:
        ultra_pre = ultra_pre + i
    for i in rec_five:
        ultra_rec = ultra_rec + i
    for i in f1_five:
        ultra_f1 = ultra_f1 + i
    for i in mcc_five:
        ultra_mcc = ultra_mcc + i
    tp = pt.PrettyTable()
    tp.field_names = ['Test', 'accuracy', 'precision', 'recall', 'f1', 'mcc']
    tp.add_row(['Validation', ultra_acc / 5, ultra_pre / 5, ultra_rec / 5, ultra_f1 / 5, ultra_mcc / 5])
    print('the value of five:' , acc_five)
    print(pre_five)
    print(rec_five)
    print(f1_five)
    print(mcc_five)
    print(tp)
    with open('result/gcn_result.txt', 'a') as write:
        write.writelines(str(tp) + '\n')

