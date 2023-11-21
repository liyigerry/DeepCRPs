import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from models.gnn import *
from models.gat import *
from models.gcn6 import *



from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import prettytable as pt
from util import ProteinDataset
from sklearn.model_selection import KFold
from torch_geometric.loader import DataLoader
from evaluate_metrics import *

from torch.utils.data.dataset import Subset
import matplotlib.pyplot as plt

device = torch.device('cuda:0')
# device = torch.device('cpu')
from sklearn.model_selection import train_test_split, cross_val_score

writer = SummaryWriter()
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold


model_lstm = torch.load('best_model/lstm_best_model.pt')
def training(model, train_loader, optimizer, epoch, epochs):
    model_lstm.train()
    model.train()
    loop = tqdm(enumerate(train_loader), total=len(train_loader), colour='red')
    training_loss = 0.0
    for batch, data in loop:

        data.to(device)
        protein_feature, protein_index, protein_batch = data.x, data.edge_index, data.batch
        graph_node = []
        print('the shape of protein feature', protein_feature.shape)
        graph_batch_all = torch.Tensor().to(device)
        fre = 0
        batch = protein_batch
        batch = batch.detach().cpu().numpy()
        u = batch[0]
        count = 0
        # 计算 各个图的节点数
        for i in batch:
            if u == i:
                count = count + 1
            else:
                u = i
                graph_node.append(count)
                count = 1
        graph_node.append(count)
        # print('the graph node',graph_node)
        # print('the len of graph node',len(graph_node))
        # total = 0
        # for i in graph_node:
        #     total = total+i
        # print('the total ',total)
        size = int(protein_batch.max().item() + 1)
        gid = 0
        batch_label = 0
        # 将数据换做统一长度
        while gid < len(protein_batch) - 1:
            graph_a = torch.Tensor().to(device)
            counta = 0
            # countb = 0
            while batch_label == protein_batch[gid]:
                feature_shape = protein_feature[gid]
                graph_a = torch.cat((graph_a, feature_shape[None, :]), dim=0)
                gid = gid + 1

                counta = counta + 1
                if gid == len(protein_batch) - 1:
                    break
            batch_label = protein_batch[gid]

            while counta < 150:
                pad = torch.zeros(20).to(device)
                graph_a = torch.cat((graph_a, pad[None, :]), dim=0)
                counta = counta + 1

            d = graph_a[None, :]
            graph_batch_all = torch.cat((graph_batch_all, d), dim=0)
        output,x = model_lstm(graph_batch_all)
        batch_num = x.shape
        print(type(batch_num))
        batch_num = batch_num[0]
        print(batch_num)
        graph_node_all = torch.Tensor().to(device)
        for batch_id in range(batch_num):
            graph_node_tensor = torch.Tensor().to(device)
            # print(batch_id)
            seq_len_num = graph_node[batch_id]
            # print('the seq_len_num',seq_len_num)
            graph_node_tensor = torch.cat((graph_node_tensor,x[batch_id,0:seq_len_num,:]),dim = 0)
            # print('the graph_node shape',graph_node_tensor.shape)
            for node_id in range(seq_len_num):
                graph_node_temp = torch.Tensor().to(device)
                graph_node_temp = torch.cat((graph_node_temp,graph_node_tensor[node_id, :]),dim=0)
                graph_node_all = torch.cat((graph_node_all,graph_node_temp[None,:]),dim= 0)
        print('feature shape ',graph_node_all.shape)
        output = model(graph_node_all,protein_index,protein_batch)
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
    model_lstm.train()
    model.eval()
    total_preds = torch.Tensor()
    total_labels = torch.Tensor()
    total_preds4 = torch.Tensor()
    y = torch.Tensor()
    with torch.no_grad():
        loop = tqdm(enumerate(loader), total=len(loader), colour='blue')
        # for batch, data in enumerate(loader):
        for batch, data in loop:

            data.to(device)
            protein_feature, protein_index, protein_batch = data.x, data.edge_index, data.batch
            graph_node = []
            print('the shape of protein feature', protein_feature.shape)
            graph_batch_all = torch.Tensor().to(device)
            fre = 0
            batch = protein_batch
            batch = batch.detach().cpu().numpy()
            u = batch[0]
            count = 0
            # 计算 各个图的节点数
            for i in batch:
                if u == i:
                    count = count + 1
                else:
                    u = i
                    graph_node.append(count)
                    count = 1
            graph_node.append(count)
            # print('the graph node',graph_node)
            # print('the len of graph node',len(graph_node))
            # total = 0
            # for i in graph_node:
            #     total = total+i
            # print('the total ',total)
            size = int(protein_batch.max().item() + 1)
            gid = 0
            batch_label = 0
            # 将数据换做统一长度
            while gid < len(protein_batch) - 1:
                graph_a = torch.Tensor().to(device)
                counta = 0
                # countb = 0
                while batch_label == protein_batch[gid]:
                    feature_shape = protein_feature[gid]
                    graph_a = torch.cat((graph_a, feature_shape[None, :]), dim=0)
                    gid = gid + 1

                    counta = counta + 1
                    if gid == len(protein_batch) - 1:
                        break
                batch_label = protein_batch[gid]

                while counta < 150:
                    pad = torch.zeros(20).to(device)
                    graph_a = torch.cat((graph_a, pad[None, :]), dim=0)
                    counta = counta + 1

                d = graph_a[None, :]
                graph_batch_all = torch.cat((graph_batch_all, d), dim=0)
            output, x = model_lstm(graph_batch_all)
            batch_num = x.shape
            print(type(batch_num))
            batch_num = batch_num[0]
            print(batch_num)
            graph_node_all = torch.Tensor().to(device)
            for batch_id in range(batch_num):
                graph_node_tensor = torch.Tensor().to(device)
                # print(batch_id)
                seq_len_num = graph_node[batch_id]
                # print('the seq_len_num',seq_len_num)
                graph_node_tensor = torch.cat((graph_node_tensor, x[batch_id, 0:seq_len_num, :]), dim=0)
                # print('the graph_node shape',graph_node_tensor.shape)
                for node_id in range(seq_len_num):
                    graph_node_temp = torch.Tensor().to(device)
                    graph_node_temp = torch.cat((graph_node_temp, graph_node_tensor[node_id, :]), dim=0)
                    graph_node_all = torch.cat((graph_node_all, graph_node_temp[None, :]), dim=0)
            print('feature shape ', graph_node_all.shape)
            output = model(graph_node_all, protein_index, protein_batch)
            total_preds4 = torch.cat((total_preds4,output.detach().cpu()),0)
            output = output.argmax(dim=1)
            # loop.set_description(f'Testing Epoch [{epoch} / {epochs}]')
            total_preds = torch.cat((total_preds, output.detach().cpu()), 0)

            total_labels = torch.cat((total_labels, data.y.view(-1, 1).cpu()), 0)
    y4 = total_preds
    total_labels = total_labels.numpy().flatten()

    total_preds = total_preds.numpy().flatten()



    return total_labels, total_preds,total_preds4,y4  # 定义数据迭代器


if __name__ == '__main__':

    train_data = ProteinDataset(root='data', dataset='train_data20')

    epochs = 1000
    epoch = 1
    batch_size = 32
    f1_five = []
    acc_five = []
    pre_five = []
    rec_five = []
    mcc_five = []
    auc_five = []
    ultra_acc = 0
    ultra_pre = 0
    ultra_rec = 0
    ultra_f1 = 0
    ultra_mcc = 0
    ultra_auc = 0

    stratifiedKFolds = StratifiedKFold(n_splits=5, shuffle=True)

    for i, (trn_idx, val_idx) in enumerate(stratifiedKFolds.split(train_data, y)):
        train_loader = DataLoader(Subset(train_data, trn_idx), batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(Subset(train_data, val_idx), batch_size=batch_size, shuffle=True)

        best_f1 = 0
        best_acc = 0
        best_pre = 0
        best_rec = 0
        best_mcc = 0
        best_auc = 0
        model = GCN1(512, hidden_dim=64).to(device)
        print(model)
        learning_rate = 0.003521
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        criterion = nn.CrossEntropyLoss().to(device)
        for epoch in range(1, epochs + 1):
            training(model, train_loader, optimizer, epoch, epochs)
            # scheduler.step()

            val_labels, val_preds,pre_y4,y4 = validation(model, test_loader, epoch, epochs)

            print(pre_y4.shape)
            print(y4.shape)
            # test_labels, test_preds = validation(model, compound_test_loader, epoch, epochs)
            val_result = [accuracy_score(val_labels, val_preds), precision_score(val_labels, val_preds),
                          recall_score(val_labels, val_preds), f1_score(val_labels, val_preds),
                          mcc_score(val_labels, val_preds),auc_score(val_labels,pre_y4)]

            tb = pt.PrettyTable()
            tb.field_names = ['Epoch / Epochs', 'Test', 'accuracy', 'precision', 'recall', 'f1', 'mcc','auc']
            tb.add_row(['{} / {}'.format(epoch, epochs), 'Validation', val_result[0], val_result[1], val_result[2],
                        val_result[3], val_result[4],val_result[-1]])
            # tb.add_row(['{} / {}'.format(epoch, epochs).format(epoch, epochs), 'Test', test_result[0], test_result[1], test_result[2], test_result[3], test_result[4], test_result[-1]])
            print(tb)
            # writer.add_scalar('RMSE/Val RMSE', val_result[1], epoch)
            writer.add_scalar('RMSE/Test RMSE', val_result[1], epoch)

            if float(val_result[3]) > best_f1:
                best_acc = float(val_result[0])
                best_pre = float(val_result[1])
                best_rec = float(val_result[2])
                best_f1 = float(val_result[3])
                best_mcc = float(val_result[-2])
                best_auc = float(val_result[-1])
                torch.save(model, 'best_model/best_model.pt')
                torch.save(model.state_dict(), 'best_model/best_model_param.pt')
        acc_five.append(best_acc)
        pre_five.append(best_pre)
        rec_five.append(best_rec)
        f1_five.append(best_f1)
        mcc_five.append(best_mcc)
        auc_five.append(best_auc)

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
    for i in auc_five:
        ultra_auc = ultra_auc + i
    tp = pt.PrettyTable()
    tp.field_names = ['Test', 'accuracy', 'precision', 'recall', 'f1', 'mcc','auc']
    tp.add_row(['Validation', ultra_acc / 5, ultra_pre / 5, ultra_rec / 5, ultra_f1 / 5, ultra_mcc / 5,ultra_auc/5])
    print('the value of five:', acc_five)
    print(pre_five)
    print(rec_five)
    print(f1_five)
    print(mcc_five)
    print(auc_five)
    print(tp)
    with open('result/gcn_result.txt', 'a') as write:
        write.writelines(str(tp) + '\n')

