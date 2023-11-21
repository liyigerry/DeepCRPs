
from sklearn.model_selection import train_test_split

import torch.cuda
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from sklearn.metrics import accuracy_score, mean_absolute_error, f1_score, recall_score
from evaluate_metrics import *
from evaluate_metrics import *
import pandas as pd

import numpy as np
import torch.utils.data as Data
import warnings
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data
import torch.nn.functional as F
from vocabulary import Vocabulary
data = pd.read_csv('data/data.csv')
vocabulary = Vocabulary.get_vocabulary_from_sequences(data.Sequence.values)
def transform(sequence, max_len):
    tensor = vocabulary.seq_to_tensor(sequence)
    if len(tensor)>max_len:
        tensor = tensor[: max_len]
        tensor[-1] = 23
        list = tensor.numpy().tolist()
    else:
        list = tensor.numpy().tolist()+[0]*(max_len-len(tensor))
    return list
data_df = pd.read_csv('data/data.csv')

seq_data = data_df['Sequence']

label_data =  data_df['Label']

data_seq = []

max = 0
temp_min = 40
temp_max = 0
for i in seq_data:

    temp = transform(i,151)
    for j in temp:
        if j >temp_max:
            temp_max = j
        if j<temp_min:
            temp_min = j

    data_seq.append(temp)


data_seq = torch.tensor(data_seq)
data_label = torch.tensor(label_data)


data_seq = F.one_hot(data_seq)
class Model_lstm_bidir(nn.Module):
    def __init__(self):
        super(Model_lstm_bidir, self).__init__()
        # self.embedding = nn.Embedding(len(vocabulary), 300)    #每个残基用长度100的向量表示，这里没有用one-hot编码
        #加入LSTM
        self.lstm = nn.LSTM(input_size=23, hidden_size=100, num_layers=1, batch_first=True, bidirectional=True,dropout=0.2)
        self.fc = nn.Linear(100*2, 4)
    def forward(self, input):
        # x = self.embedding(input)   #进行embedding操作,形状[batch_size, max_len,100]
        x = input.float()
        x, (h_n, c_n) = self.lstm(x)
        #x:[batch_size,max_len,2*hidden_size]  h_n:[2*2,batch_size,hiddem_size]
        #获取两个方向最后一次的output，进行concat
        output_fw = h_n[-2,:,:] #正向最后一次的输出
        output_bw = h_n[-1,:,:] #反向最后一次的输出
        output = torch.cat([output_fw,output_bw], dim=-1)   #[batch_size,hidden_size]

        # x = output.view([-1, max_len*100])
        out = self.fc(output)
        return out
class GRU(nn.Module):
    def __init__(self):
        super(GRU, self).__init__()

        #加入GRU
        self.gru = nn.GRU(input_size=23, hidden_size=100, num_layers=1, batch_first=True, bidirectional=True, dropout=0.3)
        self.fc = nn.Linear(100*2,4)
    def forward(self, input):
        x = input.float()
        x, h_n = self.gru(x)

        output_fw = h_n[-2,:,:] #正向最后一次的输出
        output_bw = h_n[-1,:,:] #反向最后一次的输出
        output = torch.cat([output_fw,output_bw], dim=-1)

        out = self.fc(output)
        return out
from sklearn.model_selection import KFold,StratifiedKFold
sfolder = StratifiedKFold(n_splits=5,random_state=2,shuffle=True)
result = np.random.randn(5,7)
best_result = [[0,0,0,0,0,0,0],[0,0,0,0,0,0,0],[0,0,0,0,0,0,0],[0,0,0,0,0,0,0],[0,0,0,0,0,0,0]]
for i,(train_id,test_id) in enumerate(sfolder.split(data_seq,data_label)):

    Xtrain,Ytrain = data_seq[train_id],data_label[train_id]
    Xtest,Ytest = data_seq[test_id],data_label[test_id]

    train_dataset = Data.TensorDataset(torch.tensor(Xtrain),torch.tensor(Ytrain))
    test_dataset = Data.TensorDataset(torch.tensor(Xtest),torch.tensor(Ytest))
    train_loader = Data.DataLoader(
        dataset=train_dataset,      # 数据，封装进Data.TensorDataset()类的数据
        batch_size=32,      # 每块的大小
        shuffle=True,
    # 要不要打乱数据 (打乱比较好)
        num_workers=2,              # 多进程（multiprocess）来读数据
    )
    test_loader = Data.DataLoader(
        dataset=test_dataset,      # 数据，封装进Data.TensorDataset()类的数据
        batch_size=32,      # 每块的大小
        shuffle=True,# 要不要打乱数据 (打乱比较好)
        num_workers=2,              # 多进程（multiprocess）来读数据
    )
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = GRU().to(device)
    optimizer = Adam(model.parameters(), 0.001)
    criterion = nn.CrossEntropyLoss()
    def train(epoch):
        model.train()
        for idx, (input, target) in enumerate(train_loader):

            input = input.to(device)
            target = target.to(device)
            optimizer.zero_grad()
            output = model(input)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            # print(epoch, idx, loss.item())
    def test(epoch):
        model.eval()
        total_preds = torch.Tensor()
        total_y = torch.Tensor()
        total_preds4 = torch.Tensor()
        for idx,(input,target) in enumerate(test_loader):

            input = input.to(device)
            target = target.to(device)
            optimizer.zero_grad()
            output = model(input)
            pre_y = output.argmax(dim=1)
            total_preds4 = torch.cat((total_preds4,output.detach().cpu()),0)
            total_preds = torch.cat((total_preds, pre_y.detach().cpu()), 0)
            total_y = torch.cat((total_y, target.cpu()), 0)

        total_preds = total_preds.numpy().flatten()
        return total_preds, total_y, total_preds4
    for epoch in range(500):
        train(epoch)
        pre_y,y,pre_y4= test(epoch)

        print(epoch)
        # print(pre_y[0:10],y[0:10])
        # print(y.shape)
        val_result = [accuracy_score(y, pre_y), precision_score(y, pre_y), recall_score(y, pre_y), f1_score(y, pre_y),
                      mcc_score(y, pre_y), auc_score(y,pre_y4)]
        if val_result[-2] > best_result[i][-2]:
            for z in range(7):
                best_result[i][z] = val_result[z]
            print('the epoch :', epoch, 'get the best result:', best_result[i])
    print(best_result)
r = [0, 0, 0, 0, 0, 0,0]
for i in range(7):
    for j in range(5):
        r[i] = r[i] + best_result[j][i]
for i in r:
    print('the five value :', i / 5)