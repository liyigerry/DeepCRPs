import torch
import torch.nn as nn
from torch_geometric.nn import ChebConv, global_max_pool as gmp
from torch_geometric.nn import GCNConv,GATv2Conv
from torch_geometric.nn import SAGEConv
from torch_geometric.nn import GraphConv




class GAT1(nn.Module):
    def __init__(self, feature_dim, hidden_dim=32, dropout=0.4):
        super(GAT1, self).__init__()
        self.dropout = dropout
        self.hidden = hidden_dim
        self.cconv1 = GATv2Conv(feature_dim, hidden_dim, aggr='sum', K=3)


        self.relu = nn.LeakyReLU()
        self.dropout = nn.Dropout(dropout)
        self.classification = nn.Sequential(
            nn.Linear(self.hidden, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 4)
        )

    def forward(self, data):
        # 获取小分子和蛋白质输入的结构信息
        protein_feature, protein_index, protein_batch = data.x, data.edge_index, data.batch
        # 对小分子进行卷积操作
        protein_feature = self.cconv1(protein_feature, protein_index)
        protein_feature = gmp(protein_feature, protein_batch)

        protein_feature = self.classification(protein_feature)

        return protein_feature


class GAT2(nn.Module):
    def __init__(self, feature_dim, hidden_dim=32, dropout=0.4):
        super(GAT2, self).__init__()
        self.dropout = dropout
        self.hidden = hidden_dim
        self.cconv1 = GATv2Conv(feature_dim, hidden_dim, aggr='sum', K=3)
        self.cconv2 = GATv2Conv(hidden_dim, hidden_dim * 2, aggr='sum', K=3)

        self.relu = nn.LeakyReLU()
        self.dropout = nn.Dropout(dropout)
        self.classification = nn.Sequential(
            nn.Linear(self.hidden * 2, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 4)
        )

    def forward(self, data):
        # 获取小分子和蛋白质输入的结构信息
        protein_feature, protein_index, protein_batch = data.x, data.edge_index, data.batch
        # 对小分子进行卷积操作
        protein_feature = self.cconv1(protein_feature, protein_index)
        protein_feature = self.relu(protein_feature)

        protein_feature = self.cconv2(protein_feature, protein_index)



        protein_feature = gmp(protein_feature, protein_batch)

        protein_feature = self.classification(protein_feature)

        return protein_feature

class GAT3(nn.Module):
    def __init__(self, feature_dim, hidden_dim=32, dropout=0.4):
        super(GAT3, self).__init__()
        self.dropout = dropout
        self.hidden = hidden_dim
        self.cconv1 = GATv2Conv(feature_dim, hidden_dim, aggr='sum', K=3)
        self.cconv2 = GATv2Conv(hidden_dim, hidden_dim * 2, aggr='sum', K=3)
        self.cconv3 = GATv2Conv(hidden_dim * 2, hidden_dim * 4, aggr='sum', K=3)

        self.relu = nn.LeakyReLU()
        self.dropout = nn.Dropout(dropout)
        self.classification = nn.Sequential(
            nn.Linear(self.hidden * 4, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 4)
        )

    def forward(self, data):
        # 获取小分子和蛋白质输入的结构信息
        protein_feature, protein_index, protein_batch = data.x, data.edge_index, data.batch
        # 对小分子进行卷积操作
        protein_feature = self.cconv1(protein_feature, protein_index)
        protein_feature = self.relu(protein_feature)

        protein_feature = self.cconv2(protein_feature, protein_index)
        protein_feature = self.relu(protein_feature)


        protein_feature = self.cconv3(protein_feature, protein_index)


        protein_feature = gmp(protein_feature, protein_batch)

        protein_feature = self.classification(protein_feature)

        return protein_feature

class GAT4(nn.Module):
    def __init__(self, feature_dim, hidden_dim=32, dropout=0.4):
        super(GAT4, self).__init__()
        self.dropout = dropout
        self.hidden = hidden_dim
        self.cconv1 = GATv2Conv(feature_dim, hidden_dim, aggr='sum', K=3)
        self.cconv2 = GATv2Conv(hidden_dim, hidden_dim * 2, aggr='sum', K=3)
        self.cconv3 = GATv2Conv(hidden_dim * 2, hidden_dim * 4, aggr='sum', K=3)
        self.cconv4 = GATv2Conv(hidden_dim * 4, hidden_dim * 8, aggr='sum', K=3)
        self.relu = nn.LeakyReLU()
        self.dropout = nn.Dropout(dropout)
        self.classification = nn.Sequential(
            nn.Linear(self.hidden * 8, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 4)
        )

    def forward(self, data):
        # 获取小分子和蛋白质输入的结构信息
        protein_feature, protein_index, protein_batch = data.x, data.edge_index, data.batch
        # 对小分子进行卷积操作
        protein_feature = self.cconv1(protein_feature, protein_index)
        protein_feature = self.relu(protein_feature)


        protein_feature = self.cconv2(protein_feature, protein_index)
        protein_feature = self.relu(protein_feature)


        protein_feature = self.cconv3(protein_feature, protein_index)
        protein_feature = self.relu(protein_feature)

        protein_feature = self.cconv4(protein_feature, protein_index)
        protein_feature = gmp(protein_feature, protein_batch)

        protein_feature = self.classification(protein_feature)

        return protein_feature

class GAT5(nn.Module):
    def __init__(self, feature_dim, hidden_dim=32, dropout=0.4):
        super(GAT5, self).__init__()
        self.dropout = dropout
        self.hidden = hidden_dim
        self.cconv1 = GATv2Conv(feature_dim, hidden_dim, aggr='sum', K=3)
        self.cconv2 = GATv2Conv(hidden_dim, hidden_dim * 2, aggr='sum', K=3)
        self.cconv3 = GATv2Conv(hidden_dim * 2, hidden_dim * 4, aggr='sum', K=3)
        self.cconv4 = GATv2Conv(hidden_dim * 4, hidden_dim * 8, aggr='sum', K=3)
        self.cconv5 = GATv2Conv(hidden_dim * 8, hidden_dim * 16, aggr='sum', K=3)
        self.relu = nn.LeakyReLU()
        self.dropout = nn.Dropout(dropout)
        self.classification = nn.Sequential(
            nn.Linear(self.hidden * 16, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 4)
        )

    def forward(self, data):
        # 获取小分子和蛋白质输入的结构信息
        protein_feature, protein_index, protein_batch = data.x, data.edge_index, data.batch
        # 对小分子进行卷积操作
        protein_feature = self.cconv1(protein_feature, protein_index)
        protein_feature = self.relu(protein_feature)


        protein_feature = self.cconv2(protein_feature, protein_index)
        protein_feature = self.relu(protein_feature)


        protein_feature = self.cconv3(protein_feature, protein_index)
        protein_feature = self.relu(protein_feature)

        protein_feature = self.cconv4(protein_feature, protein_index)
        protein_feature = self.relu(protein_feature)
        protein_feature = self.cconv5(protein_feature, protein_index)


        protein_feature = gmp(protein_feature, protein_batch)

        protein_feature = self.classification(protein_feature)

        return protein_feature

class GAT6(nn.Module):
    def __init__(self, feature_dim, hidden_dim=32, dropout=0.4):
        super(GAT6, self).__init__()
        self.dropout = dropout
        self.hidden = hidden_dim
        self.cconv1 = GATv2Conv(feature_dim, hidden_dim, aggr='sum', K=3)
        self.cconv2 = GATv2Conv(hidden_dim, hidden_dim * 2, aggr='sum', K=3)
        self.cconv3 = GATv2Conv(hidden_dim * 2, hidden_dim * 4, aggr='sum', K=3)
        self.cconv4 = GATv2Conv(hidden_dim * 4, hidden_dim * 8, aggr='sum', K=3)
        self.cconv5 = GATv2Conv(hidden_dim * 8, hidden_dim * 16, aggr='sum', K=3)
        self.cconv6 = GATv2Conv(hidden_dim * 16, hidden_dim * 20, aggr='sum', K=3)
        self.relu = nn.LeakyReLU()
        self.dropout = nn.Dropout(dropout)
        self.classification = nn.Sequential(
            nn.Linear(self.hidden * 20, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 4)
        )

    def forward(self, data):
        # 获取小分子和蛋白质输入的结构信息
        protein_feature, protein_index, protein_batch = data.x, data.edge_index, data.batch

        protein_feature = self.cconv1(protein_feature, protein_index)
        protein_feature = self.relu(protein_feature)

        protein_feature = self.cconv2(protein_feature, protein_index)
        protein_feature = self.relu(protein_feature)
        protein_feature = self.cconv3(protein_feature, protein_index)
        protein_feature = self.relu(protein_feature)

        protein_feature = self.cconv4(protein_feature, protein_index)
        protein_feature = self.relu(protein_feature)
        protein_feature = self.cconv5(protein_feature, protein_index)
        protein_feature = self.relu(protein_feature)
        protein_feature = self.cconv6(protein_feature, protein_index)
        protein_feature = gmp(protein_feature, protein_batch)

        protein_feature = self.classification(protein_feature)

        return protein_feature
