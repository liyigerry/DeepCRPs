import torch
import torch.nn as nn
from torch_geometric.nn import ChebConv, global_max_pool as gmp
from torch_geometric.nn import GCNConv,GATv2Conv
from torch_geometric.nn import SAGEConv
from torch_geometric.nn import GraphConv,MFConv


class Cheb(nn.Module):
    def __init__(self, feature_dim, hidden_dim=32, output_dim=256, dropout=0.4, use_residue=False):
        super(Cheb, self).__init__()
        self.dropout = dropout
        self.hidden = hidden_dim
        self.cconv1 = ChebConv(feature_dim, hidden_dim, aggr='sum', K=3)
        self.cconv2 = ChebConv(hidden_dim, hidden_dim * 2, aggr='sum', K=3)
        self.cconv3 = ChebConv(hidden_dim * 2, hidden_dim * 4, aggr='sum', K=3)

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
        # protein_feature = self.bn1(protein_feature)

        protein_feature = self.cconv2(protein_feature, protein_index)
        protein_feature = self.relu(protein_feature)
        # protein_feature = self.bn2(protein_feature)

        protein_feature = self.cconv3(protein_feature, protein_index)
        # protein_feature = self.relu(protein_feature)
        # protein_feature = self.bn3(protein_feature)

        # 对卷积后的小分子进行图的最大值池化
        # print(protein_feature.view(16, -1, self.hidden * 4))
        protein_feature = gmp(protein_feature, protein_batch)

        protein_feature = self.classification(protein_feature)

        return protein_feature

class Graph(nn.Module):
    def __init__(self, feature_dim, hidden_dim, dropout=0.4):
        super(Graph, self).__init__()
        self.dropout = dropout
        self.hidden = hidden_dim
        self.cconv1 = GraphConv(feature_dim, hidden_dim, aggr='sum', K=3)
        self.cconv2 = GraphConv(hidden_dim, hidden_dim * 2, aggr='sum', K=3)
        self.cconv3 = GraphConv(hidden_dim * 2, hidden_dim * 4, aggr='sum', K=3)

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
        # protein_feature = self.bn1(protein_feature)

        protein_feature = self.cconv2(protein_feature, protein_index)
        protein_feature = self.relu(protein_feature)
        # protein_feature = self.bn2(protein_feature)

        protein_feature = self.cconv3(protein_feature, protein_index)
        # protein_feature = self.relu(protein_feature)
        # protein_feature = self.bn3(protein_feature)

        # 对卷积后的小分子进行图的最大值池化
        # print(protein_feature.view(16, -1, self.hidden * 4))
        protein_feature = gmp(protein_feature, protein_batch)

        protein_feature = self.classification(protein_feature)

        return protein_feature

class SAGE(nn.Module):
    def __init__(self, feature_dim, hidden_dim, dropout=0.4):
        super(SAGE, self).__init__()
        self.dropout = dropout
        self.hidden = hidden_dim
        self.cconv1 = SAGEConv(feature_dim, hidden_dim, aggr='sum', K=3)
        self.cconv2 = SAGEConv(hidden_dim, hidden_dim * 2, aggr='sum', K=3)
        self.cconv3 = SAGEConv(hidden_dim * 2, hidden_dim * 4, aggr='sum', K=3)

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
        # protein_feature = self.bn1(protein_feature)

        protein_feature = self.cconv2(protein_feature, protein_index)
        protein_feature = self.relu(protein_feature)
        # protein_feature = self.bn2(protein_feature)

        protein_feature = self.cconv3(protein_feature, protein_index)
        # protein_feature = self.relu(protein_feature)
        # protein_feature = self.bn3(protein_feature)

        # 对卷积后的小分子进行图的最大值池化
        # print(protein_feature.view(16, -1, self.hidden * 4))
        protein_feature = gmp(protein_feature, protein_batch)

        protein_feature = self.classification(protein_feature)

        return protein_feature

class GAT(nn.Module):
    def __init__(self, feature_dim, hidden_dim=32, dropout=0.4):
        super(GAT, self).__init__()
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
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128,32),
            nn.ReLU(),
            nn.Linear(32,4)

        )

    def forward(self, data):
        # 获取小分子和蛋白质输入的结构信息
        protein_feature, protein_index, protein_batch = data.x, data.edge_index, data.batch
        # 对小分子进行卷积操作
        protein_feature = self.cconv1(protein_feature, protein_index)
        protein_feature = self.relu(protein_feature)
        # protein_feature = self.bn1(protein_feature)

        protein_feature = self.cconv2(protein_feature, protein_index)
        protein_feature = self.relu(protein_feature)
        # protein_feature = self.bn2(protein_feature)

        protein_feature = self.cconv3(protein_feature, protein_index)
        # protein_feature = self.relu(protein_feature)
        # protein_feature = self.bn3(protein_feature)

        # 对卷积后的小分子进行图的最大值池化
        # print(protein_feature.view(16, -1, self.hidden * 4))
        protein_feature = gmp(protein_feature, protein_batch)

        protein_feature = self.classification(protein_feature)

        return protein_feature

class GCN(nn.Module):
    def __init__(self, feature_dim, hidden_dim, dropout=0.4):
        super(GCN, self).__init__()
        self.dropout = dropout
        self.hidden = hidden_dim
        self.cconv1 = GCNConv(feature_dim, hidden_dim)
        self.cconv2 = GCNConv(hidden_dim, hidden_dim * 2)
        self.cconv3 = GCNConv(hidden_dim * 2, hidden_dim * 4)

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
        # protein_feature = self.bn1(protein_feature)

        protein_feature = self.cconv2(protein_feature, protein_index)
        protein_feature = self.relu(protein_feature)
        # protein_feature = self.bn2(protein_feature)

        protein_feature = self.cconv3(protein_feature, protein_index)
        # protein_feature = self.relu(protein_feature)
        # protein_feature = self.bn3(protein_feature)

        # 对卷积后的小分子进行图的最大值池化
        # print(protein_feature.view(16, -1, self.hidden * 4))
        protein_feature = gmp(protein_feature, protein_batch)

        protein_feature = self.classification(protein_feature)

        return protein_feature
class MF(nn.Module):
    def __init__(self, feature_dim, hidden_dim, dropout=0.4):
        super(MF, self).__init__()
        self.dropout = dropout
        self.hidden = hidden_dim
        self.cconv1 = MFConv(feature_dim, hidden_dim)
        self.cconv2 = MFConv(hidden_dim, hidden_dim * 2)
        self.cconv3 = MFConv(hidden_dim * 2, hidden_dim * 4)

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
        # protein_feature = self.bn1(protein_feature)

        protein_feature = self.cconv2(protein_feature, protein_index)
        protein_feature = self.relu(protein_feature)
        # protein_feature = self.bn2(protein_feature)

        protein_feature = self.cconv3(protein_feature, protein_index)
        # protein_feature = self.relu(protein_feature)
        # protein_feature = self.bn3(protein_feature)

        # 对卷积后的小分子进行图的最大值池化
        # print(protein_feature.view(16, -1, self.hidden * 4))
        protein_feature = gmp(protein_feature, protein_batch)

        protein_feature = self.classification(protein_feature)

        return protein_feature
class MIX(nn.Module):
    def __init__(self, feature_dim, hidden_dim, dropout=0.4):
        super(MIX, self).__init__()
        self.dropout = dropout
        self.hidden = hidden_dim
        self.cconv1 = GCNConv(feature_dim, hidden_dim)
        self.cconv2 = GCNConv(hidden_dim, hidden_dim * 2)
        self.cconv3 = GCNConv(hidden_dim * 2, hidden_dim * 4)
        self.cconv4 = GATv2Conv(feature_dim, hidden_dim, aggr='sum', K=3)
        self.cconv5 = GATv2Conv(hidden_dim, hidden_dim * 2, aggr='sum', K=3)
        self.cconv6 = GATv2Conv(hidden_dim * 2, hidden_dim * 4, aggr='sum', K=3)
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
        protein_feature1, protein_index1, protein_batch = data.x, data.edge_index, data.batch
        # 对小分子进行卷积操作
        protein_feature1 = self.cconv1(protein_feature1, protein_index1)
        protein_feature1 = self.relu(protein_feature1)
        # protein_feature = self.bn1(protein_feature)

        protein_feature1 = self.cconv2(protein_feature1, protein_index1)
        protein_feature1 = self.relu(protein_feature1)
        # protein_feature = self.bn2(protein_feature)

        protein_feature1 = self.cconv3(protein_feature1, protein_index1)
        # protein_feature = self.relu(protein_feature)
        # protein_feature = self.bn3(protein_feature)

        # 对卷积后的小分子进行图的最大值池化
        # print(protein_feature.view(16, -1, self.hidden * 4))
        protein_feature1 = gmp(protein_feature1, protein_batch)
        protein_feature2, protein_index2, protein_batch = data.x, data.edge_index, data.batch
        # 对小分子进行卷积操作
        protein_feature2 = self.cconv1(protein_feature2, protein_index2)
        protein_feature2 = self.relu(protein_feature2)
        # protein_feature = self.bn1(protein_feature)

        protein_feature2 = self.cconv2(protein_feature2, protein_index2)
        protein_feature2= self.relu(protein_feature2)
        # protein_feature = self.bn2(protein_feature)

        protein_feature2 = self.cconv3(protein_feature2, protein_index2)
        # protein_feature = self.relu(protein_feature)
        # protein_feature = self.bn3(protein_feature)

        # 对卷积后的小分子进行图的最大值池化
        # print(protein_feature.view(16, -1, self.hidden * 4))
        protein_feature2 = gmp(protein_feature2, protein_batch)
        protein_feature = torch.concat((protein_feature1,protein_feature2),1)
        protein_feature = self.classification(protein_feature)

        return protein_feature