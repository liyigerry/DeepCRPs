import torch
import torch.nn as nn
from torch_geometric.nn import ChebConv, global_max_pool as gmp

class GraphConv(nn.Module):
    def __init__(self, feature_dim, hidden_dim, output_dim=256, dropout=0.4, use_residue=False):
        super(GraphConv, self).__init__()
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

        compound_feature, compound_index, compound_batch = data.x, data.edge_index, data.batch
        # print(compound_feature.shape)
        # 对小分子进行卷积操作
        compound_feature = self.cconv1(compound_feature, compound_index)
        compound_feature = self.relu(compound_feature)
        # compound_feature = self.bn1(compound_feature)

        compound_feature = self.cconv2(compound_feature, compound_index)
        compound_feature = self.relu(compound_feature)
        # compound_feature = self.bn2(compound_feature)

        compound_feature = self.cconv3(compound_feature, compound_index)
        # compound_feature = self.relu(compound_feature)
        # compound_feature = self.bn3(compound_feature)

        # 对卷积后的小分子进行图的最大值池化
        # print(compound_feature.view(16, -1, self.hidden * 4))
        compound_feature = gmp(compound_feature, compound_batch)

        compound_feature = self.classification(compound_feature)

        return compound_feature