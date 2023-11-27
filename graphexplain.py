from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.nn import global_mean_pool
from torch_geometric.datasets import TUDataset
from util import ProteinDataset
from torch_geometric.loader import DataLoader
import torch
import os
from torch_geometric.explain import Explainer, PGExplainer

class GCN(torch.nn.Module):
    def __init__(self, hidden_channels):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(dataset.num_node_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels*2)
        self.conv3 = GCNConv(hidden_channels*2, hidden_channels*4)
        self.lin1 = Linear(hidden_channels*4, 1024)
        self.lin2 = Linear(1024, 512)
        self.lin3 = Linear(512, 4)

    def forward(self, x, edge_index, batch):
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)
        x = x.relu()
        x = self.conv3(x, edge_index)

        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin1(x)
        x = x.relu()
        x = self.lin2(x)
        x = x.relu()
        x = self.lin3(x)

        x = global_mean_pool(x, batch)
        return x



path = str(os.getcwd())

dataset = ProteinDataset(root='data', dataset='train_data20')

loader2 = DataLoader(dataset, batch_size=32, shuffle=True)
loader1 = DataLoader(dataset,batch_size=1,shuffle=False)



model = GCN(hidden_channels=32)
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
criterion = torch.nn.CrossEntropyLoss()



# Initialize explainer with the model and PGExplainer algorithm

explainer = Explainer(
    model=model,
    algorithm=PGExplainer(epochs=1500, lr=0.00003),
    explanation_type="phenomenon",
    edge_mask_type="object",
    model_config=dict(mode="multiclass_classification", task_level="graph",
                      return_type="raw"),
    threshold_config=dict(threshold_type='topk', value=10),
)
for epoch in range(1500):
    print(epoch)
    trainloss = 0.0
    for batch in loader2:

        loss = explainer.algorithm.train(
            epoch, model, batch.x, batch.edge_index, target=batch.y.to(torch.int64),batch = batch.batch)
        trainloss+=loss
    print('the loss:',trainloss/1719)
with open('result/trainloss1.txt', 'a') as write:
    write.writelines(str(trainloss/1719) + '\n')

for data1 in loader1:
    print('the value of data1:',data1)
    id = data1['id']
    id = str(id)
    id = id[1:-6]
    path = (id)+'.pdf'
    print(path)
    explanation = explainer(data1.x, data1.edge_index,target=data1.y.to(torch.int64), batch=data1.batch, index=0)
    explanation.visualize_graph(path,'networkx')
    print(f"Subgraph visualization plot has been saved to '{path}'")
    print(explanation.edge_mask)