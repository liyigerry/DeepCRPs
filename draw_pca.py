import matplotlib.pyplot as plt
import torch

from util import ProteinDataset
from sklearn import decomposition
data = ProteinDataset(root='data', dataset='train_data1024')
print(data[0])

X = data.x

print('the shape of X',X.shape)

y = []
count = 0
for i in data:

    node = i.x.shape[0]
    while node>0:
        y.append(i.y)

        node = node-1
y = torch.tensor(y)
print(y.shape)
# print('the shape of X',y.shape)
pca = decomposition.PCA(n_components=2)
pca.fit(X)
x_after = pca.transform(X)
print('the shape of x_after ',x_after.shape)
colors = ['lightcoral','chartreuse','turquoise','darkorange']
# colors =['magenta','lightpink','moccasin','mediumspringgreen']
target = ['Na-CRPs','K-CRPs','nAChRs-CRPs','Ca-CRPs']
# target = ['nAChRs-CRPs','Ca-CRPs','K-CRPs','Na-CRPs']
# 'FFC082','B0FFB0','FCA69E','E26FFF'
font_dict=dict(fontsize=22,
              )
# 0.0000000001
for i,target_name in zip([3,2,0,1],target):
    plt.scatter(x_after[y==i,0],x_after[y==i,1],color = colors[i],lw = 0.00001,label = target_name)
plt.xlabel('First Principal Component',fontsize = 22)
plt.ylabel('Second Principal Component',fontsize = 22)
plt.xticks(size = 22)
plt.yticks(size = 22)
plt.legend(fontsize = 22)
plt.show()