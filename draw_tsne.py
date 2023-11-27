import matplotlib.pyplot as plt
import torch

from util import ProteinDataset
from sklearn import manifold
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
tsene = manifold.TSNE(n_components=2,init='pca',random_state=10)

x_after =tsene.fit_transform(X)

print('the shape of x_after ',x_after.shape)
# x_min,x_max = x_after.min(0),x_after.max(0)
# x_final = (x_after-x_min)/(x_after-x_max)

colors = ['lightcoral','chartreuse','turquoise','darkorange']
# colors =['magenta','lightpink','moccasin','mediumspringgreen']
# target = ['ach','ca','k','na']
target = ['Na-CRPs','K-CRPs','nAChRs-CRPs','Ca-CRPs']
# 'FFC082','B0FFB0','FCA69E','E26FFF'
font_dict=dict(fontsize=22,
              )
# for i,target_name in zip([0,1,2,3],target):
for i,target_name in zip([3,2,0,1],target):
    plt.scatter(x_after[y==i,0],x_after[y==i,1],color = colors[i],lw = 0.0000000001,label = target_name)
plt.xlabel('tSNE_1',fontsize = 22)
plt.ylabel('tSNE_2',fontsize = 22)
plt.xticks(size = 22)
plt.yticks(size = 22)
plt.legend(fontsize = 22)
plt.show()