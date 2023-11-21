from sklearn import metrics

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
# from torchmetrics.classification import MulticlassAUROC
# from torchmetrics import AUROC
import numpy
# 分类模型评价指标
def accuracy_score(y, f):
    return metrics.accuracy_score(y, f)

def precision_score(y, f):
    return metrics.precision_score(y, f, average='macro')

def recall_score(y, f):
    return metrics.recall_score(y, f, average='macro')

def f1_score(y, f):
    return metrics.f1_score(y, f, average='macro')

def mcc_score(y, f):
    return metrics.matthews_corrcoef(y, f)
# def macro_auc(preds, targets):
#
#     scores = []
#     for i in range(targets.max() + 1):
#         # Create one-vs-all targets and predictions
#         target_i = (targets == i).long()
#         preds_i = preds[:, i]
#         # Only calculate AUC if there are both positive and negative samples for this class
#         if target_i.sum() > 0 and (1 - target_i).sum() > 0:
#             scores.append(roc_auc_score(target_i.numpy(), preds_i.numpy()))
#     return np.mean(scores)
def auc_score(y, f):

    f = torch.nn.Softmax(dim=1)(f)
    f = f.numpy()

    return metrics.roc_auc_score(y, f, average='macro', multi_class='ovo')
# def caculateAUC(AUC_labels,AUC_out):
#     row, col = AUC_labels.shape
#     temp = []
#     ROC = 0
#     for i in range(col):
#         try:
#             ROC = roc_auc_score(AUC_out[:, i], AUC_labels[:, i], average='macro')
#         except ValueError:
#             pass
#         # print("%d th AUROC: %f" % (i, ROC))
#         temp.append(ROC)
#     for i in range(col):
#         ROC += float(temp[i])
#     return ROC / (col + 1)
 # 计算的时候调用上方函数，其中AUC_out为网络输出，AUC_labels为监督标签
# def mc_auroc(preds,target):
#     metric = MulticlassAUROC(num_classes=4,average='macro')
#     return metric(preds,target)
# def auroc(preds,target):
#     metric = AUROC(task="multiclass",num_classes=4)
#     return metric(preds,target)
import numpy as np
import torch
from sklearn.metrics import roc_auc_score
import torch
import numpy as np
from sklearn.metrics import roc_auc_score

def multi_label_auc(y_true, y_pred):
    """
    计算多标签AUC的函数
    :param y_true: 真实标签，形状为[N, num_classes]
    :param y_pred: 预测标签，形状为[N, num_classes]
    :return: 多标签AUC
    """
    # 将标签转换为numpy数组
    y_true = y_true.cpu().numpy()
    y_pred = y_pred.cpu().numpy()

    # 初始化多标签AUC值
    total_auc = 0.

    # 计算每个标签的AUC值，并对所有标签的AUC值求平均
    for i in range(y_true.shape[1]):
        try:
            auc = roc_auc_score(y_true[:, i], y_pred[:, i])
        except ValueError:
            auc = 0.5  # 如果标签中只有一个类别，则返回0.5
        total_auc += auc

    multi_auc = total_auc / y_true.shape[1]

    return multi_auc


def auc2(y_true,y_pred):
    auc = multi_label_auc(y_true, y_pred)
    return auc
