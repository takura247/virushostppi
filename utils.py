import os
import pandas as pd
import numpy as np
from sklearn import metrics

def standardize_dir(dir):
    res_dir = dir
    if not res_dir.endswith('/') and not res_dir.endswith('\\'):
        res_dir += '/'

    if not os.path.exists(res_dir):
        os.mkdir(res_dir)
    return res_dir

def get_threshold(targets, preds):
    sorted_list = sorted(preds, reverse=True)
    if type(targets) != list:
        targets = targets.tolist()
    N = targets.count(1.0)
    return sorted_list[N]

def get_score(targets, preds, topk=100):
    auc_score = metrics.roc_auc_score(targets, preds, average='micro')

    aupr_score = metrics.average_precision_score(targets, preds, average='micro')
    
    y_preds = [0 if pred < get_threshold(targets, preds) else 1 for pred in preds]
    cm = metrics.confusion_matrix(targets, y_preds)
    tn, fp, fn, tp = cm.ravel()
    sn = round(float(tp) / (tp + fn),4)
    sp = round(float(tn) / (tn + fp),4)

    acc = round(metrics.accuracy_score(targets, y_preds),4)

    sorted_idx = np.argsort(preds)[::-1]
    ranks = list()
    for i, lbl in enumerate(targets):
        if int(lbl) == 1:
            ranks.append(sorted_idx[i])

    top_list = list()
    for rank in ranks:
        if rank <= topk:
            top_list.append(rank)
    count = len(top_list)

    return auc_score, aupr_score, sn, sp, acc, count

