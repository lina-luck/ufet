import numpy as np
from torch import nn
import sys
import os
project_path = os.path.split(os.path.abspath(os.path.realpath(__file__)))[0] + "/../../"
sys.path.append(os.path.abspath(project_path))
from src.common.constant import *
from scipy.optimize import brute
from sklearn.metrics import f1_score, precision_score, recall_score
import torch


def f1(p, r):
    '''
    compute f1 score
    :param p: precision
    :param r: recall
    :return: f1
    '''
    if r == 0.:
        return 0.
    return 2 * p * r / float(p + r)


def macro_f1(th, y_true, y_score):
    '''
    compute macro f1 score with threshold as input
    :param th: threshold
    :param y_true: true label
    :param y_score: predicted score
    :return: f1
    '''
    y_pred = (y_score > th) * 1
    return -f1_score(y_true, y_pred, average='macro')


def optimal_threshold(y_true, y_score):
    bounds = [(np.min(y_score), np.max(y_score))]
    # print(bounds)
    result = brute(macro_f1, args=(y_true, y_score), ranges=bounds, full_output=True, Ns=20, workers=2)
    # print(result)
    return result[0][0], -macro_f1(result[0][0], y_true, y_score)


def get_gold_pred_str(pred_idx, gold, goal='open'):
    """
    Given predicted ids and gold ids, generate a list of (gold, pred) pairs of length batch_size.
    """
    id2word_dict = ID2ANS_DICT[goal]
    gold_strs = []
    for gold_i in gold:
        gold_strs.append([id2word_dict[int(i)] for i in gold_i])
    pred_strs = []
    for pred_idx1 in pred_idx:
        pred_strs.append([(id2word_dict[int(ind)]) for ind in pred_idx1])
    return list(zip(gold_strs, pred_strs))


def record_metrics(label, predict):
    sample_num = label.size(0)
    # strict metric: P==R==F1
    correct_num = (torch.abs(predict - label).sum(1) == 0).sum().item()
    acc = correct_num / sample_num

    # micro metric
    micro_p = (label * predict).sum() / predict.sum()
    micro_r = (label * predict).sum() / label.sum()
    micro_f1 = f1(micro_p, micro_r)

    macro_p = torch.nan_to_num(((label * predict).sum(1) / predict.sum(1))).mean()
    macro_r = torch.nan_to_num(((label * predict).sum(1) / label.sum(1))).mean()
    macro_f1 = f1(macro_p, macro_r)
    # print(acc, macro_f1, micro_f1)
    return macro_p, macro_r, macro_f1

