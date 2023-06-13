import os
import sys
import torch.nn as nn

project_path = os.path.split(os.path.abspath(os.path.realpath(__file__)))[0] + "/../../"
sys.path.append(os.path.abspath(project_path))
from src.common.constant import *
import torch


def write_gold_pred_str(file_name, gold_pred):
    with open(file_name, 'w', encoding="utf-8") as f:
        f.write('gold\tpred\n')
        for g, p in gold_pred:
            f.write(','.join(g) + '\t' + ','.join(p) + '\n')


def init_logging_path(log_path, file_name):
    '''
    build log file
    :param log_path: log path
    :param file_name: log file
    :return:
    '''
    dir_log = os.path.join(log_path, f"{file_name}/")
    if os.path.exists(dir_log) and os.listdir(dir_log):
        dir_log += f'{file_name}_{len(os.listdir(dir_log))}.log'
        with open(dir_log, 'w'):
            os.utime(dir_log, None)
    if not os.path.exists(dir_log):
        os.makedirs(dir_log)
        dir_log += f'{file_name}_{len(os.listdir(dir_log))}.log'
        with open(dir_log, 'w'):
            os.utime(dir_log, None)
    return dir_log


def ufet_multitask_loss(logits, targets):
    '''
    compute multi-task loss, loss = BCE(gen) + BCE(fine) + BCE(finer) + BCE(dom)
    :param logits: logits from network
    :param targets: golden label
    :return: loss
    '''
    loss_func = nn.BCEWithLogitsLoss(reduction='sum')
    gen_cutoff, fine_cutoff, final_cutoff = ANSWER_NUM_DICT['gen'], ANSWER_NUM_DICT['kb'], ANSWER_NUM_DICT['open']

    loss = 0.0
    comparison_tensor = torch.Tensor([1.0])
    use_gpu = logits.device.type == 'cuda'
    if use_gpu:
        comparison_tensor = comparison_tensor.to('cuda')

    gen_targets = targets[:, :gen_cutoff]
    fine_targets = targets[:, gen_cutoff: fine_cutoff]
    finer_targets = targets[:, fine_cutoff: final_cutoff]

    gen_target_sum = torch.sum(gen_targets, 1)
    fine_target_sum = torch.sum(fine_targets, 1)
    finer_target_sum = torch.sum(finer_targets, 1)

    if torch.sum(gen_target_sum.data) > 0:
        gen_mask = torch.squeeze(torch.nonzero(torch.min(gen_target_sum.data, comparison_tensor)), dim=1)
        gen_logit_masked = logits[:, :gen_cutoff][gen_mask, :]
        gen_mask = torch.autograd.Variable(gen_mask)
        if use_gpu:
            gen_mask = gen_mask.to('cuda')
        gen_target_masked = gen_targets.index_select(0, gen_mask)
        gen_loss = loss_func(gen_logit_masked, gen_target_masked)
        loss += gen_loss

    if torch.sum(fine_target_sum.data) > 0:
        fine_mask = torch.squeeze(torch.nonzero(torch.min(fine_target_sum.data, comparison_tensor)), dim=1)
        fine_logit_masked = logits[:, gen_cutoff:fine_cutoff][fine_mask, :]
        fine_mask = torch.autograd.Variable(fine_mask)
        if use_gpu:
            fine_mask = fine_mask.to('cuda')
        fine_target_masked = fine_targets.index_select(0, fine_mask)
        fine_loss = loss_func(fine_logit_masked, fine_target_masked)
        loss += fine_loss

    if torch.sum(finer_target_sum.data) > 0:
        finer_mask = torch.squeeze(torch.nonzero(torch.min(finer_target_sum.data, comparison_tensor)), dim=1)
        finer_logit_masked = logits[:, fine_cutoff: final_cutoff][finer_mask, :]
        finer_mask = torch.autograd.Variable(finer_mask)
        if use_gpu:
            finer_mask = finer_mask.to('cuda')
        finer_target_masked = finer_targets.index_select(0, finer_mask)
        finer_loss = loss_func(finer_logit_masked, finer_target_masked)
        loss += finer_loss

    type_num = targets.shape[1]
    if type_num > final_cutoff:
        dom_targets = targets[:, final_cutoff:]
        dom_target_sum = torch.sum(dom_targets, 1)
        if torch.sum(dom_target_sum.data) > 0:
            dom_mask = torch.squeeze(torch.nonzero(torch.min(dom_target_sum.data, comparison_tensor)), dim=1)
            dom_logit_masked = logits[:, final_cutoff:][dom_mask, :]
            dom_mask = torch.autograd.Variable(dom_mask)
            if use_gpu:
                dom_mask = dom_mask.to('cuda')
            dom_target_masked = dom_targets.index_select(0, dom_mask)
            dom_loss = loss_func(dom_logit_masked, dom_target_masked)
            loss += dom_loss
    return loss


# y type = torch.tensor
def y2domain(y, dom2types):
    '''
    convert original y to domain y
    :param y: original y idx
    :param config: parameters
    :return: y_domain, y_type
    '''
    y_domain = y.mm(dom2types.T)
    y_domain[y_domain > 1] = 1
    return torch.cat([y, y_domain], dim=1)