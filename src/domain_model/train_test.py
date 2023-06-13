import logging
import sys
import os
import copy

import torch

project_path = os.path.split(os.path.abspath(os.path.realpath(__file__)))[0] + "/../../"
sys.path.append(os.path.abspath(project_path))
from src.domain_model.models import *
from tqdm import tqdm, trange
from src.common.early_stop import *
from src.common.eval_metric import brute, record_metrics, get_gold_pred_str
from src.common.utils import write_gold_pred_str
from transformers.optimization import AdamW, get_cosine_schedule_with_warmup
from transformers import AutoTokenizer
from src.common.dataset import *
from torch.utils.data import DataLoader
from src.common.utils import y2domain

def filtering_with_cn(pred_label, prob, cn_mapping):
    '''
    filtering predicted labels.
    If predicted types in a same domain are concept neighbors, choose one giving max prob
    :param pred_label: prediceted labels from domain model
    :param prob: probabilities with shape (num_samp, num_type)
    :param cn_type: concept neighbor types used
    :return:
    '''
    cn_mapping = torch.tensor(cn_mapping, dtype=torch.float).to(prob.device)
    for i in range(pred_label.size()[0]):
        pred = copy.deepcopy(pred_label[i])
        m_cn = pred * cn_mapping   # cn2org matrix, a 0-1 matrix, column is org types
        # if sum of col > 1,
        # means there are predicted types are concept netighbors,
        # need to be filtered
        cn_ii = torch.where(m_cn.sum(1) > 1)[0]
        prob_cn = m_cn[cn_ii] * prob[i]  # make prob = 0 if predicted label = 0
        max_ti = torch.argmax(prob_cn, dim=1)
        pred[m_cn[cn_ii].sum(0) > 0] = 0
        pred[max_ti] = 1
        pred_label[i] = pred
    return pred_label


def binarization(predict, th=0.5):
    '''
    prediction
    :param predict: predicted prob
    :param th: threshold
    :return:
    predicted labels
    '''
    pred_label = copy.deepcopy(predict)
    max_index = torch.argmax(predict, dim=1)
    for dim, i in enumerate(max_index):
        pred_label[dim, i] = 1
    pred_label[pred_label > th] = 1
    pred_label[pred_label != 1] = 0
    return pred_label


def macro_f1(th, y_true, y_score, mapping_dom2types, cut_off):
    y_pred = prediction(y_score, mapping_dom2types, cut_off, y_true, th=th)[0]
    f1 = record_metrics(torch.tensor(y_true), torch.tensor(y_pred))[2]
    logging.info('th = %.4f, f1 = %.4f', th, -f1)
    return -f1


def optimal_threshold(y_true, y_score, mapping_dom2types, cut_off):
    bounds = [(np.min(y_score), np.max(y_score))]
    logging.info("min vs max prob: %.4f, %.4f", np.min(y_score), np.max(y_score))
    result = brute(macro_f1, args=(y_true, y_score, mapping_dom2types, cut_off), ranges=bounds, full_output=True, Ns=20, workers=1)
    return result[0][0], -macro_f1(result[0][0], y_true, y_score, mapping_dom2types, cut_off)


def prediction(prob, mapping_dom2types, cut_off, true_label=None, optim_th=False, th=0.5):
    '''
    prediction
    :param prob: probabilities of all types (org + domain)
    :param mapping_dom2types: domain to types mapping matrix
    :return: predicted labels
    '''
    if optim_th:
        logging.info('optimizing threshold')
        th = optimal_threshold(true_label.detach().data.numpy(), prob.detach().data.numpy(), mapping_dom2types, cut_off)[0]
        logging.info('optimized th = %.4f', th)
    prob = torch.tensor(prob)
    if isinstance(th, np.ndarray):
        th = th[0]

    logging.info('predicted all labels (original + domain)')
    pred_all = binarization(prob, th)  # bsz * all types

    logging.info('split predicted all labels')
    pred_org = pred_all[:, :cut_off]   # bsz * org_types
    pred_dom = pred_all[:, cut_off:]   # bsz * dom_types

    logging.info('predict from domain')
    pred_dom_from_org = pred_org.mm(mapping_dom2types.T)   # bsz * dom_types
    pred_dom_from_org[pred_dom_from_org > 1] = 1
    pred_dom_only = pred_dom - pred_dom_from_org   # predicted domain that none of types in domain is predicted
    pred_dom_only[pred_dom_only != 1] = 0

    pred_types_by_dom = pred_dom_only.mm(mapping_dom2types)  # bsz * org_types, predicted types by domain
    prob_by_dom_only = prob[:, :cut_off] * pred_types_by_dom  # bsz * org_types, prob = 0 if pred_types_by_dom = 0

    pred_by_dom = torch.zeros_like(pred_org)  # final prediction from domain
    pred_dom_only_sum = torch.sum(pred_dom_only, dim=1)   # bsz
    logging.info('totoal predicted domain labels: %d', int(torch.sum(pred_dom_only_sum)))

    # case 1: sum == 1, namely only one domain predicted
    # argmax directly
    logging.info("process cases that only one domain predicted")
    samp_idx = torch.where(pred_dom_only_sum == 1)[0]
    if samp_idx.shape[0] > 0:  # if samp_idx is not empty
        logging.info(str(samp_idx.shape[0]) + " samples have single domain predicted")
        type_idx = torch.argmax(prob_by_dom_only[samp_idx], dim=1)
        pred_by_dom[samp_idx, type_idx] = 1

    # case 2: sum > 1, namely multiple domains predicted
    # handle each domain separately
    logging.info("process cases that more than one domains predicted")
    samp_idx = torch.where(pred_dom_only_sum > 1)[0]
    if samp_idx.shape[0] > 0:  # if samp_idx is not empty
        logging.info(str(samp_idx.shape[0]) + " samples have multiple domain predicted")
        for si in samp_idx:
            dom_idx = torch.where(pred_dom_only[si] == 1)[0]
            for di in dom_idx:
                ti_in_di = torch.where(mapping_dom2types[di] == 1)[0]
                prob_si = torch.zeros_like(prob[0])
                prob_si[ti_in_di] = prob[si][ti_in_di]
                ti = torch.argmax(prob_si)
                pred_by_dom[si, ti] = 1

    predict = pred_org + pred_by_dom
    predict[predict > 1] = 1
    predict[predict != 1] = 0
    logging.info('prediction done')
    return predict, th


def evaluation(args,
               model,
               use_gpu,
               data,
               mapping_dom2types,
               cn_mapping=None,
               optim_th=False,
               is_test=False,
               th=0.5,
               ds='dev'):
    logging.info("evaluation")
    model.eval()
    ground_truth = []
    # true_domain = []
    prob_domain = []
    dev_loss = 0
    step = 0
    with torch.no_grad():
        for _, input_ids, attention_mask, mask_position, labels in data:
            step += 1
            ground_truth.append(labels)
            # convert org y to domain y
            y_domain = y2domain(labels, mapping_dom2types)
            del labels

            if use_gpu:
                input_ids = input_ids.to("cuda")
                attention_mask = attention_mask.to("cuda")
                mask_position = mask_position.to("cuda")
                y_domain = y_domain.to("cuda")

            # true_domain.append(y_domain)

            logits, loss = model(input_ids, attention_mask, y_domain, mask_position)
            if isinstance(model, torch.nn.DataParallel):
                loss = loss.mean()
                prob_domain.append(model.module.sigmoid_fn(logits))
            else:
                prob_domain.append(model.sigmoid_fn(logits))
            dev_loss += loss.item()

    logging.info("Predicted scores computed")
    prob_domain = torch.cat(prob_domain, dim=0)
    ground_truth = torch.cat(ground_truth, dim=0)
    # true_domain = torch.cat(true_domain, dim=0)

    prob_domain = prob_domain.detach().cpu()
    # mapping matrix
    mapping_dom2types = mapping_dom2types.to(prob_domain.device)  # dom * types_in_dom
    org_type_num = ANSWER_NUM_DICT[args.goal]

    # prediction
    logging.info("post-processing start")
    predict, th = prediction(prob_domain, mapping_dom2types, org_type_num, true_label=ground_truth, optim_th=optim_th, th=th)
    # predict = binarization(prob_domain, th)[:, :org_type_num]
    macro_p, macro_r, macro_f1 = record_metrics(ground_truth, predict)

    result_path = os.path.join(project_path, 'output', 'results', args.model_id, args.dataset)
    if is_test:
        pred_idx = [torch.where(pp == 1)[0] for pp in predict]
        gold_idx = [torch.where(gg == 1)[0] for gg in ground_truth]
        gold_pred_str = get_gold_pred_str(pred_idx, gold_idx, args.goal)
        write_gold_pred_str(os.path.join(result_path, 'res_' + ds + '.txt'), gold_pred_str)

    # filtering based on concept neighbors
    # rebuild prob matrx
    if cn_mapping is not None:
        logging.info("before filter: avg pred: %.4f | macro_p: %.4f | macro_r: %.4f | macro_f1: %.4f", predict.sum(1).mean(), macro_p, macro_r, macro_f1)
        print('before', predict.sum(1).mean(), macro_p, macro_r, macro_f1)

        # filtering
        predict = filtering_with_cn(predict, prob_domain[:, :org_type_num], cn_mapping)
        macro_p, macro_r, macro_f1 = record_metrics(ground_truth, predict)
        logging.info("after filter: avg pred: %.4f | macro_p: %.4f | macro_r: %.4f | macro_f1: %.4f", predict.sum(1).mean(), macro_p, macro_r, macro_f1)
        print('after', predict.sum(1).mean(), macro_p, macro_r, macro_f1)
    dev_loss /= step

    gold_pred_str = None
    if is_test:
        pred_idx = [torch.where(pp == 1)[0] for pp in predict]
        gold_idx = [torch.where(gg == 1)[0] for gg in ground_truth]
        gold_pred_str = get_gold_pred_str(pred_idx, gold_idx, args.goal)
    return macro_p, macro_r, macro_f1, dev_loss, gold_pred_str, th


def run_domain_model(args):
    logging.info(str(args))
    org_type2id = ANS2ID_DICT[args.goal]
    type2id = load_vocab_dict(
        os.path.join(VOCAB_DIR_DICT[args.d_goal],  args.dfn_postfix + ".txt"))

    args.label_num = len(type2id)

    mapping_dom2types = np.load(os.path.join(VOCAB_DIR_DICT[args.d_goal],
                                             'dom2types_' + args.dfn_postfix + '.npy'))
    mapping_dom2types = torch.tensor(mapping_dom2types, dtype=torch.float)

    cn_mapping = None
    if args.cn:
        cn_mapping = np.load(os.path.join(VOCAB_DIR_DICT[args.d_goal], 'cn2org_wanli.npy'))
        cn_mapping = torch.tensor(cn_mapping, dtype=torch.float)

    # tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.bert_version)
    # add prompt placeholder
    tokenizer.add_special_tokens({'additional_special_tokens': ['[PROMPT]']})
    args.prompt_placeholder_id = tokenizer.additional_special_tokens_ids[0]
    args.unk_id = tokenizer.unk_token_id

    # create dataset
    logging.info("build dataset")
    train = DataLoader(
            dataset=UFET(os.path.join(DATA_ROOT, args.train), org_type2id),
            batch_size=args.train_batch_size,
            shuffle=True,
            collate_fn=lambda x: UFET.collate_fn(x, tokenizer),
            drop_last=False
        )
    valid = DataLoader(
        dataset=UFET(os.path.join(DATA_ROOT, args.valid), org_type2id),
        batch_size=args.test_batch_size,
        shuffle=False,
        collate_fn=lambda x: UFET.collate_fn(x, tokenizer),
        drop_last=False
    )
    test = DataLoader(
        dataset=UFET(os.path.join(DATA_ROOT, args.test), org_type2id),
        batch_size=args.test_batch_size,
        shuffle=False,
        collate_fn=lambda x: UFET.collate_fn(x, tokenizer),
        drop_last=False
    )

    # model
    use_gpu = torch.cuda.is_available()
    model = DomainModel(args, use_gpu)

    if args.tune_all:
        tuned_parameters = filter(lambda p: p.requires_grad, model.parameters())
    else:
        unfreeze_layers = ['dfc']
        tuned_parameters = [
            {'params': [param for name, param in model.named_parameters() if any(ul in name for ul in unfreeze_layers)]}]

    # optimizer
    optimizer = AdamW(
        tuned_parameters,
        lr=args.lr,
    )
    lr_schedular = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=len(train),
                                                   num_training_steps=len(train) * args.num_epoch,
                                                   num_cycles=1.5)

    # random batch training seed (shuffle) to ensure reproducibility
    # torch.manual_seed(args.seed)
    # random.seed(args.seed)
    # np.random.seed(args.seed)

    # model file name
    model_fname = '{0:s}/{1:s}.pt'.format(EXP_ROOT, args.model_id)
    early_stopping = EarlyStopping(patience=10, verbose=True, path=model_fname)

    # use gpu
    if use_gpu:
        logging.info("use gpu")
        n_gpu = torch.cuda.device_count()
        if n_gpu > 1:
            logging.info("use multiple GPUs")
            model = torch.nn.DataParallel(model)
        model.to('cuda')

    # training
    if args.do_train:
        # if os.path.exists(model_fname):
        #     model.load_state_dict(torch.load(model_fname))

        logging.info("start training...")
        for epoch in trange(int(args.num_epoch), desc='Epoch'):
            tr_loss = 0
            for step, batch in enumerate(tqdm(train)):
                model.train()
                idx, input_ids, attention_mask, mask_position, labels = batch

                # convert org y to domain y
                y_domain = y2domain(labels, mapping_dom2types)
                del labels
                if use_gpu:
                    input_ids = input_ids.to("cuda")
                    attention_mask = attention_mask.to("cuda")
                    mask_position = mask_position.to("cuda")
                    y_domain = y_domain.to("cuda")

                # forward
                optimizer.zero_grad()
                loss = model(input_ids, attention_mask, y_domain, mask_position)[1]
                if isinstance(model, torch.nn.DataParallel):
                    loss = loss.mean()

                # backward
                loss.backward()
                optimizer.step()
                lr_schedular.step()

                tr_loss += loss.item()

            tr_loss /= (step + 1)

            # valid
            _, _, f1, dev_loss, _, _ = evaluation(args, model, use_gpu, valid,
                                                  mapping_dom2types,
                                                  cn_mapping=cn_mapping, optim_th=False)

            logging.info('Epoch %d: train_loss: %.4f | dev_loss: %.4f | dev_f1: %.4f', epoch, tr_loss, dev_loss, f1)

            early_stopping(-f1, model)
            if early_stopping.early_stop:
                logging.info("Early stopping. Model trained")
                break

    if not args.do_train:
        result_path = os.path.join(project_path, 'output', 'results', args.model_id, args.dataset)
        if not os.path.exists(result_path):
            os.makedirs(result_path)

        model.load_state_dict(torch.load(model_fname))
        ma_p_dev, ma_r_dev, ma_f1_dev, _, gold_pred_str, th = evaluation(args, model, use_gpu, valid,
                                                                     mapping_dom2types,
                                                                      cn_mapping=cn_mapping,
                                                                      optim_th=args.optim_th,
                                                                      is_test=True, ds='dev')
        write_gold_pred_str(os.path.join(result_path, 'res_dev_cn.txt'), gold_pred_str)

        ma_p_test, ma_r_test, ma_f1_test, _, gold_pred_str, _ = evaluation(args, model, use_gpu, test,
                                                                       mapping_dom2types,
                                                                       cn_mapping=cn_mapping,
                                                                       optim_th=args.optim_th,
                                                                       is_test=True, ds='test')
        write_gold_pred_str(os.path.join(result_path, 'res_test_cn.txt'), gold_pred_str)

        logging.info("dev: macro_p: %.4f | macro_r: %.4f | macro_f1: %.4f", ma_p_dev, ma_r_dev, ma_f1_dev)
        logging.info("test: macro_p: %.4f | macro_r: %.4f | macro_f1: %.4f", ma_p_test, ma_r_test, ma_f1_test)

        out_str = str(args) + '\n'
        out_str += "dev: macro_p: %.4f | macro_r: %.4f | macro_f1: %.4f\n" % (ma_p_dev, ma_r_dev, ma_f1_dev)
        out_str += "test: macro_p: %.4f | macro_r: %.4f | macro_f1: %.4f\n" % (ma_p_test, ma_r_test, ma_f1_test)

        with open(os.path.join(result_path, args.dataset + '.txt'), 'a+', encoding='utf-8') as f:
            f.write(out_str)

        logging.info("done")
