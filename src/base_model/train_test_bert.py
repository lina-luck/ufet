import sys
import os
project_path = os.path.split(os.path.abspath(os.path.realpath(__file__)))[0] + "/../../"
sys.path.append(os.path.abspath(project_path))
from src.base_model.models import *
from tqdm import tqdm, trange
from src.common.early_stop import *
from src.common.eval_metric import *
from src.common.utils import write_gold_pred_str
from transformers.optimization import AdamW, get_cosine_schedule_with_warmup
from transformers import AutoTokenizer
from src.common.dataset import *
from src.common.constant import *
from torch.utils.data import DataLoader
import copy


def binarization(pred, th=0.5):
    predict = copy.deepcopy(pred)
    max_index = torch.argmax(predict, dim=1)
    for dim, i in enumerate(max_index):
        predict[dim, i] = 1
    predict[predict > th] = 1
    predict[predict != 1] = 0
    return predict


def evaluation(model, use_gpu, data, th=0.5, optim_th=False, pred_str=False):
    logging.info("evaluation")
    model.eval()
    ground_truth = []
    predict = []
    dev_loss = 0
    step = 0
    with torch.no_grad():
        for _, input_ids, attention_mask, mask_position, labels in data:
            step += 1
            if use_gpu:
                input_ids = input_ids.to("cuda")
                attention_mask = attention_mask.to("cuda")
                mask_position = mask_position.to("cuda")
                labels = labels.to("cuda")

            logits, loss = model(input_ids, attention_mask, labels, mask_position)
            predict.append(model.sigmoid_fn(logits))
            if isinstance(model, torch.nn.DataParallel):
                loss = loss.mean()
            dev_loss += loss.item()
            ground_truth.append(labels)
    logging.info("Predicted scores computed")
    predict = torch.cat(predict, dim=0)
    ground_truth = torch.cat(ground_truth, dim=0)

    if optim_th:
        th = optimal_threshold(ground_truth.detach().data.numpy(), predict.detach().data.numpy())[0]
    predict = binarization(predict, th)
    macro_p, macro_r, macro_f1 = record_metrics(ground_truth, predict)

    dev_loss /= step

    gold_pred_str = None
    if pred_str:
        pred_idx = [torch.where(pp == 1)[0] for pp in predict]
        gold_idx = [torch.where(gg == 1)[0] for gg in ground_truth]
        gold_pred_str = get_gold_pred_str(pred_idx, gold_idx)
    return macro_p, macro_r, macro_f1, dev_loss, gold_pred_str, th


def fintune_mlm_model(args):
    logging.info(str(args))
    type2id = ANS2ID_DICT[args.goal]

    # tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.bert_version)
    # add prompt placeholder
    tokenizer.add_special_tokens({'additional_special_tokens': ['[PROMPT]']})
    args.prompt_placeholder_id = tokenizer.additional_special_tokens_ids[0]
    args.unk_id = tokenizer.unk_token_id

    # create dataset
    logging.info("build dataset")
    train = DataLoader(
            dataset=UFET(os.path.join(DATA_ROOT, args.train), type2id),
            batch_size=args.train_batch_size,
            shuffle=True,
            collate_fn=lambda x: UFET.collate_fn(x, tokenizer),
            drop_last=False
        )
    valid = DataLoader(
        dataset=UFET(os.path.join(DATA_ROOT, args.valid), type2id),
        batch_size=args.test_batch_size,
        shuffle=False,
        collate_fn=lambda x: UFET.collate_fn(x, tokenizer),
        drop_last=False
    )
    test = DataLoader(
        dataset=UFET(os.path.join(DATA_ROOT, args.test), type2id),
        batch_size=args.test_batch_size,
        shuffle=False,
        collate_fn=lambda x: UFET.collate_fn(x, tokenizer),
        drop_last=False
    )

    # model
    use_gpu = torch.cuda.is_available()
    model = BertMaskModel(args, use_gpu, init_weight=True)

    # optimizer
    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
    )
    lr_schedular = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=len(train),
                                                   num_training_steps=len(train)*args.num_epoch,
                                                   num_cycles=0.5)

    # random batch training seed (shuffle) to ensure reproducibility
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    # model file name
    model_fname = '{0:s}/{1:s}.pt'.format(EXP_ROOT, args.model_id)
    early_stopping = EarlyStopping(patience=10, verbose=True, path=model_fname)

    # use gpu
    if use_gpu:
        n_gpu = torch.cuda.device_count()
        if n_gpu >= 1:
            logging.info("use multiple GPUs")
            model = torch.nn.DataParallel(model)
        model.to('cuda')

    # training
    model.load_state_dict(torch.load(model_fname))
    logging.info("start training...")
    for epoch in trange(int(args.num_epoch), desc='Epoch'):
        tr_loss = 0
        for step, batch in enumerate(tqdm(train)):
            model.train()
            idx, input_ids, attention_mask, mask_position, labels = batch
            if use_gpu:
                input_ids = input_ids.to("cuda")
                attention_mask = attention_mask.to("cuda")
                mask_position = mask_position.to("cuda")
                labels = labels.to("cuda")

            # forward
            optimizer.zero_grad()
            logits, loss = model(input_ids, attention_mask, labels, mask_position)
            if isinstance(model, torch.nn.DataParallel):
                loss = loss.mean()

            # backward
            loss.backward()
            optimizer.step()
            lr_schedular.step()

            tr_loss += loss.item()

        tr_loss /= (step + 1)

        # valid
        _, _, f1, dev_loss, _, _ = evaluation(model, use_gpu, valid)
        logging.info('Epoch %d: train_loss: %.4f | dev_loss: %.4f | dev_f1: %.4f', epoch, tr_loss, dev_loss, f1)

        early_stopping(-f1, model)
        if early_stopping.early_stop:
            logging.info("Early stopping. Model trained")
            break

    result_path = os.path.join(project_path, 'output', 'results', args.dataset, args.model_id)
    if not os.path.exists(result_path):
        os.makedirs(result_path)

    model.load_state_dict(torch.load(model_fname))
    logging.info("lambda = " + str(model.lamda))

    ma_p_dev, ma_r_dev, ma_f1_dev, _, gold_pred_str, th = evaluation(model, use_gpu, valid, optim_th=False, pred_str=True)
    write_gold_pred_str(os.path.join(result_path, 'dev.txt'), gold_pred_str)
    ma_p_test, ma_r_test, ma_f1_test, _, gold_pred_str, _ = evaluation(model, use_gpu, test, th=th, pred_str=True)
    write_gold_pred_str(os.path.join(result_path, 'test.txt'), gold_pred_str)

    logging.info("dev: macro_p: %.4f | macro_r: %.4f | macro_f1: %.4f", ma_p_dev, ma_r_dev, ma_f1_dev)
    logging.info("test: macro_p: %.4f | macro_r: %.4f | macro_f1: %.4f", ma_p_test, ma_r_test, ma_f1_test)

    out_str = str(args) + '\n'
    out_str += "th = " + str(th) + '\n'
    out_str += "dev: macro_p: %.4f | macro_r: %.4f | macro_f1: %.4f\n" % (ma_p_dev, ma_r_dev, ma_f1_dev)
    out_str += "test: macro_p: %.4f | macro_r: %.4f | macro_f1: %.4f\n" % (ma_p_test, ma_r_test, ma_f1_test)

    with open(os.path.join(result_path, args.dataset + '.txt'), 'a+', encoding='utf-8') as f:
        f.write(out_str)

    logging.info("done")
