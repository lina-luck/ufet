import numpy as np
from transformers import RobertaTokenizer, RobertaForSequenceClassification
import torch
import sys
import os
project_path = os.path.split(os.path.abspath(os.path.realpath(__file__)))[0] + "/../../"
sys.path.append(os.path.abspath(project_path))

from faiss import IndexFlatIP, normalize_L2, IndexFlatL2
from src.common.constant import *
import argparse
import json
from sklearn.preprocessing import MultiLabelBinarizer


def read_y(file_name):
    with open(file_name) as f:
        line_elems = [json.loads(sent.strip()) for sent in f.readlines()]
        y_str_list = [line_elem["y_str"] for line_elem in line_elems]
    return y_str_list


def get_prediction(t1, t2, model, tokenizer):
    x = tokenizer(t1, t2, return_tensors='pt', max_length=32, truncation=True)
    logits = model(**x).logits
    probs = logits.softmax(dim=1).squeeze(0)
    label_id = torch.argmax(probs).item()
    prediction = model.config.id2label[label_id]
    return prediction


def knn(emb, k, metric='cos'):
    emb_ = np.array(emb).copy().astype('float32')
    dim = emb_.shape[1]

    # cosine similarity
    if metric == 'l2':
        index = IndexFlatL2(dim)
    else:
        normalize_L2(emb_)   # normalize
        index = IndexFlatIP(dim)    # inner product

    index.add(emb_)
    _, I = index.search(emb_, k+1)

    np.save(os.path.join(DATA_ROOT, 'ontology/domain/mirrorbert/crowd/knn_idx.npy'), I)

    knn_pairs = []
    for i in range(emb_.shape[0]):
        for ii in I[i]:
            if ii != i and [i, ii] not in knn_pairs and [ii, i] not in knn_pairs:
                knn_pairs.append([i, ii])
    return knn_pairs


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-goal", help="dataset", default="open")
    parser.add_argument("-o", help="output file name", default="ontology/domain/biencoder/crowd/cn_wanli.txt")
    parser.add_argument("-d", help="data path", default="figer")
    parser.add_argument("-e", help="type embeddings", default='type_embeddings/crowd/biencoder.npy')
    parser.add_argument("-k", help="knn", default=5, type=int)
    parser.add_argument("-model", help="wanli model path", default="../../pretrained_models/roberta-large-wanli")
    args = parser.parse_args()

    # find knn, i.e. candidate cn pairs
    emb_type = np.load(os.path.join(DATA_ROOT, args.e))
    # emb_type = torch.load(os.path.join(DATA_ROOT, args.e)).numpy()
    candidate_pairs_idx = knn(emb_type, args.k, 'cos')
    print(str(len(candidate_pairs_idx)) + " candidate pairs generated")

    model = RobertaForSequenceClassification.from_pretrained(args.model)
    tokenizer = RobertaTokenizer.from_pretrained(args.model)

    ystr = read_y(os.path.join(DATA_ROOT, args.d, 'train.json'))
    ystr += read_y(os.path.join(DATA_ROOT, args.d, 'dev.json'))
    ystr += read_y(os.path.join(DATA_ROOT, args.d, 'test.json'))

    cn_pairs = []
    id2y = ID2ANS_DICT[args.goal]
    cnt = 0
    for i1, i2 in candidate_pairs_idx:
        ignore = False
        if cnt % 1000 == 0:
            print(str(cnt) + " of " + str(len(candidate_pairs_idx)) + " processed")
        t1 = id2y[i1]
        t2 = id2y[i2]
        cnt += 1
        # filtering co-appear types
        for y in ystr:
            if t1 in y and t2 in y:
                ignore = True
                break
        if ignore:
            continue
        t12 = get_prediction(t1, t2, model, tokenizer)
        if t12 == 'contradiction':
            t21 = get_prediction(t2, t1, model, tokenizer)
            if t21 == 'contradiction':
                cn_pairs.append([t1, t2])

    with open(os.path.join(DATA_ROOT, args.o), 'w', encoding='utf-8') as f:
        f.write('\n'.join(['#'.join(tt) for tt in cn_pairs]))

    with open(os.path.join(DATA_ROOT, os.path.dirname(args.o), 'cn2org_wanli.txt'), 'w', encoding='utf-8') as f:
        f.write('\n'.join([','.join([str(ANS2ID_DICT[args.goal][t1]), str(ANS2ID_DICT[args.goal][t2])]) for t1, t2 in cn_pairs]))

    binarizer = MultiLabelBinarizer(classes=list(range(ANSWER_NUM_DICT[args.goal])))
    cn_mapping = binarizer.fit_transform([[ANS2ID_DICT[args.goal][t1], ANS2ID_DICT[args.goal][t2]] for t1, t2 in cn_pairs])
    np.save(os.path.join(DATA_ROOT, os.path.dirname(args.o), 'cn2org_wanli.npy'), cn_mapping)