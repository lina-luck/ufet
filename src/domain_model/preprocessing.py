import argparse
import sys
import os
import numpy as np
project_path = os.path.split(os.path.abspath(os.path.realpath(__file__)))[0] + "/../../"
sys.path.append(os.path.abspath(project_path))
from src.common.constant import *
from sklearn.preprocessing import MultiLabelBinarizer


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-dom", help="domain file", default="ontology/domain/biencoder_prop/crowd/types_mul.txt")
    parser.add_argument("-goal", help="goal", default='open')
    parser.add_argument("-out_dir", help="output path", default="ontology/domain/biencoder_prop/crowd")
    args = parser.parse_args()

    # load vocab
    dom_dict = load_vocab_dict(os.path.join(DATA_ROOT, args.dom))
    org_dict = ANS2ID_DICT[args.goal]

    dom2org_id = dict()
    for d in dom_dict:
        if "#" in d:  # dom type
            y = d.split("#")
            yid_org = [org_dict[yy] for yy in y]
            dom2org_id[d] = yid_org

    out_path = os.path.join(DATA_ROOT, args.out_dir)
    if not os.path.exists(out_path):
        os.makedirs(out_path)

    with open(os.path.join(out_path, "dom2types_mul.txt"), 'w', encoding='utf-8') as f:
        f.write('\n'.join([str(k) + '\t' +
                           ','.join([str(item) for item in dom2org_id[k]]) for k in dom2org_id]))

    binarizer = MultiLabelBinarizer(classes=list(range(ANSWER_NUM_DICT[args.goal])))
    mapping = binarizer.fit_transform(dom2org_id.values())
    np.save(os.path.join(out_path, 'dom2types_mul.npy'), mapping)



