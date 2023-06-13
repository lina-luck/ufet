import argparse
import sys
import os
project_path = os.path.split(os.path.abspath(os.path.realpath(__file__)))[0] + "/../../"
sys.path.append(os.path.abspath(project_path))

from src.common.constant import *
import torch


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-model", help="file name of base model", default="bert_base_cased.pt")
    parser.add_argument("-mapping", help="file name of domain to types mapping", default="ontology/domain/biencoder_prop/crowd/dom2types_mul.txt")
    parser.add_argument("-out_dir", help="output path", default="biencoder_prop/crowd")
    args = parser.parse_args()

    weight_proto = torch.load(os.path.join(EXP_ROOT, args.model))['fc.weight']
    dom2org = load_mapping(os.path.join(DATA_ROOT, args.mapping))

    weight_init_dom = torch.zeros((len(dom2org), weight_proto.shape[1]))
    for i in range(len(dom2org)):
        weight_init_dom[i] = torch.mean(weight_proto[dom2org[i]], dim=0)

    weight_init = torch.cat((weight_proto, weight_init_dom))

    out_path = os.path.join(INIT_WEIGHTS_ROOT, args.out_dir)
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    torch.save(weight_init, os.path.join(out_path, 'bert_base_cased_dfc_mul.pt'))
