import os
from collections import namedtuple


def load_mapping(map_file):
    dom2org = []
    with open(map_file, 'r', encoding='utf-8') as f:
        for line in f:
            dstr, org_id = line.strip().split("\t")
            dom2org.append([int(i) for i in org_id.split(',')])
    return dom2org


def load_vocab_dict(vocab_file_name, vocab_max_size=None, start_vocab_count=None):
    with open(vocab_file_name) as f:
        text = [x.strip() for x in f.readlines()]
        if vocab_max_size:
            text = text[:vocab_max_size]
        if start_vocab_count:
            file_content = dict(zip(text, range(0 + start_vocab_count, len(text) + start_vocab_count)))
        else:
            file_content = dict(zip(text, range(0, len(text))))
    return file_content


project_path = os.path.split(os.path.abspath(os.path.realpath(__file__)))[0] + "/../../"
project_path = os.path.abspath(project_path)
DATA_ROOT = os.path.join(project_path, 'data')
EXP_ROOT = os.path.join(project_path, 'output', 'trained_models')
RES_ROOT = os.path.join(project_path, 'output', 'results')
INIT_WEIGHTS_ROOT = os.path.join(project_path, 'init_weights')

# KB_VOCAB = load_vocab_dict(DATA_ROOT + "/ontology/types.txt", 130)
# WIKI_VOCAB = load_vocab_dict(DATA_ROOT + "/ontology/types.txt", 4600)
CROWD_VOCAB = load_vocab_dict(DATA_ROOT + "/ontology/types.txt")
ONTO_VOCAB = load_vocab_dict(DATA_ROOT + '/ontology/onto_ontology.txt')
FIGER_VOCAB = load_vocab_dict(DATA_ROOT + '/ontology/types_figer.txt')

ANS2ID_DICT = {"open": CROWD_VOCAB,
               "figer": FIGER_VOCAB,
               # "kb": KB_VOCAB,
               "onto": ONTO_VOCAB}

ANSWER_NUM_DICT = {"open": len(CROWD_VOCAB),
                   "onto": 89,
                   "figer": 113,
                   # "wiki": 4600,
                   "kb": 130,
                   "gen": 9
                   }

open_id2ans = {v: k for k, v in CROWD_VOCAB.items()}
# wiki_id2ans = {v: k for k, v in WIKI_VOCAB.items()}
# kb_id2ans = {v: k for k, v in KB_VOCAB.items()}
onto_id2ans = {v: k for k, v in ONTO_VOCAB.items()}
figer_id2ans = {v: k for k, v in FIGER_VOCAB.items()}

ID2ANS_DICT = {"open": open_id2ans, "onto": onto_id2ans, "figer": figer_id2ans}
# label_string = namedtuple("label_types", ["head", "wiki", "kb"])
# LABEL = label_string("HEAD", "WIKI", "KB")

VOCAB_DIR_DICT = {"nb": os.path.join(DATA_ROOT, 'ontology/domain/numberbatch'),
                  "filter_5": os.path.join(DATA_ROOT, 'ontology/domain/filter_5nn'),
                  "filter_10": os.path.join(DATA_ROOT, 'ontology/domain/filter_10'),
                  "mirrorbert": os.path.join(DATA_ROOT, 'ontology/domain/mirrorbert'),
                  "property": os.path.join(DATA_ROOT, 'ontology/domain/property'),
                  "glove": os.path.join(DATA_ROOT, 'ontology/domain/glove'),
                  "skipgram": os.path.join(DATA_ROOT, 'ontology/domain/skipgram'),
                  "syngcn": os.path.join(DATA_ROOT, 'ontology/domain/syngcn'),
                  "w2s": os.path.join(DATA_ROOT, 'ontology/domain/word2sense'),
                  "mirrorwic": os.path.join(DATA_ROOT, 'ontology/domain/mirrorwic'),
                  "biencoder_large": os.path.join(DATA_ROOT, 'ontology/domain/biencoder_large'),
                  "biencoder_prop": os.path.join(DATA_ROOT, 'ontology/domain/biencoder_prop'),
                  "cl_cluster_gpt": os.path.join(DATA_ROOT, 'ufet_domain_types_from_clusters/contrastive_bert_large/cnetp_chatgpt100k_vocab'),
                  "cl_cluster": os.path.join(DATA_ROOT, 'ufet_domain_types_from_clusters/contrastive_bert_large/cnetp_vocab'),
                  "ce_cluster_gpt": os.path.join(DATA_ROOT, 'ufet_domain_types_from_clusters/cross_entropy_bert_large/cnetp_chatgpt100k_vocab'),
                  "ce_cluster": os.path.join(DATA_ROOT, 'ufet_domain_types_from_clusters/cross_entropy_bert_large/cnetp_vocab'),
                  "rel_cluster": os.path.join(DATA_ROOT, 'ufet_domain_types_from_clusters/relational_property'),
                  "rel_cluster_0.75_0.9": os.path.join(DATA_ROOT, 'ufet_domain_types_from_clusters/relational_property_filter_thresh_0.75_0.90')}

DATA_DIR_DICT = {"ufet": "crowd", "onto": "ontonotes", "figer": "figer"}