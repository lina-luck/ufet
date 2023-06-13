import torch
from sklearn.cluster import AffinityPropagation
import argparse
import os
import sys
project_path = os.path.split(os.path.abspath(os.path.realpath(__file__)))[0] + "/../../"
sys.path.append(os.path.abspath(project_path))
from src.common.constant import *
from sklearn import metrics
import numpy as np


def ap_clustering(emb):
    '''
    Affinity Propagation clustering
    :param emb: features
    :return: cluster results
    '''
    best_cluster = None
    best_score = -1e10
    for damp in range(5, 10):
        cluster = AffinityPropagation(damping=damp / 10.0, max_iter=1000)
        cluster.fit_predict(emb)
        if len(cluster.cluster_centers_) <= 1:
            print("Number of labels is 1")
            continue
        # evaluation of clustering
        silhouette_score = metrics.silhouette_score(emb, cluster.labels_)  # higher is better
        davies_bouldin_score = metrics.davies_bouldin_score(emb, cluster.labels_)  # lower is better
        metric_i = silhouette_score - davies_bouldin_score  # higher is better
        # print(metric_i)
        if metric_i > best_score:
            best_cluster = cluster.labels_
            best_score = metric_i
    return [best_cluster]


def multi_ap_clustering(emb):
    '''
    Affinity Propagation clustering
    :param emb: features
    :return: cluster results
    '''
    clusters = []
    for damp in range(5, 10):
        cluster = AffinityPropagation(damping=damp / 10.0, max_iter=1000)
        cluster.fit_predict(emb)
        clusters.append(cluster.labels_)
    return clusters


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-type_emb", help="vectors of type name", default="type_embeddings/crowd/biencoder.npy")
    parser.add_argument("-out_file", help="output file name", default="ontology/domain/biencoder/crowd/types_mul.txt")
    parser.add_argument("-clu_type", help="which kind of type to be cluster", default="all",
                        choices=['gen', 'fine', 'ultra', 'all'])
    parser.add_argument("-mul_clu", help="generate one cluster or multiple clusters", default=True, action='store_true')

    config = parser.parse_args()

    # load embedding
    emb = np.load(os.path.join(DATA_ROOT, config.type_emb))

    if config.clu_type == 'all':
        flag = "open"
        new_labels = '\n'.join(ANS2ID_DICT[flag].keys()) + '\n'
        if config.mul_clu:
            best_cluster = multi_ap_clustering(emb)
        else:
            best_cluster = ap_clustering(emb)

        domain_types = []
        for cluster in best_cluster:
            cnt = 0
            for l in np.unique(cluster):
                ind = np.where(cluster == l)[0]
                if len(ind) < 2:
                    continue

                if l == -1:
                    cnt += ind.shape[0]
                else:
                    domain = "#".join([ID2ANS_DICT[flag][i] for i in ind])
                    domain_types.append(domain)
                    # new_labels += domain + '\n'
                    cnt += 1

    domain_types = list(set(domain_types))

    new_labels += '\n'.join(domain_types)

    #  write out
    out_file = os.path.join(DATA_ROOT, config.out_file)
    out_path = os.path.dirname(out_file)
    if not os.path.exists(out_path):
        os.makedirs(out_path)

    with open(os.path.join(DATA_ROOT, config.out_file), 'w', encoding='utf-8') as f:
        f.write(new_labels)



