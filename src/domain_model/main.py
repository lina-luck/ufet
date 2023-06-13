import sys
import os
project_path = os.path.split(os.path.abspath(os.path.realpath(__file__)))[0] + "/../../"
sys.path.append(os.path.abspath(project_path))
from src.domain_model.train_test import *
import argparse
from src.common.utils import *

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-model_id", help="Identifier for model", default=None)
    parser.add_argument("-dataset", help="dataset", default="ufet")
    parser.add_argument("-goal", help="dataset, open, figer or ontonotes", default="open", choices=["open", "figer", "onto"])
    parser.add_argument("-d_goal", help="type vocab", default=None)
    parser.add_argument("-dfn_postfix", help="postfix of domain type file name", default=None)
    parser.add_argument("-cn", help="use concept neighbor or not", default=False, action="store_true")
    parser.add_argument("-bert_version", help="pretrained model name and version", default="bert-base-cased")
    parser.add_argument('-train', help="Train file", default='crowd/train.json')
    parser.add_argument('-valid', help="Valid file", default='crowd/dev.json')
    parser.add_argument('-test', help="Test file", default='crowd/test.json')
    parser.add_argument('-train_batch_size', help="training batch size", type=int, default=16)
    parser.add_argument('-test_batch_size', help="test batch size", type=int, default=16)
    parser.add_argument('-prompt_num', help="additional prompt number", default=3, type=int)
    parser.add_argument('-in_dim', help="embedding dimension", default=768, type=int)
    parser.add_argument('-multitask', help="whether to use multi-task loss", action="store_true", default=False)
    parser.add_argument('-dfc_param', type=str, default=None)
    parser.add_argument('-bert_param', type=str, default='bert_base_cased_4.pt')  # 'bert_base_cased.pt'
    parser.add_argument("-num_epoch", help="The number of epoch", default=500, type=int)
    parser.add_argument("-lr", help="learning rate", default=2e-5, type=float)
    parser.add_argument("-tune_all", help="finetune all param of model or classification layer only",
                        action="store_true", default=False)
    parser.add_argument("-do_train", help="whether to train the model", default=False, action="store_true")
    parser.add_argument("-optim_th", help="whether to optimize th", default=False, action="store_true")
    # lle regularization
    parser.add_argument("-reg", help="weight(mu) file name if use lle regularization",
                        default=None )# "lle_weights/crowd/biencoder_large_3_512.pt")  # "lle_weights/crowd/numberbatch.pt"
    parser.add_argument("-lamda", help="hyperparameter of lle reg, "
                                       "if regularization is not used, lamda is not used, too",
                        type=float, default=1e-1)

    args = parser.parse_args()

    log_file_path = init_logging_path("log", 'domain')
    logging.basicConfig(filename=log_file_path,
                        level=10,
                        format='%(asctime)s %(levelname)s %(message)s',
                        datefmt='%m/%d/%Y %I:%M:%S %p',
                        )
    import time
    s = time.time()
    run_domain_model(args)
    print(time.time() - s)