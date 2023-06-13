import sys
import os
project_path = os.path.split(os.path.abspath(os.path.realpath(__file__)))[0] + "/../../"
sys.path.append(os.path.abspath(project_path))
from src.base_model.train_test_bert import *
import argparse
from src.common.utils import *


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-model_id", help="Identifier for model", default="bert_base_cased")
    parser.add_argument("-dataset", help="dataset", default="ufet")
    parser.add_argument("-goal", help="Limiting vocab to smaller vocabs (either ontonote or figer)", default="open",
                        choices=["open", "onto", "wiki", 'kb'])
    parser.add_argument("-bert_version", help="pretrained model name and version", default="bert-base-cased")
    parser.add_argument('-train', help="Train file", default='crowd/train.json')
    parser.add_argument('-valid', help="Valid file", default='crowd/dev.json')
    parser.add_argument('-test', help="Test file", default='crowd/test.json')
    parser.add_argument('-train_batch_size', help="training batch size", type=int, default=32)
    parser.add_argument('-test_batch_size', help="test batch size", type=int, default=32)
    parser.add_argument('-prompt_num', help="additional prompt number", default=3, type=int)
    parser.add_argument('-in_dim', help="embedding dimension", default=768, type=int)
    parser.add_argument('-multitask', help="whether to use multi-task loss", action="store_true", default=True)
    parser.add_argument('-dense_param', type=str, default='bert-base-cased/dense.pth')
    parser.add_argument('-ln_param', type=str, default='bert-base-cased/ln.pth')
    parser.add_argument('-fc_param', type=str, default='bert-base-cased/ultra_fc.pth')
    parser.add_argument("-num_epoch", help="The number of epoch", default=500, type=int)
    parser.add_argument("-lr", help="learning rate", default=2e-5, type=float)
    parser.add_argument("-reg", help="weight(mu) file name if use lle regularization", default=None) # "lle_weights/crowd/bert_mask_large_filter_5nn.pt")
    parser.add_argument("-lamda", help="hyperparameter of lle reg, "
                                       "if regularization is not used, lamda is not used, too",
                        type=float, default=1e-2)

    args = parser.parse_args()

    log_file_path = init_logging_path("log", 'bert')
    logging.basicConfig(filename=log_file_path,
                        level=10,
                        format='%(asctime)s %(levelname)s %(message)s',
                        datefmt='%m/%d/%Y %I:%M:%S %p',
                        )

    fintune_mlm_model(args)
