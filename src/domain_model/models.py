import sys
import os
project_path = os.path.split(os.path.abspath(os.path.realpath(__file__)))[0] + "/../../"
sys.path.append(os.path.abspath(project_path))
from src.base_model.models import *


class DomainModel(nn.Module):
    def __init__(self, args, use_gpu=False):
        super(DomainModel, self).__init__()
        self.in_dim = args.in_dim
        self.use_gpu = use_gpu
        self.multitask = args.multitask
        self.ufet_bert = BertMaskModel(args, self.use_gpu, False)
        self.dataset = args.dataset
        self.goal = args.goal

        self.dfc = nn.Linear(self.in_dim, args.label_num, bias=True)
        if args.bert_param is not None and args.dfc_param is not None:
            self.init_weight(args.bert_param, args.dfc_param)

        self.sigmoid_fn = nn.Sigmoid()

        self.loss_func = nn.BCEWithLogitsLoss(reduction='sum')

    def define_loss(self, logits, targets):
        if self.multitask and self.dataset == 'ufet':
            return ufet_multitask_loss(logits, targets)
        return self.loss_func(logits, targets)

    def init_weight(self, bert_param, dfc_param):
        self.ufet_bert.load_state_dict(torch.load(os.path.join(EXP_ROOT, bert_param)))
        print(os.path.normpath('%s/%s' % (INIT_WEIGHTS_ROOT, dfc_param)), os.path.join(INIT_WEIGHTS_ROOT, dfc_param))
        self.dfc.weight = nn.Parameter(torch.load(os.path.join(INIT_WEIGHTS_ROOT, dfc_param)))

    def forward(self, token_ids, input_mask, targets, mask_position=None):
        input_embeds = self.ufet_bert.prompt_encoder(token_ids)
        hidden_bert = \
        self.ufet_bert.bert(inputs_embeds=input_embeds, attention_mask=input_mask).hidden_states[-1]

        mask_position = mask_position.unsqueeze(1).repeat(1, 1, self.in_dim)
        hidden_bert = torch.gather(hidden_bert, dim=1, index=mask_position).squeeze(1)  # bsz * dim
        mask_output = self.ufet_bert.ln(self.ufet_bert.dense(hidden_bert))
        logits = self.dfc(mask_output)   # bsz * type_all

        loss = self.define_loss(logits, targets)

        if self.ufet_bert.mu is not None:
            reg = self.ufet_bert.lamda * self.ufet_bert.lle_reg(self.dfc.weight[:, :ANSWER_NUM_DICT[self.goal]])
            loss += reg
        return logits, loss
