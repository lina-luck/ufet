import torch.nn as nn
import torch
import os
import sys
project_path = os.path.split(os.path.abspath(os.path.realpath(__file__)))[0] + "/../../"
sys.path.append(os.path.abspath(project_path))
from src.common.utils import ufet_multitask_loss
from transformers import AutoModelForMaskedLM
from src.common.constant import *


class BertBase(nn.Module):
    def __init__(self, args, init_weight=False):
        super().__init__()
        self.loss_func = nn.BCEWithLogitsLoss(reduction='sum')
        self.sigmoid_fn = nn.Sigmoid()
        self.in_dim = args.in_dim
        self.type_num = ANSWER_NUM_DICT[args.goal]
        self.multitask = args.multitask
        self.dataset = args.dataset
        self.dense = nn.Linear(self.in_dim, self.in_dim, bias=True)
        self.ln = nn.LayerNorm(self.in_dim, elementwise_affine=True)
        self.fc = nn.Linear(self.in_dim, self.type_num, bias=True)
        if init_weight:
            self.init_weight(os.path.join(INIT_WEIGHTS_ROOT, args.dense_param),
                             os.path.join(INIT_WEIGHTS_ROOT, args.ln_param),
                             os.path.join(INIT_WEIGHTS_ROOT, args.fc_param))
        self.mu = None
        self.lamda = None
        self.indices = None
        if args.reg is not None:
            self.lamda = nn.Parameter(torch.tensor(args.lamda))
            tmp = torch.load(os.path.join(DATA_ROOT, args.reg))
            self.mu = tmp['mu'].unsqueeze(2)
            self.indices = tmp['indices']
            del tmp

    def init_weight(self, dense_param, ln_param, fc_param):
        self.dense.load_state_dict(torch.load(dense_param))
        self.ln.load_state_dict(torch.load(ln_param))
        self.fc.load_state_dict(torch.load(fc_param))

    def define_loss(self, logits, targets):
        if self.multitask and self.dataset == 'ufet':
            return ufet_multitask_loss(logits, targets)
        return self.loss_func(logits, targets)

    def forward(self, token_ids, input_mask, targets, word_indices=None, type_ids=None):
        pass

    def save_weights(self, file):
        type_weights = self.fc.weight.data.cpu()
        torch.save(type_weights, file)

    def lle_reg(self, weights_type):
        self.mu = self.mu.to(weights_type.device)
        self.indices = self.indices.to(weights_type.device)
        weights_nn = (self.mu * weights_type[self.indices]).sum(1)
        return torch.sum((weights_nn - weights_type) ** 2)


class prompt(nn.Module):
    def __init__(self, prompt_length, embedding_dim, bert_embedding, prompt_placeholder_id, unk_id, init_template=None):
        super(prompt, self).__init__()
        self.prompt_length = prompt_length
        self.embedding_dim = embedding_dim
        self.bert_embedding = bert_embedding
        self.prompt_placeholder_id = prompt_placeholder_id
        self.unk_id = unk_id
        self.prompt = nn.Parameter(torch.randn(prompt_length, embedding_dim))
        if init_template is not None:
            self.prompt = nn.Parameter(self.bert_embedding(init_template).clone())

    def forward(self, input_ids):
        bz = input_ids.shape[0]
        raw_embedding = input_ids.clone()
        raw_embedding[raw_embedding == self.prompt_placeholder_id] = self.unk_id
        raw_embedding = self.bert_embedding(raw_embedding)  # (bz, len, embedding_dim)

        prompt_idx = torch.nonzero(input_ids == self.prompt_placeholder_id, as_tuple=False)
        prompt_idx = prompt_idx.reshape(bz, self.prompt_length, -1)[:, :, 1]    # (bz, prompt_len)
        for b in range(bz):
            for i in range(self.prompt_length):
                raw_embedding[b, prompt_idx[b, i], :] = self.prompt[i, :]
        return raw_embedding


class BertMaskModel(BertBase):
    def __init__(self, args, use_gpu=False, init_weight=False):
        super(BertMaskModel, self).__init__(args, init_weight)
        self.goal = args.goal
        self.use_gpu = use_gpu
        self.bert = AutoModelForMaskedLM.from_pretrained(args.bert_version, output_hidden_states=True)
        self.prompt_encoder = prompt(args.prompt_num, args.in_dim, self.bert.get_input_embeddings(),
                                     args.prompt_placeholder_id, args.unk_id)

    def forward(self, token_ids, input_mask, targets, mask_position=None, type_ids=None):
        input_embeds = self.prompt_encoder(token_ids)
        hidden_bert = self.bert(inputs_embeds=input_embeds, attention_mask=input_mask, token_type_ids=type_ids).hidden_states[-1]

        mask_position = mask_position.unsqueeze(1).repeat(1, 1, self.in_dim)
        hidden_bert = torch.gather(hidden_bert, dim=1, index=mask_position).squeeze(1)
        mask_output = self.ln(self.dense(hidden_bert))
        logits = self.fc(mask_output)
        loss = self.define_loss(logits, targets)

        if self.mu is not None:
            reg = self.lamda * self.lle_reg(self.fc.weight)
            loss += reg

        return logits, loss