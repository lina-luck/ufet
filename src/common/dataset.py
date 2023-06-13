import torch
import json
import pyarrow as pa
from torch.utils.data import Dataset
import random
import numpy as np


class UFET(Dataset):
    def __init__(self, file_path, type2id):
        super(UFET, self).__init__()
        self.type2id = type2id
        # load fet data from .json file
        # use pyarrow to avoid memory leak
        with open(file_path, 'r') as f:
            self.full_data = pa.array([line.strip() for line in f.readlines()])

        if len(self.full_data) >= 2000000:
            select_idx = random.sample(list(range(len(self.full_data))), 100000)
            self.data = np.array(self.full_data)[select_idx]
        else:
            self.data = self.full_data
        self.len = len(self.data)

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        sample = json.loads(str(self.data[idx]))

        left_context = " ".join(sample['left_context_token'])
        mention = sample['mention_span']
        right_context = " ".join(sample['right_context_token'])
        sentence = f'{left_context} {mention} {right_context} [PROMPT] {mention} [PROMPT] [PROMPT] '

        # get label
        label = torch.LongTensor([self.type2id[foo] for foo in sample['y_str'] if foo in self.type2id])
        label = torch.zeros(len(self.type2id)).scatter_(0, label, 1)
        return idx, sentence, label

    @staticmethod
    def collate_fn(train_data, tokenizer):
        mask_token = tokenizer.mask_token
        bz = len(train_data)
        idx = torch.LongTensor([data[0] for data in train_data])
        sentences = [data[1] + mask_token for data in train_data]
        samples = tokenizer(sentences, truncation=True, padding=True, return_tensors='pt')
        input_ids, attention_mask = samples['input_ids'], samples['attention_mask']
        mask_position = torch.nonzero(input_ids == tokenizer.mask_token_id, as_tuple=False)[:, 1].reshape(bz, 1)
        labels = torch.stack([data[2] for data in train_data], dim=0)
        return idx, input_ids, attention_mask, mask_position, labels

