from torch.utils.data import Dataset
from transformers import BertTokenizer
import json
from tqdm import tqdm
from torch.nn.utils.rnn import pad_sequence
from collections import defaultdict
import torch
import numpy as np
import random


def find_head_idx(source, target):
    target_len = len(target)
    for i in range(len(source)):
        if source[i: i + target_len] == target:
            return i
    return -1


class dataset_CasRelS(Dataset):
    def __init__(self, sentence_max_len=128, bert_name_or_path="bert-base-chinese",
                 infile="train.json", rel2id_file="rel2id.json"):
        self.sentence_max_len = sentence_max_len
        self.bert_name_or_path = bert_name_or_path
        self.infile = infile
        self.tokenizer = BertTokenizer.from_pretrained(self.bert_name_or_path)
        self.rel2id_file = rel2id_file
        self.rel2id, self.id2rel = self.get_rel_info()
        self.datas = self.get_datas()

    def get_rel_info(self):
        content = open(self.rel2id_file).read()
        info = json.loads(content)
        return info[1], info[0]

    def get_datas(self):
        datas = []
        for line in tqdm(open(self.infile)):
            info = json.loads(line)
            token_info = self.tokenizer(info["sentText"],
                                        max_length=self.sentence_max_len,
                                        truncation=True)
            tokens_raw, masks = token_info["input_ids"], token_info["attention_mask"]
            tokens = torch.tensor(tokens_raw, dtype=torch.long)
            masks = torch.tensor(masks, dtype=torch.bool)
            tokens_len = len(tokens)
            sub_heads = torch.zeros(tokens_len)
            relations = torch.zeros(len(self.rel2id))
            selected_sub_head, selected_sub_tail = torch.zeros(tokens_len), torch.zeros(tokens_len)
            selected_relation = torch.zeros(len(self.rel2id))
            selected_obj_head, selected_obj_tail = torch.zeros(tokens_len), torch.zeros(tokens_len)
            s2ro = defaultdict(list)
            for spo in info['relationMentions']:
                triple = (self.tokenizer(spo['em1Text'], add_special_tokens=False)['input_ids'],
                          self.rel2id[spo['label']],
                          self.tokenizer(spo['em2Text'], add_special_tokens=False)['input_ids'])
                sub_head_idx = find_head_idx(tokens_raw, triple[0])
                obj_head_idx = find_head_idx(tokens_raw, triple[2])
                sub_tail_idx = sub_head_idx + len(triple[0]) - 1
                obj_tail_idx = obj_head_idx + len(triple[2]) - 1
                s2ro[(sub_head_idx, sub_tail_idx)].append((obj_head_idx, obj_tail_idx, triple[1]))
            if s2ro:
                for sub_head_idx, sub_tail_idx in s2ro:
                    sub_heads[sub_head_idx] = 1
                selected_sub = random.choice(list(s2ro.keys()))
                for _, _, r in s2ro[selected_sub]:
                    relations[r] = 1
                selected_sub_head[selected_sub[0]] = 1
                selected_sub_tail[selected_sub[1]] = 1
                selected_triple = random.choice(list(s2ro[selected_sub]))
                selected_relation[selected_triple[2]] = 1
                selected_obj_head[selected_triple[0]] = 1
                selected_obj_tail[selected_triple[1]] = 1
            data_info = {
                "text": info["sentText"],
                "spo_list": info['relationMentions'],
                "input_ids": tokens,
                "attention_mask": masks,
                "sub_heads": sub_heads,
                "relations": relations,
                "selected_sub_head": selected_sub_head,
                "selected_sub_tail": selected_sub_tail,
                "selected_relation": selected_relation,
                "selected_obj_head": selected_obj_head,
                "selected_obj_tail": selected_obj_tail
            }
            datas.append(data_info)
        np.random.shuffle(datas)
        return datas

    def __len__(self):
        return len(self.datas)

    def __getitem__(self, idx):
        data = self.datas[idx]
        return data

    def collate_data(self, batch_data):
        bs_data = {}
        need_padding = ["input_ids", "attention_mask", "sub_heads", "relations", \
                        "selected_sub_head", "selected_sub_tail", "selected_relation", \
                        "selected_obj_head", "selected_obj_tail"]
        for key in batch_data[0]:
            if key not in need_padding:
                bs_data[key] = [item[key] for item in batch_data]
            else:
                bs_data[key] = pad_sequence([item[key] for item in batch_data], batch_first=True)
        return bs_data