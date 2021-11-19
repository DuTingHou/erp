import torch.nn as nn
import torch
from transformers import BertModel
import pickle


class CasRelS(nn.Module):
    def __init__(self, bert_name_or_path="bert-base-chinese",
                 bert_hidden_dim=768,
                 rel_cnt=49):
        super(CasRelS, self).__init__()
        self.bert = BertModel.from_pretrained(bert_name_or_path)
        self.hidden_dim = bert_hidden_dim
        self.rel_cnt = rel_cnt
        self.rel2emb = nn.Embedding(self.rel_cnt, self.hidden_dim, max_norm=True)
        self.to_sub_heads = nn.Linear(self.hidden_dim, 1)  # context to sub heads
        self.to_relations = nn.Linear(self.hidden_dim, self.rel_cnt)  # sub_head to relations
        self.to_ent_tail = nn.Linear(self.hidden_dim, 1)  # context head to tail
        self.to_obj_head = nn.Linear(self.hidden_dim, 1)  # context sub_head relation to obj_head

    def get_encoded_text(self, token_ids, mask):
        encoded_text = self.bert(token_ids, attention_mask=mask)[0]
        return encoded_text

    def get_special_id_enc(self, encoded_text, selected_idx):
        special_enc = torch.matmul(selected_idx.unsqueeze(1), encoded_text)
        return special_enc

    def get_sub_heads(self, encoded_text):
        sub_heads = self.to_sub_heads(encoded_text)
        return sub_heads

    def get_ent_tail(self, encoded_text, selected_head):
        sub_head = self.get_special_id_enc(encoded_text, selected_head)
        encoded_text_head = encoded_text + sub_head
        encoded_sub_tail = self.to_ent_tail(encoded_text_head).reshape(-1, encoded_text.size(1))
        encoded_sub_tail = nn.Softmax(encoded_sub_tail, dim=1)
        return encoded_sub_tail

    def get_obj_head(self, encoded_text, selected_sub_head, selected_rel):
        sub_head = self.get_special_id_enc(encoded_text, selected_sub_head)
        encoded_text_head = encoded_text + sub_head
        encoded_text_head_rel = encoded_text_head + self.rel2emb(selected_rel)
        encoded_obj_head = self.to_obj_head(encoded_text_head_rel).reshape(-1, encoded_text.size(1))
        encoded_obj_head = nn.Softmax(encoded_obj_head, dim=1)
        return encoded_obj_head

    def forward(self, token_ids, mask, selected_sub_head, selected_obj_head, selected_rel):
        encoded_text = self.get_encoded_text(token_ids, mask)
        sub_heads = self.get_sub_heads(encoded_text)
        sub_tail = self.get_ent_tail(encoded_text, selected_sub_head)
        obj_head = self.get_obj_head(encoded_text, selected_sub_head, selected_rel)
        obj_tail = self.get_ent_tail(encoded_text, selected_obj_head)
        sub_head = self.get_special_id_enc(encoded_text, selected_sub_head)
        relation = self.to_relations(sub_head)
        return sub_heads, sub_tail, relation, obj_head, obj_tail
