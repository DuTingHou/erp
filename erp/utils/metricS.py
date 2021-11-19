import torch
from transformers import BertTokenizer
import json


class MetricS(object):
    def __init__(self, model, bert_name_or_path, rel2id_file, thred=0.5, thred2=0.35):
        self.tokenizer = BertTokenizer.from_pretrained(bert_name_or_path)
        self.model = model
        self.thred = thred
        self.thred2 = thred2
        rels = json.loads(open(rel2id_file).read())
        self.rel2id = rels[1]
        self.id2rel = rels[0]

    def predict(self, text):
        tokens = self.tokenizer(text)
        words = self.tokenizer.tokenize(text)
        pred_triples = []
        token_torch = torch.tensor([tokens["input_ids"]]).long().cuda()
        mask_torch = torch.tensor([tokens["attention_mask"]]).long().cuda()
        encoded_text = self.model.get_encoded_text(token_torch, mask_torch)
        sub_heads_pred = self.model.get_sub_heads(encoded_text)
        sub_head_idxs = torch.where(sub_heads_pred[0] > self.thred)[0]
        for sub_head_idx in sub_head_idxs:
            sub_head_idxt = torch.zeros((1, len(tokens["input_ids"])))
            sub_head_idxt[0][sub_head_idx] = 1
            sub_tail_pred = self.model.get_ent_tail(encoded_text, sub_head_idxt.cuda())
            sub_tail_idx = sub_tail_pred.argmax(dim=1).cpu().numpy()[0]
            if sub_tail_idx >= sub_head_idx:
                sub_head = self.model.get_special_id_enc(encoded_text, sub_head_idxt)
                relation = self.model.to_relations(sub_head)
                relation_idxs = torch.where(relation[0] > self.thred2)[0]
                for relation_idx in relation_idxs:
                    relationt = torch.zeros((1, len(self.rel2id))).long()
                    relationt[0][relationt] = 1
                    obj_head = self.model.get_obj_head(encoded_text, sub_head_idxt, relationt.cuda())
                    obj_head_idx = obj_head.argmax(dim=1).cpu().numpy()[0]
                    obj_head_idxt = torch.zeros((1, len(tokens["input_ids"])))
                    obj_head_idxt[0][obj_head_idx] = 1
                    obj_tail = self.model.get_ent_tail(encoded_text, obj_head_idxt.cuda())
                    obj_tail_idx = obj_tail.argmax(dim=1).cpu().numpy()[0]
                    if obj_tail_idx >= obj_head_idx:
                        pred_triples.append(("".join(words[sub_head_idx:sub_tail_idx]),
                                             self.id2rel[relation_idx],
                                             "".join(words[obj_head_idx:obj_tail_idx]))
                                            )
        return pred_triples
