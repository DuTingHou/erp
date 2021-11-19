import torch
from torch import nn
from torch.nn import functional as F


class Loss_CasRelS(nn.Module):
    def __init__(self, a, b):
        super(Loss_CasRelS, self).__init__()
        self.a = a
        self.b = b
        self.pos_weight = torch.tensor(a)

    def forward(self,
                pred_sub_heads,
                pred_sub_tail,
                pred_relations,
                pred_obj_head,
                pred_obj_tail,
                label_sub_heads,
                label_relations,
                label_sub_tail,
                label_obj_head,
                label_obj_tail):
        sub_heads_loss = F.binary_cross_entropy(pred_sub_heads, label_sub_heads)
        sub_relations_loss = F.binary_cross_entropy(pred_relations, label_relations)
        sub_tail_loss = F.cross_entropy(pred_sub_tail, label_sub_tail)
        obj_head_loss = F.cross_entropy(pred_obj_head, label_obj_head)
        obj_tail_loss = F.cross_entropy(pred_obj_tail, label_obj_tail)
        return sub_heads_loss + sub_relations_loss + sub_tail_loss + obj_head_loss + obj_tail_loss
