import os, sys

import torch

sys.path.append(os.getcwd())
from erp.utils.dataset_CasRelS import dataset_CasRelS
from erp.model.CasRelS import CasRelS
from erp.loss.Loss_CasRelS import Loss_CasRelS
from torch.utils.data import DataLoader
from torch.optim import Adam, SGD
from tqdm import tqdm
from example.test_metric import demo as metric_demo


def demo():
    data_dir = "./data/oth"
    bert_name_or_path = "./source/temp"
    bert_name_or_path = "./source/bert-base-chinese"
    loss = Loss_CasRelS(1.0, 1.5)
    dataset = dataset_CasRelS(infile="{}/train.json".format(data_dir), bert_name_or_path=bert_name_or_path,
                              rel2id_file="{}/rel2id.json".format(data_dir))
    erp = CasRelS(bert_name_or_path=bert_name_or_path, bert_hidden_dim=768, rel_cnt=len(dataset.rel2id)).cuda()
    opti = Adam(erp.parameters(), lr=1e-5)
    for e in range(0, 15):
        if e > 0:
            dataset.datas = dataset.get_datas()
        dl = DataLoader(dataset=dataset, batch_size=32,
                        shuffle=False, num_workers=2,
                        collate_fn=dataset.collate_data
                        )
        pbar = tqdm(dl)
        for batch in pbar:
            batch_ids, batch_atten_mask, sub_heads, b_relations, \
            selected_sub_head, selected_sub_tail, selected_obj_head, selected_obj_tail, \
            selected_relation = batch["input_ids"].cuda(), batch["attention_mask"].cuda(), \
                                batch["sub_heads"].cuda(), batch["relations"].cuda(), \
                                batch["selected_sub_head"].cuda(), batch["selected_sub_tail"].cuda(), \
                                batch["selected_obj_head"].cuda(), batch["selected_obj_tail"].cuda(), \
                                batch["selected_relation"].cuda()

            pred_sub_heads, pred_sub_tail, pred_relations, pred_obj_head, pred_obj_tail = \
                erp(batch_ids, batch_atten_mask, selected_sub_head, selected_obj_head, selected_relation)

            opti.zero_grad()
            bloss = loss(pred_sub_heads, pred_sub_tail, pred_relations, pred_obj_head, pred_obj_tail, \
                         sub_heads, selected_sub_tail, b_relations, selected_obj_head, selected_obj_tail)
            bloss.backward()
            opti.step()
            loss_str = round(loss.detach().cpu().numpy().tolist(), 5)
            pbar.set_description("loss:{}".format(loss_str))
        model_dir = "models"
        torch.save(erp, "{}/{}_{}".format("models", e, "erp.bin"))
        metric_demo(e, bert_name_or_path, data_dir, thred1=0.5, thred2=0.5, model_dir=model_dir)


if __name__ == '__main__':
    demo()
