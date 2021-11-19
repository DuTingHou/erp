import json
import os, sys

sys.path.append(os.getcwd())

import torch
from erp.utils.metricS import MetricS
from erp.model.CasRelS import CasRelS
from tqdm import tqdm


def demo(e, bert_name_or_path, data_dir, thred1, thred2, model_dir="./models"):
    preds_all, preds_right, all_cnt = 0.0, 0.0, 0.0
    erp = torch.load("{}/{}_{}".format(model_dir, e, "erp.bin")).cuda()
    metric = MetricS(erp, bert_name_or_path, "{}/rel2id.json".format(data_dir), thred=thred1, thred2=thred2)
    tok = metric.tokenizer
    with torch.no_grad():
        cnt = 0
        for line in tqdm(open("{}/test.json".format(data_dir))):
            if cnt > 1000:
                break
            cnt += 1

            info = json.loads(line)
            text = info["sentText"]
            triples = info["relationMentions"]
            uncase_triple = []
            for triple in triples:
                emb1, emb2 = triple["em1Text"], triple["em2Text"]
                emb1 = "".join(metric.tokenizer.decode(metric.tokenizer.encode(emb1)[1:-1]).split())
                emb2 = "".join(metric.tokenizer.decode(metric.tokenizer.encode(emb2)[1:-1]).split())
                rel = triple["label"]
                uncase_triple.append((emb1.lower(), rel, emb2.lower()))
            result = metric.predict(text)
            result_new = []
            for item in result:
                pred_label = item[1]
                if item[1] == "成立日期2":
                    pred_label = "成立日期"
                item_new = (item[0].replace(" ", ""), pred_label, item[2].replace(" ", ""))
                result_new.append(item_new)
            result = result_new
            comm_set = set(result).intersection(uncase_triple)
            preds_all += len(result)
            all_cnt += len(uncase_triple)
            preds_right += len(comm_set)
            # return
        precise = round(preds_right / (preds_all + 0.0001), 3)
        recall = round(preds_right / (all_cnt + 0.001), 3)
        f1_score = round(2 * precise * recall / (precise + recall + 0.001), 3)
        print(preds_right, preds_all, all_cnt)
        print("precise:{}\trecall:{}\tf1_score:{}".format(precise, recall, f1_score))


if __name__ == '__main__':
    data_dir = "./data/oth"
    bert_name_or_path = "./source/bert-base-chinese"
    e = int(sys.argv[3])
    a, b = float(sys.argv[1]), float(sys.argv[2])
    demo(e, bert_name_or_path, data_dir, a, b)
