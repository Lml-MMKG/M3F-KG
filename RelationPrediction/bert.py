#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import unicode_literals
from __future__ import division
from __future__ import print_function
import time
import warnings

from transformers import BertTokenizer, BertForSequenceClassification

warnings.filterwarnings("ignore")
import argparse
from torch.optim import AdamW
from utils import *
from model import *

class KGProcessor(object):
    """Processor for knowledge graph data set."""
    def __init__(self,data_dir):
        self.labels = set()
        self.data_dir=data_dir
        self.train_triples =self.get_triples(data_dir+"train")
        self.test_triples = self.get_triples(data_dir+"test")
        self.ent2id=self.get_object2id(data_dir + "ent2id")
        self.rel2id = self.get_object2id(data_dir + "rel2id")
        self.ent2text = self.get_object2text(data_dir + "entity2text.txt")
        self.rel2text= self.get_object2text(data_dir + "relation2text.txt")

    def get_triples(self, file):
        """Gets training triples."""
        triples = self._read_triples(file)
        return triples

    def get_object2id(self,file):
        """Gets all labels (relations) in the knowledge graph."""
        ob2id= self._read_dict1to2(file)
        return ob2id

    def get_object2id1(self, file):
        """Gets all labels (relations) in the knowledge graph."""
        ob2id = self._read_dict2to1(file)
        return ob2id

    def get_object2text(self, file):
        """Gets all labels (relations) in the knowledge graph."""
        ob2text = self._read_dict_text(file)
        return ob2text

    def get_ills(self,file):
        print('loading a ills file...   ' + file)
        ret = []
        with open(file, "r", encoding='utf-8') as f:
            for line in f:
                th = line.strip('\n').split('\t')
                x = []
                for i in range(len(th)):
                    x.append(int(th[i]))
                ret.append(tuple(x))
        return ret


    def _read_triples(cls, file):
        print('loading a triples file...   ' + file)
        lines = set()
        """Reads a tab separated value file."""
        with open(file, "r", encoding="utf-8") as fr:
            for line in fr:
                params = line.strip("\n").split("\t")
                lines.add(tuple([int(x) for x in params]))
        return lines

    def _read_dict1to2(self,file):
        print('loading a file...   ' + file)
        dict = {}
        with open(file, "r", encoding="utf-8") as fr:
            for line in fr:
                params = line.strip("\n").split("\t")
                dict[params[0]] = int(params[1])
        return dict

    def _read_dict2to1(self,file):
        print('loading a file...   ' + file)
        dict = {}
        with open(file, "r", encoding="utf-8") as fr:
            for line in fr:
                params = line.strip("\n").split("\t")
                dict[params[1]] = int(params[0])
        return dict

    def _read_dict_text(self,file):
        print('loading a file...   ' + file)
        dict = {}
        with open(file, "r", encoding="utf-8") as fr:
            for line in fr:
                params = line.strip("\n").split("\t")
                dict[params[0]] = params[1]
        return dict


def train_bert_completion(args, REL_NUM, triples, model, tokenizer, optimizer, ent2text, id2ent, device):
    batch_size = args.train_batch_size
    triples_num = len(triples)
    triple_steps = math.ceil(triples_num / batch_size)
    triples_loss = 0
    for step in tqdm(range(triple_steps)):
        start = step * batch_size
        end = start + batch_size
        if end > len(triples):
            end = len(triples)
        batch_triples = list(triples)[start: end]
        h_ids = [x[0] for x in batch_triples]
        r_ids = [x[1] for x in batch_triples]
        t_ids = [x[2] for x in batch_triples]
        lable_ids = torch.tensor(r_ids).to(device)

        input_ids, input_masks, segment_ids = convert_examples_to_features(h_ids, t_ids, args.max_seq_length,
                                                                           ent2text, id2ent, tokenizer)
        input_ids = input_ids.to(device)
        input_masks = input_masks.to(device)
        segment_ids = segment_ids.to(device)
        logits = model(input_ids, input_masks, segment_ids).logits
        loss_fct = CrossEntropyLoss()
        loss = loss_fct(logits.view(-1, REL_NUM), lable_ids.view(-1))
        loss.backward(retain_graph=True)
        triples_loss += loss.item()
        optimizer.step()
        optimizer.zero_grad()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("triples_loss: {:f}s".format(triples_loss))

def test_bert_completion(args,epoch, model, tokenizer, train_triples, test_triples, ent2text,id2ent,device,type):
    all_triples = np.vstack((list(train_triples), list(test_triples)))
    all_triples_str_set = set()
    for triple in all_triples:
        triple_str = str(triple[0]) + "\t" + str(triple[1]) + "\t" + str(triple[2])
        all_triples_str_set.add(triple_str)
    print("***** Running bert completion testing "+type+" *****")
    print("  num test triples = ", len(test_triples))
    print("  batch size = ", args.test_batch_size)
    model.to(device)
    model.eval()
    preds = []
    test_triples = np.array(list(test_triples), dtype=np.int32)
    for si in tqdm(np.arange(0, test_triples.shape[0], args.test_batch_size)):
        batch_triples = torch.tensor(test_triples[si:si + args.test_batch_size], dtype=torch.long)
        h_ids = batch_triples[:, 0]
        t_ids = batch_triples[:, 2]
        input_ids, input_masks, segment_ids = convert_examples_to_features(h_ids, t_ids, args.max_seq_length,
                                                                           ent2text, id2ent, tokenizer)
        input_ids = input_ids.to(device)
        input_masks = input_masks.to(device)
        segment_ids = segment_ids.to(device)
        with torch.no_grad():
            logits = model(input_ids, input_masks, segment_ids).logits
        if len(preds) == 0:
            preds.append(logits.detach().cpu().numpy())
        else:
            preds[0] = np.append(
                preds[0], logits.detach().cpu().numpy(), axis=0)
    preds = preds[0]
    all_label_ids = test_triples[:, 1]
    ranks = []
    filter_ranks = []
    hits = []
    hits_filter = []
    for i in range(10):
        hits.append([])
        hits_filter.append([])
    for i, pred in enumerate(preds):
        rel_values = torch.tensor(pred)
        _, argsort1 = torch.sort(rel_values, descending=True)
        argsort1 = argsort1.cpu().numpy()
        rank = np.where(argsort1 == all_label_ids[i])[0][0]
        ranks.append(rank + 1)
        test_triple = test_triples[i]
        filter_rank = rank
        for tmp_label_id in argsort1[:rank]:
            tmp_triple = [test_triple[0], tmp_label_id, test_triple[2]]
            tmp_triple_str = str(tmp_triple[0]) + "\t" + str(tmp_triple[1]) + "\t" + str(tmp_triple[2])
            if tmp_triple_str in all_triples_str_set:
                filter_rank -= 1
        filter_ranks.append(filter_rank + 1)
        for hits_level in range(10):
            if rank <= hits_level:
                hits[hits_level].append(1.0)
            else:
                hits[hits_level].append(0.0)

            if filter_rank <= hits_level:
                hits_filter[hits_level].append(1.0)
            else:
                hits_filter[hits_level].append(0.0)
    num = len(filter_ranks)
    sum = 0
    for r in filter_ranks:
        sum += 1 / r
    MRR = sum / num
    preds = np.argmax(preds, axis=1)
    result = compute_metrics(preds, all_label_ids)
    result["Raw mean rank: "] = np.mean(ranks)
    result["Filtered mean rank: "] = np.mean(filter_ranks)
    result["Raw Hits @1"] = np.mean(hits[0])
    result["hits_filter Hits @1"] = np.mean(hits_filter[0])
    result["Raw Hits @3"] = np.mean(hits[2])
    result["hits_filter Hits @3"] = np.mean(hits_filter[2])
    result["Raw Hits @10"] = np.mean(hits[9])
    result["hits_filter Hits @10"] = np.mean(hits_filter[9])
    result["mrr"] = MRR

    save_path=args.test_path+"completion/"
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    output_eval_file = os.path.join(save_path, str(epoch) + "_results_"+type+".txt")

    with open(output_eval_file, "w") as writer:
        print("***** Test results *****")
        for key in result.keys():
            print("  %s = %s",key, str(result[key]))
            writer.write("%s = %s\n" % (key, str(result[key])))

    if torch.cuda.is_available():
        torch.cuda.empty_cache()



def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir",
                        default='../data/WN18/',
                        type=str,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--bert_model", default='bert-base-cased', type=str,
                        help="Bert pre-trained model selected in the list: bert-base-uncased, "
                             "bert-large-uncased, bert-base-cased, bert-large-cased, bert-base-multilingual-uncased, "
                             "bert-base-multilingual-cased, bert-base-chinese.")
    parser.add_argument("--save_path",
                        default='../save_model/WN18/bert/',
                        type=str,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--test_path",
                        default='../results/WN18/bert/',
                        type=str,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--max_seq_length",
                        default=128,
                        type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument("--epochs",
                        default=20,
                        type=int,
                        help="epochs")
    parser.add_argument("--train_batch_size",
                        default=64,
                        type=int,
                        help="Total batch size for training.")
    parser.add_argument("--test_batch_size",
                        default=256,
                        type=int,
                        help="Total batch size for eval.")
    parser.add_argument("--lr",
                        default=1e-5,
                        type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument('--seed',
                        type=int,
                        default=2024,
                        help="random seed for initialization")
    parser.add_argument("--epsilon",
                        default=0.9,
                        type=float,
                        help="epsilon")
    parser.add_argument("--nums_threads",
                        default=10,
                        type=int,
                        help="the number of threads")
    parser.add_argument("--nums_neg",
                        default=10,
                        type=int,
                        help="nums_neg")
    parser.add_argument("--lambda_1",
                        default=0.01,
                        type=float,
                        help="lambda_1")
    parser.add_argument("--lambda_2",
                        default=2.0,
                        type=float,
                        help="lambda_2")
    parser.add_argument("--mu_1",
                        default=0.2,
                        type=float,
                        help="mu_1")

    args = parser.parse_args()
    fixed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device: ", device)
    if os.path.exists(args.save_path) and os.listdir(args.save_path):
        raise ValueError("Output directory ({}) already exists and is not empty.".format(args.save_path))

    processor = KGProcessor(args.data_dir)
    rel2id = list(processor.rel2id.keys())
    REL_NUM = len(rel2id)
    ent2id = processor.ent2id
    ENT_NUM = len(ent2id)
    ent2text = processor.ent2text
    id2ent = {value: key for key, value in ent2id.items()}

    processor = KGProcessor(args.data_dir)
    rel2id = list(processor.rel2id.keys())
    REL_NUM = len(rel2id)
    ent2id = processor.ent2id
    id2ent = {value: key for key, value in ent2id.items()}
    ENT_NUM = len(ent2id)
    ent2text = processor.ent2text
    ent2id = processor.ent2id
    entids_list = {value: key for key, value in ent2id.items()}
    entids_list = list(entids_list.keys())
    train_triples = processor.train_triples
    test_triples = processor.test_triples
    print("-----dataset summary-----")
    print("dataset:\t", args.data_dir)
    print("entity num:\t", ENT_NUM)
    print("relation num:\t", REL_NUM)
    print("-------------------------")




    tokenizer = BertTokenizer.from_pretrained("../bert-base-multilingual-cased")
    bert_model = BertForSequenceClassification.from_pretrained("../bert-base-multilingual-cased", num_labels=REL_NUM)
    bert_model.to(device)
    param_optimizer = list(bert_model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    bert_optimizer = AdamW(optimizer_grouped_parameters, lr=args.lr)

    t_total = time.time()
    print("[start training]")
    print("  num train tripless = ", len(train_triples))
    print("  batch size = ", args.train_batch_size)

    for epoch in tqdm(range(args.epochs)):
        bert_model.train()
        train_bert_completion(args, REL_NUM, train_triples, bert_model, tokenizer, bert_optimizer, ent2text,
                                     id2ent,
                                     device)
        test_bert_completion(args, epoch, bert_model, tokenizer, train_triples, test_triples, ent2text, id2ent, device,
                             "WN18")

        if (epoch + 1) % 5 == 0:
            save_bert(bert_model, epoch + 1, args.save_path)

    print("[optimization finished!]")
    print("[total time elapsed: {:.4f} s]".format(time.time() - t_total))


if __name__ == "__main__":
    main()
