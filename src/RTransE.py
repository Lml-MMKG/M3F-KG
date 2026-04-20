#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import unicode_literals
from __future__ import division
from __future__ import print_function

import warnings

from torch.optim import AdamW
from tqdm import tqdm
import os
import pickle
import scipy.sparse as sp
import time
import math
import numpy as np
import torch
import random
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
warnings.filterwarnings("ignore")
import argparse
from torch.optim import AdamW
from utils import *
from model import *

def train_alignment(epoch,train_ill,criterion_gcn,bsize,optimizer,device,entity_emb,input_idx):
    loss_sum_all=0
    train_ill = np.array(train_ill)
    np.random.shuffle(train_ill)
    for si in np.arange(0, train_ill.shape[0], bsize):
        emb = entity_emb(input_idx)
        loss_transe= criterion_gcn(emb, train_ill[si:si + bsize], [], device=device)
        loss_transe.backward()
        loss_sum_all = loss_sum_all + loss_transe.item()
        optimizer.step()
        optimizer.zero_grad()
    print("[epoch {:d}] loss_all: {:f}".format(epoch, loss_sum_all))

def train_transe_completion(ent_embeddings,rel_embeddings,linear,triples,ent_list, nbours,batch_size,nums_neg,lambda_1,lambda_2,mu_1,optimizer,ent_idx,rel_idx,device):
    ent_embedding = F.normalize(ent_embeddings(ent_idx), p=2, dim=1)
    rel_embedding = F.normalize(rel_embeddings(rel_idx), p=2, dim=1)
    triples_loss = 0
    triples_num = len(triples)
    triple_steps = math.ceil(triples_num / batch_size)
    for step in tqdm(range(triple_steps)):
        start = step * batch_size
        end = start + batch_size
        if end > len(triples):
            end = len(triples)
        batch_triples =list(triples)[start: end]
        pos_hs=[x[0] for x in batch_triples]
        pos_rs=[x[1] for x in batch_triples]
        pos_ts=[x[2] for x in batch_triples]
        lable_ids = torch.tensor(pos_rs).to(device)
        batch_neg = generate_batch_via_neighbour(batch_triples, triples, ent_list, nbours, multi=nums_neg)
        neg_hs = [x[0] for x in batch_neg]
        neg_rs=[x[1] for x in batch_neg]
        neg_ts=[x[2] for x in batch_neg]
        phs = ent_embedding[pos_hs]
        prs = rel_embedding[pos_rs]
        pts = ent_embedding[pos_ts]
        nhs = ent_embedding[neg_hs]
        nrs = rel_embedding[neg_rs]
        nts = ent_embedding[neg_ts]
        pos_loss, neg_loss = generate_loss(phs, prs, pts, nhs, nrs, nts, lambda_1, lambda_2, mu_1, device)
        logits = pts - phs
        joint_emb = F.normalize(logits, p=2, dim=1)
        joint_emb = linear(joint_emb)
        joint_emb = F.softmax(joint_emb)
        loss_fct = CrossEntropyLoss()
        loss = loss_fct(joint_emb, lable_ids) + pos_loss + neg_loss
        loss.backward(retain_graph=True)
        optimizer.step()
        triples_loss += loss.item() / (2 * len(batch_neg))
        optimizer.zero_grad()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("triples_loss: {:f}s".format(triples_loss))
def test_transe_completion(args, epoch, ent_embeddings, rel_embeddings, linear, ent_idx, rel_idx, train_triples, test_triples,
                   device,type):
        all_triples = np.vstack((list(train_triples), list(test_triples)))
        all_triples_str_set = set()
        for triple in all_triples:
            triple_str = str(triple[0]) + "\t" + str(triple[1]) + "\t" + str(triple[2])
            all_triples_str_set.add(triple_str)

        print("***** Running completion testing *****")
        print("  num test triples = ", len(test_triples))
        print("  batch size = ", args.test_batch_size)
        ent_embeddings.to(device)
        ent_embeddings.eval()
        rel_embeddings.to(device)
        rel_embeddings.eval()
        linear.to(device)
        linear.eval()
        ent_embedding = F.normalize(ent_embeddings(ent_idx), p=2, dim=1)
        rel_embedding = F.normalize(rel_embeddings(rel_idx), p=2, dim=1)
        preds = []
        test_triples = np.array(list(test_triples), dtype=np.int32)
        for si in tqdm(np.arange(0, test_triples.shape[0], args.test_batch_size)):
            batch_triples = torch.tensor(test_triples[si:si + args.test_batch_size], dtype=torch.long)
            h_ids = batch_triples[:, 0]
            r_ids = batch_triples[:, 1]
            t_ids = batch_triples[:, 2]
            phs = ent_embedding[h_ids]
            pts = ent_embedding[t_ids]
            with torch.no_grad():
                r = pts - phs
                joint_emb = F.normalize(r, p=2, dim=1)
                joint_emb = linear(joint_emb)
                logits = F.softmax(joint_emb)
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

        save_path = args.test_path + "completion/"
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        output_eval_file = os.path.join(save_path, str(epoch) + "_results_" + type + ".txt")

        with open(output_eval_file, "w") as writer:
            print("***** Test results *****")
            for key in result.keys():
                print("  %s = %s", key, str(result[key]))
                writer.write("%s = %s\n" % (key, str(result[key])))

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir",
                        default='../data/dbp15k/zh_en/',
                        type=str,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--img_path",
                        default='../data/dbp15k/pkl/zh_en_GA_id_img_feature_dict.pkl',
                        type=str,
                        help="The output directory where the model predictions and checkpoints will be written.")

    parser.add_argument("--bert_model", default='bert-base-cased', type=str,
                        help="Bert pre-trained model selected in the list: bert-base-uncased, "
                             "bert-large-uncased, bert-base-cased, bert-large-cased, bert-base-multilingual-uncased, "
                             "bert-base-multilingual-cased, bert-base-chinese.")
    parser.add_argument("--save_path",
                        default='../save_model/dbp15k/zh_en/RTransE/',
                        type=str,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--test_path",
                        default='../results/dbp15k/zh_en/RTransE/',
                        type=str,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--max_seq_length",
                        default=128,
                        type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument("--epochs",
                        default=50,
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
                        default=0.0005,
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

    print("-----dataset summary-----")
    print("dataset:\t", args.data_dir)
    print("entity num:\t", ENT_NUM)
    print("relation num:\t", REL_NUM)
    print("-------------------------")

    ent2id=processor.ent2id
    entids_list= {value: key for key, value in ent2id.items()}
    entids_list=list(entids_list.keys())
    train_triples1 = processor.train_triples1
    test_triples1 = processor.test_triples1
    train_triples2 = processor.train_triples2
    test_triples2 = processor.test_triples2
    train_cross = processor.train_cross_triples
    test_cross = processor.test_cross_triples
    train_ills_triples = processor.train_ills_triples
    test_ills_triples = processor.test_ills_triples
    train_triples = train_triples1 | train_triples2 | train_cross | train_ills_triples
    test_triples = test_triples1 | test_triples2 | test_cross | test_ills_triples

    train_ill = processor.train_ills
    alignment_loss = NCA_loss(alpha=5, beta=10, ep=0.0)

    dim=2048
    ent_embeddings = nn.Embedding(ENT_NUM, dim)
    entity_emb = load_img(ENT_NUM, args.img_path)
    ent_embeddings.weight.data.copy_(torch.tensor(entity_emb))

    ent_embeddings.requires_grad = True
    ent_embeddings = ent_embeddings.to(device)
    rel_embeddings = nn.Embedding(REL_NUM,dim)
    nn.init.normal_(rel_embeddings.weight, mean=0, std=1.0 / REL_NUM)

    rel_embeddings.requires_grad = True
    rel_embeddings = rel_embeddings.to(device)
    linear =  nn.Linear(dim, REL_NUM, True).to(device)
    ent_idx = torch.LongTensor(np.arange(ENT_NUM)).to(device)
    rel_idx = torch.LongTensor(np.arange(REL_NUM)).to(device)

    optimizer_grouped_parameters = [
        {"params":
             [ent_embeddings.weight] +
             [rel_embeddings.weight] +
             list(linear.parameters())
         }
    ]

    transe_optimizer = AdamW(optimizer_grouped_parameters, lr=args.lr)

    if args.epsilon > 0:
        trunc_ent_num = int(len(entids_list) * (1 - args.epsilon))
        assert trunc_ent_num > 0
        print("trunc ent num:", trunc_ent_num)
    else:
        trunc_ent_num = 0
        assert not trunc_ent_num > 0

    t_total = time.time()
    print("[start training]")
    print("  num train tripless = ", len(train_triples))
    print("  batch size = ", args.train_batch_size)
    for epoch in tqdm(range(args.epochs)):
        ent_embeddings.train()
        rel_embeddings.train()
        linear.train()

        train_alignment(epoch, train_ill, alignment_loss, args.train_batch_size, transe_optimizer, device,
                        ent_embeddings,
                        ent_idx)

        ent_embedding = F.normalize(ent_embeddings(ent_idx), p=2, dim=1)

        nbours = generate_neighbours_multi_embed(ent_embedding[entids_list].detach(),
                                                 entids_list, trunc_ent_num, args.nums_threads)

        train_transe_completion(ent_embeddings, rel_embeddings,linear,train_triples,entids_list,nbours,
                                    args.train_batch_size, args.nums_neg, args.lambda_1,args.lambda_2, args.mu_1, transe_optimizer,ent_idx,rel_idx, device)

        if nbours is not None:
            del nbours
        del ent_embedding

        test_transe_completion(args, epoch, ent_embeddings, rel_embeddings, linear, ent_idx, rel_idx, train_triples,
                               test_triples1, device, "triples1")
        test_transe_completion(args, epoch, ent_embeddings, rel_embeddings, linear, ent_idx, rel_idx, train_triples,
                               test_triples2, device, "triples2")

        test_transe_completion(args, epoch, ent_embeddings, rel_embeddings, linear, ent_idx, rel_idx, train_triples,
                               test_ills_triples, device, "ills")
        if (epoch+1)%5==0:
            test_transe_completion(args, epoch, ent_embeddings, rel_embeddings, linear, ent_idx, rel_idx, train_triples,
                                   test_cross, device, "cross")

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        if (epoch + 1) % 10 == 0:
            save_transe(ent_embeddings, rel_embeddings,linear, epoch+1, args.save_path)

    print("[optimization finished!]")
    print("[total time elapsed: {:.4f} s]".format(time.time() - t_total))

if __name__ == "__main__":
    main()
