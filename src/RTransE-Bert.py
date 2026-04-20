#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import unicode_literals
from __future__ import division
from __future__ import print_function

import pickle
import time
import warnings

import torch
from transformers import BertTokenizer, BertForSequenceClassification
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

def train_completion(args,REL_NUM, train_triples, model, tokenizer,linear,optimizer,weight_raw,ent_idx,rel_idx,ent2text,id2ent,ent_embeddings,rel_embeddings,ent_list, nbours, nums_neg, lambda_1, lambda_2,mu_1, device):
    print("***** Running completion training *****")
    print("  num train tripless = ", len(train_triples))
    print("  batch size = ", args.train_batch_size)

    ent_embedding = F.normalize(ent_embeddings(ent_idx), p=2, dim=1)
    rel_embedding = F.normalize(rel_embeddings(rel_idx), p=2, dim=1)

    bert_loss = 0
    transe_loss = 0
    all_loss = 0
    triples_num = len(train_triples)
    triple_steps = math.ceil(triples_num / args.train_batch_size)
    for step in tqdm(range(triple_steps)):
        start = step * args.train_batch_size
        end = start + args.train_batch_size
        if end > len(train_triples):
            end = len(train_triples)
        batch_triples = list(train_triples)[start: end]
        pos_hs = [x[0] for x in batch_triples]
        pos_rs = [x[1] for x in batch_triples]
        pos_ts = [x[2] for x in batch_triples]

        lable_ids = torch.tensor( pos_rs).to(device)
        input_ids, input_masks, segment_ids = convert_examples_to_features(pos_hs,  pos_ts, args.max_seq_length,
                                                                           ent2text, id2ent, tokenizer)
        input_ids = input_ids.to(device)
        input_masks = input_masks.to(device)
        segment_ids = segment_ids.to(device)
        logits = model(input_ids, input_masks, segment_ids).logits
        loss_fct = CrossEntropyLoss()
        bert_loss = loss_fct(logits.view(-1, REL_NUM), lable_ids.view(-1))

        batch_neg = generate_batch_via_neighbour(batch_triples,train_triples,ent_list, nbours, multi=nums_neg)
        neg_hs = [x[0] for x in batch_neg]
        neg_rs = [x[1] for x in batch_neg]
        neg_ts = [x[2] for x in batch_neg]
        phs = ent_embedding[pos_hs]
        prs = rel_embedding[pos_rs]
        pts = ent_embedding[pos_ts]
        nhs = ent_embedding[neg_hs]
        nrs = rel_embedding[neg_rs]
        nts = ent_embedding[neg_ts]

        pos_loss, neg_loss = generate_loss(phs, prs, pts, nhs, nrs, nts, lambda_1, lambda_2, mu_1, device)
        r= pts - phs
        joint_emb =  F.normalize(r, p=2, dim=1)
        joint_emb = linear(joint_emb)
        joint_emb = F.softmax(joint_emb)
        transe_loss =  loss_fct(joint_emb,lable_ids.view(-1)) + pos_loss + neg_loss


        joint_log =0.1 * F.normalize(joint_emb, p=2, dim=1)+0.9 *  F.normalize(logits, p=2, dim=1)
        joint_loss = loss_fct(joint_log, lable_ids.view(-1))

        loss=transe_loss +bert_loss+joint_loss

        loss.backward(retain_graph=True)
        all_loss += loss.item()
        optimizer.step()
        optimizer.zero_grad()


    print("  bert_loss = %f", bert_loss)
    print("  transe_loss = %f", transe_loss)
    print("  all_loss = %f", all_loss)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def test_completion(args,epoch, model, tokenizer, train_triples, test_triples, ent2text,id2ent,ent_embeddings,rel_embeddings,linear, ent_idx,rel_idx,weight_raw,device,type):
    ent_embeddings.to(device)
    ent_embeddings.eval()
    model.to(device)
    model.eval()
    rel_embeddings.to(device)
    rel_embeddings.eval()
    linear.to(device)
    linear.eval()
    ent_embedding = F.normalize(ent_embeddings(ent_idx), p=2, dim=1)
    rel_embedding = F.normalize(rel_embeddings(rel_idx), p=2, dim=1)
    all_triples = np.vstack((list(train_triples), list(test_triples)))
    all_triples_str_set = set()
    for triple in all_triples:
        triple_str = str(triple[0]) + "\t" + str(triple[1]) + "\t" + str(triple[2])
        all_triples_str_set.add(triple_str)
    print("***** Running completion testing *****")
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

        phs = ent_embedding[h_ids]

        pts = ent_embedding[t_ids]


        with torch.no_grad():
            logits_bert = model(input_ids, input_masks, segment_ids).logits
            r = pts - phs
            joint_emb = F.normalize(r, p=2, dim=1)
            joint_emb = linear(joint_emb)
            logits_transe = F.softmax(joint_emb)
            logits =0.1* F.normalize( logits_transe , p=2, dim=1) +0.9* F.normalize(logits_bert,p=2, dim=1)
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
                        default='../save_model/dbp15k/zh_en/RTransE-Bert/',
                        type=str,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--test_path",
                        default='../results/dbp15k/zh_en/RTransE-Bert/',
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
    print("-----dataset summary-----")
    print("dataset:\t", args.data_dir)
    print("entity num:\t", ENT_NUM)
    print("relation num:\t", REL_NUM)
    print("-------------------------")

    ent2id = processor.ent2id
    entids_list= {value: key for key, value in ent2id.items()}
    entids_list= list(entids_list.keys())

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

    dim = 2048
    ent_embeddings = nn.Embedding(ENT_NUM, dim)
    entity_emb = load_img(ENT_NUM,args.img_path )
    ent_embeddings.weight.data.copy_(torch.tensor(entity_emb))

    ent_embeddings.requires_grad = True
    ent_embeddings = ent_embeddings.to(device)
    rel_embeddings = nn.Embedding(REL_NUM, dim)
    nn.init.normal_(rel_embeddings.weight, mean=0, std=1.0 / REL_NUM)

    rel_embeddings.requires_grad = True
    rel_embeddings = rel_embeddings.to(device)
    linear = nn.Linear(dim, REL_NUM, True).to(device)
    ent_idx = torch.LongTensor(np.arange(ENT_NUM)).to(device)
    rel_idx = torch.LongTensor(np.arange(REL_NUM)).to(device)

    weight_raw = torch.tensor([0.1, 0.9], requires_grad=False, device=device)

    tokenizer = BertTokenizer.from_pretrained("../bert-base-multilingual-cased")
    bert_model = BertForSequenceClassification.from_pretrained("../bert-base-multilingual-cased", num_labels=REL_NUM)
    bert_model.to(device)
    param_optimizer = list(bert_model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']


    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01,'lr':1e-5},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0,'lr':1e-5},
        {"params":[ent_embeddings.weight] +[rel_embeddings.weight] +list(linear.parameters()),'lr':0.0005},
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=1e-5)


    align_parameters = [
        {"params":
             [ent_embeddings.weight]
         }
    ]
    align_optimizer = AdamW(align_parameters, lr=args.lr)


    if args.epsilon > 0:
        trunc_ent_num = int(len(entids_list) * (1 - args.epsilon))
        assert trunc_ent_num > 0
        print("trunc ent num:", trunc_ent_num)
    else:
        trunc_ent_num = 0
        assert not trunc_ent_num > 0

    t_total = time.time()
    print("[start training]")
    print("  num train tripless = ", len(train_triples1))
    print("  batch size = ", args.train_batch_size)


    for epoch in tqdm(range(args.epochs)):
        bert_model.train()
        ent_embeddings.train()
        rel_embeddings.train()
        linear.train()

        train_alignment(epoch, train_ill, alignment_loss, args.train_batch_size, align_optimizer, device,
                        ent_embeddings,
                        ent_idx)

        ent_embedding = F.normalize(ent_embeddings(ent_idx), p=2, dim=1)
        nbours = generate_neighbours_multi_embed(ent_embedding[entids_list].detach(),
                                                 entids_list, trunc_ent_num, args.nums_threads)
        train_completion(args,REL_NUM, train_triples, bert_model, tokenizer,
                                linear,optimizer,weight_raw,ent_idx,rel_idx,ent2text,
                                id2ent,ent_embeddings,rel_embeddings,entids_list, nbours, args.nums_neg,
                                args.lambda_1, args.lambda_2,args.mu_1, device)

        test_completion(args, epoch, bert_model, tokenizer, train_triples, test_triples1, ent2text, id2ent, ent_embeddings,
                        rel_embeddings, linear, ent_idx, rel_idx, weight_raw, device, "triples1")

        test_completion(args, epoch, bert_model, tokenizer, train_triples, test_triples2, ent2text, id2ent, ent_embeddings,
                        rel_embeddings, linear, ent_idx, rel_idx, weight_raw, device, "triples2")

        test_completion(args, epoch, bert_model, tokenizer, train_triples, test_ills_triples, ent2text, id2ent, ent_embeddings,
                        rel_embeddings, linear, ent_idx, rel_idx, weight_raw, device, "ills")

        if (epoch+1)%5==0:
            test_completion(args, epoch, bert_model, tokenizer, train_triples, test_cross, ent2text, id2ent, ent_embeddings,
                            rel_embeddings, linear, ent_idx, rel_idx, weight_raw, device, "cross")

        if (epoch + 1) % 10 == 0:
            test_completion(args, epoch, bert_model, tokenizer, train_triples, test_triples, ent2text, id2ent,
                            ent_embeddings,
                            rel_embeddings, linear, ent_idx, rel_idx, weight_raw, device, "all")
        if nbours is not None:
            del nbours
        del ent_embedding

        if (epoch + 1) % 5 == 0:
            save(bert_model,ent_embeddings, rel_embeddings, linear,weight_raw, epoch,args.save_path)

    print("[optimization finished!]")
    print("[total time elapsed: {:.4f} s]".format(time.time() - t_total))



if __name__ == "__main__":
    main()
