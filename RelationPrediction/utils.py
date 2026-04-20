import math
import os
import pickle
import random
import torch
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss









def save(bert,ent_embeddings, rel_embeddings, linear, weight_raw,epoch,path):
    save_path=path
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    bert.eval()
    ent_embeddings.eval()
    rel_embeddings.eval()
    linear.eval()
    torch.save(bert.state_dict(), save_path + "model_epoch_" + str(epoch) + '.p')
    torch.save(ent_embeddings.state_dict(), save_path+"ent_embeddings_"+"epoch_" + str(epoch) + '.p')
    torch.save(rel_embeddings.state_dict(), save_path+"rel_embeddings_"+"epoch_" + str(epoch) + '.p')
    torch.save(linear.state_dict(), save_path+"linear_"+"epoch_" + str(epoch) + '.p')
    torch.save(weight_raw, save_path+"weight_raw_epoch_" + str(epoch) + '.p')
    print("Model {} save in: ".format(epoch), save_path )

def save_bert(model,epoch,path):
    save_path=path+"bert/"
    if not os.path.exists(save_path):
        os.makedirs( save_path)
    model.eval()
    torch.save(model.state_dict(),save_path+ "model_epoch_" + str(epoch) + '.p')
    print("Model {} save in: ".format(epoch), save_path + "model_epoch_" + str(epoch) + '.p')

def save_transe(ent_embeddings, rel_embeddings, linear, epoch,path):
    save_path=path+"transe/"
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    ent_embeddings.eval()
    rel_embeddings.eval()
    linear.eval()
    torch.save(ent_embeddings.state_dict(), save_path+"ent_embeddings_"+"epoch_" + str(epoch) + '.p')
    torch.save(rel_embeddings.state_dict(), save_path+"rel_embeddings_"+"epoch_" + str(epoch) + '.p')
    torch.save(linear.state_dict(), save_path+"linear_"+"epoch_" + str(epoch) + '.p')
    print("Model {} save in: ".format(epoch), save_path )




def load_img(e_num, path):
    img_dict = pickle.load(open(path, "rb"))
    # init unknown img vector with mean and std deviation of the known's
    imgs_np = np.array(list(img_dict.values()))  # shape=(30035,2048)
    mean = np.mean(imgs_np, axis=0) # 求每一列的均值  shape=(2048,)
    std = np.std(imgs_np, axis=0) # 计算每列的标准差    shape=(2048,)
    img_embd = np.array([img_dict[i] if i in img_dict else np.random.normal(mean, std, mean.shape[0]) for i in range(e_num)])
    # 根据每列的均值和标准差为没有图像的实体随机生成随图像向量
    print ("%.2f%% entities have images" % (100 * len(img_dict)/e_num))
    return img_embd

def fixed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

def simple_accuracy(preds, labels):
    return (preds == labels).mean()

def compute_metrics( preds, labels):
    assert len(preds) == len(labels)
    return {"acc": simple_accuracy(preds, labels)}

def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()

def convert_examples_to_features(h_ids, t_ids, max_seq_length, ent2text, id2ent, tokenizer):
    """Loads a data file into a list of `InputBatch`s."""
    input_ids = []
    input_masks = []
    segment_ids = []
    for i in range(len(h_ids)):
        tokens_a = tokenizer.tokenize(ent2text[id2ent[int(h_ids[i])]])
        tokens_b = tokenizer.tokenize(ent2text[id2ent[int(t_ids[i])]])
        _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
        tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
        segment_id = [0] * len(tokens)
        tokens += tokens_b + ["[SEP]"]
        segment_id += [1] * (len(tokens_b) + 1)
        input_id = tokenizer.convert_tokens_to_ids(tokens)
        input_mask = [1] * len(input_id)
        # Zero-pad up to the sequence length.
        padding = [0] * (max_seq_length - len(input_id))
        input_id += padding
        input_mask += padding
        segment_id += padding
        assert len(input_id) == max_seq_length
        assert len(input_mask) == max_seq_length
        input_ids.append(input_id)
        input_masks.append(input_mask)
        segment_ids.append(segment_id)
    return torch.tensor(input_ids), torch.tensor(input_masks), torch.tensor(segment_ids)
#******************************************
def div_list(ls, n):
    ls_len = len(ls)
    if n <= 0 or 0 == ls_len:
        return []
    if n > ls_len:
        return []
    elif n == ls_len:
        return [[i] for i in ls]
    else:
        j = ls_len // n
        k = ls_len % n
        ls_return = []
        for i in range(0, (n - 1) * j, j):
            ls_return.append(ls[i:i + j])
        ls_return.append(ls[(n - 1) * j:])
        return ls_return
def cal_neighbours_embed(frags, ent_list, sub_embed, embed, k):
    dic = dict()

    sub_embed=sub_embed.cpu().numpy().tolist()
    embed=embed.T
    embed=embed.cpu().numpy().tolist()
    sim_mat = np.matmul(sub_embed, embed)
    for i in range(sim_mat.shape[0]):
        sort_index = np.argpartition(-sim_mat[i, :], k + 1)
        dic[frags[i]] = ent_list[sort_index[0:k + 1]]
    del sim_mat

    return dic

def merge_dic(dic1, dic2):
    return {**dic1, **dic2}

def generate_neighbours_multi_embed(embed, ent_list, k, nums_threads):
    # 在同一个图谱中找与自己相似的前k实体
    #  embe实体向量(KG1 或者KG2的) ent_list 实体      k 截断后相似实体集合数量
    # 将实体ent_frags分成nums_threads组，一组1938个实体 实体从0-10499 21000-29887  最后一组1946个实体
    ent_frags = div_list(np.array(ent_list), nums_threads)
    # 给实体编号从0-19387，与上面对应分成十组
    ent_frag_indexes = div_list(np.array(range(len(ent_list))), nums_threads)

    results = list()
    for i in range(len(ent_frags)):
        results.append(cal_neighbours_embed(ent_frags[i], np.array(ent_list), embed[ent_frag_indexes[i], :], embed, k))
    dic = dict()
    for res in results:
        dic = merge_dic(dic, res)
    del embed

    return dic
#******************************************

def trunc_sampling_multi(pos_triples, all_triples, dic, ent_list, multi):
    neg_triples = list()
    ent_list = np.array(ent_list)
    for (h, r, t) in pos_triples:
        choice = random.randint(0, 999)
        if choice < 500:
            candidates = dic.get(h, ent_list)
            index = random.sample(range(0, len(candidates)), multi)
            h2s = candidates[index]
            for h2 in h2s:
                if (h2, r, t) not in all_triples:
                    neg_triples.append((h2, r, t))
        elif choice >= 500:
            candidates = dic.get(t, ent_list)
            index = random.sample(range(0, len(candidates)), multi)
            t2s = candidates[index]
            for t2 in t2s:
                if (h, r, t2) not in all_triples:
                    neg_triples.append((h, r, t2))
    return neg_triples

def trunc_sampling(pos_triples, all_triples, dic, ent_list):
    neg_triples = list()
    i=0
    for (h, r, t) in pos_triples:
        print(i)
        i+=1
        h2, r2, t2 = h, r, t
        while True:
            choice = random.randint(0, 999)
            if choice < 500:
                candidates = dic.get(h, ent_list)
                index = random.sample(range(0, len(candidates)), 1)[0]
                h2 = candidates[index]
            elif choice >= 500:
                candidates = dic.get(t, ent_list)
                index = random.sample(range(0, len(candidates)), 1)[0]
                t2 = candidates[index]
            if (h2, r2, t2) not in all_triples:
                break
        neg_triples.append((h2, r2, t2))
    return neg_triples

def generate_loss(phs, prs, pts, nhs, nrs, nts, lambda_1, lambda_2, mu_1, device):
    pos_score = torch.sum((phs + prs - pts) ** 2, dim=1)
    neg_score = torch.sum((nhs + nrs - nts) ** 2, dim=1)
    pos_loss = torch.sum(torch.maximum(pos_score - lambda_1, torch.tensor(0.).to(device)))
    neg_loss = mu_1 * torch.sum(torch.maximum(lambda_2 - neg_score, torch.tensor(0.).to(device)))
    return pos_loss, neg_loss

def generate_batch_via_neighbour(pos_triples,all_triples, ent_list, neighbours_dic, multi=1):
    assert multi >= 1
    neg_triples = list()
    if len(ent_list) < 10000:
        for i in range(multi):
            neg_triples.extend(trunc_sampling(pos_triples, all_triples, neighbours_dic, ent_list))
    else:
        neg_triples.extend(
            trunc_sampling_multi(pos_triples, all_triples, neighbours_dic, ent_list, multi))

    return neg_triples



# def train_completion_divide(args,REL_NUM, train_triples, model, tokenizer,linear,optimizer,weight_raw,ent_idx,rel_idx,ent2text,id2ent,ent_embeddings,rel_embeddings,ent_list, nbours, nums_neg, lambda_1, lambda_2,mu_1, device):
#     print("***** Running completion training *****")
#     print("  num train tripless = ", len(train_triples))
#     print("  batch size = ", args.train_batch_size)
#
#     ent_embedding = F.normalize(ent_embeddings(ent_idx), p=2, dim=1)
#     rel_embedding = F.normalize(rel_embeddings(rel_idx), p=2, dim=1)
#
#     bert_loss = 0
#     transe_loss = 0
#     all_loss = 0
#     triples_num = len(train_triples)
#     triple_steps = math.ceil(triples_num / args.train_batch_size)
#     for step in tqdm(range(triple_steps)):
#         start = step * args.train_batch_size
#         end = start + args.train_batch_size
#         if end > len(train_triples):
#             end = len(train_triples)
#         batch_triples = list(train_triples)[start: end]
#         pos_hs = [x[0] for x in batch_triples]
#         pos_rs = [x[1] for x in batch_triples]
#         pos_ts = [x[2] for x in batch_triples]
#
#         lable_ids = torch.tensor( pos_rs).to(device)
#         input_ids, input_masks, segment_ids = convert_examples_to_features(pos_hs,  pos_ts, args.max_seq_length,
#                                                                            ent2text, id2ent, tokenizer)
#         input_ids = input_ids.to(device)
#         input_masks = input_masks.to(device)
#         segment_ids = segment_ids.to(device)
#         logits = model(input_ids, input_masks, segment_ids).logits
#         loss_fct = CrossEntropyLoss()
#         bert_loss = loss_fct(logits.view(-1, REL_NUM), lable_ids.view(-1))
#
#         batch_neg = generate_batch_via_neighbour(batch_triples,train_triples,ent_list, nbours, multi=nums_neg)
#         neg_hs = [x[0] for x in batch_neg]
#         neg_rs = [x[1] for x in batch_neg]
#         neg_ts = [x[2] for x in batch_neg]
#         phs = ent_embedding[pos_hs]
#         prs = rel_embedding[pos_rs]
#         pts = ent_embedding[pos_ts]
#         nhs = ent_embedding[neg_hs]
#         nrs = rel_embedding[neg_rs]
#         nts = ent_embedding[neg_ts]
#         pos_loss, neg_loss = generate_loss(phs, prs, pts, nhs, nrs, nts, lambda_1, lambda_2, mu_1, device)
#         r= pts - phs
#         joint_emb = torch.cat((F.normalize(prs, p=2, dim=1), F.normalize(r, p=2, dim=1)), dim=1)
#         joint_emb = linear(joint_emb)
#         joint_emb = F.softmax(joint_emb)
#         transe_loss =  loss_fct(joint_emb,lable_ids.view(-1)) + pos_loss + neg_loss
#
#
#         w_normalized = F.softmax(weight_raw, dim=0)
#         joint_log = w_normalized[0] * F.normalize(joint_emb, p=2, dim=1)+w_normalized[1] *  F.normalize(logits, p=2, dim=1)
#         joint_loss = loss_fct(joint_log, lable_ids.view(-1))
#         loss=transe_loss +bert_loss+joint_loss
#         loss.backward(retain_graph=True)
#         all_loss += loss.item()
#
#         optimizer.step()
#         optimizer.zero_grad()
#
#     print("weight:",w_normalized[0].item(), w_normalized[1].item())
#     print("  bert_loss = %f", bert_loss)
#     print("  transe_loss = %f", transe_loss)
#     print("  all_loss = %f", all_loss)
#     if torch.cuda.is_available():
#         torch.cuda.empty_cache()
#
# def train_completion_divide(args,REL_NUM, train_triples, model, tokenizer,linear,optimizer,weight_raw,ent_idx,rel_idx,ent2text,id2ent,ent_embeddings,rel_embeddings,ent_list, nbours, nums_neg, lambda_1, lambda_2,mu_1, device):
#     print("***** Running completion training *****")
#     print("  num train tripless = ", len(train_triples))
#     print("  batch size = ", args.train_batch_size)
#
#     ent_embedding = F.normalize(ent_embeddings(ent_idx), p=2, dim=1)
#     rel_embedding = F.normalize(rel_embeddings(rel_idx), p=2, dim=1)
#
#     bert_loss = 0
#     transe_loss = 0
#     all_loss = 0
#     triples_num = len(train_triples)
#     triple_steps = math.ceil(triples_num / args.train_batch_size)
#     for step in tqdm(range(triple_steps)):
#         start = step * args.train_batch_size
#         end = start + args.train_batch_size
#         if end > len(train_triples):
#             end = len(train_triples)
#         batch_triples = list(train_triples)[start: end]
#         pos_hs = [x[0] for x in batch_triples]
#         pos_rs = [x[1] for x in batch_triples]
#         pos_ts = [x[2] for x in batch_triples]
#
#         lable_ids = torch.tensor( pos_rs).to(device)
#         input_ids, input_masks, segment_ids = convert_examples_to_features(pos_hs,  pos_ts, args.max_seq_length,
#                                                                            ent2text, id2ent, tokenizer)
#         input_ids = input_ids.to(device)
#         input_masks = input_masks.to(device)
#         segment_ids = segment_ids.to(device)
#         logits = model(input_ids, input_masks, segment_ids).logits
#         loss_fct = CrossEntropyLoss()
#         bert_loss = loss_fct(logits.view(-1, REL_NUM), lable_ids.view(-1))
#
#         batch_neg = generate_batch_via_neighbour(batch_triples,train_triples,ent_list, nbours, multi=nums_neg)
#         neg_hs = [x[0] for x in batch_neg]
#         neg_rs = [x[1] for x in batch_neg]
#         neg_ts = [x[2] for x in batch_neg]
#         phs = ent_embedding[pos_hs]
#         prs = rel_embedding[pos_rs]
#         pts = ent_embedding[pos_ts]
#         nhs = ent_embedding[neg_hs]
#         nrs = rel_embedding[neg_rs]
#         nts = ent_embedding[neg_ts]
#         pos_loss, neg_loss = generate_loss(phs, prs, pts, nhs, nrs, nts, lambda_1, lambda_2, mu_1, device)
#         r= pts - phs
#         joint_emb = torch.cat((F.normalize(prs, p=2, dim=1), F.normalize(r, p=2, dim=1)), dim=1)
#         joint_emb = linear(joint_emb)
#         joint_emb = F.softmax(joint_emb)
#         transe_loss =  loss_fct(joint_emb,lable_ids.view(-1)) + pos_loss + neg_loss
#
#
#         w_normalized = F.softmax(weight_raw, dim=0)
#         joint_log = w_normalized[0] * F.normalize(joint_emb, p=2, dim=1)+w_normalized[1] *  F.normalize(logits, p=2, dim=1)
#         joint_loss = loss_fct(joint_log, lable_ids.view(-1))
#         loss=transe_loss +bert_loss+joint_loss
#         loss.backward(retain_graph=True)
#         all_loss += loss.item()
#
#         optimizer.step()
#         optimizer.zero_grad()
#
#     print("weight:",w_normalized[0].item(), w_normalized[1].item())
#     print("  bert_loss = %f", bert_loss)
#     print("  transe_loss = %f", transe_loss)
#     print("  all_loss = %f", all_loss)
#     if torch.cuda.is_available():
#         torch.cuda.empty_cache()