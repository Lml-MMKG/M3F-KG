import math
import os
import numpy as np
import random
import torch.nn.functional as F
from torch import nn
import torch
from torch.nn import init


def cosine_sim(im, s):
    """Cosine similarity between all the image and sentence pairs
    """
    return im.mm(s.t())
class NCA_loss(nn.Module):
    def __init__(self, alpha, beta, ep):
        super(NCA_loss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.ep = ep
        self.sim = cosine_sim

    def forward(self, emb, train_links, test_links, device):
        emb = F.normalize(emb)
        num_ent = emb.shape[0]
        im = emb[train_links[:, 0]].to(device)
        s = emb[train_links[:, 1]]

        im_neg_scores=0
        s_neg_scores=0
        if len(test_links) != 0:
            test_links = test_links[random.sample([x for x in np.arange(0, len(test_links))], 4500)]
            im_neg_scores = self.sim(im, emb[test_links[:, 1]])
            s_neg_scores = self.sim(s, emb[test_links[:, 0]])

        bsize = im.size()[0]
        scores = self.sim(im, s)  # + 1
        tmp = torch.eye(bsize).to(device)
        s_diag = tmp * scores

        alpha = self.alpha
        alpha_2 = alpha  # / 3.0
        beta = self.beta
        ep = self.ep
        S_ = torch.exp(alpha * (scores - ep))
        S_ = S_ - S_ * tmp  # clear diagnal

        if len(test_links) != 0:
            S_1 = torch.exp(alpha * (im_neg_scores - ep))
            S_2 = torch.exp(alpha * (s_neg_scores - ep))

        loss_diag = - torch.log(1 + F.relu(s_diag.sum(0)))

        loss = torch.sum(
            torch.log(1 + S_.sum(0)) / alpha
            + torch.log(1 + S_.sum(1)) / alpha
            + loss_diag * beta \
            ) / bsize

        loss_global_neg=0
        if len(test_links) != 0:
            loss_global_neg = (torch.sum(torch.log(1 + S_1.sum(0)) / alpha_2
                                         + torch.log(1 + S_2.sum(0)) / alpha_2)
                               + torch.sum(torch.log(1 + S_1.sum(1)) / alpha_2
                                           + torch.log(1 + S_2.sum(1)) / alpha_2)) / 4500
        if len(test_links) != 0:
            return loss + loss_global_neg
        return loss
class MlP(nn.Module):
    def __init__(self,input_dim,hidden_dim):
        super(MlP, self).__init__()
        self.dense1 = nn.Linear(input_dim, 2*hidden_dim, True)
        self.dense2 = nn.Linear(2*hidden_dim, hidden_dim, True)
        self.dense3 = nn.Linear(hidden_dim, hidden_dim, True)
        init.xavier_normal_(self.dense1.weight)
        init.xavier_normal_(self.dense2.weight)
    def forward(self,features):
        x = self.dense1(features)#[B,h]
        x = F.relu(x)
        x = self.dense2(x)#[B,1]
        x=F.relu(x)
        x = self.dense3(x)  # [B,1]
        x = torch.tanh(x)
        return x
class KGProcessor(object):
    """Processor for knowledge graph data set."""
    def __init__(self,data_dir):
        self.labels = set()
        self.data_dir=data_dir
        self.train_triples1 =self.get_triples(data_dir+"train_triples1")
        self.train_triples2 = self.get_triples(data_dir+"train_triples2")


        self.train_cross_triples = self.get_triples(data_dir+"train_cross_triples")
        self.train_ills_triples = self.get_triples(data_dir+"train_ills_triples")

        self.test_triples1 = self.get_triples(data_dir+"test_triples1")
        self.test_triples2 =self.get_triples(data_dir+"test_triples2")

        self.test_cross_triples = self.get_triples(data_dir + "test_cross_triples")
        self.test_ills_triples = self.get_triples(data_dir + "test_ills_triples")

        self.ent2id=self.get_object2id(data_dir + "ent2id")
        self.ent2id_1 = self.get_object2id1(data_dir + "ent_ids_1")
        self.ent2id_2 =self.get_object2id1(data_dir + "ent_ids_2")
        self.rel2id = self.get_object2id(data_dir + "rel2id")
        self.rel2id_1 = self.get_object2id1(data_dir + "rel_ids_1")
        self.rel2id_2 = self.get_object2id1(data_dir + "rel_ids_2")

        self.ent2text = self.get_object2text(data_dir + "entity2text.txt")
        # self.rel2text= self.get_object2text(data_dir + "rel2text.txt")

        self.train_ills=self.get_ills(data_dir+"train_ills")
        self.test_ills= self.get_ills(data_dir+"test_ills")


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



