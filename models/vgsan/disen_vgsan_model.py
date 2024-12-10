# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from .modules import SelfAttention
from .gnn import GCNLayer
from . import config


class Encoder(nn.Module):
    def __init__(self, num_items, args):
        super(Encoder, self).__init__()
        self.encoder_mu = SelfAttention(num_items, args)
        self.encoder_logvar = SelfAttention(num_items, args)

    def forward(self, seqs, seqs_data):
        """
        seqs: (batch_size, seq_len, hidden_size)
        seqs_data: (batch_size, seq_len)
        """
        mu = self.encoder_mu(seqs, seqs_data)
        logvar = self.encoder_logvar(seqs, seqs_data)
        return mu, logvar


class Decoder(nn.Module):
    def __init__(self, num_items, args):
        super(Decoder, self).__init__()
        self.decoder = SelfAttention(num_items, args)

    def forward(self, seqs, seqs_data):
        """
        seqs: (batch_size, seq_len, hidden_size)
        seqs_data: (batch_size, seq_len)
        """
        feat_seq = self.decoder(seqs, seqs_data)
        return feat_seq


class DisenVGSAN(nn.Module):
    def __init__(self, c_id, args, num_items_list):
        super(DisenVGSAN, self).__init__()
        self.device = "cuda:%s" % args.gpu if args.cuda else "cpu"
        self.num_items_list = num_items_list
        self.c_id = c_id
        # Item embeddings cannot be shared between clients, because the number
        # of items in each domain is different.
        self.item_emb_e_list = nn.ModuleList(
            [nn.Embedding(num_items_list[c_id] + 1, config.hidden_size, padding_idx=num_items_list[c_id]) for i in range(len(num_items_list))])
        self.pos_emb_e_list = nn.ModuleList(
            [nn.Embedding(args.max_seq_len, config.hidden_size) for i in range(len(num_items_list))])
        self.GNN_encoder_e_list = nn.ModuleList(
            [GCNLayer(args) for i in range(len(num_items_list))])

        self.encoder_e_list = nn.ModuleList(
            [Encoder(num_items_list[i], args) for i in range(len(num_items_list))])

        self.linear_local = nn.Linear(
            config.hidden_size, num_items_list[c_id])
        self.linear_pad_local = nn.Linear(config.hidden_size, 1)

        self.linear_shared = nn.Linear(
            config.hidden_size * len(self.num_items_list), num_items_list[c_id])
        self.linear_pad_shared = nn.Linear(
            config.hidden_size * len(self.num_items_list), 1)

        self.LayerNorm_e_list = nn.ModuleList(
            [nn.LayerNorm(config.hidden_size, eps=1e-12) for i in range(len(num_items_list))])
        self.dropout_list = nn.ModuleList(
            [nn.Dropout(config.dropout_rate) for i in range(len(num_items_list))])

        self.Distribution_Aligner_list = nn.ModuleList([
            # Only apply a simple linear transformation
            nn.Linear(config.hidden_size, config.hidden_size)
            for i in range(len(num_items_list))
        ])

    def my_index_select_embedding(self, memory, index):
        tmp = list(index.size()) + [-1]
        index = index.view(-1)
        ans = memory(index)
        ans = ans.view(tmp)
        return ans

    def my_index_select(self, memory, index):
        tmp = list(index.size()) + [-1]
        index = index.view(-1)
        ans = torch.index_select(memory, 0, index)
        ans = ans.view(tmp)
        return ans

    def graph_convolution(self, adj):
        self.item_graph_embs_e_list = []
        for i in range(len(self.num_items_list)):
            self.item_index_e = torch.arange(
                0, self.item_emb_e_list[i].num_embeddings, 1).to(self.device)
            item_embs_e = self.my_index_select_embedding(
                self.item_emb_e_list[i], self.item_index_e)
            self.item_graph_embs_e_list.append(
                self.GNN_encoder_e_list[i](item_embs_e, adj))

    def get_position_ids(self, seqs):
        seq_length = seqs.size(1)
        position_ids = torch.arange(
            seq_length, dtype=torch.long, device=seqs.device)
        position_ids = position_ids.unsqueeze(0).expand_as(seqs)
        return position_ids

    def add_position_embedding_e(self, seqs, seq_embeddings, c_id):
        position_ids = self.get_position_ids(seqs)
        position_embeddings = self.pos_emb_e_list[c_id](position_ids)
        seq_embeddings += position_embeddings
        seq_embeddings = self.LayerNorm_e_list[c_id](seq_embeddings)
        seq_embeddings = self.dropout_list[c_id](seq_embeddings)
        return seq_embeddings  # (batch_size, seq_len, hidden_size)

    def forward(self, seqs, neg_seqs=None, aug_seqs=None):
        # `item_graph_embs` stores the embeddings of all items.
        # Here we need to select the embeddings of items appearing in the
        # sequence
        seqs_emb_e_list = []
        for i in range(len(self.num_items_list)):
            seqs_emb_e = self.my_index_select(
                self.item_graph_embs_e_list[i], seqs) + self.item_emb_e_list[i](seqs)
            # (batch_size, seq_len, hidden_size)
            seqs_emb_e *= self.item_emb_e_list[i].embedding_dim ** 0.5
            seqs_emb_e = self.add_position_embedding_e(
                seqs, seqs_emb_e, i)  # (batch_size, seq_len, hidden_size)
            seqs_emb_e_list.append(seqs_emb_e)

        # Here is a shortcut operation that adds up the embeddings of items
        # convolved by GNN and those that have not been convolved.
        if self.training:
            neg_seqs_emb_list = []
            aug_seqs_emb_list = []
            for i in range(len(self.num_items_list)):
                neg_seqs_emb = self.my_index_select(
                    self.item_graph_embs_e_list[i], neg_seqs) + self.item_emb_e_list[i](neg_seqs)
                aug_seqs_emb = self.my_index_select(
                    self.item_graph_embs_e_list[i], aug_seqs) + self.item_emb_e_list[i](aug_seqs)
                # (batch_size, seq_len, hidden_size)
                neg_seqs_emb *= self.item_emb_e_list[i].embedding_dim ** 0.5
                # (batch_size, seq_len, hidden_size)
                aug_seqs_emb *= self.item_emb_e_list[i].embedding_dim ** 0.5
                neg_seqs_emb = self.add_position_embedding_e(
                    neg_seqs, neg_seqs_emb, i)  # (batch_size, seq_len, hidden_size)
                aug_seqs_emb = self.add_position_embedding_e(
                    aug_seqs, aug_seqs_emb, i)  # (batch_size, seq_len, hidden_size)
                neg_seqs_emb_list.append(neg_seqs_emb)
                aug_seqs_emb_list.append(aug_seqs_emb)

        if self.training:
            neg_z_e_list = []
            aug_z_e_list = []
            for i in range(len(self.num_items_list)):
                neg_mu_e, neg_logvar_e = self.encoder_e_list[i](
                    neg_seqs_emb, neg_seqs)
                neg_z_e = self.reparameterization(neg_mu_e, neg_logvar_e)

                aug_mu_e, aug_logvar_e = self.encoder_e_list[i](
                    aug_seqs_emb, aug_seqs)
                aug_z_e = self.reparameterization(aug_mu_e, aug_logvar_e)
                neg_z_e_list.append(neg_z_e)
                aug_z_e_list.append(aug_z_e)

        mu_e_list = []
        logvar_e_list = []
        z_e_list = []
        for i in range(len(self.num_items_list)):
            mu_e, logvar_e = self.encoder_e_list[i](seqs_emb_e, seqs)
            z_e = self.reparameterization(mu_e, logvar_e)
            mu_e_list.append(mu_e)
            logvar_e_list.append(logvar_e)
            z_e_list.append(z_e)

        result = self.linear_local(z_e_list[self.c_id])
        result_pad = self.linear_pad_local(z_e_list[self.c_id])
        # reconstructed_seq_exclusive = self.decoder(z_e)
        result_exclusive = self.linear_local(z_e_list[self.c_id])
        result_exclusive_pad = self.linear_pad_local(z_e_list[self.c_id])

        z_e = []
        for i in range(len(self.num_items_list)):
            if i != self.c_id:
                # z_e.append(torch.zeros_like(z_e_list[i]))
                z = self.Distribution_Aligner_list[i](z_e_list[i])
                z_e.append(z)
            else:
                z_e.append(z_e_list[i].detach())
        z_e = torch.cat(z_e, dim=-1)

        result_shared = self.linear_shared(z_e)
        result_pad_shared = self.linear_pad_shared(z_e)
        result_exclusive_shared = self.linear_shared(z_e)
        result_exclusive_pad_shared = self.linear_pad_shared(z_e)

        if self.training:
            return torch.cat((result, result_pad), dim=-1), \
                torch.cat((result_exclusive, result_exclusive_pad), dim=-1), \
                torch.cat((result_shared, result_pad_shared), dim=-1), \
                torch.cat((result_exclusive_shared, result_exclusive_pad_shared), dim=-1), \
                mu_e_list, logvar_e_list, z_e_list, neg_z_e_list, aug_z_e_list
        else:
            return torch.cat((result, result_pad), dim=-1), \
                torch.cat((result_shared, result_pad_shared), dim=-1)

    def reparameterization(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        if self.training:
            eps = torch.randn_like(std)
            res = mu + std * eps
        else:
            res = mu
        return res
