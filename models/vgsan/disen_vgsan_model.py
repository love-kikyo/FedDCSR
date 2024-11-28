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
    def __init__(self, num_items, args):
        super(DisenVGSAN, self).__init__()
        self.device = "cuda:%s" % args.gpu if args.cuda else "cpu"

        # Item embeddings cannot be shared between clients, because the number
        # of items in each domain is different.
        self.item_emb_e = nn.ModuleList([
            nn.Embedding(num_item + 1, config.hidden_size,
                         padding_idx=num_item)
            for num_item in num_items
        ])
        self.pos_emb_e = nn.Embedding(args.max_seq_len, config.hidden_size)
        self.GNN_encoder_e = nn.ModuleList([
            GCNLayer(args) for _ in num_items
        ])

        self.encoder_e = nn.ModuleList([
            Encoder(num_item, args) for num_item in num_items
        ])
        self.decoder = nn.ModuleList([
            Decoder(num_item, args) for num_item in num_items
        ])

        # The last prediction layer cannot be shared between clients, because
        # the number of items in each domain is different.
        self.linear = nn.ModuleList([
            nn.Linear(config.hidden_size, num_item) for num_item in num_items
        ])
        self.linear_pad = nn.ModuleList([
            nn.Linear(config.hidden_size, 1) for _ in num_items
        ])

        self.LayerNorm_e = nn.LayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.dropout_rate)

        self.item_graph_embs_e = [None] * len(num_items)

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

    def graph_convolution(self, adj_list, domain_idx):
        adj = adj_list[domain_idx]
        item_index_e = torch.arange(
            0, self.item_emb_e[domain_idx].num_embeddings, 1).to(self.device)
        item_embs_e = self.my_index_select_embedding(
            self.item_emb_e[domain_idx], item_index_e)
        self.item_graph_embs_e[domain_idx] = self.GNN_encoder_e[domain_idx](
            item_embs_e, adj)

    def get_position_ids(self, seqs):
        seq_length = seqs.size(1)
        position_ids = torch.arange(
            seq_length, dtype=torch.long, device=seqs.device)
        position_ids = position_ids.unsqueeze(0).expand_as(seqs)
        return position_ids

    def add_position_embedding_e(self, seqs, seq_embeddings):
        position_ids = self.get_position_ids(seqs)
        position_embeddings = self.pos_emb_e(position_ids)
        seq_embeddings += position_embeddings
        seq_embeddings = self.LayerNorm_e(seq_embeddings)
        seq_embeddings = self.dropout(seq_embeddings)
        return seq_embeddings  # (batch_size, seq_len, hidden_size)

    def forward(self, seqs, neg_seqs=None, aug_seqs=None, domain_idx=0):
        """
        Forward pass for multi-domain embeddings.

        Args:
            seqs: 输入的序列 (batch_size, seq_len)
            neg_seqs: 负样本序列 (batch_size, seq_len)
            aug_seqs: 数据增强后的序列 (batch_size, seq_len)
            domain_idx: 当前领域索引 (int)

        Returns:
            不同模式下的输出，取决于 self.training。
        """
        # 动态选择当前领域的图嵌入和物品嵌入层
        item_graph_embs_e = self.item_graph_embs_e[domain_idx]
        item_emb_e = self.item_emb_e[domain_idx]

        # 获取序列的嵌入
        seqs_emb_e = self.my_index_select(
            item_graph_embs_e, seqs) + item_emb_e(seqs)
        # (batch_size, seq_len, hidden_size)
        seqs_emb_e *= item_emb_e.embedding_dim ** 0.5
        seqs_emb_e = self.add_position_embedding_e(
            seqs, seqs_emb_e)  # (batch_size, seq_len, hidden_size)

        if self.training:
            # 获取负样本和增强样本的嵌入
            neg_seqs_emb = self.my_index_select(
                item_graph_embs_e, neg_seqs) + item_emb_e(neg_seqs)
            aug_seqs_emb = self.my_index_select(
                item_graph_embs_e, aug_seqs) + item_emb_e(aug_seqs)
            # 缩放嵌入
            neg_seqs_emb *= item_emb_e.embedding_dim ** 0.5
            aug_seqs_emb *= item_emb_e.embedding_dim ** 0.5
            # 添加位置嵌入
            neg_seqs_emb = self.add_position_embedding_e(
                neg_seqs, neg_seqs_emb)  # (batch_size, seq_len, hidden_size)
            aug_seqs_emb = self.add_position_embedding_e(
                aug_seqs, aug_seqs_emb)  # (batch_size, seq_len, hidden_size)

        # 编码器处理
        if self.training:
            neg_mu_e, neg_logvar_e = self.encoder_e(neg_seqs_emb, neg_seqs)
            neg_z_e = self.reparameterization(neg_mu_e, neg_logvar_e)

            aug_mu_e, aug_logvar_e = self.encoder_e(aug_seqs_emb, aug_seqs)
            aug_z_e = self.reparameterization(aug_mu_e, aug_logvar_e)

        mu_e, logvar_e = self.encoder_e(seqs_emb_e, seqs)
        z_e = self.reparameterization(mu_e, logvar_e)

        # Decoder and output layers
        result = self.linear(z_e)
        result_pad = self.linear_pad(z_e)

        result_exclusive = self.linear(z_e)
        result_exclusive_pad = self.linear_pad(z_e)

        if self.training:
            return torch.cat((result, result_pad), dim=-1), \
                torch.cat((result_exclusive, result_exclusive_pad), dim=-1), \
                mu_e, logvar_e, z_e, neg_z_e, aug_z_e
        else:
            return torch.cat((result, result_pad), dim=-1)

    def reparameterization(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        if self.training:
            eps = torch.randn_like(std)
            res = mu + std * eps
        else:
            res = mu
        return res
