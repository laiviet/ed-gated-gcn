# -*- coding: utf-8 -*-
# file: BERT_SPC.py
# author: songyouwei <youwei0314@gmail.com>
# Copyright (C) 2019. All Rights Reserved.
import torch
import torch.nn as nn
from layers.squeeze_embedding import SqueezeEmbedding
from layers.dynamic_rnn import DynamicLSTM

INFINITY_NUMBER = 1e12


class BertED(nn.Module):
    def __init__(self, bert, opt):
        super(BertED, self).__init__()
        # self.squeeze_embedding = SqueezeEmbedding()
        self.device = opt.device
        self.hidden_dim = opt.hidden_dim
        self.bert = bert
        self.dropout = nn.Dropout(opt.dropout)
        self.n_layer = 12
        self.fc = nn.Linear(768 * self.n_layer, opt.polarities_dim)

    def forward(self, inputs):
        """
        bert_length: L
        original_length: T
        """

        L = inputs['cls_text_sep_length'].max().cpu().numpy().tolist()
        T = inputs['sentence_length'].max().cpu().numpy().tolist()

        text_bert_indices = inputs['cls_text_sep_indices'][:, :L]  # B x L
        bert_segments_ids = inputs['cls_text_sep_segments_ids'][:, :L]  # B x L
        transform = inputs['transform'][:, :T, :L]  # B x T x L
        anchor_index = inputs['anchor_index']  # B

        B = anchor_index.shape[0]
        # print('| L: ', L)
        # print('| T: ', T)

        x, pooled_output = self.bert(text_bert_indices,
                                     bert_segments_ids,
                                     output_all_encoded_layers=True)

        x = torch.cat(x[-self.n_layer:], dim=-1)
        # print('| bert_ed > x ', tuple(x.shape))
        # print('| bert_ed > pooled_output ', tuple(pooled_output.shape))
        # print('| bert_ed > x ', tuple(x.shape))
        x = torch.bmm(transform, x)  # B x T x D

        # print('| BertED > x(bmm)', tuple(x.shape))
        # print('| BertED > anchor_index', tuple(anchor_index.shape))

        mask = torch.LongTensor([i for i in range(T)]).to(self.device).repeat(B, 1)  # B x T
        # print('| BertED > mask', tuple(mask.shape))

        mask = mask == anchor_index.unsqueeze(1)
        mask = mask.unsqueeze(dim=2).expand(-1, -1, 768 * self.n_layer)
        # print('| BertED > mask', tuple(mask.shape))

        anchor_rep = torch.masked_select(x, mask).view(B, -1)
        # print('| BertED > anchor_rep', tuple(anchor_rep.shape))

        anchor_rep = self.dropout(anchor_rep)
        logits = self.fc(anchor_rep)
        # print('| BertED > logits', tuple(logits.shape))

        return logits, 0, 0, None