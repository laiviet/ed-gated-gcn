# -*- coding: utf-8 -*-
# file: BERT_SPC.py
# author: songyouwei <youwei0314@gmail.com>
# Copyright (C) 2019. All Rights Reserved.
import torch
import torch.nn as nn
from layers.squeeze_embedding import SqueezeEmbedding
from layers.dynamic_rnn import DynamicLSTM
from models.gcn import GraphConvolution

INFINITY_NUMBER = 1e12


class BertAmir(nn.Module):
    def __init__(self, bert, opt):
        super(BertAmir, self).__init__()
        # self.squeeze_embedding = SqueezeEmbedding()
        self.device = opt.device
        self.bert = bert
        self.dropout = nn.Dropout(opt.dropout)
        self.dense = nn.Linear((opt.n_layer) * opt.bert_dim + 2 * opt.hidden_dim, opt.polarities_dim)
        # self.dense = nn.Linear(2*opt.hidden_dim, opt.polarities_dim)
        self.n_layer = opt.n_layer
        self.text_lstm = DynamicLSTM(opt.bert_dim * self.n_layer, opt.hidden_dim, num_layers=2,
                                     batch_first=True,
                                     bidirectional=True)

        self.gc1 = GraphConvolution(2 * opt.hidden_dim, 2 * opt.hidden_dim, opt)
        self.gc2 = GraphConvolution(2 * opt.hidden_dim, 2 * opt.hidden_dim, opt)

        self.gate1 = nn.Sequential(nn.Linear(opt.hidden_dim * 2, opt.hidden_dim * 2), nn.Sigmoid(),
                                   nn.Linear(opt.hidden_dim * 2, opt.hidden_dim * 2), nn.Sigmoid(),
                                   nn.Linear(opt.hidden_dim * 2, opt.hidden_dim * 2), nn.Sigmoid())
        self.gate2 = nn.Sequential(nn.Linear(opt.hidden_dim * 2, opt.hidden_dim * 2), nn.Sigmoid(),
                                   nn.Linear(opt.hidden_dim * 2, opt.hidden_dim * 2), nn.Sigmoid(),
                                   nn.Linear(opt.hidden_dim * 2, opt.hidden_dim * 2), nn.Sigmoid())

        self.fc = nn.Linear(2 * 2 * opt.hidden_dim, opt.polarities_dim)

    def get_all_layers(self, inputs):
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
        return anchor_rep

    def forward(self, inputs):
        """
        bert_length: L
        original_length: T
        """

        L = inputs['cls_text_sep_aspect_sep_length'].max().cpu().numpy().tolist()
        T = inputs['sentence_length'].max().cpu().numpy().tolist()
        B = inputs['sentence_length'].shape[0]

        text_bert_indices = inputs['cls_text_sep_aspect_sep_indices'][:, :L]  # B x L
        bert_segments_ids = inputs['cls_text_sep_aspect_sep_segments_ids'][:, :L]  # B x L
        adj = inputs['dependency_graph'][:, :T, :T]  # B x T x T
        aspect_mask = inputs['cls_text_sep_aspect_sep_aspect_mask'][:, :L]  # B x L
        dist_to_target = inputs['dist_to_target'][:, :T]  # B x T
        mask = inputs['cls_text_sep_aspect_sep_mask'][:, :T]  # B x T
        transform = inputs['transform'][:, :T, :L]  # B x T x L
        # print('| L: ', L)
        # print('| T: ', T)

        x, pooled_output = self.bert(text_bert_indices, bert_segments_ids, output_all_encoded_layers=True)
        # print('| Bert_amir > len(x)', len(x))
        # print('| Bert_amir > x[0]', tuple(x[0].shape))

        bert_x = torch.cat(x[-self.n_layer:], dim=-1)

        # anchor_rep = self.get_all_layers(inputs)

        x, (_, _) = self.text_lstm(bert_x, inputs['cls_text_sep_aspect_sep_length'])

        # print('| Bert_amir > x ', tuple(x.shape))
        # print('| Bert_amir > transform ', tuple(transform.shape))
        # print('| Bert_amir > aspect_mask ', tuple(aspect_mask.shape))

        aspect = torch.max(x.masked_fill(aspect_mask.bool().unsqueeze(2), -INFINITY_NUMBER), 1)[0]
        # print('| Bert_amir > aspect ', tuple(aspect.shape))

        x = torch.bmm(transform, x)  # B x T x D
        # print('| Bert_amir > x (bmm) ', tuple(x.shape))
        # print('| Bert_amir > adj', tuple(adj.shape))

        # Anchor extract

        gate1 = self.gate1(aspect).repeat(1, x.shape[1]).view(x.shape)
        gate2 = self.gate2(aspect).repeat(1, x.shape[1]).view(x.shape)

        gate1 = self.dropout(gate1)
        gate2 = self.dropout(gate2)
        gcn1 = self.gc1(x, adj)
        gcngate1 = gcn1 * gate1

        # print('| Bert_amir > gcngate1', tuple(gcngate1.shape))

        gcngate2 = gcn1 * gate2

        # print('| Bert_amir > gcngate2', tuple(gcngate2.shape))

        x1 = torch.max(gcngate1.masked_fill(mask.bool().unsqueeze(2), -INFINITY_NUMBER), 1)[0]
        y1 = torch.max(gcngate2.masked_fill(mask.bool().unsqueeze(2), -INFINITY_NUMBER), 1)[0]
        sf1 = nn.Softmax(1)
        xy = (x1 * y1).sum(1).mean()
        x = gate2 * self.gc2(gcn1, adj)
        out = torch.max(x, dim=1)[0]
        pooled_output = self.dropout(pooled_output)
        out = self.dropout(out)
        logits = self.dense(torch.cat([pooled_output, out], dim=1))

        output_w = self.fc(torch.cat([x, aspect.repeat(1, x.shape[1]).view(x.shape)], dim=2))
        scores = (logits.repeat(1, x.shape[1]).view(x.shape[0], x.shape[1], -1) * output_w).sum(2)
        # sf2 = nn.Softmax(2)
        kl = (sf1(scores) * sf1(dist_to_target.float())).sum(1).mean()

        return logits, xy, kl
        # return logits, xy, 0
        # return logits, 0, 0




class BertAmir2(nn.Module):
    def __init__(self, bert, opt):
        super(BertAmir2, self).__init__()
        # self.squeeze_embedding = SqueezeEmbedding()
        self.device = opt.device
        self.bert = bert
        self.dropout = nn.Dropout(opt.dropout)
        self.dense = nn.Linear(opt.n_layer * opt.bert_dim + 2 * opt.hidden_dim, opt.polarities_dim)
        # self.dense = nn.Linear(2*opt.hidden_dim, opt.polarities_dim)
        self.n_layer = opt.n_layer
        self.text_lstm = DynamicLSTM(opt.bert_dim * self.n_layer, opt.hidden_dim, num_layers=2,
                                     batch_first=True,
                                     bidirectional=True)

        self.gc1 = GraphConvolution(2 * opt.hidden_dim, 2 * opt.hidden_dim, opt)
        self.gc2 = GraphConvolution(2 * opt.hidden_dim, 2 * opt.hidden_dim, opt)

        self.gate1 = nn.Sequential(nn.Sigmoid(),
                                   nn.Linear(opt.hidden_dim * 2, opt.hidden_dim * 2), nn.Sigmoid(),
                                   nn.Linear(opt.hidden_dim * 2, opt.hidden_dim * 2), nn.Sigmoid(),
                                   nn.Linear(opt.hidden_dim * 2, opt.hidden_dim * 2), nn.Sigmoid())
        self.gate2 = nn.Sequential(nn.Sigmoid(),
                                   nn.Linear(opt.hidden_dim * 2, opt.hidden_dim * 2), nn.Sigmoid(),
                                   nn.Linear(opt.hidden_dim * 2, opt.hidden_dim * 2), nn.Sigmoid(),
                                   nn.Linear(opt.hidden_dim * 2, opt.hidden_dim * 2), nn.Sigmoid())

        self.fc = nn.Linear(2 * 2 * opt.hidden_dim, opt.polarities_dim)

    def get_all_layers(self, x, inputs, L,T):
        """
        bert_length: L
        original_length: T
        """

        transform = inputs['transform'][:, :T, :L]  # B x T x L
        anchor_index = inputs['anchor_index']  # B

        B = anchor_index.shape[0]
        # print('| L: ', L)
        # print('| T: ', T)

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
        return anchor_rep

    def forward(self, inputs):
        """
        bert_length: L
        original_length: T
        """

        L = inputs['cls_text_sep_aspect_sep_length'].max().cpu().numpy().tolist()
        T = inputs['sentence_length'].max().cpu().numpy().tolist()
        B = inputs['sentence_length'].shape[0]

        text_bert_indices = inputs['cls_text_sep_aspect_sep_indices'][:, :L]  # B x L
        bert_segments_ids = inputs['cls_text_sep_aspect_sep_segments_ids'][:, :L]  # B x L
        adj = inputs['dependency_graph'][:, :T, :T]  # B x T x T
        aspect_mask = inputs['cls_text_sep_aspect_sep_aspect_mask'][:, :L]  # B x L
        dist_to_target = inputs['dist_to_target'][:, :T]  # B x T
        mask = inputs['cls_text_sep_aspect_sep_mask'][:, :T]  # B x T
        transform = inputs['transform'][:, :T, :L]  # B x T x L
        # print('| L: ', L)
        # print('| T: ', T)

        x, pooled_output = self.bert(text_bert_indices, bert_segments_ids, output_all_encoded_layers=True)
        # print('| Bert_amir > len(x)', len(x))
        # print('| Bert_amir > x[0]', tuple(x[0].shape))

        bert_x = torch.cat(x[-self.n_layer:], dim=-1)

        anchor_rep = self.get_all_layers(bert_x, inputs, L,T)

        x, (_, _) = self.text_lstm(bert_x, inputs['cls_text_sep_aspect_sep_length'])

        # print('| Bert_amir > x ', tuple(x.shape))
        # print('| Bert_amir > transform ', tuple(transform.shape))
        # print('| Bert_amir > aspect_mask ', tuple(aspect_mask.shape))

        aspect = torch.max(x.masked_fill(aspect_mask.bool().unsqueeze(2), -INFINITY_NUMBER), 1)[0]
        # print('| Bert_amir > aspect ', tuple(aspect.shape))

        x = torch.bmm(transform, x)  # B x T x D
        # print('| Bert_amir > x (bmm) ', tuple(x.shape))
        # print('| Bert_amir > adj', tuple(adj.shape))

        # Anchor extract

        gate1 = self.gate1(aspect).repeat(1, x.shape[1]).view(x.shape)
        gate2 = self.gate2(aspect).repeat(1, x.shape[1]).view(x.shape)

        gate1 = self.dropout(gate1)
        gate2 = self.dropout(gate2)
        gcn1 = self.gc1(x, adj)
        gcngate1 = gcn1 * gate1

        # print('| Bert_amir > gcngate1', tuple(gcngate1.shape))

        gcngate2 = gcn1 * gate2

        # print('| Bert_amir > gcngate2', tuple(gcngate2.shape))

        x1 = torch.max(gcngate1.masked_fill(mask.bool().unsqueeze(2), -INFINITY_NUMBER), 1)[0]
        y1 = torch.max(gcngate2.masked_fill(mask.bool().unsqueeze(2), -INFINITY_NUMBER), 1)[0]
        sf1 = nn.Softmax(1)
        xy = (x1 * y1).sum(1).mean()
        x = gate2 * self.gc2(gcn1, adj)
        out = torch.max(x, dim=1)[0]
        # pooled_output = self.dropout(pooled_output)
        out = self.dropout(out)
        logits = self.dense(torch.cat([anchor_rep, out], dim=1))

        output_w = self.fc(torch.cat([x, aspect.repeat(1, x.shape[1]).view(x.shape)], dim=2))
        scores = (logits.repeat(1, x.shape[1]).view(x.shape[0], x.shape[1], -1) * output_w).sum(2)
        # sf2 = nn.Softmax(2)
        kl = (sf1(scores) * sf1(dist_to_target.float())).sum(1).mean()

        return logits, xy, kl
        # return logits, xy, 0
        # return logits, 0, 0
