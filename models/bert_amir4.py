import torch
import torch.nn as nn
from layers.squeeze_embedding import SqueezeEmbedding
from layers.dynamic_rnn import DynamicLSTM
from models.gcn import GraphConvolution

INFINITY_NUMBER = 1e12


class BertAmir4(nn.Module):
    def __init__(self, bert, opt):
        super(BertAmir4, self).__init__()
        # self.squeeze_embedding = SqueezeEmbedding()
        self.device = opt.device
        self.bert = bert
        self.dropout = nn.Dropout(opt.dropout)
        self.dense = nn.Linear(2*2 * opt.hidden_dim, opt.polarities_dim)
        # self.dense = nn.Linear(2*opt.hidden_dim, opt.polarities_dim)
        self.n_layer = opt.n_layer
        self.hidden_dim= opt.hidden_dim
        self.text_lstm = DynamicLSTM(opt.bert_dim * self.n_layer, opt.hidden_dim, num_layers=1,
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

    def forward(self, inputs):
        """
        bert_length: L
        original_length: T
        """
        B = inputs['sentence_length'].shape[0]
        L = inputs['cls_text_sep_length'].max().cpu().numpy().tolist()
        T = inputs['sentence_length'].max().cpu().numpy().tolist()

        text_bert_indices = inputs['cls_text_sep_indices'][:, :L]  # B x L
        bert_segments_ids = inputs['cls_text_sep_segments_ids'][:, :L]  # B x L
        transform = inputs['transform'][:, :T, :L]  # B x T x L
        anchor_index = inputs['anchor_index']  # B
        dist_to_target = inputs['dist_to_target'][:, :T]  # B x T

        adj = inputs['dependency_graph'][:, :T, :T]   # B x T x T

        x, pooled_output = self.bert(text_bert_indices,
                                     bert_segments_ids,
                                     output_all_encoded_layers=True)
        # print('| bert_amir4 > x[0] ', tuple(x[0].shape))

        x = torch.cat(x[-self.n_layer:], dim=-1)
        # print('| bert_amir4 > x ', tuple(x.shape))
        # print('| bert_amir4 > pooled_output ', tuple(pooled_output.shape))
        # print('| bert_amir4 > x ', tuple(x.shape))
        x = torch.bmm(transform, x)  # B x T x D

        # print('| bert_amir4 > x(bmm)', tuple(x.shape))
        # print('| bert_amir4 > inputs[sentence_length]', tuple(inputs['sentence_length'].shape))


        x, (_, _) = self.text_lstm(x, inputs['sentence_length'])

        # print('| bert_amir4 > anchor_index', tuple(anchor_index.shape))

        mask = torch.LongTensor([i for i in range(T)]).to(self.device).repeat(B, 1)  # B x T
        # print('| bert_amir4 > mask', tuple(mask.shape))

        mask = mask == anchor_index.unsqueeze(1)
        mask = mask.unsqueeze(dim=2).expand(-1, -1, self.hidden_dim * 2)
        # print('| bert_amir4 > mask', tuple(mask.shape))

        aspect = torch.masked_select(x, mask).view(B, -1)
        # print('| bert_amir4 > aspect', tuple(aspect.shape))


        # ANchor extract

        gate1 = self.gate1(aspect).repeat(1, x.shape[1]).view(x.shape)
        gate2 = self.gate2(aspect).repeat(1, x.shape[1]).view(x.shape)

        gate1 = self.dropout(gate1)
        gate2 = self.dropout(gate2)
        gcn1 = self.gc1(x, adj)
        gcngate1 = gcn1 * gate1

        # print('| bert_amir4 > gcngate1', tuple(gcngate1.shape))

        gcngate2 = gcn1 * gate2

        # print('| bert_amir4 > gcngate2', tuple(gcngate2.shape))

        x1 = torch.max(gcngate1, 1)[0]
        y1 = torch.max(gcngate2, 1)[0]
        sf1 = nn.Softmax(1)
        xy = (x1 * y1).sum(1).mean()
        x = gate2 * self.gc2(gcn1, adj)
        out = torch.max(x, dim=1)[0]
        # pooled_output = self.dropout(pooled_output)
        out = self.dropout(out)
        logits = self.dense(torch.cat([aspect, out], dim=1))

        output_w = self.fc(torch.cat([x, aspect.repeat(1, x.shape[1]).view(x.shape)], dim=2))
        scores = (logits.repeat(1, x.shape[1]).view(x.shape[0], x.shape[1], -1) * output_w).sum(2)
        # sf2 = nn.Softmax(2)
        kl = (sf1(scores) * sf1(dist_to_target.float())).sum(1).mean()

        return logits, xy, kl
        # return logits, xy, 0
        # return logits, 0, 0
