import torch
import torch.nn as nn
from layers.squeeze_embedding import SqueezeEmbedding
from layers.dynamic_rnn import DynamicLSTM
from models.gcn import GraphConvolution

INFINITY_NUMBER = 1e12


class BertAmir5(nn.Module):
    def __init__(self, bert, opt):
        super(BertAmir5, self).__init__()
        # self.squeeze_embedding = SqueezeEmbedding()
        self.device = opt.device
        self.bert = bert
        self.dropout = nn.Dropout(opt.dropout)
        self.hidden_dim = hidden_dim = 128

        self.dense = nn.Linear(2 * 2 * hidden_dim + 768, opt.polarities_dim)
        # self.dense = nn.Linear(2*hidden_dim, opt.polarities_dim)
        self.n_layer = 12
        self.lstm = nn.LSTM(self.n_layer * 768, hidden_dim, bidirectional=True, batch_first=True, num_layers=1)

        self.gc1 = GraphConvolution(2 * hidden_dim, 2 * hidden_dim, opt)
        self.gc2 = GraphConvolution(2 * hidden_dim, 2 * hidden_dim, opt)

        self.gate1 = nn.Sequential(nn.Linear(hidden_dim * 2, hidden_dim * 2), nn.Sigmoid(),
                                   nn.Linear(hidden_dim * 2, hidden_dim * 2), nn.Sigmoid())
        self.gate2 = nn.Sequential(nn.Linear(hidden_dim * 2, hidden_dim * 2), nn.Sigmoid(),
                                   nn.Linear(hidden_dim * 2, hidden_dim * 2), nn.Sigmoid())

        self.fc = nn.Linear(2 * 2 * hidden_dim, opt.polarities_dim)

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

        adj = inputs['dependency_graph'][:, :T, :T]  # B x T x T

        x, pooled_output = self.bert(text_bert_indices,
                                     bert_segments_ids,
                                     output_all_encoded_layers=True)
        # print('| bert_amir5 > x[0] ', tuple(x[0].shape))

        x = torch.cat(x[-self.n_layer:], dim=-1)
        # print('| bert_amir5 > x ', tuple(x.shape))
        # print('| bert_amir5 > pooled_output ', tuple(pooled_output.shape))
        # print('| bert_amir5 > x ', tuple(x.shape))
        x = torch.bmm(transform, x)  # B x T x D

        # print('| bert_amir5 > x(bmm)', tuple(x.shape))
        # print('| bert_amir5 > inputs[sentence_length]', tuple(inputs['sentence_length'].shape))

        x, _ = self.lstm(x)

        # print('| bert_amir5 > anchor_index', tuple(anchor_index.shape))

        mask = torch.LongTensor([i for i in range(T)]).to(self.device).repeat(B, 1)  # B x T
        # print('| bert_amir5 > mask', tuple(mask.shape))

        mask = mask == anchor_index.unsqueeze(1)
        mask = mask.unsqueeze(dim=2).expand(-1, -1, self.hidden_dim * 2)
        # print('| bert_amir5 > mask', tuple(mask.shape))

        aspect = torch.masked_select(x, mask).view(B, -1)
        # print('| bert_amir5 > aspect', tuple(aspect.shape))

        gate1 = self.gate1(aspect).repeat(1, x.shape[1]).view(x.shape)
        gate2 = self.gate2(aspect).repeat(1, x.shape[1]).view(x.shape)

        gate1 = self.dropout(gate1)
        gate2 = self.dropout(gate2)
        gcn1 = self.gc1(x, adj)
        gcngate1 = gcn1 * gate1

        # print('| bert_amir5 > gcngate1', tuple(gcngate1.shape))

        gcngate2 = gcn1 * gate2

        # print('| bert_amir5 > gcngate2', tuple(gcngate2.shape))

        x1 = torch.max(gcngate1, 1)[0]
        y1 = torch.max(gcngate2, 1)[0]
        sf1 = nn.Softmax(1)
        xy = (x1 * y1).sum(1).mean()
        x = gate2 * self.gc2(gcn1, adj)
        out = torch.max(x, dim=1)[0]
        pooled_output = self.dropout(pooled_output)
        out = self.dropout(out)
        logits = self.dense(torch.cat([aspect, out, pooled_output], dim=1))

        output_w = self.fc(torch.cat([x, aspect.repeat(1, x.shape[1]).view(x.shape)], dim=2))
        scores = (logits.repeat(1, x.shape[1]).view(x.shape[0], x.shape[1], -1) * output_w).sum(2)

        kl = (sf1(scores) * sf1(dist_to_target.float())).sum(1).mean()

        return logits, xy, kl


class BertAmir51(nn.Module):
    """
    No pooled output
    Hidden dim 100
    """

    def __init__(self, bert, opt):
        super(BertAmir51, self).__init__()
        # self.squeeze_embedding = SqueezeEmbedding()
        self.device = opt.device
        self.bert = bert
        self.dropout = nn.Dropout(opt.dropout)
        self.hidden_dim = hidden_dim = 100

        self.dense = nn.Linear(2 * 2 * hidden_dim, opt.polarities_dim)
        # self.dense = nn.Linear(2*hidden_dim, opt.polarities_dim)
        self.n_layer = 12
        self.lstm = nn.LSTM(self.n_layer * 768, hidden_dim, bidirectional=True, batch_first=True, num_layers=1)

        self.gc1 = GraphConvolution(2 * hidden_dim, 2 * hidden_dim, opt)
        self.gc2 = GraphConvolution(2 * hidden_dim, 2 * hidden_dim, opt)

        self.gate1 = nn.Sequential(nn.Linear(hidden_dim * 2, hidden_dim * 2), nn.Sigmoid(),
                                   nn.Linear(hidden_dim * 2, hidden_dim * 2), nn.Sigmoid())
        self.gate2 = nn.Sequential(nn.Linear(hidden_dim * 2, hidden_dim * 2), nn.Sigmoid(),
                                   nn.Linear(hidden_dim * 2, hidden_dim * 2), nn.Sigmoid())

        self.fc = nn.Linear(2 * 2 * hidden_dim, opt.polarities_dim)

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

        adj = inputs['dependency_graph'][:, :T, :T]  # B x T x T

        x, pooled_output = self.bert(text_bert_indices,
                                     bert_segments_ids,
                                     output_all_encoded_layers=True)
        # print('| bert_amir5 > x[0] ', tuple(x[0].shape))

        x = torch.cat(x[-self.n_layer:], dim=-1)
        # print('| bert_amir5 > x ', tuple(x.shape))
        # print('| bert_amir5 > pooled_output ', tuple(pooled_output.shape))
        # print('| bert_amir5 > x ', tuple(x.shape))
        x = torch.bmm(transform, x)  # B x T x D

        # print('| bert_amir5 > x(bmm)', tuple(x.shape))
        # print('| bert_amir5 > inputs[sentence_length]', tuple(inputs['sentence_length'].shape))

        x, _ = self.lstm(x)

        # print('| bert_amir5 > anchor_index', tuple(anchor_index.shape))

        mask = torch.LongTensor([i for i in range(T)]).to(self.device).repeat(B, 1)  # B x T
        # print('| bert_amir5 > mask', tuple(mask.shape))

        mask = mask == anchor_index.unsqueeze(1)
        mask = mask.unsqueeze(dim=2).expand(-1, -1, self.hidden_dim * 2)
        # print('| bert_amir5 > mask', tuple(mask.shape))

        aspect = torch.masked_select(x, mask).view(B, -1)
        # print('| bert_amir5 > aspect', tuple(aspect.shape))

        gate1 = self.gate1(aspect).repeat(1, x.shape[1]).view(x.shape)
        gate2 = self.gate2(aspect).repeat(1, x.shape[1]).view(x.shape)

        gate1 = self.dropout(gate1)
        gate2 = self.dropout(gate2)
        gcn1 = self.gc1(x, adj)
        gcngate1 = gcn1 * gate1

        # print('| bert_amir5 > gcngate1', tuple(gcngate1.shape))

        gcngate2 = gcn1 * gate2

        # print('| bert_amir5 > gcngate2', tuple(gcngate2.shape))

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

        kl = (sf1(scores) * sf1(dist_to_target.float())).sum(1).mean()

        return logits, xy, kl


class BertAmir52(nn.Module):
    def __init__(self, bert, opt):
        super(BertAmir52, self).__init__()
        # self.squeeze_embedding = SqueezeEmbedding()
        self.device = opt.device
        self.bert = bert
        self.dropout = nn.Dropout(opt.dropout)
        self.hidden_dim = hidden_dim = 128

        self.dense = nn.Linear(2 * 2 * hidden_dim + 768, opt.polarities_dim)
        # self.dense = nn.Linear(2*hidden_dim, opt.polarities_dim)
        self.n_layer = 12
        self.lstm = nn.LSTM(self.n_layer * 768, hidden_dim, bidirectional=True, batch_first=True, num_layers=1)

        self.gc1 = GraphConvolution(2 * hidden_dim, 2 * hidden_dim, opt)
        self.gc2 = GraphConvolution(2 * hidden_dim, 2 * hidden_dim, opt)

        self.gate1 = nn.Sequential(nn.Sigmoid(),
                                   nn.Linear(hidden_dim * 2, hidden_dim * 2),
                                   nn.Sigmoid(),
                                   nn.Linear(hidden_dim * 2, hidden_dim * 2),
                                   nn.Sigmoid())
        self.gate2 = nn.Sequential(nn.Sigmoid(),
                                   nn.Linear(hidden_dim * 2, hidden_dim * 2),
                                   nn.Sigmoid(),
                                   nn.Linear(hidden_dim * 2, hidden_dim * 2),
                                   nn.Sigmoid())

        self.fc = nn.Linear(2 * 2 * hidden_dim, opt.polarities_dim)

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

        adj = inputs['dependency_graph'][:, :T, :T]  # B x T x T

        x, pooled_output = self.bert(text_bert_indices,
                                     bert_segments_ids,
                                     output_all_encoded_layers=True)
        # print('| bert_amir5 > x[0] ', tuple(x[0].shape))

        x = torch.cat(x[-self.n_layer:], dim=-1)
        # print('| bert_amir5 > x ', tuple(x.shape))
        # print('| bert_amir5 > pooled_output ', tuple(pooled_output.shape))
        # print('| bert_amir5 > x ', tuple(x.shape))
        x = torch.bmm(transform, x)  # B x T x D

        # print('| bert_amir5 > x(bmm)', tuple(x.shape))
        # print('| bert_amir5 > inputs[sentence_length]', tuple(inputs['sentence_length'].shape))

        x, _ = self.lstm(x)

        # print('| bert_amir5 > anchor_index', tuple(anchor_index.shape))

        mask = torch.LongTensor([i for i in range(T)]).to(self.device).repeat(B, 1)  # B x T
        # print('| bert_amir5 > mask', tuple(mask.shape))

        mask = mask == anchor_index.unsqueeze(1)
        mask = mask.unsqueeze(dim=2).expand(-1, -1, self.hidden_dim * 2)
        # print('| bert_amir5 > mask', tuple(mask.shape))

        anchor_rep = torch.masked_select(x, mask).view(B, -1)
        # print('| bert_amir5 > aspect', tuple(aspect.shape))

        gate1 = self.gate1(anchor_rep).repeat(1, x.shape[1]).view(x.shape)
        gate2 = self.gate2(anchor_rep).repeat(1, x.shape[1]).view(x.shape)

        gate1 = self.dropout(gate1)
        gate2 = self.dropout(gate2)
        gcn1 = self.gc1(x, adj)
        gcngate1 = gcn1 * gate1

        # print('| bert_amir5 > gcngate1', tuple(gcngate1.shape))

        gcngate2 = gcn1 * gate2

        # print('| bert_amir5 > gcngate2', tuple(gcngate2.shape))

        x1 = torch.max(gcngate1, 1)[0]
        y1 = torch.max(gcngate2, 1)[0]
        sf1 = nn.Softmax(1)
        xy = (x1 * y1).sum(1).mean()
        x = gate2 * self.gc2(gcn1, adj)
        out = torch.max(x, dim=1)[0]
        pooled_output = self.dropout(pooled_output)
        out = self.dropout(out)
        logits = self.dense(torch.cat([anchor_rep, out, pooled_output], dim=1))

        output_w = self.fc(torch.cat([x, anchor_rep.repeat(1, x.shape[1]).view(x.shape)], dim=2))
        scores = (logits.repeat(1, x.shape[1]).view(x.shape[0], x.shape[1], -1) * output_w).sum(2)

        kl = (sf1(scores) * sf1(dist_to_target.float())).sum(1).mean()

        return logits, xy, kl


class BertAmir53(nn.Module):
    def __init__(self, bert, opt):
        super(BertAmir53, self).__init__()
        # self.squeeze_embedding = SqueezeEmbedding()
        self.device = opt.device
        self.bert = bert
        self.dropout = nn.Dropout(opt.dropout)
        self.hidden_dim = hidden_dim = 128

        self.dense = nn.Linear(2 * 2 * hidden_dim + 768, opt.polarities_dim)
        # self.dense = nn.Linear(2*hidden_dim, opt.polarities_dim)
        self.n_layer = 12
        self.lstm = nn.LSTM(self.n_layer * 768, hidden_dim, bidirectional=True, batch_first=True, num_layers=1)

        self.gc1 = GraphConvolution(2 * hidden_dim, 2 * hidden_dim, opt)
        self.gc2 = GraphConvolution(2 * hidden_dim, 2 * hidden_dim, opt)

        self.gate1 = nn.Sequential(nn.Sigmoid(),
                                   nn.Linear(hidden_dim * 2, hidden_dim * 2),
                                   nn.Sigmoid(),
                                   nn.Linear(hidden_dim * 2, hidden_dim * 2),
                                   nn.Sigmoid())
        self.gate2 = nn.Sequential(nn.Sigmoid(),
                                   nn.Linear(hidden_dim * 2, hidden_dim * 2),
                                   nn.Sigmoid(),
                                   nn.Linear(hidden_dim * 2, hidden_dim * 2),
                                   nn.Sigmoid())

        self.fc1 = nn.Linear(2 * hidden_dim + 768, 2 * hidden_dim)
        self.fc2 = nn.Linear(2 * 2 * hidden_dim, opt.polarities_dim)

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

        adj = inputs['dependency_graph'][:, :T, :T]  # B x T x T

        x, pooled_output = self.bert(text_bert_indices,
                                     bert_segments_ids,
                                     output_all_encoded_layers=True)
        # print('| bert_amir5 > x[0] ', tuple(x[0].shape))

        x = torch.cat(x[-self.n_layer:], dim=-1)
        # print('| bert_amir5 > x ', tuple(x.shape))
        # print('| bert_amir5 > pooled_output ', tuple(pooled_output.shape))
        # print('| bert_amir5 > x ', tuple(x.shape))
        x = torch.bmm(transform, x)  # B x T x D

        # print('| bert_amir5 > x(bmm)', tuple(x.shape))
        # print('| bert_amir5 > inputs[sentence_length]', tuple(inputs['sentence_length'].shape))

        x, _ = self.lstm(x)

        # print('| bert_amir5 > anchor_index', tuple(anchor_index.shape))

        mask = torch.LongTensor([i for i in range(T)]).to(self.device).repeat(B, 1)  # B x T
        # print('| bert_amir5 > mask', tuple(mask.shape))

        mask = mask == anchor_index.unsqueeze(1)
        mask = mask.unsqueeze(dim=2).expand(-1, -1, self.hidden_dim * 2)
        # print('| bert_amir5 > mask', tuple(mask.shape))

        anchor_rep = torch.masked_select(x, mask).view(B, -1)
        # print('| bert_amir5 > aspect', tuple(aspect.shape))

        aspect = self.fc1(torch.cat([anchor_rep, pooled_output], dim=1))

        gate1 = self.gate1(aspect).repeat(1, x.shape[1]).view(x.shape)
        gate2 = self.gate2(aspect).repeat(1, x.shape[1]).view(x.shape)

        gate1 = self.dropout(gate1)
        gate2 = self.dropout(gate2)
        gcn1 = self.gc1(x, adj)
        gcngate1 = gcn1 * gate1

        # print('| bert_amir5 > gcngate1', tuple(gcngate1.shape))

        gcngate2 = gcn1 * gate2

        # print('| bert_amir5 > gcngate2', tuple(gcngate2.shape))

        x1 = torch.max(gcngate1, 1)[0]
        y1 = torch.max(gcngate2, 1)[0]
        sf1 = nn.Softmax(1)
        xy = (x1 * y1).sum(1).mean()
        x = gate2 * self.gc2(gcn1, adj)
        out = torch.max(x, dim=1)[0]
        pooled_output = self.dropout(pooled_output)
        out = self.dropout(out)
        logits = self.dense(torch.cat([anchor_rep, out, pooled_output], dim=1))

        output_w = self.fc2(torch.cat([x, aspect.repeat(1, x.shape[1]).view(x.shape)], dim=2))
        scores = (logits.repeat(1, x.shape[1]).view(x.shape[0], x.shape[1], -1) * output_w).sum(2)

        kl = (sf1(scores) * sf1(dist_to_target.float())).sum(1).mean()

        return logits, xy, kl


class BertAmir54(nn.Module):
    def __init__(self, bert, opt):
        super(BertAmir54, self).__init__()
        # self.squeeze_embedding = SqueezeEmbedding()
        self.device = opt.device
        self.bert = bert
        self.dropout = nn.Dropout(opt.dropout)
        self.hidden_dim = hidden_dim = 128

        self.dense = nn.Sequential(
            nn.Linear(2 * 2 * hidden_dim + 768, 768),
            nn.Linear(768, opt.polarities_dim)
        )
        # self.dense = nn.Linear(2*hidden_dim, opt.polarities_dim)
        self.n_layer = 12
        self.lstm = nn.LSTM(self.n_layer * 768, hidden_dim, bidirectional=True, batch_first=True, num_layers=1)

        self.gc1 = GraphConvolution(2 * hidden_dim, 2 * hidden_dim, opt)
        self.gc2 = GraphConvolution(2 * hidden_dim, 2 * hidden_dim, opt)

        self.gate1 = nn.Sequential(nn.Sigmoid(),
                                   nn.Linear(hidden_dim * 2, hidden_dim * 2),
                                   nn.Sigmoid(),
                                   nn.Linear(hidden_dim * 2, hidden_dim * 2),
                                   nn.Sigmoid())
        self.gate2 = nn.Sequential(nn.Sigmoid(),
                                   nn.Linear(hidden_dim * 2, hidden_dim * 2),
                                   nn.Sigmoid(),
                                   nn.Linear(hidden_dim * 2, hidden_dim * 2),
                                   nn.Sigmoid())
        self.fc = nn.Sequential(nn.Sigmoid(),
                                nn.Linear(2 * 2 * hidden_dim, opt.polarities_dim))

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

        adj = inputs['dependency_graph'][:, :T, :T]  # B x T x T

        x, pooled_output = self.bert(text_bert_indices,
                                     bert_segments_ids,
                                     output_all_encoded_layers=True)
        # print('| bert_amir5 > x[0] ', tuple(x[0].shape))

        x = torch.cat(x[-self.n_layer:], dim=-1)
        # print('| bert_amir5 > x ', tuple(x.shape))
        # print('| bert_amir5 > pooled_output ', tuple(pooled_output.shape))
        # print('| bert_amir5 > x ', tuple(x.shape))
        x = torch.bmm(transform, x)  # B x T x D

        # print('| bert_amir5 > x(bmm)', tuple(x.shape))
        # print('| bert_amir5 > inputs[sentence_length]', tuple(inputs['sentence_length'].shape))

        x, _ = self.lstm(x)

        # print('| bert_amir5 > anchor_index', tuple(anchor_index.shape))

        mask = torch.LongTensor([i for i in range(T)]).to(self.device).repeat(B, 1)  # B x T
        # print('| bert_amir5 > mask', tuple(mask.shape))

        mask = mask == anchor_index.unsqueeze(1)
        mask = mask.unsqueeze(dim=2).expand(-1, -1, self.hidden_dim * 2)
        # print('| bert_amir5 > mask', tuple(mask.shape))

        aspect = torch.masked_select(x, mask).view(B, -1)
        # print('| bert_amir5 > aspect', tuple(aspect.shape))

        gate1 = self.gate1(aspect).repeat(1, x.shape[1]).view(x.shape)
        gate2 = self.gate2(aspect).repeat(1, x.shape[1]).view(x.shape)

        gate1 = self.dropout(gate1)
        gate2 = self.dropout(gate2)
        gcn1 = self.gc1(x, adj)
        gcngate1 = gcn1 * gate1

        # print('| bert_amir5 > gcngate1', tuple(gcngate1.shape))

        gcngate2 = gcn1 * gate2

        # print('| bert_amir5 > gcngate2', tuple(gcngate2.shape))

        x1 = torch.max(gcngate1, 1)[0]
        y1 = torch.max(gcngate2, 1)[0]
        sf1 = nn.Softmax(1)
        xy = (x1 * y1).sum(1).mean()
        x = gate2 * self.gc2(gcn1, adj)
        out = torch.max(x, dim=1)[0]
        pooled_output = self.dropout(pooled_output)
        out = self.dropout(out)
        logits = self.dense(torch.cat([aspect, out, pooled_output], dim=1))

        output_w = self.fc(torch.cat([x, aspect.repeat(1, x.shape[1]).view(x.shape)], dim=2))
        scores = (logits.repeat(1, x.shape[1]).view(x.shape[0], x.shape[1], -1) * output_w).sum(2)

        kl = (sf1(scores) * sf1(dist_to_target.float())).sum(1).mean()

        return logits, xy, kl, scores


class BertAmir55(nn.Module):
    def __init__(self, bert, opt):
        super(BertAmir55, self).__init__()
        # self.squeeze_embedding = SqueezeEmbedding()
        self.device = opt.device
        self.bert = bert
        self.dropout = nn.Dropout(opt.dropout)
        self.hidden_dim = hidden_dim = 128
        self.n_layer = 12


        self.dense = nn.Linear(2*2 * hidden_dim + 768*self.n_layer, opt.polarities_dim)
        # self.dense = nn.Linear(2*hidden_dim, opt.polarities_dim)
        self.lstm = nn.LSTM(self.n_layer * 768, hidden_dim, bidirectional=True, batch_first=True, num_layers=1)

        self.gc1 = GraphConvolution(2 * hidden_dim, 2 * hidden_dim, opt)
        self.gc2 = GraphConvolution(2 * hidden_dim, 2 * hidden_dim, opt)

        self.gate1 = nn.Sequential(nn.Sigmoid(),
                                   nn.Linear(hidden_dim * 2, hidden_dim * 2),
                                   nn.Sigmoid(),
                                   nn.Linear(hidden_dim * 2, hidden_dim * 2),
                                   nn.Sigmoid())
        self.gate2 = nn.Sequential(nn.Sigmoid(),
                                   nn.Linear(hidden_dim * 2, hidden_dim * 2),
                                   nn.Sigmoid(),
                                   nn.Linear(hidden_dim * 2, hidden_dim * 2),
                                   nn.Sigmoid())
        self.fc = nn.Sequential(nn.Linear(2 * 2 * hidden_dim, opt.polarities_dim))

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

        adj = inputs['dependency_graph'][:, :T, :T]  # B x T x T

        x, pooled_output = self.bert(text_bert_indices,
                                     bert_segments_ids,
                                     output_all_encoded_layers=True)
        # print('| bert_amir5 > x[0] ', tuple(x[0].shape))

        x = torch.cat(x[-self.n_layer:], dim=-1)
        # print('| bert_amir5 > x ', tuple(x.shape))
        # print('| bert_amir5 > pooled_output ', tuple(pooled_output.shape))
        # print('| bert_amir5 > x ', tuple(x.shape))
        x = torch.bmm(transform, x)  # B x T x D

        # print('| bert_amir5 > x(bmm)', tuple(x.shape))
        # print('| bert_amir5 > inputs[sentence_length]', tuple(inputs['sentence_length'].shape))
        mask = torch.LongTensor([i for i in range(T)]).to(self.device).repeat(B, 1)  # B x T
        mask = mask == anchor_index.unsqueeze(1)
        bert_mask = mask.unsqueeze(dim=2).expand(-1, -1, 768 * self.n_layer)
        anchor_rep = torch.masked_select(x, bert_mask).view(B, -1)
        anchor_rep=self.dropout(anchor_rep)

        x, _ = self.lstm(x)

        # print('| bert_amir5 > anchor_index', tuple(anchor_index.shape))


        lstm_mask = mask.unsqueeze(dim=2).expand(-1, -1, self.hidden_dim * 2)
        # print('| bert_amir5 > mask', tuple(mask.shape))

        aspect = torch.masked_select(x, lstm_mask).view(B, -1)
        # print('| bert_amir5 > aspect', tuple(aspect.shape))

        gate1 = self.gate1(aspect).repeat(1, x.shape[1]).view(x.shape)
        gate2 = self.gate2(aspect).repeat(1, x.shape[1]).view(x.shape)

        gate1 = self.dropout(gate1)
        gate2 = self.dropout(gate2)
        gcn1 = self.gc1(x, adj)
        gcngate1 = gcn1 * gate1

        # print('| bert_amir5 > gcngate1', tuple(gcngate1.shape))

        gcngate2 = gcn1 * gate2

        # print('| bert_amir5 > gcngate2', tuple(gcngate2.shape))

        x1 = torch.max(gcngate1, 1)[0]
        y1 = torch.max(gcngate2, 1)[0]
        sf1 = nn.Softmax(1)
        xy = (x1 * y1).sum(1).mean()
        x = gate2 * self.gc2(gcn1, adj)
        out = torch.max(x, dim=1)[0]
        pooled_output = self.dropout(pooled_output)
        out = self.dropout(out)
        logits = self.dense(torch.cat([anchor_rep, aspect, out], dim=1))

        output_w = self.fc(torch.cat([x, aspect.repeat(1, x.shape[1]).view(x.shape)], dim=2))
        scores = (logits.repeat(1, x.shape[1]).view(x.shape[0], x.shape[1], -1) * output_w).sum(2)

        kl = (sf1(scores) * sf1(dist_to_target.float())).sum(1).mean()

        return logits, xy, kl, scores



class BertAmir55NoGate(nn.Module):
    def __init__(self, bert, opt):
        super(BertAmir55NoGate, self).__init__()
        # self.squeeze_embedding = SqueezeEmbedding()
        self.device = opt.device
        self.bert = bert
        self.dropout = nn.Dropout(opt.dropout)
        self.hidden_dim = hidden_dim = 128
        self.n_layer = 12


        self.dense = nn.Linear(2*2 * hidden_dim + 768*self.n_layer, opt.polarities_dim)
        # self.dense = nn.Linear(2*hidden_dim, opt.polarities_dim)
        self.lstm = nn.LSTM(self.n_layer * 768, hidden_dim, bidirectional=True, batch_first=True, num_layers=1)

        self.gc1 = GraphConvolution(2 * hidden_dim, 2 * hidden_dim, opt)
        self.gc2 = GraphConvolution(2 * hidden_dim, 2 * hidden_dim, opt)

        self.gate1 = nn.Sequential(nn.Sigmoid(),
                                   nn.Linear(hidden_dim * 2, hidden_dim * 2),
                                   nn.Sigmoid(),
                                   nn.Linear(hidden_dim * 2, hidden_dim * 2),
                                   nn.Sigmoid())
        self.gate2 = nn.Sequential(nn.Sigmoid(),
                                   nn.Linear(hidden_dim * 2, hidden_dim * 2),
                                   nn.Sigmoid(),
                                   nn.Linear(hidden_dim * 2, hidden_dim * 2),
                                   nn.Sigmoid())
        self.fc = nn.Sequential(nn.Linear(2 * 2 * hidden_dim, opt.polarities_dim))

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

        adj = inputs['dependency_graph'][:, :T, :T]  # B x T x T

        x, pooled_output = self.bert(text_bert_indices,
                                     bert_segments_ids,
                                     output_all_encoded_layers=True)
        # print('| bert_amir5 > x[0] ', tuple(x[0].shape))

        x = torch.cat(x[-self.n_layer:], dim=-1)
        # print('| bert_amir5 > x ', tuple(x.shape))
        # print('| bert_amir5 > pooled_output ', tuple(pooled_output.shape))
        # print('| bert_amir5 > x ', tuple(x.shape))
        x = torch.bmm(transform, x)  # B x T x D

        # print('| bert_amir5 > x(bmm)', tuple(x.shape))
        # print('| bert_amir5 > inputs[sentence_length]', tuple(inputs['sentence_length'].shape))
        mask = torch.LongTensor([i for i in range(T)]).to(self.device).repeat(B, 1)  # B x T
        mask = mask == anchor_index.unsqueeze(1)
        bert_mask = mask.unsqueeze(dim=2).expand(-1, -1, 768 * self.n_layer)
        anchor_rep = torch.masked_select(x, bert_mask).view(B, -1)
        anchor_rep=self.dropout(anchor_rep)

        x, _ = self.lstm(x)

        # print('| bert_amir5 > anchor_index', tuple(anchor_index.shape))


        lstm_mask = mask.unsqueeze(dim=2).expand(-1, -1, self.hidden_dim * 2)
        # print('| bert_amir5 > mask', tuple(mask.shape))

        aspect = torch.masked_select(x, lstm_mask).view(B, -1)
        # print('| bert_amir5 > aspect', tuple(aspect.shape))

        # gate1 = self.gate1(aspect).repeat(1, x.shape[1]).view(x.shape)
        # gate2 = self.gate2(aspect).repeat(1, x.shape[1]).view(x.shape)
        #
        # gate1 = self.dropout(gate1)
        # gate2 = self.dropout(gate2)
        gcn1 = self.gc1(x, adj)
        # gcngate1 = gcn1 * gate1

        # print('| bert_amir5 > gcngate1', tuple(gcngate1.shape))

        # gcngate2 = gcn1 * gate2

        # print('| bert_amir5 > gcngate2', tuple(gcngate2.shape))

        # x1 = torch.max(gcngate1, 1)[0]
        # y1 = torch.max(gcngate2, 1)[0]
        sf1 = nn.Softmax(1)
        # xy = (x1 * y1).sum(1).mean()
        x = self.gc2(gcn1, adj)
        out = torch.max(x, dim=1)[0]
        # pooled_output = self.dropout(pooled_output)
        out = self.dropout(out)
        logits = self.dense(torch.cat([anchor_rep, aspect, out], dim=1))

        output_w = self.fc(torch.cat([x, aspect.repeat(1, x.shape[1]).view(x.shape)], dim=2))
        scores = (logits.repeat(1, x.shape[1]).view(x.shape[0], x.shape[1], -1) * output_w).sum(2)

        kl = (sf1(scores) * sf1(dist_to_target.float())).sum(1).mean()

        return logits, 0.0, kl, scores


#
#
# class BertAmir56(nn.Module):
#     def __init__(self, bert, opt):
#         super(BertAmir56, self).__init__()
#         # self.squeeze_embedding = SqueezeEmbedding()
#         self.device = opt.device
#         self.bert = bert
#         self.dropout = nn.Dropout(opt.dropout)
#         self.hidden_dim = hidden_dim = 128
#         self.n_layer = 12
#
#
#         self.dense = nn.Linear(2* hidden_dim + 768*self.n_layer, opt.polarities_dim)
#         # self.dense = nn.Linear(2*hidden_dim, opt.polarities_dim)
#         self.lstm = nn.LSTM(self.n_layer * 768, hidden_dim, bidirectional=True, batch_first=True, num_layers=1)
#
#         self.gc1 = GraphConvolution(2 * hidden_dim, 2 * hidden_dim, opt)
#         self.gc2 = GraphConvolution(2 * hidden_dim, 2 * hidden_dim, opt)
#
#         self.gate1 = nn.Sequential(nn.Sigmoid(),
#                                    nn.Linear(hidden_dim * 2, hidden_dim * 2),
#                                    nn.Sigmoid(),
#                                    nn.Linear(hidden_dim * 2, hidden_dim * 2),
#                                    nn.Sigmoid())
#         self.gate2 = nn.Sequential(nn.Sigmoid(),
#                                    nn.Linear(hidden_dim * 2, hidden_dim * 2),
#                                    nn.Sigmoid(),
#                                    nn.Linear(hidden_dim * 2, hidden_dim * 2),
#                                    nn.Sigmoid())
#         self.fc = nn.Sequential(nn.Linear(2 * 2 * hidden_dim, opt.polarities_dim))
#
#     def forward(self, inputs):
#         """
#         bert_length: L
#         original_length: T
#         """
#         B = inputs['sentence_length'].shape[0]
#         L = inputs['cls_text_sep_length'].max().cpu().numpy().tolist()
#         T = inputs['sentence_length'].max().cpu().numpy().tolist()
#
#         text_bert_indices = inputs['cls_text_sep_indices'][:, :L]  # B x L
#         bert_segments_ids = inputs['cls_text_sep_segments_ids'][:, :L]  # B x L
#         transform = inputs['transform'][:, :T, :L]  # B x T x L
#         anchor_index = inputs['anchor_index']  # B
#         dist_to_target = inputs['dist_to_target'][:, :T]  # B x T
#
#         adj = inputs['dependency_graph'][:, :T, :T]  # B x T x T
#
#         x, pooled_output = self.bert(text_bert_indices,
#                                      bert_segments_ids,
#                                      output_all_encoded_layers=True)
#         # print('| bert_amir5 > x[0] ', tuple(x[0].shape))
#
#         x = torch.cat(x[-self.n_layer:], dim=-1)
#         # print('| bert_amir5 > x ', tuple(x.shape))
#         # print('| bert_amir5 > pooled_output ', tuple(pooled_output.shape))
#         # print('| bert_amir5 > x ', tuple(x.shape))
#         x = torch.bmm(transform, x)  # B x T x D
#
#         # print('| bert_amir5 > x(bmm)', tuple(x.shape))
#         # print('| bert_amir5 > inputs[sentence_length]', tuple(inputs['sentence_length'].shape))
#         mask = torch.LongTensor([i for i in range(T)]).to(self.device).repeat(B, 1)  # B x T
#         mask = mask == anchor_index.unsqueeze(1)
#         bert_mask = mask.unsqueeze(dim=2).expand(-1, -1, 768 * self.n_layer)
#         anchor_rep = torch.masked_select(x, bert_mask).view(B, -1)
#         anchor_rep=self.dropout(anchor_rep)
#
#         x, _ = self.lstm(x)
#
#         # print('| bert_amir5 > anchor_index', tuple(anchor_index.shape))
#
#
#         lstm_mask = mask.unsqueeze(dim=2).expand(-1, -1, self.hidden_dim * 2)
#         # print('| bert_amir5 > mask', tuple(mask.shape))
#
#         aspect = torch.masked_select(x, lstm_mask).view(B, -1)
#         # print('| bert_amir5 > aspect', tuple(aspect.shape))
#
#         gate1 = self.gate1(aspect).repeat(1, x.shape[1]).view(x.shape)
#         gate2 = self.gate2(aspect).repeat(1, x.shape[1]).view(x.shape)
#
#         gate1 = self.dropout(gate1)
#         gate2 = self.dropout(gate2)
#         gcn1 = self.gc1(x, adj)
#         gcngate1 = gcn1 * gate1
#
#         # print('| bert_amir5 > gcngate1', tuple(gcngate1.shape))
#
#         gcngate2 = gcn1 * gate2
#
#         # print('| bert_amir5 > gcngate2', tuple(gcngate2.shape))
#
#         x1 = torch.max(gcngate1, 1)[0]
#         y1 = torch.max(gcngate2, 1)[0]
#         sf1 = nn.Softmax(1)
#         xy = (x1 * y1).sum(1).mean()
#         x = gate2 * self.gc2(gcn1, adj)
#         out = torch.max(x, dim=1)[0]
#         # pooled_output = self.dropout(pooled_output)
#         out = self.dropout(out)
#
#         anchor_rep = self.dropout(anchor_rep)
#         aspect = self.dropout(aspect)
#         logits = self.dense(torch.cat([anchor_rep, aspect], dim=1))
#
#         output_w = self.fc(torch.cat([x, aspect.repeat(1, x.shape[1]).view(x.shape)], dim=2))
#         scores = (logits.repeat(1, x.shape[1]).view(x.shape[0], x.shape[1], -1) * output_w).sum(2)
#
#         kl = (sf1(scores) * sf1(dist_to_target.float())).sum(1).mean()
#
#         return logits, xy, kl
