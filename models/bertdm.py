import torch
import torch.nn as nn


class BertDMNoAnchor(nn.Module):
    def __init__(self, bert, opt):
        super(BertDMNoAnchor, self).__init__()
        # self.squeeze_embedding = SqueezeEmbedding()
        self.device = opt.device
        self.bert = bert
        self.n_layer = opt.n_layer
        self.dropout = nn.Dropout(opt.dropout)
        self.dense = nn.Linear(opt.n_layer * opt.bert_dim * 2, opt.polarities_dim)

        self.m = torch.LongTensor([i for i in range(200)]).to(self.device)

    def get_mask(self, L, anchor_index, length):
        m = self.m[:L].repeat(anchor_index.shape[0], 1)
        # print('BertDM > get_mask > anchor_index', tuple(anchor_index.shape))

        anchor_index = anchor_index.unsqueeze(dim=1)
        # print('BertDM > get_mask > anchor_index', tuple(anchor_index.shape))

        maskL = m < (anchor_index + 1)
        maskR = m > anchor_index

        maskL = maskL.float().unsqueeze(dim=0)
        maskR = maskR.float().unsqueeze(dim=0)

        maskS = m < length.unsqueeze(dim=1)
        maskS = maskS.float().unsqueeze(dim=0)
        maskR = maskR * maskS

        # print('BertDM > get_mask > maskL', tuple(maskL.shape))
        #
        # print(maskR[0][0])
        # print(maskL[0][0])
        # print(anchor_index[0][0])
        # exit(0)

        return maskL, maskR

    def forward(self, inputs):
        """
        bert_length: L
        original_length: T
        """

        L = inputs['cls_text_sep_length'].max().cpu().numpy().tolist()
        T = inputs['sentence_length'].max().cpu().numpy().tolist()
        B = inputs['sentence_length'].shape[0]
        sentence_length = inputs['sentence_length']
        text_bert_indices = inputs['cls_text_sep_indices'][:, :L]  # B x L
        bert_segments_ids = inputs['cls_text_sep_segments_ids'][:, :L]  # B x L

        anchor_index = inputs['anchor_index']

        transform = inputs['transform'][:, :T, :L]  # B x T x L
        # print('| L: ', L)
        # print('| T: ', T)

        bert_x, pooled_output = self.bert(text_bert_indices, bert_segments_ids, output_all_encoded_layers=True)
        # print('| BertDM > len(bert_x)', len(x))

        bert_x = torch.cat(bert_x[-self.n_layer:], dim=-1)

        transform_x = torch.bmm(transform, bert_x)

        # print('| BertDM > transform_x', tuple(transform_x.shape))

        transpose_x = transform_x.transpose(1, 2).transpose(0, 1)

        # print('| BertDM > transpose_x', tuple(transpose_x.shape))

        maskL, maskR = self.get_mask(T, anchor_index, sentence_length)

        L = (transpose_x * maskL).transpose(0, 1)
        R = (transpose_x * maskR).transpose(0, 1)
        L = L + torch.ones_like(L)
        R = R + torch.ones_like(R)
        # print('| BertDM > L', tuple(L.shape))

        pooledL, _ = L.max(dim=2)
        pooledR, _ = R.max(dim=2)
        x = torch.cat((pooledL, pooledR), 1)
        x = x - torch.ones_like(x)

        # mask = torch.LongTensor([i for i in range(T)]).to(self.device).repeat(B, 1)  # B x T
        # print('| BertDM > mask', tuple(mask.shape))

        # mask = mask == anchor_index.unsqueeze(1)
        # mask = mask.unsqueeze(dim=2).expand(-1, -1, 768 * self.n_layer)
        # print('| BertDM > mask', tuple(mask.shape))

        # anchor_rep = torch.masked_select(transform_x, mask).view(B, -1)
        # print('| BertDM > anchor_rep', tuple(anchor_rep.shape))

        # anchor_rep = self.dropout(anchor_rep)
        # x = torch.cat((anchor_rep, x),1)
        logits = self.dense(self.dropout(x))

        return logits, 0.0, 0.0, 0.0

class BertDM(nn.Module):
    def __init__(self, bert, opt):
        super(BertDM, self).__init__()
        # self.squeeze_embedding = SqueezeEmbedding()
        self.device = opt.device
        self.bert = bert
        self.n_layer = opt.n_layer
        self.dropout = nn.Dropout(opt.dropout)
        self.dense = nn.Linear(opt.n_layer * opt.bert_dim * 3, opt.polarities_dim)

        self.m = torch.LongTensor([i for i in range(200)]).to(self.device)

    def get_mask(self, L, anchor_index, length):
        m = self.m[:L].repeat(anchor_index.shape[0], 1)
        # print('BertDM > get_mask > anchor_index', tuple(anchor_index.shape))

        anchor_index = anchor_index.unsqueeze(dim=1)
        # print('BertDM > get_mask > anchor_index', tuple(anchor_index.shape))

        maskL = m < (anchor_index + 1)
        maskR = m > anchor_index

        maskL = maskL.float().unsqueeze(dim=0)
        maskR = maskR.float().unsqueeze(dim=0)

        maskS = m < length.unsqueeze(dim=1)
        maskS = maskS.float().unsqueeze(dim=0)
        maskR = maskR * maskS

        # print('BertDM > get_mask > maskL', tuple(maskL.shape))
        #
        # print(maskR[0][0])
        # print(maskL[0][0])
        # print(anchor_index[0][0])
        # exit(0)

        return maskL, maskR

    def forward(self, inputs):
        """
        bert_length: L
        original_length: T
        """

        L = inputs['cls_text_sep_length'].max().cpu().numpy().tolist()
        T = inputs['sentence_length'].max().cpu().numpy().tolist()
        B = inputs['sentence_length'].shape[0]
        sentence_length = inputs['sentence_length']
        text_bert_indices = inputs['cls_text_sep_indices'][:, :L]  # B x L
        bert_segments_ids = inputs['cls_text_sep_segments_ids'][:, :L]  # B x L

        anchor_index = inputs['anchor_index']

        transform = inputs['transform'][:, :T, :L]  # B x T x L
        # print('| L: ', L)
        # print('| T: ', T)

        bert_x, pooled_output = self.bert(text_bert_indices, bert_segments_ids, output_all_encoded_layers=True)
        # print('| BertDM > len(bert_x)', len(x))

        bert_x = torch.cat(bert_x[-self.n_layer:], dim=-1)

        transform_x = torch.bmm(transform, bert_x)

        # print('| BertDM > transform_x', tuple(transform_x.shape))

        transpose_x = transform_x.transpose(1, 2).transpose(0, 1)

        # print('| BertDM > transpose_x', tuple(transpose_x.shape))

        maskL, maskR = self.get_mask(T, anchor_index, sentence_length)

        L = (transpose_x * maskL).transpose(0, 1)
        R = (transpose_x * maskR).transpose(0, 1)
        L = L + torch.ones_like(L)
        R = R + torch.ones_like(R)
        # print('| BertDM > L', tuple(L.shape))

        pooledL, _ = L.max(dim=2)
        pooledR, _ = R.max(dim=2)
        x = torch.cat((pooledL, pooledR), 1)
        x = x - torch.ones_like(x)

        mask = torch.LongTensor([i for i in range(T)]).to(self.device).repeat(B, 1)  # B x T
        # print('| BertDM > mask', tuple(mask.shape))

        mask = mask == anchor_index.unsqueeze(1)
        mask = mask.unsqueeze(dim=2).expand(-1, -1, 768 * self.n_layer)
        # print('| BertDM > mask', tuple(mask.shape))

        anchor_rep = torch.masked_select(transform_x, mask).view(B, -1)
        # print('| BertDM > anchor_rep', tuple(anchor_rep.shape))

        anchor_rep = self.dropout(anchor_rep)
        x = torch.cat((anchor_rep, x),1)
        logits = self.dense(self.dropout(x))

        return logits, 0.0, 0.0, 0.0

class BertDMLSTM(nn.Module):
    def __init__(self, bert, opt):
        super(BertDMLSTM, self).__init__()
        # self.squeeze_embedding = SqueezeEmbedding()
        self.device = opt.device
        self.bert = bert
        self.n_layer = opt.n_layer
        self.hidden_dim = opt.hidden_dim
        self.dropout = nn.Dropout(opt.dropout)
        self.dense = nn.Linear(opt.n_layer * opt.bert_dim + 6 * self.hidden_dim, opt.polarities_dim)

        self.m = torch.LongTensor([i for i in range(200)]).to(self.device)

    def get_mask(self, L, anchor_index, length):
        m = self.m[:L].repeat(anchor_index.shape[0], 1)
        # print('BertDM > get_mask > anchor_index', tuple(anchor_index.shape))

        anchor_index = anchor_index.unsqueeze(dim=1)
        # print('BertDM > get_mask > anchor_index', tuple(anchor_index.shape))

        maskL = m < (anchor_index + 1)
        maskR = m > anchor_index

        maskL = maskL.float().unsqueeze(dim=0)
        maskR = maskR.float().unsqueeze(dim=0)

        maskS = m < length.unsqueeze(dim=1)
        maskS = maskS.float().unsqueeze(dim=0)
        maskR = maskR * maskS

        # print('BertDM > get_mask > maskL', tuple(maskL.shape))
        #
        # print(maskR[0][0])
        # print(maskL[0][0])
        # print(anchor_index[0][0])
        # exit(0)

        return maskL, maskR

    def forward(self, inputs):
        """
        bert_length: L
        original_length: T
        """

        L = inputs['cls_text_sep_length'].max().cpu().numpy().tolist()
        T = inputs['sentence_length'].max().cpu().numpy().tolist()
        B = inputs['sentence_length'].shape[0]
        sentence_length = inputs['sentence_length']
        text_bert_indices = inputs['cls_text_sep_indices'][:, :L]  # B x L
        bert_segments_ids = inputs['cls_text_sep_segments_ids'][:, :L]  # B x L

        anchor_index = inputs['anchor_index']

        transform = inputs['transform'][:, :T, :L]  # B x T x L
        # print('| L: ', L)
        # print('| T: ', T)

        bert_x, pooled_output = self.bert(text_bert_indices, bert_segments_ids, output_all_encoded_layers=True)
        # print('| BertDM > len(bert_x)', len(x))

        bert_x = torch.cat(bert_x[-self.n_layer:], dim=-1)

        transform_x = torch.bmm(transform, bert_x)

        # print('| BertDM > transform_x', tuple(transform_x.shape))

        transpose_x = transform_x.transpose(1, 2).transpose(0, 1)

        # print('| BertDM > transpose_x', tuple(transpose_x.shape))

        maskL, maskR = self.get_mask(T, anchor_index, sentence_length)

        L = (transpose_x * maskL).transpose(0, 1)
        R = (transpose_x * maskR).transpose(0, 1)
        L = L + torch.ones_like(L)
        R = R + torch.ones_like(R)
        # print('| BertDM > L', tuple(L.shape))

        pooledL, _ = L.max(dim=2)
        pooledR, _ = R.max(dim=2)
        x = torch.cat((pooledL, pooledR), 1)
        x = x - torch.ones_like(x)

        mask = torch.LongTensor([i for i in range(T)]).to(self.device).repeat(B, 1)  # B x T
        # print('| BertDM > mask', tuple(mask.shape))

        mask = mask == anchor_index.unsqueeze(1)
        mask = mask.unsqueeze(dim=2).expand(-1, -1, 768 * self.n_layer)
        # print('| BertDM > mask', tuple(mask.shape))

        anchor_rep = torch.masked_select(transform_x, mask).view(B, -1)
        # print('| BertDM > anchor_rep', tuple(anchor_rep.shape))

        anchor_rep = self.dropout(anchor_rep)
        x = torch.cat((anchor_rep, x),1)
        logits = self.dense(self.dropout(x))

        return logits, 0.0, 0.0, 0.0
