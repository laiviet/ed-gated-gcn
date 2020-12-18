# -*- coding: utf-8 -*-
# file: train.py
# author: songyouwei <youwei0314@gmail.com>
# Copyright (C) 2018. All Rights Reserved.
import argparse
import math
import os
import sys
from time import strftime, localtime
import random
import numpy
import json

from pytorch_pretrained_bert import BertModel, BertTokenizer
from sklearn.metrics import precision_recall_fscore_support as prfs
from sklearn.metrics import f1_score
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from data_utils import LitbankDataset

from models.bert_ed import *
from models.bert_spc import BERT_SPC
from models.bert_amir import *
from models.bert_amir4 import *
from models.bert_amir5 import *
from models.bertdm import BertDM

from data_utils import collate_fn
import warnings
from constant import *

warnings.filterwarnings("ignore", category=DeprecationWarning)


class Instructor:
    def __init__(self, opt):
        self.opt = opt
        do_lower_case = 'uncased' in opt.pretrained_bert_name
        tokenizer = BertTokenizer.from_pretrained(opt.pretrained_bert_name, do_lower_case=do_lower_case)
        bert = BertModel.from_pretrained(opt.pretrained_bert_name).to(opt.device)
        for param in bert.parameters():
            param.requires_grad = False
        self.model = opt.model_class(bert, opt).to(opt.device)

        features = opt.inputs_cols
        self.positive_label_set = [i for i in range(1, self.opt.dataset_file['n_class'])]
        self.all_label_set = [i for i in range(0, self.opt.dataset_file['n_class'])]
        print('Positive label set: ', self.positive_label_set)
        print('All label set: ', self.all_label_set)

        self.trainset = LitbankDataset(opt.dataset_file['train'], tokenizer, features)
        self.testset = LitbankDataset(opt.dataset_file['test'], tokenizer, features)
        self.valset = LitbankDataset(opt.dataset_file['dev'], tokenizer, features)

        if opt.device.type == 'cuda':
            print('cuda memory allocated: {}'.format(torch.cuda.memory_allocated(device=opt.device.index)))
        self._print_args()

    def _print_args(self):
        n_trainable_params, n_nontrainable_params = 0, 0
        for p in self.model.parameters():
            n_params = torch.prod(torch.tensor(p.shape))
            if p.requires_grad:
                n_trainable_params += n_params
            else:
                n_nontrainable_params += n_params
        print(
            'n_trainable_params: {0}, n_nontrainable_params: {1}'.format(n_trainable_params, n_nontrainable_params))
        print('> training arguments:')
        for arg in vars(self.opt):
            print('>>> {0}: {1}'.format(arg, getattr(self.opt, arg)))

    def _reset_params(self):
        for child in self.model.children():
            if type(child) != BertModel:  # skip bert params
                for p in child.parameters():
                    if p.requires_grad:
                        if len(p.shape) > 1:
                            self.opt.initializer(p)
                        else:
                            stdv = 1. / math.sqrt(p.shape[0])
                            torch.nn.init.uniform_(p, a=-stdv, b=stdv)

    def _train(self, criterion, optimizer, train_dataloader, val_dataloader, test_dataloader):
        max_val_acc = 0
        max_val_f1 = 0
        max_test_f1 = 0.0
        global_step = 0
        path = None
        state_dict = self.model.state_dict()

        label_set = [i for i in range(0, self.opt.dataset_file['n_class'])]
        for epoch in range(self.opt.num_epoch):
            print('>' * 100)
            print('epoch: {}'.format(epoch))
            n_correct, n_total, loss_total = 0, 0, 0
            # switch model to training mode
            self.model.train()
            all_targets, all_outputs = [], []

            for i_batch, sample_batched in enumerate(train_dataloader):
                global_step += 1
                # clear gradient accumulators
                optimizer.zero_grad()

                inputs = {col: sample_batched[col].to(self.opt.device) for col in self.opt.inputs_cols}
                outputs, gate, kl, _ = self.model(inputs)
                targets = sample_batched['polarity'].to(self.opt.device)

                # print('| logits', tuple(outputs.shape))
                # print('| targets', tuple(targets.shape))

                loss = criterion(outputs, targets)

                loss += self.opt.gate_w * gate
                loss += self.opt.kl_w * kl

                loss.backward()
                optimizer.step()

                outputs = outputs.detach()
                n_correct += (torch.argmax(outputs, -1) == targets).sum().item()
                n_total += len(outputs)
                loss_total += loss.item() * len(outputs)
                all_targets.append(targets)
                all_outputs.append(torch.argmax(outputs, -1))
                if global_step % self.opt.log_step == 0:
                    all_targets = torch.cat(all_targets).view(-1).cpu().numpy()
                    all_outputs = torch.cat(all_outputs).view(-1).cpu().numpy()
                    p, r, f1 = self.prf(all_targets, all_outputs)

                    train_loss = loss_total / n_total
                    print('loss: {:.4f}, {:.4f} {:.4f} {:.4f}'.format(train_loss, p, r, f1))

                    all_targets, all_outputs = [], []
                    loss_total = 0.0

                # if global_step % (self.opt.log_step * 5) == 0:
            val_p, val_r, val_f1 = self._evaluate_acc_f1(val_dataloader)
            print('> val_f1: {:.4f} {:.4f} {:.4f}'.format(val_p, val_r, val_f1))
            test_p, test_r, test_f1 = self._evaluate_acc_f1(test_dataloader)
            print('> test_f1: {:.4f} {:.4f} {:.4f}'.format(test_p, test_r, test_f1))

            if test_f1 > max_test_f1:
                max_test_f1 = test_f1
                state_dict = self.model.state_dict()
            self.model.train()

        return state_dict

    def analyze(self, state_dict, data_loader):

        self.model.load_state_dict(state_dict)
        all_tokens = []
        all_anchor_index = []
        all_labels = []
        all_prediction = []
        all_scores = []
        with torch.no_grad():
            for i, batch_input in enumerate(data_loader):
                model_input = {col: batch_input[col].to(self.opt.device) for col in self.opt.inputs_cols}
                logits, _, _, scores = self.model(model_input)
                prediction = torch.argmax(logits, -1).cpu().numpy().tolist()
                all_prediction += prediction
                all_tokens += batch_input['token']
                all_anchor_index += batch_input['anchor_index'].numpy().tolist()
                all_labels += batch_input['polarity'].numpy().tolist()
                all_scores += scores.cpu().numpy().tolist()

        data = []
        for tokens, anchor_index, target, predict, scores in zip(all_tokens, all_anchor_index, all_labels,
                                                                 all_prediction, all_scores):
            item = {'token': tokens,
                    'score': scores,
                    'anchor_index': anchor_index,
                    'target': target,
                    'predict': predict,
                    }
            data.append(item)
        with open(self.opt.model + '.json', 'w') as f:
            json.dump(data, f)

    def prf(self, golds, preds):
        return self.sklearn_prf(golds, preds)
        # return self.paper_prf(golds, preds)

    def sklearn_prf(self, golds, preds):
        p, r, f, _ = prfs(golds, preds, labels=self.positive_label_set, average='micro')
        return p, r, f

    def paper_prf(self, golds, preds):
        correct = 0.0
        trials = 0.0
        trues = 0.0
        for j in range(len(preds)):
            if preds[j] > 0:
                trials += 1
            if golds[j] > 0:
                trues += 1
            if preds[j] == golds[j] and preds[j] > 0:
                correct += 1

        p = 0.0
        if trials > 0:
            p = correct / trials
        r = 0.0
        if trues > 0:
            r = correct / trues

        f = 0.0
        if (p + r) > 0:
            f = (2 * p * r) / (p + r)
        return p, r, f

    def _evaluate_acc_f1(self, data_loader):
        n_correct, n_total = 0, 0
        t_targets_all, t_outputs_all = [], []
        # switch model to evaluation mode
        label_set = [i for i in range(0, self.opt.dataset_file['n_class'])]
        self.model.eval()
        with torch.no_grad():
            for t_batch, t_sample_batched in enumerate(data_loader):
                t_inputs = {col: t_sample_batched[col].to(self.opt.device) for col in self.opt.inputs_cols}
                t_targets = t_sample_batched['polarity'].cpu()
                t_outputs, _, _, _ = self.model(t_inputs)
                t_preds = torch.argmax(t_outputs, -1).cpu()

                t_targets_all.append(t_targets)
                t_outputs_all.append(t_preds)

        t_targets_all = torch.cat(t_targets_all, dim=0).numpy()
        t_outputs_all = torch.cat(t_outputs_all, dim=0).numpy()

        p, r, f = self.prf(t_targets_all, t_outputs_all)
        return p, r, f

    def run(self):
        criterion = nn.CrossEntropyLoss()
        _params = filter(lambda p: p.requires_grad, self.model.parameters())
        optimizer = self.opt.optimizer(_params, lr=self.opt.learning_rate, )
        # optimizer = torch.optim.Adam(_params)

        train_data_loader = DataLoader(dataset=self.trainset,
                                       batch_size=self.opt.batch_size,
                                       shuffle=False,
                                       num_workers=3,
                                       collate_fn=collate_fn)
        test_data_loader = DataLoader(dataset=self.testset,
                                      batch_size=self.opt.batch_size,
                                      shuffle=False,
                                      num_workers=3,
                                      collate_fn=collate_fn)
        val_data_loader = DataLoader(dataset=self.valset,
                                     batch_size=self.opt.batch_size,
                                     shuffle=False,
                                     num_workers=3,
                                     collate_fn=collate_fn)

        self._reset_params()
        best_state_dict = self._train(criterion, optimizer, train_data_loader, val_data_loader, test_data_loader)

        # self.analyze(best_state_dict, test_data_loader)


def main():
    model_classes = {
        'bert_spc': BERT_SPC,
        'bert_amir': BertAmir,
        'bert_amir2': BertAmir2,
        'bert_amir4': BertAmir4,
        'bert_amir5': BertAmir5,
        'bert_amir51': BertAmir51,
        'bert_amir52': BertAmir52,
        'bert_amir53': BertAmir53,
        'bert_amir54': BertAmir54,
        'bert_amir55': BertAmir55,
        'bert_amir55nogate': BertAmir55NoGate,
        'bert_ed': BertED,
        'bertdm': BertDM
    }

    # Hyper Parameters
    models = list(model_classes.keys())
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='bert_ed', choices=models, type=str)
    parser.add_argument('--dataset', default='litbank-cased',
                        choices=['ace34-cased', 'ace34-uncased', 'litbank-cased', 'litbank-uncased', 'ace2-uncased',
                                 'ace-cased', 'debug'], type=str)
    parser.add_argument('--optimizer', default='adam', type=str)
    parser.add_argument('--initializer', default='xavier_uniform_', type=str)
    parser.add_argument('--learning_rate', default=0.0001, type=float, help='try 5e-5, 2e-5 for BERT, 1e-3 for others')
    parser.add_argument('--dropout', default=0.25, type=float)
    parser.add_argument('--l2reg', default=0.01, type=float)
    parser.add_argument('--num_epoch', default=30, type=int, help='try larger number for non-BERT models')
    parser.add_argument('--batch_size', default=256, type=int, help='try 16, 32, 64 for BERT models')
    parser.add_argument('--log_step', default=200, type=int)
    parser.add_argument('--embed_dim', default=300, type=int)
    parser.add_argument('--gate_w', default=0.01, type=float)
    parser.add_argument('--kl_w', default=0.01, type=float)
    parser.add_argument('--n_layer', default=4, type=int)
    parser.add_argument('--hidden_dim', default=256, type=int)
    parser.add_argument('--bert_dim', default=768, type=int)
    parser.add_argument('--hops', default=3, type=int)
    parser.add_argument('--device', default=None, type=str, help='e.g. cuda:0')
    parser.add_argument('--seed', default=14181, type=int, help='set seed for reproducibility')
    parser.add_argument('--class_weight', default=1, type=float, help='Class weighting')
    # The following parameters are only valid for the lcf-bert model
    parser.add_argument('--local_context_focus', default='cdm', type=str, help='local context focus mode, cdw or cdm')
    # semantic-relative-distance, see the paper of LCF-BERT model
    parser.add_argument('--SRD', default=3, type=int, help='set SRD')
    opt = parser.parse_args()

    if opt.seed is not None:
        random.seed(opt.seed)
        numpy.random.seed(opt.seed)
        torch.manual_seed(opt.seed)
        torch.cuda.manual_seed(opt.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    dataset_files = {
        'litbank-cased': LITBANK_CASE,
        'litbank-uncased': LITBANK_UNCASE,
        'ace34-cased': ACE34_CASE,
        'ace34-uncased': ACE34_UNCASE,
        'ace2-uncased': ACE2_UNCASE,
        'ace-cased': ACE_CASE,
        'debug': DEBUG
    }
    input_colses = {
        'bert_spc': ['cls_text_sep_aspect_sep_indices',
                     'cls_text_sep_aspect_sep_segments_ids'],
        'bert_amir': ['cls_text_sep_aspect_sep_indices',
                      'cls_text_sep_aspect_sep_length',
                      'cls_text_sep_aspect_sep_segments_ids',
                      'cls_text_sep_aspect_sep_aspect_mask',
                      'cls_text_sep_aspect_sep_mask',
                      'cls_text_sep_indices',
                      'cls_text_sep_length',
                      'cls_text_sep_segments_ids',
                      'transform',
                      'anchor_index',
                      'sentence_length',
                      'dependency_graph',
                      'mask',
                      'dist_to_target',
                      'anchor_index'],
        'bert_amir3': ['cls_text_sep_aspect_sep_indices',
                       'cls_text_sep_aspect_sep_length',
                       'cls_text_sep_aspect_sep_segments_ids',
                       'cls_text_sep_aspect_sep_aspect_mask',
                       'cls_text_sep_aspect_sep_mask',
                       'cls_text_sep_indices',
                       'cls_text_sep_length',
                       'cls_text_sep_segments_ids',
                       'transform',
                       'anchor_index',
                       'sentence_length',
                       'dependency_graph',
                       'mask',
                       'dist_to_target',
                       'anchor_index'],
        'bert_amir2': ['cls_text_sep_aspect_sep_indices',
                       'cls_text_sep_aspect_sep_length',
                       'cls_text_sep_aspect_sep_segments_ids',
                       'cls_text_sep_aspect_sep_aspect_mask',
                       'cls_text_sep_aspect_sep_mask',
                       'cls_text_sep_indices',
                       'cls_text_sep_length',
                       'cls_text_sep_segments_ids',
                       'transform',
                       'anchor_index',
                       'sentence_length',
                       'dependency_graph',
                       'mask',
                       'dist_to_target', 'anchor_index'],
        'bert_ed': [
            'cls_text_sep_indices',
            'cls_text_sep_length',
            'cls_text_sep_segments_ids',
            'transform',
            'sentence_length',
            'anchor_index'
        ],
        'bert_amir4': [
            'cls_text_sep_indices',
            'cls_text_sep_length',
            'cls_text_sep_segments_ids',
            'transform',
            'sentence_length',
            'dependency_graph',
            'anchor_index',
            'dist_to_target'
        ],
        'bert_amir5': [
            'cls_text_sep_indices',
            'cls_text_sep_length',
            'cls_text_sep_segments_ids',
            'transform',
            'sentence_length',
            'dependency_graph',
            'anchor_index',
            'dist_to_target'
        ],
        'bert_lstm': [
            'cls_text_sep_indices',
            'cls_text_sep_length',
            'cls_text_sep_segments_ids',
            'transform',
            'sentence_length',
            'anchor_index'
        ],
        'bertdm': [
            'cls_text_sep_indices',
            'cls_text_sep_length',
            'cls_text_sep_segments_ids',
            'transform',
            'sentence_length',
            'anchor_index'
        ]
    }

    input_colses['bert_amir51'] = input_colses['bert_amir5']
    input_colses['bert_amir52'] = input_colses['bert_amir5']
    input_colses['bert_amir53'] = input_colses['bert_amir5']
    input_colses['bert_amir54'] = input_colses['bert_amir5']
    input_colses['bert_amir55'] = input_colses['bert_amir5']
    input_colses['bert_amir55nogate'] = input_colses['bert_amir5']
    input_colses['bert_amir56'] = input_colses['bert_amir5']

    initializers = {
        'xavier_uniform_': torch.nn.init.xavier_uniform_,
        'xavier_normal_': torch.nn.init.xavier_normal,
        'orthogonal_': torch.nn.init.orthogonal_,
    }
    optimizers = {
        'adadelta': torch.optim.Adadelta,  # default lr=1.0
        'adagrad': torch.optim.Adagrad,  # default lr=0.01
        'adam': torch.optim.Adam,  # default lr=0.001
        'adamax': torch.optim.Adamax,  # default lr=0.002
        'asgd': torch.optim.ASGD,  # default lr=0.01
        'rmsprop': torch.optim.RMSprop,  # default lr=0.01
        'sgd': torch.optim.SGD,
    }
    opt.polarities_dim = dataset_files[opt.dataset]['n_class']
    opt.pretrained_bert_name = dataset_files[opt.dataset]['pretrained_bert_name']
    opt.model_class = model_classes[opt.model]
    opt.dataset_file = dataset_files[opt.dataset]
    opt.inputs_cols = input_colses[opt.model]
    opt.initializer = initializers[opt.initializer]
    opt.optimizer = optimizers[opt.optimizer]
    opt.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') \
        if opt.device is None else torch.device(opt.device)

    print('--------------------')

    ins = Instructor(opt)
    ins.run()


if __name__ == '__main__':
    main()
