# -*- coding: utf-8 -*-
# file: data_utils.py
# author: songyouwei <youwei0314@gmail.com>
# Copyright (C) 2018. All Rights Reserved.

import sys
import os
import pickle
import torch
import collections
import spacy
from torch.utils.data import Dataset
from pytorch_pretrained_bert import BertTokenizer
import time
from tqdm import tqdm
import os
import multiprocessing
import numpy as np
from constant import *


def build_litbank_tokenizer(max_seq_len, dat_fname):
    if os.path.exists(dat_fname):
        print('loading tokenizer:', dat_fname)
        tokenizer = pickle.load(open(dat_fname, 'rb'))
    else:
        files = [os.path.join('datasets/litbank/train', x) for x in os.listdir('datasets/litbank/train')]
        files += [os.path.join('datasets/litbank/dev', x) for x in os.listdir('datasets/litbank/dev')]
        files += [os.path.join('datasets/litbank/test', x) for x in os.listdir('datasets/litbank/test')]
        text = ''
        for fname in files:
            fin = open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
            lines = fin.readlines()
            fin.close()
            tokens = [x.split('\t')[0] for x in lines if '\t' in x]
            text = text + ' ' + ' '.join(tokens)

        tokenizer = Tokenizer(max_seq_len)
        tokenizer.fit_on_text(text)
        pickle.dump(tokenizer, open(dat_fname, 'wb'))
    return tokenizer


def _load_word_vec(path, word2idx=None):
    fin = open(path, 'r', encoding='utf-8', newline='\n', errors='ignore')
    word_vec = {}
    for line in fin:
        tokens = line.rstrip().split()
        if word2idx is None or tokens[0] in word2idx.keys():
            word_vec[tokens[0]] = np.asarray(tokens[1:], dtype='float32')
    return word_vec


def build_embedding_matrix(word2idx, embed_dim, dat_fname):
    if os.path.exists(dat_fname):
        print('loading embedding_matrix:', dat_fname)
        embedding_matrix = pickle.load(open(dat_fname, 'rb'))
    else:
        print('loading word vectors...')
        embedding_matrix = np.zeros((len(word2idx) + 2, embed_dim))  # idx 0 and len(word2idx)+1 are all-zeros
        fname = '../dataset/embedding/glove.6B.300d.txt'
        word_vec = _load_word_vec(fname, word2idx=word2idx)
        print('building embedding_matrix:', dat_fname)
        for word, i in word2idx.items():
            vec = word_vec.get(word)
            if vec is not None:
                # words not found in embedding index will be all-zeros.
                embedding_matrix[i] = vec
        pickle.dump(embedding_matrix, open(dat_fname, 'wb'))
    return embedding_matrix


def pad_and_truncate(sequence, maxlen, dtype='int64', padding='post', truncating='post', value=0):
    x = (np.ones(maxlen) * value).astype(dtype)
    if truncating == 'pre':
        trunc = sequence[-maxlen:]
    else:
        trunc = sequence[:maxlen]
    trunc = np.asarray(trunc, dtype=dtype)
    if padding == 'post':
        x[:len(trunc)] = trunc
    else:
        x[-len(trunc):] = trunc
    return x


class Tokenizer(object):
    def __init__(self, max_seq_len, lower=True):
        self.lower = lower
        self.max_seq_len = max_seq_len
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 1

    def fit_on_text(self, text):
        if self.lower:
            text = text.lower()
        words = text.split()
        for word in words:
            if word not in self.word2idx:
                self.word2idx[word] = self.idx
                self.idx2word[self.idx] = word
                self.idx += 1

    def text_to_sequence(self, text, reverse=False, padding='post', truncating='post'):
        if self.lower:
            text = text.lower()
        words = text.split()
        unknownidx = len(self.word2idx) + 1
        sequence = [self.word2idx[w] if w in self.word2idx else unknownidx for w in words]
        if len(sequence) == 0:
            sequence = [0]
        if reverse:
            sequence = sequence[::-1]
        return pad_and_truncate(sequence, self.max_seq_len, padding=padding, truncating=truncating)


EVENT_TAGSET = {
    'O': 0,
    'EVENT': 1,
}


def read_token_from_file(path):
    data = []
    target_counter = collections.Counter()
    with open(path, 'r') as f:
        text = f.read()
    sample_texts = text.split('\n\n')

    for i, sample_text in enumerate(sample_texts):
        sample_id = (path, i)
        lines = [x.strip() for x in sample_text.split('\n')]
        lines = [x for x in lines if len(x) > 0]
        if len(lines) == 0:
            continue
        line_parts = [x.split('\t') for x in lines if '\t' in x]
        tokens = [x[0] for x in line_parts]
        data.append((sample_id, tokens))
    return data


def read_folder(path):
    data = []
    target_counter = collections.Counter()
    files = [os.path.join(path, x) for x in os.listdir(path) if x.endswith('tsv')]
    for fpath in files:
        with open(fpath, 'r') as f:
            text = f.read()
        sample_texts = text.split('\n\n')

        for i, sample_text in enumerate(sample_texts):
            sample_id = (fpath, i)
            lines = [x.strip() for x in sample_text.split('\n')]
            lines = [x for x in lines if len(x) > 0]
            if len(lines) == 0:
                continue
            line_parts = [x.split('\t') for x in lines if '\t' in x]
            tokens = [x[0] for x in line_parts]
            labels = [x[1] for x in line_parts]
            targets = [EVENT_TAGSET[x] for x in labels]
            target_counter.update(targets)
            assert len(tokens) == len(labels)
            # if len(tokens)> 80:
            #     print(len(tokens), sample_id)
            #     print(' '.join(tokens))
            data.append((sample_id, tokens, labels, targets))
    print('Dataset: ', path, target_counter)
    return data


def read_litbank_file(path):
    data = []
    with open(path, 'r') as f:
        text = f.read()
    sample_texts = text.split('\n\n')

    for i, sample_text in enumerate(sample_texts):
        sample_id = (path, i)
        lines = [x.strip() for x in sample_text.split('\n')]
        lines = [x for x in lines if len(x) > 0]
        if len(lines) == 0:
            continue
        line_parts = [x.split('\t') for x in lines if '\t' in x]
        tokens = [x[0] for x in line_parts]
        labels = [x[1] for x in line_parts]
        targets = [EVENT_TAGSET[x] for x in labels]
        assert len(tokens) == len(labels)
        data.append((sample_id, tokens, labels, targets))

        if len(tokens) > 82:
            print(sample_id)
            print(len(tokens))
            print(' '.join(tokens))
    return data


def read_ace2_file(path):
    data = []
    with open(path, 'r') as f:
        text = f.read()
    sample_texts = text.split('\n\n')
    TAGSET = {
        'ANCHOR': 0,
        'EVENT': 1
    }

    for i, sample_text in enumerate(sample_texts):
        sample_id = (path, i)
        lines = [x.strip() for x in sample_text.split('\n')]
        lines = [x for x in lines if len(x) > 0]
        if len(lines) == 0:
            continue
        line_parts = [x.split('\t') for x in lines if '\t' in x]
        tokens = [x[0] for x in line_parts]
        labels = [x[1] for x in line_parts]
        anchor_index = [i for i, x in enumerate(labels) if x != 'O'][0]
        label = labels[anchor_index]

        assert label in ['ANCHOR', 'EVENT'], 'No label founded.'
        target = TAGSET[label]
        data.append((sample_id, tokens, label, target, anchor_index))
    return data


def read_ace34_file(path):
    data = []
    with open(path, 'r') as f:
        text = f.read()
    sample_texts = text.split('\n\n')
    TAGSET = {'Justice:Acquit': 33, 'Justice:Sue': 11, 'Movement:Transport': 1,
              'Life:Marry': 8, 'Justice:Convict': 24, 'Justice:Pardon': 28,
              'Other': 0, 'Justice:Arrest-Jail': 16, 'Business:Merge-Org': 30,
              'Life:Die': 15, 'Contact:Meet': 7, 'Life:Divorce': 32,
              'Justice:Appeal': 29, 'Personnel:Start-Position': 3, 'Business:End-Org': 13,
              'Justice:Release-Parole': 26, 'Justice:Extradite': 31, 'Transaction:Transfer-Money': 10,
              'Justice:Execute': 18, 'Personnel:Nominate': 4, 'Justice:Fine': 27,
              'Business:Start-Org': 23, 'Business:Declare-Bankruptcy': 25, 'Personnel:End-Position': 6,
              'Justice:Charge-Indict': 22, 'Conflict:Demonstrate': 12, 'Transaction:Transfer-Ownership': 17,
              'Justice:Trial-Hearing': 19, 'Justice:Sentence': 20, 'Life:Be-Born': 21,
              'Contact:Phone-Write': 9, 'Life:Injure': 14, 'Conflict:Attack': 5,
              'Personnel:Elect': 2}

    for i, sample_text in enumerate(sample_texts):
        sample_id = (path, i)
        lines = [x.strip() for x in sample_text.split('\n')]
        lines = [x for x in lines if len(x) > 0]
        if len(lines) == 0:
            continue
        line_parts = [x.split('\t') for x in lines if '\t' in x]
        tokens = [x[0] for x in line_parts]
        labels = [x[1] for x in line_parts]
        anchor_index = [i for i, x in enumerate(labels) if x != 'O'][0]
        label = labels[anchor_index]

        assert label in TAGSET, 'No label founded.'
        target = TAGSET[label]
        data.append((sample_id, tokens, label, target, anchor_index))
    return data


def read_ace_file(path):
    TAGSET = {'Justice:Acquit': 33, 'Justice:Sue': 11, 'Movement:Transport': 1,
              'Life:Marry': 8, 'Justice:Convict': 24, 'Justice:Pardon': 28,
              'Other': 0, 'Justice:Arrest-Jail': 16, 'Business:Merge-Org': 30,
              'Life:Die': 15, 'Contact:Meet': 7, 'Life:Divorce': 32,
              'Justice:Appeal': 29, 'Personnel:Start-Position': 3, 'Business:End-Org': 13,
              'Justice:Release-Parole': 26, 'Justice:Extradite': 31, 'Transaction:Transfer-Money': 10,
              'Justice:Execute': 18, 'Personnel:Nominate': 4, 'Justice:Fine': 27,
              'Business:Start-Org': 23, 'Business:Declare-Bankruptcy': 25, 'Personnel:End-Position': 6,
              'Justice:Charge-Indict': 22, 'Conflict:Demonstrate': 12, 'Transaction:Transfer-Ownership': 17,
              'Justice:Trial-Hearing': 19, 'Justice:Sentence': 20, 'Life:Be-Born': 21,
              'Contact:Phone-Write': 9, 'Life:Injure': 14, 'Conflict:Attack': 5,
              'Personnel:Elect': 2, 'O': 100}

    data = []
    with open(path, 'r') as f:
        text = f.read()
    sample_texts = text.split('\n\n')

    max_len = 0

    for i, sample_text in enumerate(sample_texts):
        sample_id = (path, i)
        lines = [x.strip() for x in sample_text.split('\n')]
        lines = [x for x in lines if len(x) > 0]
        if len(lines) == 0:
            continue
        line_parts = [x.split('\t') for x in lines if '\t' in x]
        tokens = [x[0] for x in line_parts]
        labels = [x[1] for x in line_parts]
        targets = [TAGSET[x] for x in labels]
        assert len(tokens) == len(labels)

        if len(tokens) > max_len:
            max_len = len(tokens)
        data.append((sample_id, tokens, labels, targets))
    print('Max len: ', max_len)
    return data


def get_dist(i, target, adj, seen):
    seen.append(i)
    if i == target:
        return 0
    else:
        children = []
        for j in range(len(adj)):
            if adj[i][j] == 1 and i != j and j not in seen:
                children.append(j)
        md = 100000
        for c in children:
            d = get_dist(c, target, adj, seen) + 1
            if d < md:
                md = d
        return md


def get_dist_to_target(adj, target, dist, length):
    for i in range(length):
        dist[i] = get_dist(i, target, adj, [])
    assert all(d != -1 for d in dist), dist
    return [d + 1 for d in dist]


class LitbankDataset(Dataset):

    def __init__(self, folder, tokenizer, features, sort=True):
        super(LitbankDataset, self).__init__()
        self.features = features + [ 'polarity']
        self.data = []
        self.tokenizer = tokenizer
        preprocessed_files = sorted([os.path.join(folder, x) for x in os.listdir(folder) if x.endswith('.proc')])
        for path in preprocessed_files:
            print('Loading preprocessed file: ', path)
            with open(path, 'rb') as f:
                self.data += pickle.load(f)
        self.length = len(self.data)
        print('Total: ', self.length)

        if sort:
            import time
            start = time.time()
            print('Sorting... ', end='')
            self.data = sorted(self.data, key=lambda x: x['sentence_length'])
            print(time.time() - start)

    def __getitem__(self, idx):
        item = self.data[idx]
        return {
            k: item[k]
            for k in self.features
        }

    def __len__(self):
        return self.length


def dont_change(x):
    return x

tensor_type = {
    'cls_text_sep_aspect_sep_indices': torch.LongTensor,
    'cls_text_sep_aspect_sep_length': torch.LongTensor,
    'cls_text_sep_aspect_sep_segments_ids': torch.LongTensor,
    'cls_text_sep_aspect_sep_aspect_mask': torch.FloatTensor,
    'cls_text_sep_aspect_sep_mask': torch.FloatTensor,
    'cls_text_sep_indices': torch.LongTensor,
    'cls_text_sep_length': torch.LongTensor,
    'cls_text_sep_segments_ids': torch.LongTensor,
    'anchor_index': torch.LongTensor,
    'transform': torch.FloatTensor,
    'sentence_length': torch.LongTensor,
    'bert_length': torch.LongTensor,
    'polarity': torch.LongTensor,
    'dependency_graph': torch.FloatTensor,
    'mask': torch.FloatTensor,
    'dist_to_target': torch.LongTensor,
    'token': dont_change
}


def collate_fn(items):
    # for item in items:
    #     print('----')
    #     for k, v in item.items():
    #         if type(v) == list:
    #             print(k, len(v))
    keys = items[0].keys()
    # print(keys)
    results = {}
    for k in keys:
        # try:
        results[k] = tensor_type[k]([x[k] for x in items])
        # except:
        #     print(k)
        #     x = [torch.Tensor(x[k]) for x in items]
        #     for y in x:
        #         print(y.shape)
        # exit(0)

    return results


def zeros(M, N, zero):
    zeros = []
    for i in range(M):
        zeros.append([zero for _ in range(N)])
    return zeros


def process_litbank_file(path):
    if 'uncased' in path:
        LITBANK = LITBANK_UNCASE
        tokenizer = BertTokenizer.from_pretrained(LITBANK['pretrained_bert_name'])
    else:
        LITBANK = LITBANK_CASE
        tokenizer = BertTokenizer.from_pretrained(LITBANK['pretrained_bert_name'], do_lower_case=False)

    BERT_ML = LITBANK['bert_ml']
    ORI_ML = LITBANK['ori_ml']

    fin = open('{}.graph'.format(path), 'rb')
    idx2gragh = pickle.load(fin)
    fin.close()

    raw_data = read_litbank_file(path)
    data = []
    for sample_id, tokens, labels, targets in raw_data:
        sent_len = len(tokens)
        # On raw text
        tok_bert_indices = []
        for tok in tokens:
            bert_tokens = tokenizer.tokenize(tok)
            tok_bert_indices.append(tokenizer.convert_tokens_to_ids(bert_tokens))
        bert_len = sum([len(x) for x in tok_bert_indices])

        transform = zeros(ORI_ML, BERT_ML, 0.0)

        # Create transform to convert tokenized length to original length
        offset = 1
        raw_text_bert_indices = []
        # print(sample_id)
        # print(sum([len(x) for x in tok_bert_indices]))
        # print(' '.join(tokens))
        for i, indices in enumerate(tok_bert_indices):
            l = len(indices)
            raw_text_bert_indices += indices
            for j in range(l):
                transform[i][offset + j] = 1 / l
            offset += l

        dep_matrix = idx2gragh[sample_id]

        offset = 1  # Because of the CLS
        for anchor_index, (aspect_indices, target) in enumerate(zip(tok_bert_indices, targets)):
            if sample_id[0].split('/')[3].startswith('gpt') and target== 0:
                continue

            #   CLS + Sentence + SEP + aspect + SEP
            cls_text_sep_aspect_sep_indices = [101] + raw_text_bert_indices + [102] + aspect_indices + [102]
            cls_text_sep_aspect_sep_length = len(cls_text_sep_aspect_sep_indices)
            cls_text_sep_aspect_sep_mask = [0 for _ in range(cls_text_sep_aspect_sep_length)] + [1 for _ in range(
                BERT_ML - cls_text_sep_aspect_sep_length)]
            cls_text_sep_aspect_sep_indices += [0 for _ in range(BERT_ML - cls_text_sep_aspect_sep_length)]
            cls_text_sep_aspect_sep_segments_ids = [0 for _ in range(BERT_ML)]
            for i in range(bert_len + 2, bert_len + 2 + len(aspect_indices) + 1):
                cls_text_sep_aspect_sep_segments_ids[i] = 1

            cls_text_sep_aspect_sep_aspect_mask = [1 for _ in range(BERT_ML)]
            for i in range(offset, offset + len(aspect_indices)):
                cls_text_sep_aspect_sep_aspect_mask[i] = 0

            offset += len(aspect_indices)

            # CLS + sentence + SEP
            cls_text_sep_indices = [101] + raw_text_bert_indices + [102]
            cls_text_sep_length = len(cls_text_sep_indices)
            cls_text_sep_indices += [0 for x in range(BERT_ML - cls_text_sep_length)]
            cls_text_sep_segments_ids = [0 for _ in range(BERT_ML)]

            # Original sentence
            mask = [0 for _ in range(sent_len)] + [1 for _ in range(ORI_ML - sent_len)]
            dist = [-1 for i in range(sent_len)]
            dist_to_target = get_dist_to_target(dep_matrix, anchor_index, dist, sent_len)
            max_dist = max(dist_to_target) + 1
            dist_padding = [max_dist] * (ORI_ML - len(dist_to_target))
            pad_dist_to_target = dist_to_target + dist_padding

            item = {
                'cls_text_sep_aspect_sep_indices': cls_text_sep_aspect_sep_indices,
                'cls_text_sep_aspect_sep_length': cls_text_sep_aspect_sep_length,
                'cls_text_sep_aspect_sep_segments_ids': cls_text_sep_aspect_sep_segments_ids,
                'cls_text_sep_aspect_sep_aspect_mask': cls_text_sep_aspect_sep_aspect_mask,
                'cls_text_sep_aspect_sep_mask': cls_text_sep_aspect_sep_mask,
                'cls_text_sep_indices': cls_text_sep_indices,
                'cls_text_sep_length': cls_text_sep_length,
                'cls_text_sep_segments_ids': cls_text_sep_segments_ids,
                'anchor_index': anchor_index,
                'transform': transform,
                'sentence_length': sent_len,
                'bert_length': bert_len,
                'polarity': target,
                'dependency_graph': dep_matrix,
                'mask': mask,
                'dist_to_target': pad_dist_to_target
            }
            data.append(item)
    preprocessed_file = path.replace('.tsv', '.proc')
    with open(preprocessed_file, 'wb') as f:
        pickle.dump(data, f)
    print(preprocessed_file)
    # Ends here


def process_ace2_file(path):
    BERT_ML = ACE2_UNCASE['bert_ml']
    ORI_ML = ACE2_UNCASE['ori_ml']

    pretrained_bert_name = 'bert-base-uncased'
    tokenizer = BertTokenizer.from_pretrained(pretrained_bert_name)
    fin = open('{}.graph'.format(path), 'rb')
    idx2gragh = pickle.load(fin)
    fin.close()

    raw_data = read_ace2_file(path)
    data = []
    for sample_id, tokens, label, target, anchor_index in raw_data:
        sent_len = len(tokens)
        # On raw text
        tok_bert_indices = []
        for tok in tokens:
            tok = tok.replace('...', '.').replace('...', '.').replace('...', '.')
            bert_tokens = tokenizer.tokenize(tok)
            tok_bert_indices.append(tokenizer.convert_tokens_to_ids(bert_tokens))
        bert_len = sum([len(x) for x in tok_bert_indices])

        assert bert_len <= BERT_ML, 'Bert length: {}\n{}'.format(bert_len, tokens)
        assert sent_len <= ORI_ML, 'Ori length: {}'.format(sent_len)

        transform = zeros(ORI_ML, BERT_ML, 0.0)

        # Create transform to convert tokenized length to original length
        offset = 1
        raw_text_bert_indices = []
        # print(sample_id)
        # print(sum([len(x) for x in tok_bert_indices]))
        # print(' '.join(tokens))
        for i, indices in enumerate(tok_bert_indices):
            l = len(indices)
            raw_text_bert_indices += indices
            for j in range(l):
                assert i <= ORI_ML, '| i={}'.format(i)
                assert offset + j < BERT_ML, '| offset={} j={}, sum={}'.format(offset, j, offset + j)
                transform[i][offset + j] = 1 / l
            # if i == anchor_index:
            #     bert_anchor_index = offset
            offset += l

        dep_matrix = idx2gragh[sample_id]

        aspect_indices = tok_bert_indices[anchor_index]

        #   CLS + Sentence + SEP + aspect + SEP
        cls_text_sep_aspect_sep_indices = [101] + raw_text_bert_indices + [102] + aspect_indices + [102]

        cls_text_sep_aspect_sep_length = len(cls_text_sep_aspect_sep_indices)
        assert cls_text_sep_aspect_sep_length <= BERT_ML, 'CLS indices length: {}\n{}'.format(
            cls_text_sep_aspect_sep_length, tokens)
        cls_text_sep_aspect_sep_mask = [0 for _ in range(cls_text_sep_aspect_sep_length)] + [1 for _ in range(
            BERT_ML - cls_text_sep_aspect_sep_length)]
        cls_text_sep_aspect_sep_indices += [0 for _ in range(BERT_ML - cls_text_sep_aspect_sep_length)]
        cls_text_sep_aspect_sep_segments_ids = [0 for _ in range(BERT_ML)]

        for i in range(bert_len + 2, bert_len + 2 + len(aspect_indices) + 1):
            cls_text_sep_aspect_sep_segments_ids[i] = 1

        cls_text_sep_aspect_sep_aspect_mask = [1 for _ in range(BERT_ML)]
        for i in range(offset, offset + len(aspect_indices)):
            cls_text_sep_aspect_sep_aspect_mask[i] = 0

        # CLS + sentence + SEP
        cls_text_sep_indices = [101] + raw_text_bert_indices + [102]
        cls_text_sep_length = len(cls_text_sep_indices)
        cls_text_sep_indices += [0 for x in range(BERT_ML - cls_text_sep_length)]
        cls_text_sep_segments_ids = [0 for _ in range(BERT_ML)]

        # Original sentence
        mask = [0 for _ in range(sent_len)] + [1 for _ in range(ORI_ML - sent_len)]
        assert len(mask) == ORI_ML, "Mask length mismatch"
        dist = [-1 for i in range(sent_len)]
        dist_to_target = get_dist_to_target(dep_matrix, anchor_index, dist, sent_len)
        dist_padding = [0] * (ORI_ML - len(dist_to_target))
        pad_dist_to_target = dist_to_target + dist_padding

        # pad_aspect_indices = aspect_indices + [BERT_ML - len(aspect_indices)]

        item = {
            'cls_text_sep_aspect_sep_indices': cls_text_sep_aspect_sep_indices,
            'cls_text_sep_aspect_sep_length': cls_text_sep_aspect_sep_length,
            'cls_text_sep_aspect_sep_segments_ids': cls_text_sep_aspect_sep_segments_ids,
            'cls_text_sep_aspect_sep_aspect_mask': cls_text_sep_aspect_sep_aspect_mask,
            'cls_text_sep_aspect_sep_mask': cls_text_sep_aspect_sep_mask,
            'cls_text_sep_indices': cls_text_sep_indices,
            'cls_text_sep_length': cls_text_sep_length,
            'cls_text_sep_segments_ids': cls_text_sep_segments_ids,
            'anchor_index': anchor_index,
            'transform': transform,
            'sentence_length': sent_len,
            'bert_length': bert_len,
            'polarity': target,
            'dependency_graph': dep_matrix,
            'mask': mask,
            'dist_to_target': pad_dist_to_target
        }
        offset += len(aspect_indices)

        data.append(item)
    preprocessed_file = path.replace('.tsv', '.proc')
    with open(preprocessed_file, 'wb') as f:
        pickle.dump(data, f)
    print(preprocessed_file)


def process_ace_file(path):
    BERT_ML = ACE_CASE['bert_ml']
    ORI_ML = ACE_CASE['ori_ml']
    MAX_TARGET_VALUE = ACE_CASE['n_class']
    tokenizer = BertTokenizer.from_pretrained(ACE_CASE['pretrained_bert_name'], do_lower_case=False)

    fin = open('{}.graph'.format(path), 'rb')
    idx2gragh = pickle.load(fin)
    fin.close()

    raw_data = read_ace_file(path)
    data = []
    for sample_id, tokens, labels, targets in raw_data:
        sent_len = len(tokens)
        # On raw text
        tok_bert_indices = []
        for tok in tokens:
            bert_tokens = tokenizer.tokenize(tok)
            tok_bert_indices.append(tokenizer.convert_tokens_to_ids(bert_tokens))
        bert_len = sum([len(x) for x in tok_bert_indices])

        transform = zeros(ORI_ML, BERT_ML, 0.0)

        # Create transform to convert tokenized length to original length
        offset = 1
        raw_text_bert_indices = []
        # print(sample_id)
        # print(sum([len(x) for x in tok_bert_indices]))
        # print(' '.join(tokens))
        for i, indices in enumerate(tok_bert_indices):
            l = len(indices)
            raw_text_bert_indices += indices
            for j in range(l):
                transform[i][offset + j] = 1 / l
            offset += l

        dep_matrix = idx2gragh[sample_id]

        offset = 1  # Because of the CLS
        for anchor_index, (aspect_indices, target) in enumerate(zip(tok_bert_indices, targets)):

            # Discard O label (it is not Other label)
            if target > MAX_TARGET_VALUE:
                continue
            #   CLS + Sentence + SEP + aspect + SEP
            cls_text_sep_aspect_sep_indices = [101] + raw_text_bert_indices + [102] + aspect_indices + [102]
            cls_text_sep_aspect_sep_length = len(cls_text_sep_aspect_sep_indices)
            cls_text_sep_aspect_sep_mask = [0 for _ in range(cls_text_sep_aspect_sep_length)] + [1 for _ in range(
                BERT_ML - cls_text_sep_aspect_sep_length)]
            cls_text_sep_aspect_sep_indices += [0 for _ in range(BERT_ML - cls_text_sep_aspect_sep_length)]
            cls_text_sep_aspect_sep_segments_ids = [0 for _ in range(BERT_ML)]
            for i in range(bert_len + 2, bert_len + 2 + len(aspect_indices) + 1):
                cls_text_sep_aspect_sep_segments_ids[i] = 1

            cls_text_sep_aspect_sep_aspect_mask = [1 for _ in range(BERT_ML)]
            for i in range(offset, offset + len(aspect_indices)):
                cls_text_sep_aspect_sep_aspect_mask[i] = 0

            offset += len(aspect_indices)

            # CLS + sentence + SEP
            cls_text_sep_indices = [101] + raw_text_bert_indices + [102]
            cls_text_sep_length = len(cls_text_sep_indices)
            cls_text_sep_indices += [0 for x in range(BERT_ML - cls_text_sep_length)]
            cls_text_sep_segments_ids = [0 for _ in range(BERT_ML)]

            # Original sentence
            mask = [0 for _ in range(sent_len)] + [1 for _ in range(ORI_ML - sent_len)]
            dist = [-1 for i in range(sent_len)]
            dist_to_target = get_dist_to_target(dep_matrix, anchor_index, dist, sent_len)
            max_dist = max(dist_to_target) + 1
            dist_padding = [max_dist] * (ORI_ML - len(dist_to_target))
            pad_dist_to_target = dist_to_target + dist_padding

            item = {
                'cls_text_sep_aspect_sep_indices': cls_text_sep_aspect_sep_indices,
                'cls_text_sep_aspect_sep_length': cls_text_sep_aspect_sep_length,
                'cls_text_sep_aspect_sep_segments_ids': cls_text_sep_aspect_sep_segments_ids,
                'cls_text_sep_aspect_sep_aspect_mask': cls_text_sep_aspect_sep_aspect_mask,
                'cls_text_sep_aspect_sep_mask': cls_text_sep_aspect_sep_mask,
                'cls_text_sep_indices': cls_text_sep_indices,
                'cls_text_sep_length': cls_text_sep_length,
                'cls_text_sep_segments_ids': cls_text_sep_segments_ids,
                'anchor_index': anchor_index,
                'transform': transform,
                'sentence_length': sent_len,
                'bert_length': bert_len,
                'polarity': target,
                'dependency_graph': dep_matrix,
                'mask': mask,
                'dist_to_target': pad_dist_to_target
            }
            data.append(item)
    preprocessed_file = path.replace('.tsv', '.proc')
    with open(preprocessed_file, 'wb') as f:
        pickle.dump(data, f)
    print(preprocessed_file)


def process_ace34_file(path):
    BERT_ML = ACE34_UNCASE['bert_ml']
    ORI_ML = ACE34_UNCASE['ori_ml']

    pretrained_bert_name = ACE34_UNCASE['pretrained_bert_name']
    tokenizer = BertTokenizer.from_pretrained(pretrained_bert_name)
    fin = open('{}.graph'.format(path), 'rb')
    idx2gragh = pickle.load(fin)
    fin.close()

    raw_data = read_ace34_file(path)
    data = []
    for sample_id, tokens, label, target, anchor_index in raw_data:
        sent_len = len(tokens)
        # On raw text
        tok_bert_indices = []
        for tok in tokens:
            tok = tok.replace('...', '.').replace('...', '.').replace('...', '.')
            bert_tokens = tokenizer.tokenize(tok)
            tok_bert_indices.append(tokenizer.convert_tokens_to_ids(bert_tokens))
        bert_len = sum([len(x) for x in tok_bert_indices])

        assert bert_len <= BERT_ML, 'Bert length: {}\n{}'.format(bert_len, tokens)
        assert sent_len <= ORI_ML, 'Ori length: {}'.format(sent_len)

        transform = zeros(ORI_ML, BERT_ML, 0.0)

        # Create transform to convert tokenized length to original length
        offset = 1
        raw_text_bert_indices = []
        # print(sample_id)
        # print(sum([len(x) for x in tok_bert_indices]))
        # print(' '.join(tokens))
        for i, indices in enumerate(tok_bert_indices):
            l = len(indices)
            raw_text_bert_indices += indices
            for j in range(l):
                assert i <= ORI_ML, '| i={}'.format(i)
                assert offset + j < BERT_ML, '| offset={} j={}, sum={}'.format(offset, j, offset + j)
                transform[i][offset + j] = 1 / l
            # if i == anchor_index:
            #     bert_anchor_index = offset
            offset += l

        assert offset == 1 + len(raw_text_bert_indices), "Wrong offset"

        dep_matrix = idx2gragh[sample_id]

        aspect_indices = tok_bert_indices[anchor_index]

        #   CLS + Sentence + SEP + aspect + SEP
        cls_text_sep_aspect_sep_indices = [101] + raw_text_bert_indices + [102] + aspect_indices + [102]

        cls_text_sep_aspect_sep_length = len(cls_text_sep_aspect_sep_indices)
        assert cls_text_sep_aspect_sep_length <= BERT_ML, 'CLS indices length: {}\n{}'.format(
            cls_text_sep_aspect_sep_length, tokens)
        cls_text_sep_aspect_sep_mask = [0 for _ in range(cls_text_sep_aspect_sep_length)] + [1 for _ in range(
            BERT_ML - cls_text_sep_aspect_sep_length)]
        cls_text_sep_aspect_sep_indices += [0 for _ in range(BERT_ML - cls_text_sep_aspect_sep_length)]
        cls_text_sep_aspect_sep_segments_ids = [0 for _ in range(BERT_ML)]

        for i in range(bert_len + 2, bert_len + 2 + len(aspect_indices) + 1):
            cls_text_sep_aspect_sep_segments_ids[i] = 1

        cls_text_sep_aspect_sep_aspect_mask = [1 for _ in range(BERT_ML)]
        for i in range(offset + 1, offset + 1 + len(aspect_indices)):
            cls_text_sep_aspect_sep_aspect_mask[i] = 0

        # CLS + sentence + SEP
        cls_text_sep_indices = [101] + raw_text_bert_indices + [102]
        cls_text_sep_length = len(cls_text_sep_indices)
        cls_text_sep_indices += [0 for x in range(BERT_ML - cls_text_sep_length)]
        cls_text_sep_segments_ids = [0 for _ in range(BERT_ML)]

        # Original sentence
        mask = [0 for _ in range(sent_len)] + [1 for _ in range(ORI_ML - sent_len)]
        dist = [-1 for i in range(sent_len)]
        dist_to_target = get_dist_to_target(dep_matrix, anchor_index, dist, sent_len)
        dist_padding = [0] * (ORI_ML - len(dist_to_target))
        pad_dist_to_target = dist_to_target + dist_padding

        # pad_aspect_indices = aspect_indices + [BERT_ML - len(aspect_indices)]

        item = {
            'token': tokens,
            'cls_text_sep_aspect_sep_indices': cls_text_sep_aspect_sep_indices,
            'cls_text_sep_aspect_sep_length': cls_text_sep_aspect_sep_length,
            'cls_text_sep_aspect_sep_segments_ids': cls_text_sep_aspect_sep_segments_ids,
            'cls_text_sep_aspect_sep_aspect_mask': cls_text_sep_aspect_sep_aspect_mask,
            'cls_text_sep_aspect_sep_mask': cls_text_sep_aspect_sep_mask,
            'cls_text_sep_indices': cls_text_sep_indices,
            'cls_text_sep_length': cls_text_sep_length,
            'cls_text_sep_segments_ids': cls_text_sep_segments_ids,
            'anchor_index': anchor_index,
            'transform': transform,
            'sentence_length': sent_len,
            'bert_length': bert_len,
            'polarity': target,
            'dependency_graph': dep_matrix,
            'mask': mask,
            'dist_to_target': pad_dist_to_target
        }
        # offset += len(aspect_indices)

        data.append(item)
    preprocessed_file = path.replace('.tsv', '.proc')
    with open(preprocessed_file, 'wb') as f:
        pickle.dump(data, f)
    print(preprocessed_file)


def preprocess_litbank_all():
    folders = []
    folders += sorted(
        [os.path.join('datasets/litbank-cased/train', x) for x in os.listdir('datasets/litbank-cased/train') if
         x.endswith('.tsv')])
    folders += sorted(
        [os.path.join('datasets/litbank-cased/dev', x) for x in os.listdir('datasets/litbank-cased/dev') if
         x.endswith('.tsv')])
    folders += sorted(
        [os.path.join('datasets/litbank-cased/test', x) for x in os.listdir('datasets/litbank-cased/test') if
         x.endswith('.tsv')])
    pool = multiprocessing.Pool(32)
    pool.map(process_litbank_file, folders)
    pool.close()


def preprocess_ace2_all():
    folders = []
    folders += sorted(
        [os.path.join('datasets/ace2-uncased/train', x) for x in os.listdir('datasets/ace2-uncased/train') if
         x.endswith('.tsv')])
    folders += sorted([os.path.join('datasets/ace2-uncased/dev', x) for x in os.listdir('datasets/ace2-uncased/dev') if
                       x.endswith('.tsv')])
    folders += sorted(
        [os.path.join('datasets/ace2-uncased/test', x) for x in os.listdir('datasets/ace2-uncased/test') if
         x.endswith('.tsv')])
    pool = multiprocessing.Pool(32)
    pool.map(process_ace2_file, folders)
    pool.close()


def preprocess_ace_all():
    folders = []
    folders += sorted([os.path.join('datasets/ace-cased/train', x) for x in os.listdir('datasets/ace-cased/train') if
                       x.endswith('.tsv')])
    folders += sorted([os.path.join('datasets/ace-cased/dev', x) for x in os.listdir('datasets/ace-cased/dev') if
                       x.endswith('.tsv')])
    folders += sorted([os.path.join('datasets/ace-cased/test', x) for x in os.listdir('datasets/ace-cased/test') if
                       x.endswith('.tsv')])
    pool = multiprocessing.Pool(32)
    pool.map(process_ace_file, folders)
    pool.close()


def preprocess_ace34_all():
    folders = []
    base = 'datasets/ace34-uncased'
    train = os.path.join(base, 'train')
    dev = os.path.join(base, 'dev')
    test = os.path.join(base, 'test')
    folders += sorted([os.path.join(train, x) for x in os.listdir(train) if x.endswith('.tsv')])
    folders += sorted([os.path.join(dev, x) for x in os.listdir(dev) if x.endswith('.tsv')])
    folders += sorted([os.path.join(test, x) for x in os.listdir(test) if x.endswith('.tsv')])
    pool = multiprocessing.Pool(32)
    pool.map(process_ace34_file, folders)
    pool.close()


def analyze_length():
    raw_data = read_folder('datasets/litbank/dev')
    raw_data += read_folder('datasets/litbank/test')
    raw_data += read_folder('datasets/litbank/train')
    len_counter = collections.Counter()
    lengths = [len(x[3]) for x in raw_data]
    len_counter.update(lengths)
    dist = list(len_counter.items())
    dist = sorted(dist, key=lambda x: x[0])
    for x, y in dist:
        print(x, y)


def fix_length():
    files = []
    path = 'datasets/litbank'
    train = os.path.join(path, 'train')
    dev = os.path.join(path, 'dev')
    test = os.path.join(path, 'test')
    files += sorted([os.path.join(train, x) for x in os.listdir(train) if x.endswith('tsv')])
    files += sorted([os.path.join(dev, x) for x in os.listdir(dev) if x.endswith('tsv')])
    files += sorted([os.path.join(test, x) for x in os.listdir(test) if x.endswith('tsv')])
    # print(files)
    # pool = mdiles)
    for file in files:
        read_litbank_file(file)


def test_dataset():
    train = LitbankDataset('datasets/ace34/train', None, [])
    print(len(train))
    dev = LitbankDataset('datasets/ace34/dev', None, [])
    print(len(dev))
    test = LitbankDataset('datasets/ace34/test', None, [])
    print(len(test))


if __name__ == '__main__':
    # read_litbank_file('')
    #
    # preprocess_ace2_all()
    # preprocess_ace_all()
    # preprocess_litbank_all()
    # preprocess_ace34_all()

    files=['datasets/litbank-cased/train/gpt-{}.tsv'.format(i) for i in range(14)]

    p = multiprocessing.Pool(15)
    p.map(process_litbank_file, files)

    # test_dataset()
    # process_a_file('datasets/litbank/train/730_oliver_twist_brat.tsv')
    # process_a_file('datasets/litbank/train/8867_the_magnificent_ambersons_brat.tsv')
    # analyze_length()
    # check_data()
