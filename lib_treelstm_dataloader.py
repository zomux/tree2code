#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
from torch.nn.utils.rnn import pad_sequence
import numpy as np
import dgl

from lib_treedata import TreeDataGenerator
from nmtlab.utils import Vocab, OPTS
from nmtlab.dataset import Dataset


class BilingualTreeDataLoader(Dataset):

    def __init__(self, src_path, cfg_path, src_vocab_path, treelstm_vocab_path, cache_path=None,
                 batch_size=64, max_tokens=80,
                 part_index=0, part_num=1,
                 load_data=True,
                 truncate=None,
                 limit_datapoints=None,
                 limit_tree_depth=0):
        self._max_tokens = max_tokens
        self._src_path = src_path
        self._src_vocab_path = src_vocab_path
        self._cfg_path = cfg_path
        self._treelstm_vocab_path = treelstm_vocab_path
        self._src_vocab = Vocab(self._src_vocab_path)
        self._label_vocab = Vocab(self._treelstm_vocab_path)
        self._cache_path = cache_path
        self._truncate = truncate
        self._part_index = part_index
        self._part_num = part_num
        self._limit_datapoints = limit_datapoints
        self._limit_tree_depth = limit_tree_depth
        self._rand = np.random.RandomState(3)
        if load_data:
            train_data, valid_data = self._load_data()
        self._n_train_samples = len(train_data)
        super(BilingualTreeDataLoader, self).__init__(train_data=train_data, valid_data=valid_data, batch_size=batch_size)

    def _load_data(self):
        src_vocab = self._src_vocab
        src_lines = open(self._src_path).readlines()
        partition_size = int(len(src_lines) / self._part_num)
        src_lines = src_lines[self._part_index * partition_size: (self._part_index + 1) * partition_size]
        treegen = TreeDataGenerator(self._cfg_path, self._treelstm_vocab_path,
                                    cache_path=self._cache_path,
                                    part_index=self._part_index, part_num=self._part_num,
                                    limit_datapoints=self._limit_datapoints,
                                    limit_tree_depth=self._limit_tree_depth)
        treegen.load()
        trees = treegen.trees()
        if self._limit_datapoints > 0:
            src_lines = src_lines[:self._limit_datapoints]
        data_pairs = []
        assert len(src_lines) == len(trees)
        for src, paired_tree in zip(src_lines, trees):
            if paired_tree is None:
                continue
            enc_tree, dec_tree = paired_tree
            src_ids = src_vocab.encode(src.strip().split())
            if len(src_ids) > self._max_tokens:
                continue
            data_pairs.append((src_ids, enc_tree, dec_tree))
        if self._truncate is not None:
            data_pairs = data_pairs[:self._truncate]
        if len(data_pairs) < len(src_lines):
            missing_num = len(src_lines) - len(data_pairs)
            extra_indexes = np.random.RandomState(3).choice(np.arange(len(data_pairs)), missing_num)
            extra_data = [data_pairs[i] for i in extra_indexes.tolist()]
            data_pairs.extend(extra_data)
        np.random.RandomState(3).shuffle(data_pairs)
        valid_data = data_pairs[:1000]
        train_data = data_pairs[1000:]
        return train_data, valid_data

    def set_gpu_scope(self, scope_index, n_scopes):
        self._batch_size = int(self._batch_size / n_scopes)

    def n_train_samples(self):
        return len(self._train_data)

    def train_set(self):
        self._rand.shuffle(self._train_data)
        return self._train_iterator()

    def _train_iterator(self):
        for i in range(self.n_train_batch()):
            samples = self._train_data[i * self._batch_size: (i + 1) * self._batch_size]
            yield self.batch(samples)

    def batch(self, samples):
        src_samples = [x[0] for x in samples]
        enc_trees = [x[1] for x in samples]
        dec_trees = [x[2] for x in samples]
        src_batch = pad_sequence([torch.tensor(x) for x in src_samples], batch_first=True)
        enc_tree_batch = dgl.batch(enc_trees)
        dec_tree_batch = dgl.batch(dec_trees)
        return src_batch, enc_tree_batch, dec_tree_batch

    def valid_set(self):
        return self._valid_iterator()

    def _valid_iterator(self):
        n_batches = int(len(self._valid_data) / self._batch_size)
        for i in range(n_batches):
            samples = self._valid_data[i * self._batch_size: (i + 1) * self._batch_size]
            yield self.batch(samples)

    def src_vocab(self):
        return self._src_vocab

    def label_vocab(self):
        return self._label_vocab

    def yield_all_batches(self, batch_size=128):
        OPTS.tinydata = False
        src_vocab = self._src_vocab
        data = []
        src_lines = open(self._src_path).readlines()
        cfg_lines = open(self._cfg_path).readlines()
        assert len(src_lines) == len(cfg_lines)
        print("start to batch {} samples".format(len(src_lines)))
        treegen = TreeDataGenerator(self._cfg_path, self._treelstm_vocab_path,
                                    part_index=0, part_num=1,
                                    limit_tree_depth=self._limit_tree_depth)
        batch_samples = []
        batch_src_lines = []
        for src_line, cfg_line in zip(src_lines, cfg_lines):
            src_line = src_line.strip()
            cfg_line = cfg_line.strip()
            paired_tree = treegen.build_trees(cfg_line)
            if paired_tree is None:
                continue
            enc_tree, dec_tree = paired_tree
            src_ids = src_vocab.encode(src_line.split())
            if len(src_ids) > self._max_tokens:
                continue
            batch_samples.append((src_ids, enc_tree, dec_tree))
            batch_src_lines.append((src_line, cfg_line))
            if len(batch_samples) >= batch_size:
                src_batch, enc_tree_batch, dec_tree_batch = self.batch(batch_samples)
                src_line_batch = [x[0] for x in batch_src_lines]
                cfg_line_batch = [x[1] for x in batch_src_lines]
                yield src_line_batch, cfg_line_batch, src_batch, enc_tree_batch, dec_tree_batch
                batch_src_lines.clear()
                batch_samples.clear()
        if len(batch_samples):
            src_batch, enc_tree_batch, dec_tree_batch = self.batch(batch_samples)
            src_line_batch = [x[0] for x in batch_src_lines]
            cfg_line_batch = [x[1] for x in batch_src_lines]
            yield src_line_batch, cfg_line_batch, src_batch, enc_tree_batch, dec_tree_batch
