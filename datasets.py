#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

def get_dataset_paths(data_root, dataset_tok):
    if dataset_tok == "iwslt17":
        train_src_corpus = "{}/iwslt17.case.vi.bpe32k.filtered".format(data_root)
        train_tgt_corpus = "{}/iwslt17.case.en.bpe32k.filtered".format(data_root)
        train_cfg_corpus = "{}/iwslt17.case.en.cfg.filtered".format(data_root)
        src_vocab_path = "{}/iwslt17.case.vi.bpe32k.vocab".format(data_root)
        cfg_vocab_path = "{}/iwslt17.case.en.treelstm.vocab".format(data_root)
        assert os.path.exists(cfg_vocab_path)

    if dataset_tok == "iwslt14":
        train_src_corpus = "{}/iwslt14.de.bpe20k.filtered".format(data_root)
        train_tgt_corpus = "{}/iwslt14.case.en.bpe32k.filtered".format(data_root)
        train_cfg_corpus = "{}/iwslt14.en.cfg.filtered".format(data_root)
        src_vocab_path = "{}/iwslt14.de.bpe20k.vocab".format(data_root)
        cfg_vocab_path = "{}/iwslt14.case.en.treelstm.vocab".format(data_root)
        assert os.path.exists(cfg_vocab_path)

    if dataset_tok == "wmt14":
        train_src_corpus = "{}/wmt14.de.sp.filtered".format(data_root)
        train_tgt_corpus = "{}/wmt14.en.sp.filtered".format(data_root)
        train_cfg_corpus = "{}/wmt14.en.cfg.filtered".format(data_root)
        src_vocab_path = "{}/wmt14.de.sp.vocab".format(data_root)
        cfg_vocab_path = "{}/wmt14.treelstm.vocab".format(data_root)
        test_src_corpus = "{}/wmt14_deen_test.de.sp".format(data_root)
        test_cfg_corpus = "{}/wmt14_deen_test.en.cfg.oneline".format(data_root)

    assert os.path.exists(cfg_vocab_path)
    return dict(
        train_src_corpus=train_src_corpus,
        train_tgt_corpus=train_tgt_corpus,
        train_cfg_corpus=train_cfg_corpus,
        src_vocab_path=src_vocab_path,
        cfg_vocab_path=cfg_vocab_path,
        test_src_corpus=test_src_corpus,
        test_cfg_corpus=test_cfg_corpus
    )