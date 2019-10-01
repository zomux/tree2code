#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import networkx as nx
import os
import sys
import dgl
import re
from nmtlab.utils import Vocab
from nmtlab.utils import OPTS
from nltk.tree import Tree
import _pickle as pickle

PAD = -1
UNK = 3


class TreeDataGenerator(object):

    def __init__(self, cfg_path, treelstm_vocab_path, part_index=0, part_num=1, cache_path=None, limit_datapoints=None,
                 limit_tree_depth=0):
        if cache_path is not None:
            self._cache_path = "{}.{}in{}".format(cache_path, part_index, part_num)
        else:
            self._cache_path = None
        self._cfg_path = cfg_path
        self._cfg_lines = None
        self._part_index = part_index
        self._part_num = part_num
        self._limit_datapoints = limit_datapoints
        self._limit_tree_depth = limit_tree_depth
        self._vocab = Vocab(treelstm_vocab_path, picklable=True)
        self._trees = []

    def load(self):
        if not OPTS.smalldata and not OPTS.tinydata and self._cache_path is not None and os.path.exists(self._cache_path):
            print("loading cached trees part {} ...".format(self._part_index))
            self._trees = pickle.load(open(self._cache_path, "rb"))
            return
        self._cfg_lines = open(self._cfg_path).readlines()
        partition_size = int(len(self._cfg_lines) / self._part_num)
        self._cfg_lines = self._cfg_lines[self._part_index * partition_size: (self._part_index + 1) * partition_size]
        if self._limit_datapoints > 0:
            self._cfg_lines = self._cfg_lines[:self._limit_datapoints]
        print("building trees part {} ...".format(self._part_index))
        self._trees = self._build_batch_trees()
        if False and self._cache_path is not None:
            print("caching trees...")
            pickle.dump(self._trees, open(self._cache_path, "wb"))

    def _parse_cfg_line(self, cfg_line):
        t = cfg_line.strip()
        # Replace leaves of the form (!), (,), with (! !), (, ,)
        t = re.sub(r"\((.)\)", r"(\1 \1)", t)
        # Replace leaves of the form (tag word root) with (tag word)
        t = re.sub(r"\(([^\s()]+) ([^\s()]+) [^\s()]+\)", r"(\1 \2)", t)
        try:
            tree = Tree.fromstring(t)
        except ValueError as e:
            tree = None
        return tree

    def _build_batch_trees(self):
        trees = []
        for line in self._cfg_lines:
            paired_tree = self.build_trees(line)
            trees.append(paired_tree)
        return trees

    def build_trees(self, cfg_line):
        parse = self._parse_cfg_line(cfg_line)
        if parse is None or not parse.leaves():
            return None
        enc_g = nx.DiGraph()
        dec_g = nx.DiGraph()
        failed = False

        def _rec_build(id_enc, id_dec, node, depth=0):
            if len(node) > 10:
                return
            if self._limit_tree_depth > 0 and depth >= self._limit_tree_depth:
                return
            # Skipp all terminals
            all_terminals = True
            for child in node:
                if not isinstance(child[0], str) and not isinstance(child[0], bytes):
                    all_terminals = False
                    break
            if all_terminals:
                return
            for j, child in enumerate(node):
                cid_enc = enc_g.number_of_nodes()
                cid_dec = dec_g.number_of_nodes()

                # Avoid leaf nodes
                tagid_enc = self._vocab.encode_token("{}_1".format(child.label()))
                tagid_dec = self._vocab.encode_token("{}_{}".format(node.label(), j+1))
                # assert tagid_enc != UNK and tagid_dec != UNK
                enc_g.add_node(cid_enc, x=tagid_enc, mask=0)
                dec_g.add_node(cid_dec, x=tagid_dec, y=tagid_enc, pos=j, mask=0, depth=depth+1)
                enc_g.add_edge(cid_enc, id_enc)
                dec_g.add_edge(id_dec, cid_dec)
                if not isinstance(child[0], str) and not isinstance(child[0], bytes):
                    _rec_build(cid_enc, cid_dec, child, depth=depth + 1)

        if parse.label() == "ROOT" and len(parse) == 1:
            # Skip the root node
            parse = parse[0]
        root_tagid = self._vocab.encode_token("{}_1".format(parse.label()))
        enc_g.add_node(0, x=root_tagid, mask=1)
        dec_g.add_node(0, x=self._vocab.encode_token("ROOT_1"), y=root_tagid, pos=0, mask=1, depth=0)
        _rec_build(0, 0, parse)
        if failed:
            return None
        enc_graph = dgl.DGLGraph()
        enc_graph.from_networkx(enc_g, node_attrs=['x', 'mask'])
        dec_graph = dgl.DGLGraph()
        dec_graph.from_networkx(dec_g, node_attrs=['x', 'y', 'pos', 'mask', 'depth'])
        return enc_graph, dec_graph

    def trees(self):
        return self._trees


if __name__ == '__main__':

    cfg_content = """
(ROOT (NP (NP (NNP Rachel) (NNP Pike)) (: :) (NP (NP (DT the) (NN science)) (PP (IN behind) (NP (DT a) (NN climate) (NN headline))))))
(ROOT (S (NP (PRP they)) (VP (VBP are) (NP (NP (DT both) (CD two) (NNS branches)) (PP (IN of) (NP (NP (DT the) (JJ same) (NN field)) (PP (IN of) (NP (JJ atmospheric) (NN science))))))) (. .)))
"""
    open("/tmp/tmp_cfg.txt", "w").write(cfg_content.strip())
    treelstm_vocab_path = "{}/data/stnmt/processed_data/aspec.case.en.treelstm.vocab".format(os.getenv("HOME"))
    gen = TreeDataGenerator("/tmp/tmp_cfg.txt", treelstm_vocab_path)
    gen.load()
