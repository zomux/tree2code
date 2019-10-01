#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import dgl.init

from lib_treelstm import TreeLSTMCell
from lib_treelstm import DecoderTreeLSTMCell
from lib_treelstm_dataloader import BilingualTreeDataLoader
from lib_semhash import SemanticHashing

from nmtlab.modules.transformer_modules import TransformerEmbedding
from nmtlab.modules.transformer_modules import TransformerEncoderLayer

class TreeAutoEncoder(nn.Module):

    def __init__(self, dataset, hidden_size=256, code_bits=5, without_source=False, dropout_ratio=0.1):
        super(TreeAutoEncoder, self).__init__()
        assert isinstance(dataset, BilingualTreeDataLoader)
        self.hidden_size = hidden_size
        self._vocab_size = dataset.src_vocab().size()
        self._label_size = dataset.label_vocab().size()
        self._code_bits = code_bits
        self._without_source = without_source

        # Encoder
        self.src_embed_layer = TransformerEmbedding(self._vocab_size, self.hidden_size, dropout_ratio=dropout_ratio)
        self.encoder_norm = nn.LayerNorm(self.hidden_size)
        self.encoder_layers = nn.ModuleList()
        ff_size = hidden_size * 4
        for _ in range(3):
            layer = TransformerEncoderLayer(self.hidden_size, ff_size, n_att_head=8, dropout_ratio=dropout_ratio)
            self.encoder_layers.append(layer)

        self.label_embed_layer = nn.Embedding(self._label_size, self.hidden_size)
        self.enc_cell = TreeLSTMCell(hidden_size, hidden_size)
        self.dec_cell = DecoderTreeLSTMCell(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout_ratio)
        self.logit_nn = nn.Linear(self.hidden_size, self._label_size)
        if code_bits > 0:
            self.semhash = SemanticHashing(hidden_size, bits=code_bits)
        else:
            self.semhash = None
        self.initialize_parameters()

    def initialize_parameters(self):
        """Initialize the parameters in the model."""
        # Initialize weights
        for param in self.parameters():
            shape = param.shape
            if len(shape) > 1:
                nn.init.xavier_uniform_(param)
        for module in self.modules():
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.constant_(module.bias, 0.0)

    def encode_source(self, src_seq, src_mask, meanpool=False):
        src_seq = src_seq.long()
        x = self.src_embed_layer(src_seq)
        for l, layer in enumerate(self.encoder_layers):
            x = layer(x, src_mask)
        encoder_states = self.encoder_norm(x)
        if meanpool:
            encoder_states = encoder_states * src_mask.unsqueeze(-1)
            encoder_states = encoder_states.sum(1) / (src_mask.sum(1).unsqueeze(-1) + 10e-8)

        encoder_outputs = {
            "encoder_states": encoder_states,
            "src_mask": src_mask
        }
        return encoder_outputs

    def forward(self, src, enc_tree, dec_tree, return_code=False, **kwargs):
        self._init_graph(enc_tree, dec_tree)
        # Soruce encoding
        src_mask = torch.ne(src, 0).float()
        encoder_outputs = self.encode_source(src, src_mask, meanpool=True)
        encoder_states = encoder_outputs["encoder_states"]
        # Tree encoding
        enc_x = enc_tree.ndata["x"].cuda()
        x_embeds = self.label_embed_layer(enc_x)
        enc_tree.ndata['iou'] = self.enc_cell.W_iou(self.dropout(x_embeds))
        enc_tree.ndata['h'] = torch.zeros((enc_tree.number_of_nodes(), self.hidden_size)).cuda()
        enc_tree.ndata['c'] = torch.zeros((enc_tree.number_of_nodes(), self.hidden_size)).cuda()
        enc_tree.ndata['mask'] = enc_tree.ndata['mask'].float().cuda()
        dgl.prop_nodes_topo(enc_tree)
        # Obtain root representation
        root_mask = enc_tree.ndata["mask"].float().cuda()
        # root_idx = torch.arange(root_mask.shape[0])[root_mask > 0].cuda()
        root_h = self.dropout(enc_tree.ndata.pop("h")) * root_mask.unsqueeze(-1)
        orig_h = root_h[root_mask > 0]
        partial_h = orig_h
        if self._without_source:
            partial_h += encoder_states

        # Discretization
        if self._code_bits > 0:
            if return_code:
                codes = self.semhash(partial_h, return_code=True)
                ret = {"codes": codes}
                return ret
            else:
                partial_h = self.semhash(partial_h)
            if not self._without_source:
                partial_h += encoder_states

        root_h[root_mask > 0] = partial_h
        # Tree decoding
        dec_x = dec_tree.ndata["x"].cuda()
        dec_embeds = self.label_embed_layer(dec_x)
        dec_tree.ndata['iou'] = self.dec_cell.W_iou(self.dropout(dec_embeds))
        dec_tree.ndata['h'] = root_h
        dec_tree.ndata['c'] = torch.zeros((enc_tree.number_of_nodes(), self.hidden_size)).cuda()
        dec_tree.ndata['mask'] = dec_tree.ndata['mask'].float().cuda()
        dgl.prop_nodes_topo(dec_tree)
        # Compute logits
        all_h = self.dropout(dec_tree.ndata.pop("h"))
        logits = self.logit_nn(all_h)
        logp = F.log_softmax(logits, 1)
        # Compute loss
        y_labels = dec_tree.ndata["y"].cuda()
        monitor = {}
        loss = F.nll_loss(logp, y_labels, reduction="mean")
        acc = (logits.argmax(1) == y_labels).float().mean()
        monitor["loss"] = loss
        monitor["label_accuracy"] = acc
        return monitor

    def _init_graph(self, enc_tree, dec_tree):
        enc_tree.register_message_func(self.enc_cell.message_func)
        enc_tree.register_reduce_func(self.enc_cell.reduce_func)
        enc_tree.register_apply_node_func(self.enc_cell.apply_node_func)
        enc_tree.set_n_initializer(dgl.init.zero_initializer)
        dec_tree.register_message_func(self.dec_cell.message_func)
        dec_tree.register_reduce_func(self.dec_cell.reduce_func)
        dec_tree.register_apply_node_func(self.dec_cell.apply_node_func)
        dec_tree.set_n_initializer(dgl.init.zero_initializer)

    def load_pretrain(self, pretrain_path):
        first_param = next(self.parameters())
        device_str = str(first_param.device)
        pre_state_dict = torch.load(pretrain_path, map_location=device_str)["model_state"]
        keys = list(pre_state_dict.keys())
        for key in keys:
            if "semhash" in key:
                pre_state_dict.pop(key)
        state_dict = self.state_dict()
        state_dict.update(pre_state_dict)
        self.load_state_dict(state_dict)

    def load(self, model_path):
        first_param = next(self.parameters())
        device_str = str(first_param.device)
        state_dict = torch.load(model_path, map_location=device_str)["model_state"]
        self.load_state_dict(state_dict)
