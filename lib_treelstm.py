#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import torch
import torch as th
import torch.nn as nn
import dgl

from nmtlab.utils import OPTS


class TreeLSTMCell(nn.Module):

    def forward(self, *input):
        pass

    def __init__(self, x_size, h_size, update_masked_nodes=True):
        super(TreeLSTMCell, self).__init__()
        self.W_iou = nn.Linear(x_size, 3 * h_size, bias=False)
        self.U_iou = nn.Linear(10 * h_size, 3 * h_size, bias=False)
        self.b_iou = nn.Parameter(th.zeros(1, 3 * h_size))
        self.U_f = nn.Linear(10 * h_size, 10 * h_size)
        self.h_size = h_size
        self._update_masked_nodes = update_masked_nodes

    def message_func(self, edges):
        ret = {'h': edges.src['h'], 'c': edges.src['c']}
        return ret

    def reduce_func(self, nodes):
        # concatenate h_jl for equation (1), (2), (3), (4)
        h_cat = nodes.mailbox['h'].view(nodes.mailbox['h'].size(0), -1)
        last_dim = h_cat.shape[-1]
        n_children = nodes.mailbox['h'].size(1)
        # equation (2)
        u_result = torch.matmul(h_cat, self.U_f.weight[:last_dim, :last_dim]) + self.U_f.bias[:last_dim].unsqueeze(0)
        f = th.sigmoid(u_result).view(*nodes.mailbox['h'].size())
        # second term of equation (5)
        c = th.sum(f * nodes.mailbox['c'], 1)
        iou_result = torch.matmul(h_cat, self.U_iou.weight[:, :last_dim].transpose(0, 1))
        if OPTS.w_childnorm:
            iou_result /= n_children
        return {'iou': iou_result, 'c': c}

    def apply_node_func(self, nodes):
        # equation (1), (3), (4)
        iou = nodes.data['iou'] + self.b_iou
        i, o, u = th.chunk(iou, 3, 1)
        i, o, u = th.sigmoid(i), th.sigmoid(o), th.tanh(u)
        # equation (5)
        c = i * u + nodes.data['c']
        # equation (6)
        h = o * th.tanh(c)
        if not self._update_masked_nodes:
            mask = nodes.data["mask"].unsqueeze(-1)
            h = nodes.data["h"] * mask + (1 - mask) * h
            c = nodes.data["c"] * mask + (1 - mask) * c
        return {'h' : h, 'c' : c}


class DecoderTreeLSTMCell(TreeLSTMCell):

    def __init__(self, *args):
        super(DecoderTreeLSTMCell, self).__init__(*args, update_masked_nodes=False)
        self.U_f = nn.Linear(self.h_size, 10 * self.h_size)

    def reduce_func(self, nodes):
        # concatenate h_jl for equation (1), (2), (3), (4)
        h_cat = nodes.mailbox['h'].view(nodes.mailbox['h'].size(0), -1)
        c = nodes.mailbox['c']
        last_dim = h_cat.shape[-1]
        depth_mask = nodes.data["depth"]
        if OPTS.w_cleargen:
            h_cat *= 0.
            c *= 0.

        if "extra_input_depth_2" in nodes.data and int((depth_mask == 2).any().numpy()) > 0:
            extra_input_depth_2 = nodes.data["extra_input_depth_2"]
            select_mask = (depth_mask == 2)[:, None].float().cuda()
            extra_input_depth_2 = extra_input_depth_2 * select_mask
            h_cat = h_cat + extra_input_depth_2
        if "extra_input_depth_1" in nodes.data and int((depth_mask == 1).any().numpy()) > 0:
            extra_input_depth_1 = nodes.data["extra_input_depth_1"]
            select_mask = (depth_mask == 1)[:, None].float().cuda()
            extra_input_depth_1 = extra_input_depth_1 * select_mask
            h_cat = h_cat + extra_input_depth_1
        # equation (2)
        u_result = h_cat.new_zeros(h_cat.shape)
        iou_result = h_cat.new_zeros((h_cat.shape[0], self.h_size * 3))
        pos_data = nodes.data["pos"]
        for pos in range(pos_data.max() + 1):
            pos_mask = (nodes.data["pos"] == pos)
            pos_h_cat = h_cat[pos_mask]
            pos_result = torch.matmul(pos_h_cat, self.U_f.weight[pos * self.h_size: (pos+1) * self.h_size])
            pos_result += self.U_f.bias[pos * self.h_size: (pos+1) * self.h_size].unsqueeze(0)
            u_result[pos_mask] = pos_result
            pos_iou_result = torch.matmul(pos_h_cat, self.U_iou.weight[:, pos * self.h_size: (pos+1) * self.h_size].transpose(0, 1))
            iou_result[pos_mask] = pos_iou_result
        if OPTS.w_cleargen:
            return {"reduced_h": u_result, 'c': c.sum(1)}
        f = th.sigmoid(u_result).view(*nodes.mailbox['h'].size())
        # second term of equation (5)
        c = th.sum(f * c, 1)
        return {'iou': iou_result, 'c': c}

    def apply_node_func(self, nodes):
        if OPTS.w_cleargen:
            if "reduced_h" in nodes.data.keys():
                h = nodes.data["reduced_h"]
            else:
                h = nodes.data["h"]
            c = nodes.data["c"]
            return {"h": h, "c": c}
        else:
            return super(DecoderTreeLSTMCell, self).apply_node_func(nodes)
