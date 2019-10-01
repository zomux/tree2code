#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F


class GumbelEncoder(nn.Module):
    
    def __init__(self, hidden_size, bits=8, tau=1.):
        super(GumbelEncoder, self).__init__()
        num_logits = 2**bits
        self.encode_transform = nn.Linear(hidden_size, num_logits)
        self.decode_transform = nn.Linear(num_logits, hidden_size)
        self._max_gumbel_probs = None
        self.tau = tau
    
    def coding(self, x):
        logits = self.encode_transform(x)
        logits = F.log_softmax(logits, dim=-1)
        return logits
    
    def decode(self, logits):
        B, T, H = logits.shape
        gumbel_softmax = F.gumbel_softmax(logits.view(B, T * H), self.tau).view(B, T, -1)
        out_vectors = self.decode_transform(gumbel_softmax)
        self._max_gumbel_probs = torch.max(gumbel_softmax)
        return out_vectors
    
    def forward(self, x, return_code=False):
        logits = self.coding(x)
        if return_code:
            return torch.argmax(logits, -1)
        else:
            out_vectors = self.decode(logits)
            return out_vectors
    
    def max_gumbel_probs(self):
        return self._max_gumbel_probs
