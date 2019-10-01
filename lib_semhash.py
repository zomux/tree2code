#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
from torch.autograd import Function


class MagicMasking(Function):
    """
    Gradient always goes to val1.
    """
    
    @staticmethod
    def forward(ctx, val1, val2):
        mask = val1.new_empty(val1.shape[:-1]).fill_(0.5).bernoulli_()
        mask = mask.unsqueeze(-1)
        return mask * val1 + (1. - mask) * val2
    
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None
    
magic_masking = MagicMasking.apply

def saturating_sigmoid(x):
    return torch.clamp(1.2 * torch.sigmoid(x) - 0.1, min=0, max=1)


class SemanticHashing(nn.Module):
    
    def __init__(self, input_size, bits=16):
        super(SemanticHashing, self).__init__()
        self._input_size = input_size
        self.enc_dense = nn.Linear(input_size, bits)
        self.h1a_dense = nn.Linear(bits, input_size * 8)
        self.h1b_dense = nn.Linear(bits, input_size * 8)
        self.h2_dense = nn.Linear(input_size * 8, input_size * 8)
        self.result_dense = nn.Linear(input_size * 8, input_size)

    def encode(self, x):
        v = self.enc_dense(x)
        means = torch.zeros_like(v)
        noise = torch.normal(means)
        v_n = v + noise
        v_1 = saturating_sigmoid(v_n)
        v_2 = torch.gt(v_n, 0).float()
        if self.training:
            v_d = magic_masking(v_1, v_2)
        else:
            v_d = torch.gt(v, 0).float()
        return v_d
    
    def decode(self, v_d):
        h1a = self.h1a_dense(v_d)
        h1b = self.h1b_dense(1.0-v_d)
        h1 = h1a + h1b
        h2 = self.h2_dense(torch.relu(h1))
        result = self.result_dense(torch.relu(h2))
        return result
    
    def forward(self, input_vector, return_code=False):
        v_d = self.encode(input_vector)
        if return_code:
            units = 2 ** torch.arange(v_d.shape[-1])
            if torch.cuda.is_available():
                units = units.cuda().float()
            codes = (v_d.float() * units).sum(-1)
            return codes
        else:
            result_vector = self.decode(v_d)
            return result_vector
    

if __name__ == '__main__':
    import torch
    tensor = torch.randn((3, 6, 256)).float().cuda()
    semhash = SemanticHashing(256)
    semhash.cuda()
    semhash.forward(tensor)
    print("testing backward")
    t1 = torch.randn(3, 3, requires_grad=True)
    t2 = torch.randn(3, 3)
    print("saturated sigmoid")
    print(saturating_sigmoid(t1))
    print("binarize")
    print(t1 > 0)
    out = magic_masking(t1, t2)
    print(t1)
    print(t2)
    print(out)
    out.sum().backward()
    print(t1.grad)
