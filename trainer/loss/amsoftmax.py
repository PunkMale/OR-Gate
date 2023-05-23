#! /usr/bin/python
# -*- encoding: utf-8 -*-

import torch
import torch.nn as nn
from .utils import accuracy
from parser import args


class AMSoftmax(nn.Module):
    def __init__(self, embedding_dim, num_classes, margin=0.2, scale=30, **kwargs):
        super(AMSoftmax, self).__init__()

        self.m = margin
        self.s = scale
        self.in_feats = embedding_dim
        self.W = torch.nn.Parameter(torch.randn(embedding_dim, num_classes), requires_grad=True)
        self.ce = nn.CrossEntropyLoss()
        nn.init.xavier_normal_(self.W, gain=1)

        print('Initialised AM-Softmax m=%.3f s=%.3f'%(self.m, self.s))

    def forward(self, x, label=None):
        assert len(x.shape) == 3
        label = label.repeat_interleave(x.shape[1])
        x = x.reshape(-1, self.in_feats)
        assert x.size()[0] == label.size()[0]
        assert x.size()[1] == self.in_feats

        x_norm = torch.norm(x, p=2, dim=1, keepdim=True).clamp(min=1e-12)
        x_norm = torch.div(x, x_norm)
        w_norm = torch.norm(self.W, p=2, dim=0, keepdim=True).clamp(min=1e-12)
        w_norm = torch.div(self.W, w_norm)
        costh = torch.mm(x_norm, w_norm)
        label_view = label.view(-1, 1)
        if label_view.is_cuda: label_view = label_view.cpu()
        delt_costh = torch.zeros(costh.size()).scatter_(1, label_view, self.m)
        if x.is_cuda: delt_costh = delt_costh.to(costh.device)
        costh_m = costh - delt_costh
        costh_m_s = self.s * costh_m
        loss    = self.ce(costh_m_s, label)
        prec1   = accuracy(costh_m_s.detach(), label.detach(), topk=(1,))[0]
        return loss, prec1, costh_m_s

