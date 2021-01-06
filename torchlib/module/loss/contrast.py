#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2019-07-24 18:29:48
# @Author  : Zhi Liu (zhiliu.mind@gmail.com)
# @Link    : http://iridescent.ink
# @Version : $1.0$

import torch as th
from tsar.utils.const import EPS


class ContrastLoss(th.nn.Module):
    r"""ContrastLoss

    ``'way2'``

    .. math::
        C = \frac{{\rm E}(|I|^2)}{[E(|I|)]^2}

    see section 13.4.1 in "Ian G. Cumming's SAR book"



    """

    def __init__(self, mode='way1', reduction='mean'):
        super(ContrastLoss, self).__init__()
        self.mode = mode
        self.reduction = reduction

    def forward(self, X):

        if th.is_complex(X):
            X = ((X * X.conj()).real).sqrt()
        elif X.size(-1) == 2:
            X = X.pow(2).sum(axis=-1).sqrt()

        D = X.dim()
        dim = list(range(1, D))

        if X.dtype is not th.float32 or th.double:
            X = X.to(th.float32)

        if self.mode in ['way1', 'WAY1']:
            C = X.std(axis=dim) / (X.mean(axis=dim) + EPS)
        if self.mode in ['way2', 'WAY2']:
            C = X.pow(2).mean(axis=dim) / (X.mean(axis=dim) + EPS)

        if self.reduction == 'mean':
            C = th.mean(C)
        if self.reduction == 'sum':
            C = th.sum(C)
        return C
