#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2019-07-24 18:29:48
# @Author  : Zhi Liu (zhiliu.mind@gmail.com)
# @Link    : http://iridescent.ink
# @Version : $1.0$

import torch as th
from torchlib.utils.const import EPS


class Contrast(th.nn.Module):
    r"""Contrast

    way1 is defined as follows, see [1]:

    .. math::
       C = \frac{\sqrt{{\rm E}\left(|I|^2 - {\rm E}(|I|^2)\right)^2}}{{\rm E}(|I|^2)}


    way2 is defined as follows, see [2]:

    .. math::
        C = \frac{{\rm E}(|I|^2)}{\left({\rm E}(|I|)\right)^2}

    [1] Efficient Nonparametric ISAR Autofocus Algorithm Based on Contrast Maximization and Newton
    [2] section 13.4.1 in "Ian G. Cumming's SAR book"


    """

    def __init__(self, mode='way1', reduction='mean'):
        super(Contrast, self).__init__()
        self.mode = mode
        self.reduction = reduction

    def forward(self, X):

        if th.is_complex(X):
            X = (X * X.conj()).real
        elif X.size(-1) == 2:
            X = X.pow(2).sum(axis=-1)

        D = X.dim()
        axis = list(range(1, D))

        if X.dtype is not th.float32 or th.double:
            X = X.to(th.float32)

        if self.mode in ['way1', 'WAY1']:
            Xmean = X.mean(axis=axis, keepdims=True)
            C = (X - Xmean).pow(2).mean(axis=axis, keepdims=True).sqrt() / (Xmean + EPS)
        if self.mode in ['way2', 'WAY2']:
            C = X.mean(axis=axis, keepdims=True) / ((X.sqrt().mean(axis=axis, keepdims=True)).pow(2) + EPS)

        if self.reduction == 'mean':
            C = th.mean(C)
        if self.reduction == 'sum':
            C = th.sum(C)
        return C


if __name__ == '__main__':

    c_func = Contrast(reduction='mean')
    X = th.randn(1, 3, 4, 2)
    V = c_func(X)
    print(V)

    X = X[:, :, :, 0] + 1j * X[:, :, :, 1]
    V = c_func(X)
    print(V)
