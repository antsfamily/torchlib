#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2019-07-24 18:29:48
# @Author  : Zhi Liu (zhiliu.mind@gmail.com)
# @Link    : http://iridescent.ink
# @Version : $1.0$

import torch as th
from torchtool.utils.const import EPS


class ContrastLoss(th.nn.Module):
    r"""ContrastLoss


    .. math::
        C = \frac{{\rm E}(|I|^2)}{[E(|I|)]^2}

    see section 13.4.1 in "Ian G. Cumming's SAR book"

    .. math::
        l = -\log(C)

    """

    def __init__(self, reduction='mean'):
        super(ContrastLoss, self).__init__()
        self.reduction = reduction

    def forward(self, X):

        if th.is_complex(X):
            X = ((X * X.conj()).real).sqrt()
        elif X.size(-1) == 2:
            X = X.pow(2).sum(axis=-1).sqrt()

        D = X.dim()
        dim = list(range(1, D))

        C = X.pow(2).mean(axis=dim) / X.mean(axis=dim)

        if self.reduction == 'mean':
            C = th.mean(C)
        if self.reduction == 'sum':
            C = th.sum(C)
        return -th.log(C)
        # return 1. / (C + EPS)


class Contrast0(th.nn.Module):
    r"""Contrast



    """

    def __init__(self, reduction='mean'):
        super(Contrast0, self).__init__()
        self.reduction = reduction

    def forward(self, X):

        if th.is_complex(X):
            X = (X * X.conj()).real
        elif X.size(-1) == 2:
            X = th.sum(X.pow(2), axis=-1)

        D = X.dim()
        dim = list(range(1, D))

        varv = th.std(X, dim=dim, unbiased=True, keepdim=True)
        meanv = th.mean(X, axis=dim, keepdim=True)
        C = varv / meanv

        if self.reduction == 'mean':
            C = th.mean(C)
        if self.reduction == 'sum':
            C = th.sum(C)

        return -C
        # return 1. / (C + EPS)


if __name__ == '__main__':

    c_func = Contrast(reduction='mean')
    X = th.randn(1, 3, 4, 2)
    V = c_func(X)
    print(V)

    X = X[:, :, :, 0] + 1j * X[:, :, :, 1]
    V = c_func(X)
    print(V)
