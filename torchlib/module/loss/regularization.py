#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2019-07-24 18:29:48
# @Author  : Zhi Liu (zhiliu.mind@gmail.com)
# @Link    : http://iridescent.ink
# @Version : $1.0$

import torch as th
from torchlib.utils.const import EPS


class LogSparseLoss(th.nn.Module):
    r"""Log Sparse loss



    """

    def __init__(self, p=1., axis=None, caxis=None, reduction='mean'):
        super(LogSparseLoss, self).__init__()
        self.p = p
        self.axis = axis
        self.caxis = caxis
        self.reduction = reduction

    def forward(self, X):
        if th.is_complex(X):
            X = (X * X.conj()).real
        elif (self.caxis is None) or X.shape[-1] == 2:
            X = th.sum(X.pow(2), axis=-1, keepdims=True)
        else:
            X = th.sum(X.pow(2), axis=self.caxis, keepdims=True)

        if self.axis is None:
            D = X.dim()
            axis = list(range(1, D)) if D > 2 else list(range(0, D))
        else:
            axis = self.axis

        S = th.sum(th.log2(1 + X / self.p), axis)
        if self.reduction == 'mean':
            S = th.mean(S)
        if self.reduction == 'sum':
            S = th.sum(S)

        return S


if __name__ == '__main__':

    ent_func = LogSparseLoss()
    ent_func1 = LogSparseLoss(p=1, caxis=1)
    X = th.randn(1, 3, 4, 2)
    S = ent_func(X)
    print(S)

    Y = X.permute(0, 3, 1, 2)
    S = ent_func1(Y)
    print(S)

    X = X[:, :, :, 0] + 1j * X[:, :, :, 1]
    S = ent_func(X)
    print(S)

