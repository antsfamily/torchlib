#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2019-07-24 18:29:48
# @Author  : Zhi Liu (zhiliu.mind@gmail.com)
# @Link    : http://iridescent.ink
# @Version : $1.0$

import torch as th
import torchlib as tl


class LogSparseLoss(th.nn.Module):
    """Log sparse loss

    """

    def __init__(self, cdim=None, dim=None, p=1., reduction='mean'):
        super(LogSparseLoss, self).__init__()
        self.p = p
        self.dim = dim
        self.cdim = cdim
        self.reduction = reduction

    def forward(self, X):
        if th.is_complex(X):  # complex in complex
            pass
        else:
            if self.cdim is None:  # real
                pass
            else:  # complex in real
                d = X.ndim
                idxreal = tl.sl(d, axis=self.cdim, idx=[[0]])
                idximag = tl.sl(d, axis=self.cdim, idx=[[1]])
                X = X[idxreal] + 1j * X[idximag]

        X = X.abs()
        S = th.sum(th.log2(1 + X / self.p), self.dim)
        if self.reduction == 'mean':
            S = th.mean(S)
        if self.reduction == 'sum':
            S = th.sum(S)

        return S


class FourierLogSparseLoss(th.nn.Module):
    r"""FourierLogSparseLoss



    """

    def __init__(self, cdim=None, dim=None, p=1., reduction='mean'):
        super(FourierLogSparseLoss, self).__init__()
        self.p = p
        self.dim = dim
        self.cdim = cdim
        self.reduction = reduction

    def forward(self, X):
        if th.is_complex(X):  # complex in complex
            pass
        else:
            if self.cdim is None:  # real
                pass
            else:  # complex in real
                d = X.ndim
                idxreal = tl.sl(d, axis=self.cdim, idx=[[0]])
                idximag = tl.sl(d, axis=self.cdim, idx=[[1]])
                X = X[idxreal] + 1j * X[idximag]

        for a in self.dim:
            X = th.fft.fft(X, n=None, dim=a)

        X = X.abs()

        S = th.sum(th.log2(1 + X / self.p), self.dim)
        if self.reduction == 'mean':
            S = th.mean(S)
        if self.reduction == 'sum':
            S = th.sum(S)

        return S


if __name__ == '__main__':

    p = 1
    p = 2
    p = 0.5
    X = th.randn(1, 3, 4, 2)
    X = X[:, :, :, 0] + 1j * X[:, :, :, 1]

    sparse_func = LogSparseLoss(p=p)
    sparse_func = LogSparseLoss(p=p, dim=None, cdim=-1)
    sparse_func1 = LogSparseLoss(p=p, dim=None, cdim=1)
    S = sparse_func(X)
    print(S)

    Y = th.view_as_real(X)
    S = sparse_func(Y)
    print(S)

    Y = Y.permute(0, 3, 1, 2)
    S = sparse_func1(Y)
    print(S)

    # print(X)

    sparse_func = FourierLogSparseLoss(p=p)
    sparse_func = FourierLogSparseLoss(p=p, dim=(1, 2), cdim=-1)
    sparse_func1 = FourierLogSparseLoss(p=p, dim=(2, 3), cdim=1)
    S = sparse_func(X)
    print(S)

    Y = th.view_as_real(X)
    S = sparse_func(Y)
    print(S)

    Y = Y.permute(0, 3, 1, 2)
    S = sparse_func1(Y)
    print(S)

    # print(X)
