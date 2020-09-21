#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2020-02-18 11:06:13
# @Author  : Zhi Liu (zhiliu.mind@gmail.com)
# @Link    : http://iridescent.ink
# @Version : $1.0$

import torch as th
from torchtool.utils.const import EPS


def shannon_entropy(X):

    if X.dim() == 2:
        P = th.sum(X)
        p = X / (P + EPS)
        S = - th.sum(p * th.log2(p + EPS))
    if X.dim() == 4:
        P = th.sum(X, (1, 2, 3), keepdims=True)
        p = X / (P + EPS)
        S = - th.sum(p * th.log2(p + EPS), (1, 2, 3), keepdims=True)

    return S


def natural_entropy(X):

    if X.dim() == 2:
        P = th.sum(X)
        p = X / (P + EPS)
        S = - th.sum(p * th.log(p + EPS))
    if X.dim() == 4:
        P = th.sum(X, (1, 2, 3), keepdims=True)
        p = X / (P + EPS)
        S = - th.sum(p * th.log(p + EPS), (1, 2, 3), keepdims=True)
    return S


if __name__ == "__main__":

    H, W = (8, 8)

    X = th.randn(4, 1, H, W) + 1j * th.randn(4, 1, H, W)

    X = th.conj(X) * X

    S1 = shannon_entropy(X)
    S2 = natural_entropy(X)

    print(S1, S2)
