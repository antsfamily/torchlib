#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2019-07-24 18:29:48
# @Author  : Zhi Liu (zhiliu.mind@gmail.com)
# @Link    : http://iridescent.ink
# @Version : $1.0$

import torch as th


def accuracy(X, Y, TH=None):
    r"""compute accuracy



    Parameters
    ----------
    X : {Torch Tensor}
        Predicted one hot matrix
    Y : {Torch Tensor}
        Referenced one hot matrix
    TH : {[type]}, optional
        threshold: X > TH --> 1, X <= TH --> 0
    """

    if TH is not None:
        X = (X > TH).float()
    maxY = th.max(Y)
    if maxY != 1:
        Y = (Y == maxY).float()

    acc = th.mean((X == Y).float()).item()

    return acc


if __name__ == '__main__':
    import numpy as np
    import torchtool as tht

    P = np.array([[1, 1, 1], [0, 1, 0]])
    R = np.array([[0, 0, 1], [0, 0, 1]])

    P = th.Tensor(P)
    R = th.Tensor(R)
    prec = tht.precision(P, R)
    print(prec)
    acc = accuracy(P, R)
    print(acc)
