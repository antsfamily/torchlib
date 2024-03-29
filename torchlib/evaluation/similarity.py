#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2019-07-24 18:29:48
# @Author  : Zhi Liu (zhiliu.mind@gmail.com)
# @Link    : http://iridescent.ink
# @Version : $1.0$

import torch as th
from torchlib.utils.const import EPS


def jaccard_index(X, Y, TH=None):
    r"""Jaccard similarity coefficient

    .. math::
        \mathrm{J}(\mathrm{A}, \mathrm{B})=\frac{|A \cap B|}{|A \cup B|}

    Parameters
    ----------
    X : tensor
        retrieval results, retrieved-->1, not retrieved-->0
    Y : tensor
        referenced, positive-->1, negative-->0
    TH : float
        X > TH --> 1, X <= TH --> 0

    Returns
    -------
    JS : float
        the jaccard similarity coefficient.

    """

    if TH is not None:
        X = (X > TH).float()

    X = (X > 0.5)
    Y = (Y > 0.5)

    X = th.as_tensor(X, dtype=th.float32)
    Y = th.as_tensor(Y, dtype=th.float32)

    Inter = th.sum((X + Y) == 2)
    Union = th.sum((X + Y) >= 1)

    JS = float(Inter) / (float(Union) + EPS)

    return JS


def dice_coeff(X, Y, TH=0.5):
    r"""Dice coefficient

    .. math::
        s = \frac{2|Y \cap X|}{|X|+|Y|}

    Parameters
    ----------
    X : tensor
        retrieval results, retrieved-->1, not retrieved-->0
    Y : tensor
        referenced, positive-->1, negative-->0
    TH : float
        X > TH --> 1, X <= TH --> 0

    Returns
    -------
    DC : float
        the dice coefficient.
    """

    if TH is not None:
        X = (X > TH).float()

    X = (X > 0.5)
    Y = (Y > 0.5)

    X = th.as_tensor(X, dtype=th.float32)
    Y = th.as_tensor(Y, dtype=th.float32)
    Inter = th.sum((X + Y) == 2)
    DC = float(2 * Inter) / (float(th.sum(X) + th.sum(Y)) + EPS)

    return DC


if __name__ == '__main__':
    import numpy as np
    import torchlib as tl

    X = np.array([[1, 1, 1], [0, 1, 0]])
    Y = np.array([[0, 0, 1], [0, 0, 1]])
    TH = None

    X = th.Tensor(X)
    Y = th.Tensor(Y)
    prec = tl.precision(X, Y)
    print("prec: ", prec)
    acc = tl.accuracy(X, Y)
    print("acc: ", acc)

    JS = jaccard_index(X, Y, TH=TH)
    DC = dice_coeff(X, Y, TH=TH)

    print("JS: ", JS)
    print("DC: ", DC)
    print("2JS/(1+JS)", 2.0 * JS / (1.0 + JS))
    print("DC/(2-DC)", DC / (2.0 - DC))
