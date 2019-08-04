#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2019-07-24 18:29:48
# @Author  : Zhi Liu (zhiliu.mind@gmail.com)
# @Link    : http://iridescent.ink
# @Version : $1.0$

import torch as th
from torchtool.utils.const import EPS


def true_positive(X, Y):
    """Find true positive elements

    true_positive(X, Y) returns those elements that are positive classes in Y
    and retrieved as positive in X.

    Parameters
    ----------
    X : {Torch Tensor}
        retrieval results, retrieved-->1, not retrieved-->0
    Y : {Torch Tensor}
        referenced, positive-->1, negative-->0

    Returns
    -------
    TP: {Torch Tensor}
        a torch tensor which has the same type with :attr:`X` or :attr:`Y`.
        In TP, true positive elements are ones, while others are zeros.
    """
    TP = ((X == 1) + (Y == 1)) == 2
    return TP


def false_positive(X, Y):
    """Find false positive elements

    false_positive(X, Y) returns elements that are negative classes in Y
    and retrieved as positive in X.

    Parameters
    ----------
    X : {Torch Tensor}
        retrieval results, retrieved-->1, not retrieved-->0
    Y : {Torch Tensor}
        referenced, positive-->1, negative-->0

    Returns
    -------
    FP: {Torch Tensor}
        a torch tensor which has the same type with :attr:`X` or :attr:`Y`.
        In FP, false positive elements are ones, while others are zeros.
    """
    FP = ((X == 1) + (Y == 0)) == 2
    return FP


def true_negative(X, Y):
    """Find true negative elements

    true_negative(X, Y) returns elements that are negative classes in Y
    and retrieved as negative in X.

    Parameters
    ----------
    X : {Torch Tensor}
        retrieval results, retrieved-->1, not retrieved-->0
    Y : {Torch Tensor}
        referenced, positive-->1, negative-->0

    Returns
    -------
    TN: {Torch Tensor}
        a torch tensor which has the same type with :attr:`X` or :attr:`Y`.
        In TN, true negative elements are ones, while others are zeros.
    """
    TN = ((X == 0) + (Y == 0)) == 2
    return TN


def false_negative(X, Y):
    """Find false negative elements

    true_negative(X, Y) returns elements that are positive classes in Y
    and retrieved as negative in X.

    Parameters
    ----------
    X : {Torch Tensor}
        retrieval results, retrieved-->1, not retrieved-->0
    Y : {Torch Tensor}
        referenced, positive-->1, negative-->0

    Returns
    -------
    FN: {Torch Tensor}
        a torch tensor which has the same type with :attr:`X` or :attr:`Y`.
        In FN, false negative elements are ones, while others are zeros.
    """
    FN = ((X == 0) + (Y == 1)) == 2
    return FN


def precision(X, Y, TH=None):
    r"""Compute precision

    .. math::
       {\rm PPV} = {P} = \frac{\rm TP}{{\rm TP} + {\rm FP}}
       :label: equ-Precision

    Parameters
    ----------
    X : {Torch Tensor}
        retrieval results, retrieved-->1, not retrieved-->0
    Y : {Torch Tensor}
        referenced, positive-->1, negative-->0
    TH : {float number}
        X > TH --> 1, X <= TH --> 0

    Returns
    -------
    P: {float number}
        precision
    """

    maxX = th.max(X)
    if maxX != 1:
        X = (X == maxX)
    maxY = th.max(Y)
    if maxY != 1:
        Y = (Y == maxY)

    # TP : True Positive
    # FP : False Positive
    TP = true_positive(X, Y)
    FP = false_positive(X, Y)
    P = float(th.sum(TP)) / (float(th.sum(TP + FP)) + EPS)
    return P


def recall(X, Y, TH=None):
    r"""Compute recall(sensitivity)

    .. math::
       {\rm TPR} = {R} = \frac{\rm TP}{{\rm TP} + {\rm FN}}
       :label: equ-Recall

    Parameters
    ----------
    X : {Torch Tensor}
        retrieval results, retrieved-->1, not retrieved-->0
    Y : {Torch Tensor}
        referenced, positive-->1, negative-->0
    TH : {float number}
        X > TH --> 1, X <= TH --> 0

    Returns
    -------
    R: {float number}
        recall
    """

    maxX = th.max(X)
    if maxX != 1:
        X = (X == maxX)
    maxY = th.max(Y)
    if maxY != 1:
        Y = (Y == maxY)

    # TP : True Positive
    # FN : False Negative
    TP = true_positive(X, Y)
    FN = false_negative(X, Y)
    R = float(th.sum(TP)) / (float(th.sum(TP + FN)) + EPS)
    return R


def sensitivity(X, Y, TH=None):
    r"""Compute sensitivity(recall)

    .. math::
       {\rm TPR} = {R} = \frac{\rm TP}{{\rm TP} + {\rm FN}}
       :label: equ-Recall

    Parameters
    ----------
    X : {Torch Tensor}
        retrieval results, retrieved-->1, not retrieved-->0
    Y : {Torch Tensor}
        referenced, positive-->1, negative-->0
    TH : {float number}
        X > TH --> 1, X <= TH --> 0

    Returns
    -------
    R: {float number}
        recall
    """

    maxX = th.max(X)
    if maxX != 1:
        X = (X == maxX)
    maxY = th.max(Y)
    if maxY != 1:
        Y = (Y == maxY)

    # TP : True Positive
    # FN : False Negative
    TP = true_positive(X, Y)
    FN = false_negative(X, Y)
    R = float(th.sum(TP)) / (float(th.sum(TP + FN)) + EPS)
    return R


def selectivity(X, Y, TH=None):
    r"""Compute selectivity or specificity

    .. math::
       {\rm TNR} = {S} = \frac{\rm TN}{{\rm TN} + {\rm FP}}
       :label: equ-selectivity

    Parameters
    ----------
    X : {Torch Tensor}
        retrieval results, retrieved-->1, not retrieved-->0
    Y : {Torch Tensor}
        referenced, positive-->1, negative-->0
    TH : {float number}
        X > TH --> 1, X <= TH --> 0

    Returns
    -------
    S: {float number}
        selectivity
    """

    maxX = th.max(X)
    if maxX != 1:
        X = (X == maxX)
    maxY = th.max(Y)
    if maxY != 1:
        Y = (Y == maxY)

    # TN : True Negative
    # FP : False Positive
    TN = true_negative(X, Y)
    FP = false_positive(X, Y)
    S = float(th.sum(TN)) / (float(th.sum(TN + FP)) + EPS)
    return S


def fmeasure(X, Y, TH=None, beta=1.0):
    r"""Compute F-measure

    .. math::
       F_{\beta} = \frac{(1+\beta^2)鈰匬鈰匯}{\beta^2鈰匬 + R}
       :label: equ-F-measure

    Parameters
    ----------
    X : {Torch Tensor}
        retrieval results, retrieved-->1, not retrieved-->0
    Y : {Torch Tensor}
        referenced, positive-->1, negative-->0
    TH : {float number}
        X > TH --> 1, X <= TH --> 0
    beta : {float number}
        X > TH --> 1, X <= TH --> 0
    Returns
    -------
    F: {float number}
        F-measure
    """
    if TH is not None:
        X = X > TH

    maxX = th.max(X)
    if maxX != 1:
        X = (X == maxX)
    maxY = th.max(Y)
    if maxY != 1:
        Y = (Y == maxY)

    TP = true_positive(X, Y)
    FP = false_positive(X, Y)
    P = float(th.sum(TP)) / (float(th.sum(TP + FP)) + EPS)

    TP = true_positive(X, Y)
    FN = false_negative(X, Y)
    R = float(th.sum(TP)) / (float(th.sum(TP + FN)) + EPS)
    F = (1 + beta**2) * P * R / ((beta**2) * P + R + EPS)
    return F


def false_alarm_rate(X, Y, TH=None):
    r"""Compute false alarm rate or False Discovery Rate

    .. math::
       {\rm FDR} = \frac{\rm FP}{{\rm TP} + {\rm FP}} = 1 - P
       :label: equ-FalseDiscoveryRate

    Parameters
    ----------
    X : {Torch Tensor}
        retrieval results, retrieved-->1, not retrieved-->0
    Y : {Torch Tensor}
        referenced, positive-->1, negative-->0
    TH : {float number}
        X > TH --> 1, X <= TH --> 0

    Returns
    -------
    FDR: {float number}
        False Discovery Rate
    """

    if TH is not None:
        X = X > TH
    maxY = th.max(Y)
    if maxY != 1:
        Y = (Y == maxY)

    # FP : False Positive
    # TP : True Positive
    TP = true_positive(X, Y)
    FP = false_positive(X, Y)
    FDR = float(th.sum(FP)) / (float(th.sum(TP + FP)) + EPS)
    return FDR


def miss_alarm_rate(X, Y, TH=None):
    r"""Compute miss alarm rate or False Negative Rate

    .. math::
       {\rm FNR} = \frac{\rm FN}{{\rm FN} + {\rm TP}} = 1 - R
       :label: equ-FalseNegativeRate

    Parameters
    ----------
    X : {Torch Tensor}
        retrieval results, retrieved-->1, not retrieved-->0
    Y : {Torch Tensor}
        referenced, positive-->1, negative-->0
    TH : {float number}
        X > TH --> 1, X <= TH --> 0

    Returns
    -------
    FNR: {float number}
        False Negative Rate
    """

    if TH is not None:
        X = X > TH
    maxY = th.max(Y)
    if maxY != 1:
        Y = (Y == maxY)

    # FP : False Negative
    # TP : True Positive
    TP = true_positive(X, Y)
    FN = false_negative(X, Y)
    FNR = float(th.sum(FN)) / (float(th.sum(TP + FN)) + EPS)
    return FNR


if __name__ == '__main__':
    import numpy as np
    import torchtool as tht

    X = np.array([[1, 1, 1], [0, 1, 0]])
    X = np.array([[0, 0, 1], [0, 0, 1]])
    X = np.array([[0.2, 0, 1], [0, 0.2, 1]])
    Y = np.array([[0, 0, 1], [0, 0, 1]])
    TH = 0.1
    beta = 1.0

    X = th.Tensor(X)
    Y = th.Tensor(Y)
    prec = tht.precision(X, Y)
    print("prec: ", prec)
    acc = tht.accuracy(X, Y)
    print("acc: ", acc)

    P = precision(X, Y, TH=TH)
    R = recall(X, Y, TH=TH)
    F = fmeasure(X, Y, TH=TH, beta=1.0)
    FDR = false_alarm_rate(X, Y, TH=None)
    FNR = miss_alarm_rate(X, Y, TH=None)

    print("P: ", P)
    print("R: ", R)
    print("F: ", F)
    print("FDR: ", FDR)
    print("FNR: ", FNR)
    print("1.0-FDR: ", 1.0 - FDR)
    print("1.0-FNR: ", 1.0 - FNR)
