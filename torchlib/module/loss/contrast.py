#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2019-07-24 18:29:48
# @Author  : Zhi Liu (zhiliu.mind@gmail.com)
# @Link    : http://iridescent.ink
# @Version : $1.0$

import torch as th
from tsar.utils.const import EPS


class ContrastReciprocalLoss(th.nn.Module):
    r"""ContrastReciprocalLoss

    way1 is defined as follows, see [1]:

    .. math::
       C = \frac{{\rm E}(|I|^2)}{\sqrt{{\rm E}\left(|I|^2 - {\rm E}(|I|^2)\right)^2}}


    way2 is defined as follows, see [2]:

    .. math::
        C = \frac{\left({\rm E}(|I|)\right)^2}{{\rm E}(|I|^2)}

    [1] Efficient Nonparametric ISAR Autofocus Algorithm Based on Contrast Maximization and Newton
    [2] section 13.4.1 in "Ian G. Cumming's SAR book"


    """

    def __init__(self, mode='way1', reduction='mean'):
        super(ContrastReciprocalLoss, self).__init__()
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
            Xmean = X.mean(axis=axis)
            C = Xmean / ((X - Xmean).pow(2).mean(axis=axis).sqrt() + EPS)
        if self.mode in ['way2', 'WAY2']:
            C = (X.sqrt().mean(axis=axis)).pow(2) / (X.mean(axis=axis) + EPS)

        if self.reduction == 'mean':
            C = th.mean(C)
        if self.reduction == 'sum':
            C = th.sum(C)
        return C


class NegativeContrastLoss(th.nn.Module):
    r"""NegativeContrastLoss

    way1 is defined as follows, see [1]:

    .. math::
       C = -\frac{\sqrt{{\rm E}\left(|I|^2 - {\rm E}(|I|^2)\right)^2}}{{\rm E}(|I|^2)}


    way2 is defined as follows, see [2]:

    .. math::
        C = -\frac{{\rm E}(|I|^2)}{\left({\rm E}(|I|)\right)^2}

    [1] Efficient Nonparametric ISAR Autofocus Algorithm Based on Contrast Maximization and Newton
    [2] section 13.4.1 in "Ian G. Cumming's SAR book"


    """

    def __init__(self, mode='way1', reduction='mean'):
        super(NegativeContrastLoss, self).__init__()
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
            Xmean = X.mean(axis=axis)
            C = (X - Xmean).pow(2).mean(axis=axis).sqrt() / (Xmean + EPS)
        if self.mode in ['way2', 'WAY2']:
            C = X.mean(axis=axis) / ((X.sqrt().mean(axis=axis)).pow(2) + EPS)

        if self.reduction == 'mean':
            C = th.mean(C)
        if self.reduction == 'sum':
            C = th.sum(C)
        return -C


class ContrastLoss(th.nn.Module):
    r"""ContrastLoss

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
        super(ContrastLoss, self).__init__()
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
            Xmean = X.mean(axis=axis)
            C = (X - Xmean).pow(2).mean(axis=axis).sqrt() / (Xmean + EPS)
        if self.mode in ['way2', 'WAY2']:
            C = X.mean(axis=axis) / ((X.sqrt().mean(axis=axis)).pow(2) + EPS)

        if self.reduction == 'mean':
            C = th.mean(C)
        if self.reduction == 'sum':
            C = th.sum(C)
        return C
