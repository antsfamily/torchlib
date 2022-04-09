#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2019-07-24 18:29:48
# @Author  : Zhi Liu (zhiliu.mind@gmail.com)
# @Link    : http://iridescent.ink
# @Version : $1.0$

import torch as th
import torchlib as tl


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

    Parameters
    ----------
    X : numpy ndarray
        The image array.
    mode : str, optional
        ``'way1'`` or ``'way2'``
    axis : tuple, None, optional
        the dimensions for compute entropy. by default None (if input's dimension > 2, then all but the first, else all).
    caxis : int or None
        If :attr:`X` is complex-valued, :attr:`caxis` is ignored. If :attr:`X` is real-valued and :attr:`caxis` is integer
        then :attr:`X` will be treated as complex-valued, in this case, :attr:`caxis` specifies the complex axis;
        otherwise (None), :attr:`X` will be treated as real-valued
    reduction : str, optional
        The operation in batch dim, ``'None'``, ``'mean'`` or ``'sum'`` (the default is 'mean')

    Returns
    -------
    scalar
        The contrast value of input.

    Examples
    --------

    ::

        th.manual_seed(2020)
        X = th.randn(1, 3, 4, 2)
        ctst_func = Contrast(mode='way1', axis=(1, 2), caxis=-1, reduction='mean')
        V = ctst_func(X)
        print(V)

        X = X[:, :, :, 0] + 1j * X[:, :, :, 1]
        ctst_func = Contrast(mode='way1', axis=(1, 2), caxis=None, reduction='mean')
        V = ctst_func(X)
        print(V)

        ctst_func = Contrast(mode='way1', axis=None, caxis=None, reduction='mean')
        V = ctst_func(X)
        print(V)

        ctst_func = Contrast(mode='way1', axis=(2), caxis=None, reduction='mean')
        V = ctst_func(X)
        print(V)

        ctst_func = Contrast(mode='way1', axis=(2), caxis=None, reduction=None)
        V = ctst_func(X)
        print(V)
    
        # output
        tensor(1.2694)
        tensor(1.2694)
        tensor(1.2694)
        tensor(0.7724)
        tensor([[[0.9093],
                [1.0752],
                [0.3326]]])
    """

    def __init__(self, mode='way1', axis=None, caxis=None, reduction='mean'):
        super(Contrast, self).__init__()
        self.mode = mode
        self.axis = axis
        self.caxis = caxis
        self.reduction = reduction

    def forward(self, X):

        return tl.contrast(X, mode=self.mode, axis=self.axis, caxis=self.caxis, reduction=self.reduction)


if __name__ == '__main__':

    th.manual_seed(2020)
    X = th.randn(1, 3, 4, 2)
    ctst_func = Contrast(mode='way1', axis=(1, 2), caxis=-1, reduction='mean')
    V = ctst_func(X)
    print(V)

    X = X[:, :, :, 0] + 1j * X[:, :, :, 1]
    ctst_func = Contrast(mode='way1', axis=(1, 2), caxis=None, reduction='mean')
    V = ctst_func(X)
    print(V)

    ctst_func = Contrast(mode='way1', axis=None, caxis=None, reduction='mean')
    V = ctst_func(X)
    print(V)

    ctst_func = Contrast(mode='way1', axis=(2), caxis=None, reduction='mean')
    V = ctst_func(X)
    print(V)

    ctst_func = Contrast(mode='way1', axis=(2), caxis=None, reduction=None)
    V = ctst_func(X)
    print(V)
