#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2019-07-24 18:29:48
# @Author  : Zhi Liu (zhiliu.mind@gmail.com)
# @Link    : http://iridescent.ink
# @Version : $1.0$

import torch as th
from torchlib.utils.const import EPS


def entropy(X, mode='shannon', axis=None, caxis=None, reduction='mean'):
    r"""compute the entropy of the inputs

    .. math::
        {\rm S} = -\sum_{n=0}^N p_i{\rm log}_2 p_n

    where :math:`N` is the number of pixels, :math:`p_n=\frac{|X_n|^2}{\sum_{n=0}^N|X_n|^2}`.

    Parameters
    ----------
    X : tensor
        The complex or real inputs, for complex inputs, both complex and real representations are surpported.
    mode : str, optional
        The entropy mode: ``'shannon'`` or ``'natural'`` (the default is 'shannon')
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
    S : tensor
        The entropy of the inputs.
    
    Examples
    --------

    ::

        th.manual_seed(2020)
        X = th.randn(1, 3, 4, 2)
        V = entropy(X, mode='shannon', axis=(1, 2), caxis=-1, reduction='mean')
        print(V)

        X = X[:, :, :, 0] + 1j * X[:, :, :, 1]
        V = entropy(X, mode='shannon', axis=(1, 2), caxis=None, reduction='mean')
        print(V)

        V = entropy(X, mode='shannon', axis=None, caxis=None, reduction='mean')
        print(V)

        V = entropy(X, mode='shannon', axis=(2), caxis=None, reduction='mean')
        print(V)

        V = entropy(X, mode='shannon', axis=(2), caxis=None, reduction=None)
        print(V)

        # output
        tensor(2.8302)
        tensor(2.8302)
        tensor(2.8302)
        tensor(1.5349)
        tensor([[[1.3829],
                [1.3055],
                [1.9163]]])
    """

    if mode in ['Shannon', 'shannon', 'SHANNON']:
        logfunc = th.log2
    if mode in ['Natural', 'natural', 'NATURAL']:
        logfunc = th.log

    if th.is_complex(X):
        X = (X * X.conj()).real
    else:
        if type(caxis) is int:
            if X.shape[caxis] != 2:
                raise ValueError('The complex input is represented in real-valued formation, but you specifies wrong axis!')
            X = th.pow(X, 2).sum(axis=caxis, keepdims=True)
        if caxis is None:
            X = th.pow(X, 2)

    if axis is None:
        D = X.dim()
        axis = tuple(range(1, D)) if D > 2 else tuple(range(0, D))

    P = th.sum(X, axis, keepdims=True)
    p = X / (P + EPS)
    S = -th.sum(p * logfunc(p + EPS), axis, keepdims=True)
    if reduction == 'mean':
        S = th.mean(S)
    if reduction == 'sum':
        S = th.sum(S)

    return S


if __name__ == '__main__':

    th.manual_seed(2020)
    X = th.randn(1, 3, 4, 2)
    V = entropy(X, mode='shannon', axis=(1, 2), caxis=-1, reduction='mean')
    print(V)

    X = X[:, :, :, 0] + 1j * X[:, :, :, 1]
    V = entropy(X, mode='shannon', axis=(1, 2), caxis=None, reduction='mean')
    print(V)

    V = entropy(X, mode='shannon', axis=None, caxis=None, reduction='mean')
    print(V)

    V = entropy(X, mode='shannon', axis=(2), caxis=None, reduction='mean')
    print(V)

    V = entropy(X, mode='shannon', axis=(2), caxis=None, reduction=None)
    print(V)