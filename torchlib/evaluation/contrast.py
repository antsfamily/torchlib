#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2019-07-24 18:29:48
# @Author  : Zhi Liu (zhiliu.mind@gmail.com)
# @Link    : http://iridescent.ink
# @Version : $1.0$

import torch as th
from torchlib.utils.const import EPS


def contrast(X, mode='way1', cdim=None, dim=None, keepcdim=False, reduction='mean'):
    r"""Compute contrast of an complex image

    ``'way1'`` is defined as follows, see [1]:

    .. math::
       C = \frac{\sqrt{{\rm E}\left(|I|^2 - {\rm E}(|I|^2)\right)^2}}{{\rm E}(|I|^2)}


    ``'way2'`` is defined as follows, see [2]:

    .. math::
        C = \frac{{\rm E}(|I|^2)}{\left({\rm E}(|I|)\right)^2}

    [1] Efficient Nonparametric ISAR Autofocus Algorithm Based on Contrast Maximization and Newton
    [2] section 13.4.1 in "Ian G. Cumming's SAR book"

    Parameters
    ----------
    X : torch tensor
        The image array.
    mode : str, optional
        ``'way1'`` or ``'way2'``
    cdim : int or None
        If :attr:`X` is complex-valued, :attr:`cdim` is ignored. If :attr:`X` is real-valued and :attr:`cdim` is integer
        then :attr:`X` will be treated as complex-valued, in this case, :attr:`cdim` specifies the complex axis;
        otherwise (None), :attr:`X` will be treated as real-valued
    dim : tuple, None, optional
        The dimension axis (if :attr:`keepcdim` is :obj:`False` then :attr:`cdim` is not included) for computing contrast. 
        The default is :obj:`None`, which means all.
    keepcdim : bool
        If :obj:`True`, the complex dimension will be keeped. Only works when :attr:`X` is complex-valued tensor 
        and :attr:`dim` is not :obj:`None` but represents in real format. Default is :obj:`False`.
    reduction : str, optional
        The operation in batch dim, ``'None'``, ``'mean'`` or ``'sum'`` (the default is 'mean')

    Returns
    -------
    C : scalar or tensor
        The contrast value of input.
    
    Examples
    --------

    ::

        th.manual_seed(2020)
        X = th.randn(5, 2, 3, 4)

        # real
        C1 = contrast(X, cdim=None, dim=(-2, -1), mode='way1', reduction=None)
        C2 = contrast(X, cdim=None, dim=(-2, -1), mode='way1', reduction='sum')
        C3 = contrast(X, cdim=None, dim=(-2, -1), mode='way1', reduction='mean')
        print(C1, C2, C3)

        # complex in real format
        C1 = contrast(X, cdim=1, dim=(-2, -1), mode='way1', reduction=None)
        C2 = contrast(X, cdim=1, dim=(-2, -1), mode='way1', reduction='sum')
        C3 = contrast(X, cdim=1, dim=(-2, -1), mode='way1', reduction='mean')
        print(C1, C2, C3)

        # complex in complex format
        X = X[:, 0, ...] + 1j * X[:, 1, ...]
        C1 = contrast(X, cdim=None, dim=(-2, -1), mode='way1', reduction=None)
        C2 = contrast(X, cdim=None, dim=(-2, -1), mode='way1', reduction='sum')
        C3 = contrast(X, cdim=None, dim=(-2, -1), mode='way1', reduction='mean')
        print(C1, C2, C3)


        # output
        tensor([[1.2612, 1.1085],
                [1.5992, 1.2124],
                [0.8201, 0.9887],
                [1.4376, 1.0091],
                [1.1397, 1.1860]]) tensor(11.7626) tensor(1.1763)
        tensor([0.6321, 1.1808, 0.5884, 1.1346, 0.6038]) tensor(4.1396) tensor(0.8279)
        tensor([0.6321, 1.1808, 0.5884, 1.1346, 0.6038]) tensor(4.1396) tensor(0.8279)
    """

    if th.is_complex(X):  # complex in complex
        X = (X * X.conj()).real
    else:
        if cdim is None:  # real
            X = X**2
        else:  # complex in real
            X = th.sum(X**2, axis=cdim, keepdims=keepcdim)

    if X.dtype is not th.float32 or th.double:
        X = X.to(th.float32)

    if mode in ['way1', 'WAY1']:
        Xmean = X.mean(dim=dim, keepdims=True)
        C = (X - Xmean).pow(2).mean(dim=dim, keepdims=True).sqrt() / (Xmean + EPS)
        C = th.sum(C, dim=dim, keepdims=False)
    if mode in ['way2', 'WAY2']:
        C = X.mean(dim=dim, keepdims=True) / ((X.sqrt().mean(dim=dim, keepdims=True)).pow(2) + EPS)
        C = th.sum(C, dim=dim, keepdims=False)

    if reduction == 'mean':
        C = th.mean(C)
    if reduction == 'sum':
        C = th.sum(C)
    return C


if __name__ == '__main__':

    th.manual_seed(2020)
    X = th.randn(5, 2, 3, 4)

    # real
    C1 = contrast(X, cdim=None, dim=(-2, -1), mode='way1', reduction=None)
    C2 = contrast(X, cdim=None, dim=(-2, -1), mode='way1', reduction='sum')
    C3 = contrast(X, cdim=None, dim=(-2, -1), mode='way1', reduction='mean')
    print(C1, C2, C3)

    # complex in real format
    C1 = contrast(X, cdim=1, dim=(-2, -1), mode='way1', reduction=None)
    C2 = contrast(X, cdim=1, dim=(-2, -1), mode='way1', reduction='sum')
    C3 = contrast(X, cdim=1, dim=(-2, -1), mode='way1', reduction='mean')
    print(C1, C2, C3)

    # complex in complex format
    X = X[:, 0, ...] + 1j * X[:, 1, ...]
    C1 = contrast(X, cdim=None, dim=(-2, -1), mode='way1', reduction=None)
    C2 = contrast(X, cdim=None, dim=(-2, -1), mode='way1', reduction='sum')
    C3 = contrast(X, cdim=None, dim=(-2, -1), mode='way1', reduction='mean')
    print(C1, C2, C3)
