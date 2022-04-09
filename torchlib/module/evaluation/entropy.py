#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2019-07-24 18:29:48
# @Author  : Zhi Liu (zhiliu.mind@gmail.com)
# @Link    : http://iridescent.ink
# @Version : $1.0$

import torch as th
import torchlib as tl


class Entropy(th.nn.Module):
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

        mode = 'natural'
        mode = 'shannon'
        th.manual_seed(2020)
        X = th.randn(1, 3, 4, 2)
        ent_func = Entropy(mode=mode, axis=(1, 2), caxis=-1, reduction='mean')
        V = ent_func(X)
        print(V)

        X = X[:, :, :, 0] + 1j * X[:, :, :, 1]
        ent_func = Entropy(mode=mode, axis=(1, 2), caxis=None, reduction='mean')
        V = ent_func(X)
        print(V)

        ent_func = Entropy(mode=mode, axis=None, caxis=None, reduction='mean')
        V = ent_func(X)
        print(V)

        ent_func = Entropy(mode=mode, axis=(2), caxis=None, reduction='mean')
        V = ent_func(X)
        print(V)

        ent_func = Entropy(mode=mode, axis=(2), caxis=None, reduction=None)
        V = ent_func(X)
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

    def __init__(self, mode='shannon', axis=None, caxis=None, reduction='mean'):
        super(Entropy, self).__init__()
        self.mode = mode
        self.axis = axis
        self.caxis = caxis
        self.reduction = reduction

    def forward(self, X):

        return tl.entropy(X, mode=self.mode, axis=self.axis, caxis=self.caxis, reduction=self.reduction)


if __name__ == '__main__':

    mode = 'natural'
    mode = 'shannon'
    th.manual_seed(2020)
    X = th.randn(1, 3, 4, 2)
    ent_func = Entropy(mode=mode, axis=(1, 2), caxis=-1, reduction='mean')
    V = ent_func(X)
    print(V)

    X = X[:, :, :, 0] + 1j * X[:, :, :, 1]
    ent_func = Entropy(mode=mode, axis=(1, 2), caxis=None, reduction='mean')
    V = ent_func(X)
    print(V)

    ent_func = Entropy(mode=mode, axis=None, caxis=None, reduction='mean')
    V = ent_func(X)
    print(V)

    ent_func = Entropy(mode=mode, axis=(2), caxis=None, reduction='mean')
    V = ent_func(X)
    print(V)

    ent_func = Entropy(mode=mode, axis=(2), caxis=None, reduction=None)
    V = ent_func(X)
    print(V)
