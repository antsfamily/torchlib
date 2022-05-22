#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2019-07-24 18:29:48
# @Author  : Zhi Liu (zhiliu.mind@gmail.com)
# @Link    : http://iridescent.ink
# @Version : $1.0$

import torch as th
import torchlib as tl


class FnormLoss(th.nn.Module):
    r"""F-norm Loss

    Both complex and real representation are supported.

    .. math::
       {\rm norm}({\bf X}) = \|{\bf X}\|_2 = \left(\sum_{x_i\in {\bf X}}|x_i|^2\right)^{\frac{1}{2}}

    where, :math:`u, v` are the real and imaginary part of x, respectively.

    Parameters
    ----------
    X : tensor
        input
    cdim : int or None
        If :attr:`X` is complex-valued, :attr:`cdim` is ignored. If :attr:`X` is real-valued and :attr:`cdim` is integer
        then :attr:`X` will be treated as complex-valued, in this case, :attr:`cdim` specifies the complex axis;
        otherwise (None), :attr:`X` will be treated as real-valued
    dim : int or None
        The dimension axis (:attr:`cdim` is not included) for computing norm. The default is ``None``, which means all. 
    reduction : str, None or optional
        The operation in batch dim, ``None``, ``'mean'`` or ``'sum'`` (the default is 'mean')

    Returns
    -------
    tensor
         the inputs's f-norm.

    Examples
    ---------

    ::

        th.manual_seed(2020)
        X = th.randn(5, 2, 3, 4)
        print('---norm')

        # real
        F1 = FnormLoss(cdim=None, dim=(-2, -1), reduction=None)(X, Y)
        F2 = FnormLoss(cdim=None, dim=(-2, -1), reduction='sum')(X, Y)
        F3 = FnormLoss(cdim=None, dim=(-2, -1), reduction='mean')(X, Y)
        print(F1, F2, F3)

        # complex in real format
        F1 = FnormLoss(cdim=1, dim=(-2, -1), reduction=None)(X, Y)
        F2 = FnormLoss(cdim=1, dim=(-2, -1), reduction='sum')(X, Y)
        F3 = FnormLoss(cdim=1, dim=(-2, -1), reduction='mean')(X, Y)
        print(F1, F2, F3)

        # complex in complex format
        X = X[:, 0, ...] + 1j * X[:, 1, ...]
        F1 = FnormLoss(cdim=None, dim=(-2, -1), reduction=None)(X, Y)
        F2 = FnormLoss(cdim=None, dim=(-2, -1), reduction='sum')(X, Y)
        F3 = FnormLoss(cdim=None, dim=(-2, -1), reduction='mean')(X, Y)
        print(F1, F2, F3)

        ---norm
        tensor([[2.8719, 2.8263],
                [3.1785, 3.4701],
                [4.6697, 3.2955],
                [3.0992, 2.6447],
                [3.5341, 3.5779]]) tensor(33.1679) tensor(3.3168)
        tensor([4.0294, 4.7058, 5.7154, 4.0743, 5.0290]) tensor(23.5539) tensor(4.7108)
        tensor([4.0294, 4.7058, 5.7154, 4.0743, 5.0290]) tensor(23.5539) tensor(4.7108)
    """

    def __init__(self, cdim=None, dim=None, reduction='mean'):
        super(FnormLoss, self).__init__()
        self.cdim = cdim
        self.dim = dim
        self.reduction = reduction

    def forward(self, X, Y):
        return tl.fnorm(X - Y, cdim=self.cdim, dim=self.dim, reduction=self.reduction)


class PnormLoss(th.nn.Module):
    r"""obtain the p-norm of a tensor

    Both complex and real representation are supported.

    .. math::
       {\rm pnorm}({\bf X}) = \|{\bf X}\|_p = \left(\sum_{x_i\in {\bf X}}|x_i|^p\right)^{\frac{1}{p}}

    where, :math:`u, v` are the real and imaginary part of x, respectively.

    Parameters
    ----------
    X : tensor
        input
    cdim : int or None
        If :attr:`X` is complex-valued, :attr:`cdim` is ignored. If :attr:`X` is real-valued and :attr:`cdim` is integer
        then :attr:`X` will be treated as complex-valued, in this case, :attr:`cdim` specifies the complex axis;
        otherwise (None), :attr:`X` will be treated as real-valued
    dim : int or None
        The dimension axis (:attr:`cdim` is not included) for computing norm. The default is ``None``, which means all. 
    p : int
        Specifies the power. The default is 2.
    
    Returns
    -------
    tensor
         the inputs's p-norm.

    Examples
    ---------

    ::

        th.manual_seed(2020)
        X = th.randn(5, 2, 3, 4)
        print('---norm')

        # real
        F1 = PnormLoss(cdim=None, dim=(-2, -1), reduction=None)(X, Y)
        F2 = PnormLoss(cdim=None, dim=(-2, -1), reduction='sum')(X, Y)
        F3 = PnormLoss(cdim=None, dim=(-2, -1), reduction='mean')(X, Y)
        print(F1, F2, F3)

        # complex in real format
        F1 = PnormLoss(cdim=1, dim=(-2, -1), reduction=None)(X, Y)
        F2 = PnormLoss(cdim=1, dim=(-2, -1), reduction='sum')(X, Y)
        F3 = PnormLoss(cdim=1, dim=(-2, -1), reduction='mean')(X, Y)
        print(F1, F2, F3)

        # complex in complex format
        X = X[:, 0, ...] + 1j * X[:, 1, ...]
        F1 = PnormLoss(cdim=None, dim=(-2, -1), reduction=None)(X, Y)
        F2 = PnormLoss(cdim=None, dim=(-2, -1), reduction='sum')(X, Y)
        F3 = PnormLoss(cdim=None, dim=(-2, -1), reduction='mean')(X, Y)
        print(F1, F2, F3)

        ---norm
        tensor([[2.8719, 2.8263],
                [3.1785, 3.4701],
                [4.6697, 3.2955],
                [3.0992, 2.6447],
                [3.5341, 3.5779]]) tensor(33.1679) tensor(3.3168)
        tensor([4.0294, 4.7058, 5.7154, 4.0743, 5.0290]) tensor(23.5539) tensor(4.7108)
        tensor([4.0294, 4.7058, 5.7154, 4.0743, 5.0290]) tensor(23.5539) tensor(4.7108)
    """

    def __init__(self, cdim=None, dim=None, p=2, reduction='mean'):
        super(PnormLoss, self).__init__()
        self.p = p
        self.dim = dim
        self.cdim = cdim
        self.reduction = reduction

    def forward(self, X, Y):
        return tl.pnorm(X - Y, cdim=self.cdim, dim=self.dim, reduction=self.reduction)


if __name__ == '__main__':

    th.manual_seed(2020)
    X = th.randn(5, 2, 3, 4)
    Y = th.randn(5, 2, 3, 4)
    print('---norm')

    # real
    F1 = FnormLoss(cdim=None, dim=(-2, -1), reduction=None)(X, Y)
    F2 = FnormLoss(cdim=None, dim=(-2, -1), reduction='sum')(X, Y)
    F3 = FnormLoss(cdim=None, dim=(-2, -1), reduction='mean')(X, Y)
    print(F1, F2, F3)

    # complex in real format
    F1 = FnormLoss(cdim=1, dim=(-2, -1), reduction=None)(X, Y)
    F2 = FnormLoss(cdim=1, dim=(-2, -1), reduction='sum')(X, Y)
    F3 = FnormLoss(cdim=1, dim=(-2, -1), reduction='mean')(X, Y)
    print(F1, F2, F3)

    # complex in complex format
    X = X[:, 0, ...] + 1j * X[:, 1, ...]
    Y = Y[:, 0, ...] + 1j * Y[:, 1, ...]
    F1 = FnormLoss(cdim=None, dim=(-2, -1), reduction=None)(X, Y)
    F2 = FnormLoss(cdim=None, dim=(-2, -1), reduction='sum')(X, Y)
    F3 = FnormLoss(cdim=None, dim=(-2, -1), reduction='mean')(X, Y)
    print(F1, F2, F3)

    th.manual_seed(2020)
    X = th.randn(5, 2, 3, 4)
    Y = th.randn(5, 2, 3, 4)
    print('---pnorm')
    
    # real
    F1 = PnormLoss(cdim=None, dim=(-2, -1), reduction=None)(X, Y)
    F2 = PnormLoss(cdim=None, dim=(-2, -1), reduction='sum')(X, Y)
    F3 = PnormLoss(cdim=None, dim=(-2, -1), reduction='mean')(X, Y)
    print(F1, F2, F3)

    # complex in real format
    F1 = PnormLoss(cdim=1, dim=(-2, -1), reduction=None)(X, Y)
    F2 = PnormLoss(cdim=1, dim=(-2, -1), reduction='sum')(X, Y)
    F3 = PnormLoss(cdim=1, dim=(-2, -1), reduction='mean')(X, Y)
    print(F1, F2, F3)

    # complex in complex format
    X = X[:, 0, ...] + 1j * X[:, 1, ...]
    Y = Y[:, 0, ...] + 1j * Y[:, 1, ...]
    F1 = PnormLoss(cdim=None, dim=(-2, -1), reduction=None)(X, Y)
    F2 = PnormLoss(cdim=None, dim=(-2, -1), reduction='sum')(X, Y)
    F3 = PnormLoss(cdim=None, dim=(-2, -1), reduction='mean')(X, Y)
    print(F1, F2, F3)
