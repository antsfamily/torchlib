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
    cdim : int or None
        If :attr:`X` is complex-valued, :attr:`caxis` is ignored. If :attr:`X` is real-valued and :attr:`caxis` is integer
        then :attr:`X` will be treated as complex-valued, in this case, :attr:`caxis` specifies the complex axis;
        otherwise (None), :attr:`X` will be treated as real-valued
    dim : tuple, None, optional
        The dimension axis (:attr:`caxis` is not included) for computing entropy. The default is ``None``, which means all. 
    mode : str, optional
        The entropy mode: ``'shannon'`` or ``'natural'`` (the default is 'shannon')
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
        X = th.randn(5, 2, 3, 4)

        # real
        S1 = Entropy(cdim=None, dim=(-2, -1), mode='shannon', reduction=None)(X)
        S2 = Entropy(cdim=None, dim=(-2, -1), mode='shannon', reduction='sum')(X)
        S3 = Entropy(cdim=None, dim=(-2, -1), mode='shannon', reduction='mean')(X)
        print(S1, S2, S3)

        # complex in real format
        S1 = Entropy(cdim=1, dim=(-2, -1), mode='shannon', reduction=None)(X)
        S2 = Entropy(cdim=1, dim=(-2, -1), mode='shannon', reduction='sum')(X)
        S3 = Entropy(cdim=1, dim=(-2, -1), mode='shannon', reduction='mean')(X)
        print(S1, S2, S3)

        # complex in complex format
        X = X[:, 0, ...] + 1j * X[:, 1, ...]
        S1 = Entropy(cdim=None, dim=(-2, -1), mode='shannon', reduction=None)(X)
        S2 = Entropy(cdim=None, dim=(-2, -1), mode='shannon', reduction='sum')(X)
        S3 = Entropy(cdim=None, dim=(-2, -1), mode='shannon', reduction='mean')(X)
        print(S1, S2, S3)

        # output
        tensor([[2.5482, 2.7150],
                [2.0556, 2.6142],
                [2.9837, 2.9511],
                [2.4296, 2.7979],
                [2.7287, 2.5560]]) tensor(26.3800) tensor(2.6380)
        tensor([3.2738, 2.5613, 3.2911, 2.7989, 3.2789]) tensor(15.2040) tensor(3.0408)
        tensor([3.2738, 2.5613, 3.2911, 2.7989, 3.2789]) tensor(15.2040) tensor(3.0408)
    """

    def __init__(self, cdim=None, dim=None, mode='shannon', reduction='mean'):
        super(Entropy, self).__init__()
        self.mode = mode
        self.dim = dim
        self.cdim = cdim
        self.reduction = reduction

    def forward(self, X):
        return tl.entropy(X, cdim=self.cdim, dim=self.dim, mode=self.mode, reduction=self.reduction)


if __name__ == '__main__':

    th.manual_seed(2020)
    X = th.randn(5, 2, 3, 4)

    # real
    S1 = Entropy(cdim=None, dim=(-2, -1), mode='shannon', reduction=None)(X)
    S2 = Entropy(cdim=None, dim=(-2, -1), mode='shannon', reduction='sum')(X)
    S3 = Entropy(cdim=None, dim=(-2, -1), mode='shannon', reduction='mean')(X)
    print(S1, S2, S3)

    # complex in real format
    S1 = Entropy(cdim=1, dim=(-2, -1), mode='shannon', reduction=None)(X)
    S2 = Entropy(cdim=1, dim=(-2, -1), mode='shannon', reduction='sum')(X)
    S3 = Entropy(cdim=1, dim=(-2, -1), mode='shannon', reduction='mean')(X)
    print(S1, S2, S3)

    # complex in complex format
    X = X[:, 0, ...] + 1j * X[:, 1, ...]
    S1 = Entropy(cdim=None, dim=(-2, -1), mode='shannon', reduction=None)(X)
    S2 = Entropy(cdim=None, dim=(-2, -1), mode='shannon', reduction='sum')(X)
    S3 = Entropy(cdim=None, dim=(-2, -1), mode='shannon', reduction='mean')(X)
    print(S1, S2, S3)
