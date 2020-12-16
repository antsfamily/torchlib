#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2019-07-24 18:29:48
# @Author  : Zhi Liu (zhiliu.mind@gmail.com)
# @Link    : http://iridescent.ink
# @Version : $1.0$

import numpy as np
import torch as th
from torchlib.utils.const import EPS


def standardization(X, mean=None, std=None, axis=None, extra=False):
    r"""standardization

    .. math::
        \bar{X} = \frac{X-\mu}{\sigma}


    Parameters
    ----------
    X : {torch tensor}
        data to be normalized,
    mean : {list or None}, optional
        mean value (the default is None, which means auto computed)
    std : {list or None}, optional
        standard deviation (the default is None, which means auto computed)
    axis : {list or int}, optional
        specify the axis for computing mean and standard deviation (the default is None, which means all elements)
    extra : {bool}, optional
        if True, also return the mean and std (the default is False, which means just return the standardized data)
    """

    if type(X) is np.ndarray:
        X = th.from_numpy(X)

    if mean is None:
        if axis is None:
            mean = th.mean(X)
        else:
            mean = th.mean(X, axis, keepdim=True)
    if std is None:
        if axis is None:
            std = th.std(X)
        else:
            std = th.std(X, axis, keepdim=True)
    if extra is True:
        return (X - mean) / (std + EPS), mean, std
    else:
        return (X - mean) / (std + EPS)


def scale(X, st=[0, 1], sf=None, istrunc=True, extra=False):
    r"""
    Scale data.

    .. math::
        x \in [a, b] \rightarrow y \in [c, d]

    .. math::
        y = (d-c)*(x-a) / (b-a) + c.

    Parameters
    ----------
    X : tensor_like
        The data to be scaled.
    st : tuple, list, optional
        Specifies the range of data after beening scaled. Default [0, 1].
    sf : tuple, list, optional
        Specifies the range of data. Default [min(X), max(X)].
    istrunc : bool
        Specifies wether to truncate the data to [a, b], For example,
        If sf == [a, b] and 'istrunc' is true,
        then X[X < a] == a and X[X > b] == b.
    extra : bool
        If ``True``, also return :attr:`st` and :attr:`sf`.

    Returns
    -------
    out : tensor
        Scaled data tensor.
    st, sf : list or tuple
        If :attr:`extra` is true, also be returned
    """

    if type(X) is np.ndarray:
        X = th.from_numpy(X)

    X = X.float()

    if not(isinstance(st, (tuple, list)) and len(st) == 2):
        raise Exception("'st' is a tuple or list, such as (-1,1)")
    if sf is not None:
        if not(isinstance(sf, (tuple, list)) and len(sf) == 2):
            raise Exception("'sf' is a tuple or list, such as (0, 255)")
    else:
        sf = [th.min(X) + 0.0, th.max(X) + 0.0]
    if sf[0] is None:
        sf = (th.min(X) + 0.0, th[1])
    if sf[1] is None:
        sf = (sf[0], th.max(X) + 0.0)

    a = sf[0] + 0.0
    b = sf[1] + 0.0
    c = st[0] + 0.0
    d = st[1] + 0.0

    if istrunc:
        X[X < a] = a
        X[X > b] = b

    if extra:
        return (X - a) * (d - c) / (b - a + EPS) + c, st, sf
    else:
        return (X - a) * (d - c) / (b - a + EPS) + c


if __name__ == '__main__':

    X = th.randn(4, 3, 5, 6)
    # X = th.randn(3, 4)
    XX = standardization(X, axis=(0, 2, 3))
    XX, meanv, stdv = standardization(X, axis=(0, 2, 3), extra=True)
    print(XX.size())
    print(meanv, stdv)

    X = np.random.randn(4, 3, 5, 6) * 255
    # X = th.randn(3, 4)
    XX = standardization(X, axis=(0, 2, 3))
    XX, meanv, stdv = standardization(X, axis=(0, 2, 3), extra=True)
    print(XX.size())
    print(meanv, stdv)
    print(XX)

    XX = scale(X, st=[0, 1])
    print(XX)
