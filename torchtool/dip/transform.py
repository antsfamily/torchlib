#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2019-07-24 18:29:48
# @Author  : Zhi Liu (zhiliu.mind@gmail.com)
# @Link    : http://iridescent.ink
# @Version : $1.0$

import numpy as np
import torch as th
from torchtool.utils.const import EPS


def normalize(X, mean=None, std=None, axis=None, ver=False):
    r"""normalization

    .. math::
        \bar{X} = \frac{X-\mu}{\sigma}


    Parameters
    ----------
    X : {torch tensor}
        data to be normalized,
    mean : {list or None}, optional
        mean value (the default is None, which auto computed)
    std : {list or None}, optional
        standard deviation (the default is None, which auto computed)
    axis : {list or int}, optional
        specify the axis for computing mean value (the default is None, which all elements)
    axis : {bool}, optional
        if True, also return the mean and std (the default is False, which all elements)
    """

    if type(X) is np.ndarray:
        X = th.from_numpy(X)

    if axis is False:
        return X / X.max()

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
    if ver is True:
        return (X - mean) / std, mean, std
    else:
        return (X - mean) / (std + EPS)


if __name__ == '__main__':

    X = th.randn(4, 3, 5, 6)
    # X = th.randn(3, 4)
    XX = normalize(X, axis=(0, 2, 3))
    XX, meanv, stdv = normalize(X, axis=(0, 2, 3), ver=True)
    print(XX.size())
    print(meanv, stdv)

    X = np.random.randn(4, 3, 5, 6) * 255
    # X = th.randn(3, 4)
    XX = normalize(X, axis=(0, 2, 3))
    XX, meanv, stdv = normalize(X, axis=(0, 2, 3), ver=True)
    print(XX.size())
    print(meanv, stdv)
    print(XX)
