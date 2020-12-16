#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2020-03-23 19:28:33
# @Author  : Yan Liu & Zhi Liu (zhiliu.mind@gmail.com)
# @Link    : http://iridescent.ink
# @Version : $1.0$
from __future__ import division, print_function, absolute_import
import torch as th
from torchlib.base.arrayops import sl
from torchlib.utils.randomfunc import setseed, randperm


def sample_tensor(x, n, axis=0, groups=1, mode='sequentially', seed=None, extra=False):
    r"""sample a tensor

    Sample a tensor sequentially/uniformly/randomly.

    Parameters
    ----------
    x : {torch tensor}
        a torch tensor to be sampled
    n : {number}
        sample number
    axis : {number}, optional
        the axis to be sampled (the default is 0)
    groups : {number}, optional
        number of groups in this tensor (the default is 1)
    mode : {str}, optional
        - ``'sequentially'``: [0, int(n/groups)] (default)
        - ``'uniformly'``: evenly spaced
        - ``'randomly'``: randomly selected, non-returned sampling
    seed : {None or number}, optional
        only work for ``'randomly'`` mode (the default is None)
    extra : {bool}, optional
        If ``True``, also return the selected indexes, the default is ``False``.

    Returns
    -------
    y : {torch tensor}
        Sampled torch tensor.
    idx : {list}
        Sampled indexes, if :attr:`extra` is ``True``, this will also be returned.


    ::

        x = th.randint(1000, (20, 3, 4))
        y1, idx1 = sample(x, 10, axis=0, groups=2, mode='sequentially', extra=True)
        y2, idx2 = sample(x, 10, axis=0, groups=2, mode='uniformly', extra=True)
        y3, idx3 = sample(x, 10, axis=0, groups=2, mode='randomly', extra=True)

        print(x.shape)
        print(y1.shape)
        print(y2.shape)
        print(y3.shape)
        print(idx1)
        print(idx2)
        print(idx3)

        the outputs are as follows:

        torch.Size([20, 3, 4])
        torch.Size([10, 3, 4])
        torch.Size([10, 3, 4])
        torch.Size([10, 3, 4])
        [0, 1, 2, 3, 4, 10, 11, 12, 13, 14]
        [0, 2, 4, 6, 8, 10, 12, 14, 16, 18]
        [3, 1, 5, 8, 7, 17, 18, 13, 16, 10]


    """

    N = x.size(axis)
    M = int(N / groups)  # each group has M samples
    m = int(n / groups)  # each group has m sampled samples

    if (M < m):
        raise ValueError('The tensor does not has enough samples')

    idx = []
    if mode in ['sequentially', 'Sequentially']:
        for g in range(groups):
            idx += list(range(int(M * g), int(M * g) + m))
    if mode in ['uniformly', 'Uniformly']:
        for g in range(groups):
            idx += list(range(int(M * g), int(M * g + M), int(M / m)))
    if mode in ['randomly', 'Randomly']:
        setseed(seed)
        for g in range(groups):
            idx += randperm(int(M * g), int(M * g + M), m)

    if extra:
        return x[sl(x.dim(), axis=axis, idx=idx)], idx
    else:
        return x[sl(x.dim(), axis=axis, idx=idx)]


def split_tensor(x, ratios=[0.7, 0.2, 0.1], axis=0, shuffle=False, seed=None):
    r"""split tensor

    split a tensor into some parts.

    Parameters
    ----------
    x : {torch Tensor}
        A torch tensor.
    ratios : {list}, optional
        split ratios (the default is [0.7, 0.2, 0.05])
    axis : {number}, optional
        split axis (the default is 0)
    shuffle : {bool}, optional
        whether shuffle (the default is False)
    seed : {number}, optional
        shuffule seed (the default is None)
    """

    y = []

    N, ns = x.size(axis), 0
    if shuffle:
        setseed(seed)
        idx = randperm(0, N, N)
    else:
        idx = list(range(N))

    for ratio in ratios:
        n = int(ratio * N)
        y.append(x[idx[ns:ns + n]])
        ns += n

    return y


if __name__ == '__main__':

    setseed(2020, 'torch')
    x = th.randint(1000, (20, 3, 4))
    y1, idx1 = sample_tensor(x, 10, axis=0, groups=3, mode='sequentially', extra=True)
    y2, idx2 = sample_tensor(x, 10, axis=0, groups=2, mode='uniformly', extra=True)
    y3, idx3 = sample_tensor(x, 10, axis=0, groups=2, mode='randomly', extra=True)

    print(x.shape)
    print(y1.shape)
    print(y2.shape)
    print(y3.shape)
    print(idx1)
    print(idx2)
    print(idx3)

    y1, y2, y3 = split_tensor(x, ratios=[0.7, 0.2, 0.1], axis=0, shuffle=False, seed=None)
    print(y3)
    y1, y2, y3 = split_tensor(x, ratios=[0.7, 0.2, 0.1], axis=0, shuffle=True, seed=None)
    print(y3)
    y1, y2, y3 = split_tensor(x, ratios=[0.7, 0.2, 0.1], axis=0, shuffle=True, seed=2021)
    print(y3)
    y1, y2, y3 = split_tensor(x, ratios=[0.7, 0.2, 0.1], axis=0, shuffle=True, seed=2021)
    print(y3)
    print(y1.shape, y2.shape, y3.shape)
