#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2020-03-23 19:28:33
# @Author  : Yan Liu & Zhi Liu (zhiliu.mind@gmail.com)
# @Link    : http://iridescent.ink
# @Version : $1.0$
from __future__ import division, print_function, absolute_import
import torch as th


def window(n, wtype=None, isperiodic=True, dtype=None, device=None, requires_grad=False):
    r"""Generates window

    Parameters
    ----------
    n : int
        The length of the window.
    wtype : str or None, optional
        The window type:
        - ``'rectangle'`` for rectangle window
        - ``'bartlett'`` for bartlett window
        - ``'blackman'`` for blackman window
        - ``'hamming x y'`` for hamming window with :math:`\alpha=x, \beta=y`, default is 0.54, 0.46.
        - ``'hanning'`` for hanning window
        - ``'kaiser x'`` for kaiser window with :math:`\beta=x`, default is 12.
    isperiodic : bool, optional
        If True (default), returns a window to be used as periodic function.
        If False, return a symmetric window.
    dtype : None, optional
        The desired data type of returned tensor.
    device : None, optional
        The desired device of returned tensor.
    requires_grad : bool, optional
        If autograd should record operations on the returned tensor. Default: False.

    Returns
    -------
    tensor
        A 1-D tensor of size (n,) containing the window
    """

    if wtype is None:
        return th.ones(n, dtype=dtype, device=device, requires_grad=requires_grad)

    wtype.lower()
    wtype = wtype.split()

    if wtype[0] == 'rectangle':
        return th.ones(n, dtype=dtype, device=device, requires_grad=requires_grad)
    if wtype[0] == 'bartlett':
        return th.bartlett_window(n, periodic=isperiodic, dtype=dtype, layout=th.strided, device=device, requires_grad=requires_grad)
    if wtype[0] == 'blackman':
        return th.blackman_window(n, periodic=isperiodic, dtype=dtype, layout=th.strided, device=device, requires_grad=requires_grad)
    if wtype[0] == 'hamming':
        alpha, beta = 0.54, 0.46
        if len(wtype) == 2:
            alpha = float(wtype[1])
        elif len(wtype) == 3:
            alpha, beta = float(wtype[1]), float(wtype[2])
        return th.hamming_window(n, periodic=isperiodic, alpha=alpha, beta=beta, dtype=dtype, layout=th.strided, device=device, requires_grad=requires_grad)
    if wtype[0] == 'hanning':
        return th.hann_window(n, periodic=isperiodic, dtype=dtype, layout=th.strided, device=device, requires_grad=requires_grad)
    if wtype[0] == 'kaiser':
        beta = 12.
        if len(wtype) > 1:
            beta = float(wtype[1])
        return th.kaiser_window(n, periodic=isperiodic, beta=beta, dtype=dtype, layout=th.strided, device=device, requires_grad=requires_grad)


def windowing(x, w, axis=None):
    """Performs windowing operation in the specified axis.

    Parameters
    ----------
    x : tensor
        The input tensor.
    w : tensor
        A 1-d window tensor.
    axis : int or None, optional
        The axis.

    Returns
    -------
    tensor
        The windowed data.

    """
    if axis is None:
        return x * w

    if type(axis) is not int:
        raise TypeError('The axis should be a integer!')

    d = x.dim()
    shape = [1] * d
    shape[axis] = len(w)

    w = w.view(shape)
    return x * w


if __name__ == '__main__':

    import matplotlib.pyplot as plt

    n = 512
    wtype = 'bartlett'
    wtype = 'blackman'
    wtype = 'hamming 0.54 0.46'
    wtype = 'hanning'
    wtype = 'kaiser 12'
    w = window(n, wtype=wtype)

    plt.figure()
    plt.grid()
    plt.plot(w)
    plt.show()
