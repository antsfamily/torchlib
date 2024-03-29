#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2019-11-07 17:00:48
# @Author  : Zhi Liu (zhiliu.mind@gmail.com)
# @Link    : http://iridescent.ink
# @Version : $1.0$

import torch as th
import numpy as np
import torchlib as tl


def unwrap(x, discont=tl.PI, axis=-1, imp=None):
    r"""Unwrap by changing deltas between values to :math:`2\pi` complement.

    Unwrap radian phase `x` by changing absolute jumps greater than
    `discont` to their :math:`2\pi` complement along the given axis.

    Parameters
    ----------
    x : Tensor or ndarray
        The input.
    discont : float, optional
        Maximum discontinuity between values, default is :math:`\pi`.
    axis : int, optional
        Axis along which unwrap will operate, default is the last axis.
    imp : str, optional
        The implenmentation way, ``'numpy'`` --> numpy, None or ``'torch'`` --> torch (default)

    Returns
    -------
    Tensor or ndarray
        The unwrapped.
    
    Examples
    --------

    ::

        x_np = np.array([3.14, -3.12, 3.12, 3.13, -3.11])
        y_np = unwrap(x_np)
        print(y_np, y_np.shape, type(y_np))
        x_th = th.Tensor(x_np)
        y_th = unwrap(x_th)
        print(y_th, y_th.shape, type(y_th))

        print("------------------------")
        x_th = th.tensor([3.14, -3.12, 3.12, 3.13, -3.11])
        x_th = th.cat((th.flip(x_th, dims=[0]), x_th), axis=0)
        print(x_th)
        y_th = unwrap2(x_th)
        print(y_th, y_th.shape, type(y_th))

        print("------------------------")
        x_np = np.array([3.14, -3.12, 3.12, 3.13, -3.11])
        x_th = th.Tensor(x_np)
        y_th = unwrap(x_th, imp='numpy')
        print(y_th, y_th.shape, type(y_th))

        y_th = unwrap(x_th, imp=None)
        print(y_th, y_th.shape, type(y_th))

        # output

        tensor([3.1400, 3.1632, 3.1200, 3.1300, 3.1732], dtype=torch.float64) torch.Size([5]) <class 'torch.Tensor'>
        tensor([3.1400, 3.1632, 3.1200, 3.1300, 3.1732]) torch.Size([5]) <class 'torch.Tensor'>
        ------------------------
        tensor([-3.1100,  3.1300,  3.1200, -3.1200,  3.1400,  3.1400, -3.1200,  3.1200,
                3.1300, -3.1100])
        tensor([3.1732, 3.1300, 3.1200, 3.1632, 3.1400, 3.1400, 3.1632, 3.1200, 3.1300,
                3.1732]) torch.Size([10]) <class 'torch.Tensor'>
        ------------------------
        tensor([3.1400, 3.1632, 3.1200, 3.1300, 3.1732]) torch.Size([5]) <class 'torch.Tensor'>
        tensor([3.1400, 3.1632, 3.1200, 3.1300, 3.1732]) torch.Size([5]) <class 'torch.Tensor'>

    """

    if discont is None:
        discont = tl.PI

    if imp in ['numpy', 'np', 'NUMPY', 'NP']:
        if type(x) is th.Tensor:
            xtype = 'torch.Tensor'
            xdtype, xdevice = x.dtype, x.device
            x = x.cpu().numpy()
        else:
            xtype = 'numpy.array'
        y = np.unwrap(x, discont=discont, axis=axis)
        if xtype == 'torch.Tensor':
            return th.tensor(y, dtype=xdtype, device=xdevice)
        else:
            return y
    elif imp is None or imp == 'torch':
        if type(x) is not th.Tensor:
            x = th.tensor(x)
        high, period = tl.PI, 2 * tl.PI
        low = -high
        pshape = list(x.shape)
        pshape[axis] = 1
        p = th.zeros(pshape, dtype=x.dtype, device=x.device)
        dx = th.diff(x, dim=axis, prepend=p)
        dxm = ((dx - low) % period) + low
        dxm[(dxm == low) & (dx > 0)] = high
        xadj = dxm - dx
        xadj[dx.abs() < discont] = 0

        return x + xadj.cumsum(axis)
    else:
        raise TypeError('Not supported!')


def unwrap2(x, discont=tl.PI, axis=-1):
    r"""Unwrap by changing deltas between values to :math:`2\pi` complement.

    Unwrap radian phase `x` by changing absolute jumps greater than
    `discont` to their :math:`2\pi` complement along the given axis. The elements
    are divided into 2 parts (with equal length) along the given axis.
    The first part is unwrapped in inverse order, while the second part
    is unwrapped in normal order.

    Parameters
    ----------
    x : Tensor
        The input.
    discont : float, optional
        Maximum discontinuity between values, default is :math:`\pi`.
    axis : int, optional
        Axis along which unwrap will operate, default is the last axis.

    Returns
    -------
    Tensor
        The unwrapped.

    see :func:`unwrap`
    """

    if type(x) is not th.Tensor:
        x = th.tensor(x)

    s = x.shape[axis]
    i = int(s / 2)
    # d = x.dim()
    # idx1 = tl.sl(d, [axis], [slice(i - 1, None, -1)])
    y1 = th.index_select(x, axis, th.tensor(range(i - 1, -1, -1), device=x.device))
    y1 = unwrap(y1, discont=discont, axis=axis, imp=None)
    # idx2 = tl.sl(d, [axis], [slice(i, s, 1)])
    y2 = th.index_select(x, axis, th.tensor(range(i, s, 1), device=x.device))
    y2 = unwrap(y2, discont=discont, axis=axis, imp=None)
    y1 = th.index_select(y1, axis, th.tensor(range(i - 1, -1, -1), device=x.device))
    y = th.cat((y1, y2), axis=axis)
    return y


if __name__ == '__main__':

    x_np = np.array([3.14, -3.12, 3.12, 3.13, -3.11])
    y_np = unwrap(x_np)
    print(y_np, y_np.shape, type(y_np))
    x_th = th.Tensor(x_np)
    y_th = unwrap(x_th)
    print(y_th, y_th.shape, type(y_th))

    print("------------------------")
    x_th = th.tensor([3.14, -3.12, 3.12, 3.13, -3.11])
    x_th = th.cat((th.flip(x_th, dims=[0]), x_th), axis=0)
    print(x_th)
    y_th = unwrap2(x_th)
    print(y_th, y_th.shape, type(y_th))

    print("------------------------")
    x_np = np.array([3.14, -3.12, 3.12, 3.13, -3.11])
    x_th = th.Tensor(x_np)
    y_th = unwrap(x_th, imp='numpy')
    print(y_th, y_th.shape, type(y_th))

    y_th = unwrap(x_th, imp=None)
    print(y_th, y_th.shape, type(y_th))
