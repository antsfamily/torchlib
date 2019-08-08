#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2019-07-24 18:29:48
# @Author  : Zhi Liu (zhiliu.mind@gmail.com)
# @Link    : http://iridescent.ink
# @Version : $1.0$


import numpy as np
import copy


def cat(shapes, axis=0):
    r"""Concatenates

    Concatenates the given sequence of seq shapes in the given dimension.
    All tensors must either have the same shape (except in the concatenating dimension) or be empty.

    Parameters
    ----------
    shapes : {tuples or lists}
        (shape1, shape2, ...)
    axis : {number}, optional
        specify the concatenated axis (the default is 0)

    Returns
    -------
    tulpe or list
        concatenated shape

    Raises
    ------
    ValueError
        Shapes are not consistent in axises except the specified one.
    """
    x = 0
    s = copy.copy(shapes[0])
    s = list(s)
    for shape in shapes:
        for ax in range(len(s)):
            if (ax != axis) and (s[ax] != shape[ax]):
                raise ValueError("All tensors must either have \
                    the same shape (except in the concatenating dimension)\
                     or be empty.")
        x += shape[axis]
        # print(x)
    print(s, x)
    s[axis] = x
    return s


def concat2(X1, X2, axis):
    r"""concat2

    concatenate [X1, X2] in aixs direction

    Parameters
    ----------
    X1 : {Torch Tensor}
        the first torch tensor for concatenating
    X2 : {Torch Tensor}
        the second torch tensor for concatenating
    axis : {integer}
        concatenating axis

    Returns
    -------
    Torch Tensor
        concatenated tensors
    """

    if X1 is None or X1 is []:
        return X2
    else:
        # print(type(X1), type(X2))
        # print(X2.shape)
        return th.cat((X1, X2), axis)


if __name__ == '__main__':
    import torchtool as tht
    import torch as th

    x = th.randn(2, 3)
    xs = x.shape
    xs = list(xs)
    print(xs)
    print('===cat')
    print(x.size())
    print('---Theoretical result')

    ys = tht.cat((xs, xs, xs), 0)
    print(ys)

    ys = tht.cat((xs, xs, xs), 1)
    print(ys)
    print('---Torch result')

    y = th.cat((x, x, x), 0)
    print(y.size())
    y = th.cat((x, x, x), 1)
    print(y.size())
