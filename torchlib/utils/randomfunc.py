#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2015-10-15 10:34:16
# @Author  : Zhi Liu (zhiliu.mind@gmail.com)
# @Link    : http://iridescent.ink
# @Version : $1.1$

import torch as th
import numpy as np


def setseed(seed=None, target='numpy'):
    r"""set seed

    Set numpy / torch / torch.random / torch.cuda seed.

    Parameters
    ----------
    seed : {integer or None}, optional
        seed for random number generator (the default is None)
    target : {str}, optional
        - ``'numpy'``: ``np.random.seed(seed)``
        - ``'random'``: ``torch.random.manual_seed(seed)`` (the default)
        - ``'torch'``: ``torch.manual_seed(seed)`` (the default)
        - ``'cuda'``: ``torch.cuda.manual_seed(seed)`` (the default)
        - ``'cudaall'``: ``torch.cuda.manual_seed_all(seed)`` (the default)

    """

    if target in ['numpy', 'np']:
        np.random.seed(seed)
    if target in ['random', 'rand']:
        th.random.manual_seed(seed)
    if target in ['torch']:
        th.manual_seed(seed)
    if target in ['cuda']:
        th.cuda.manual_seed(seed)
    if target in ['cudaall', 'cuda all']:
        th.cuda.manual_seed_all(seed)


def randgrid(start, stop, step, number):
    r"""generates non-repeated random integers

    Generates :attr:`number` non-repeated random integers from :attr:`start` to :attr:`stop` with stepsize :attr:`step`.

    Parameters
    ----------
    start : {integer}
        start sampling point
    stop : {integer}
        stop sampling point
    step : {integer}
        sampling stepsize
    number : {integer}
        the number of samples
    """

    P = np.random.permutation(range(start, stop, step))

    return list(P[0:number])


def randgrid2d(start, stop, step, number, upovr=[0.25, 0.25]):
    r"""generates non-repeated random integers

    Generates :attr:`number` non-repeated random integers from :attr:`start` to :attr:`stop` with stepsize :attr:`step`.

    Parameters
    ----------
    start : {integer list}
        start sampling point in both two dimensions.
    stop : {integer list}
        stop sampling point in both two dimensions.
    step : {integer list}
        sampling stepsize in both two dimensions.
    number : {integer}
        the number of samples
    upovr : {float list}
        the upper limit of overlap rate.


    Examples
    ----------

    ::

        import matplotlib.pyplot as plt
        R = randperm(2, 10, 8)
        print(R)

        mask = np.zeros((5, 6))
        mask[3, 4] = 0
        mask[2, 5] = 0

        y, x = randperm2d(5, 6, 4, mask=mask)

        y = randperm(0, 8192, 800)
        x = randperm(0, 8192, 800)

        y, x = randgrid2d([0, 0], [8192, 8192], [256, 256], 2048)

        plt.figure()
        plt.plot(x, y, '*')
        plt.show()


    """

    H = int((stop[0] - start[0]) / step[0])
    W = int((stop[1] - start[1]) / step[1])
    Y = np.array(range(start[0], stop[0], step[0]))
    X = np.array(range(start[1], stop[1], step[1]))

    nP = H * W
    P = np.random.permutation(range(0, nP, 1))[0:number]

    P = P.flatten()
    N = len(P)

    Ph = P // W
    Pw = P % W

    Y = Y[Ph]
    X = X[Pw]
    if int(upovr[0] * step[0]) > 0:
        Y += np.random.randint(0, int(step[0] * upovr[0]), N)
    if int(upovr[1] * step[1]) > 0:
        X += np.random.randint(0, int(step[1] * upovr[1]), N)

    return list(Y), list(X)


def randperm(start, end, number):
    r"""randperm function like matlab

    genarates diffrent random interges in range [start, end)

    Parameters
    ----------
    start : {integer}
        start sampling point
    end : {integer}
        end sampling point
    number : {integer}
        random numbers
    """

    P = np.random.permutation(range(start, end))

    return list(P[0:number])


def randperm2d(H, W, number, population=None, mask=None):
    """randperm 2d function

    genarates diffrent random interges in range [start, end)

    Parameters
    ----------
    H : {integer}
        height

    W : {integer}
        width
    number : {integer}
        random numbers
    population : {list or numpy array(1d or 2d)}
        part of population in range(0, H*W)
    """

    if population is None:
        population = np.array(range(0, H * W)).reshape(H, W)
    population = np.array(population)
    if mask is not None and np.sum(mask) != 0:
        population = population[mask > 0]

    population = population.flatten()
    population = np.random.permutation(population)

    Ph = np.floor(population / W).astype('int')
    Pw = np.floor(population - Ph * W).astype('int')

    # print(Pw + Ph * W)
    return list(Ph[0:number]), list(Pw[0:number])


if __name__ == '__main__':

    import matplotlib.pyplot as plt
    R = randperm(2, 10, 8)
    print(R)

    mask = np.zeros((5, 6))
    mask[3, 4] = 0
    mask[2, 5] = 0

    Rh, Rw = randperm2d(5, 6, 4, mask=mask)

    print(Rh)
    print(Rw)

    y = randperm(0, 8192, 800)
    x = randperm(0, 8192, 800)

    y, x = randgrid2d([0, 0], [8192, 8192], [256, 256], 400, [0, 0])
    # y = y[0:40]
    # x = x[0:40]

    plt.figure()
    plt.plot(x, y, '*')
    plt.show()
