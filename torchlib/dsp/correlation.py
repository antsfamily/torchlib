#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2020-03-05 16:36:03
# @Author  : Yan Liu & Zhi Liu (zhiliu.mind@gmail.com)
# @Link    : http://iridescent.ink
# @Version : $1.0$
from __future__ import division, print_function, absolute_import

import numpy as np
from torchlib.dsp.ffts import fft, ifft, padfft
from torchlib.base.mathops import nextpow2, ematmul
from torchlib.base.arrayops import cut


def cutfftcorr1(y, nfft, Nx, Nh, shape='same', axis=0, ftshift=False):
    r"""Throwaway boundary elements to get correlation results.

    Throwaway boundary elements to get correlation results.

    Parameters
    ----------
    y : tensor
        array after ``iff``.
    nfft : int
        number of fft points.
    Nx : int
        signal length
    Nh : int
        filter length
    shape : dstr
        output shape:
        1. ``'same' --> same size as input x``, :math:`N_x`
        2. ``'valid' --> valid correlation output``
        3. ``'full' --> full correlation output``, :math:`N_x+N_h-1`
        (the default is 'same')
    axis : int
        correlation axis (the default is 0)
    ftshift : bool
        whether to shift the frequencies (the default is False)

    Returns
    -------
    y : tensor
        array with shape specified by :attr:`same`.
    """

    nfft, Nx, Nh = np.int32([nfft, Nx, Nh])
    N = Nx + Nh - 1
    Nextra = nfft - N

    if nfft < N:
        raise ValueError("~~~To get right results, nfft must be larger than Nx+Nh-1!")

    if ftshift:
        if Nextra == 0:
            if np.mod(Nx, 2) == 0 and np.mod(Nh, 2) > 0:
                y = cut(y, ((0, N), ), axis)
            else:
                y = cut(y, ((1, nfft), (0, 1)), axis)
        else:
            if np.mod(Nx, 2) == 0 and np.mod(Nextra, 2) == 0:
                Nhead = np.int32(np.fix(Nextra / 2.))
            else:
                Nhead = np.int32(np.fix(Nextra / 2.) + 1)
            Nstart2 = Nhead
            Nend2 = np.int32(Nstart2 + N)
            y = cut(y, ((Nstart2, Nend2), ), axis)
    else:
        Nstart2 = 0
        Nend2 = Nx
        Nend1 = nfft
        Nstart1 = int(np.uint(Nend1 - (Nh - 1)))
        y = cut(y, ((Nstart1, Nend1), (Nstart2, Nend2)), axis)

    if shape in ['same', 'SAME', 'Same']:
        Nstart = np.uint(np.fix(Nh / 2.))
        Nend = np.uint(Nstart + Nx)
    elif shape in ['valid', 'VALID', 'Valid']:
        Nstart = np.uint(Nh - 1)
        Nend = np.uint(N - (Nh - 1))
    elif shape in ['full', 'FULL', 'Full']:
        Nstart, Nend = (0, N)
    y = cut(y, ((Nstart, Nend),), axis=axis)
    return y


def fftcorr1(x, h, shape='same', axis=0, nfft=None, ftshift=False, eps=None):
    """Correlation using Fast Fourier Transformation

    Correlation using Fast Fourier Transformation.

    Parameters
    ----------
    x : tensor
        data to be convolved.
    h : tensor
        filter array
    shape : dstr, optional
        output shape:
        1. ``'same' --> same size as input x``, :math:`N_x`
        2. ``'valid' --> valid correlation output``
        3. ``'full' --> full correlation output``, :math:`N_x+N_h-1`
        (the default is 'same')
    axis : int, optional
        correlation axis (the default is 0)
    nfft : int, optional
        number of fft points (the default is None, :math:`2^{nextpow2(N_x+N_h-1)}`),
        note that :attr:`nfft` can not be smaller than :math:`N_x+N_h-1`.
    ftshift : bool, optional
        whether shift frequencies (the default is False)
    eps : None or float, optional
        x[abs(x)<eps] = 0 (the default is None, does nothing)

    Returns
    -------
    y : tensor
        Correlation result array.

    """

    if np.ndim(h) == 1 and axis > 0:
        h = np.expand_dims(h, list(range(axis)))
    Nh = np.size(h, axis)
    Nx = np.size(x, axis)

    N = Nx + Nh - 1
    if nfft is None:
        nfft = 2**nextpow2(N)
    else:
        if nfft < N:
            raise ValueError("~~~To get right results, nfft must be larger than Nx+Nh-1!")

    x = padfft(x, nfft, axis, ftshift)
    h = padfft(h, nfft, axis, ftshift)
    x = fft(x, nfft, axis, norm=None, shift=ftshift)
    h = fft(h, nfft, axis, norm=None, shift=ftshift)
    h[..., 1] = -h[..., 1]  # conj
    y = ematmul(x, h)  # element-by-element complex multiplication

    y = ifft(y, nfft, axis, norm=None, shift=ftshift)
    y = cutfftcorr1(y, nfft, Nx, Nh, shape, axis, ftshift)

    if eps is not None:
        y[abs(y) < eps] = 0.

    return y


def xcorr(A, B, shape='same', axis=0):
    r"""Cross-correlation function estimates.


    Parameters
    ----------
    A : numpy array
        data1
    B : numpy array
        data2
    mod : str, optional
        - 'biased': scales the raw cross-correlation by 1/M.
        - 'unbiased': scales the raw correlation by 1/(M-abs(lags)).
        - 'coeff': normalizes the sequence so that the auto-correlations
                   at zero lag are identically 1.0.
        - 'none': no scaling (this is the default).
    """

    if np.ndim(A) == 1 and np.ndim(B) == 1:
        Ma, Mb = (1, 1)
        Na, Nb = (len(A), len(B))
    if np.ndim(A) == 2 and np.ndim(B) == 2:
        print(A.shape, B.shape)
        Ma, Na = A.shape
        Mb, Nb = B.shape
        if axis == 1 and Ma != Mb:
            raise ValueError("~~~Array A and B should have the same rows!")
        if axis == 0 and Na != Nb:
            raise ValueError("~~~Array A and B should have the same cols!")
    if shape in ['same', 'SAME']:
        Nc = max(Na, Nb)
    elif shape in ['full', 'FULL']:
        Nc = Na + Nb - 1
    elif shape in ['valid', 'VALID']:
        Nc = max(Na, Nb) - max(Na, Nb) + 1
    else:
        raise ValueError("~~~Not supported shape:" + shape + "!")

    CPLXDTYPESTR = ['complex128', 'complex64', 'complex']

    if A.dtype in CPLXDTYPESTR or B.dtype in CPLXDTYPESTR:
        dtype = 'complex'
    else:
        dtype = 'float'

    if np.ndim(A) == 1 and np.ndim(B) == 1:
        C = np.correlate(A, B, mode=shape)
    if np.ndim(A) == 2 and np.ndim(B) == 2:
        C = np.zeros((Ma, Nc), dtype=dtype)
        if axis == 0:
            for n in range(Na):
                C[:, n] = np.correlate(A[:, n], B[:, n], mode=shape)
        if axis == 1:
            for m in range(Ma):
                C[m, :] = np.correlate(A[m, :], B[m, :], mode=shape)
    return C


def accc(Sr, isplot=False):
    r"""Average cross correlation coefficient

    Average cross correlation coefficient (ACCC)

    .. math::
       \overline{C(\eta)}=\sum_{\eta} s^{*}(\eta) s(\eta+\Delta \eta)

    where, :math:`\eta, \Delta \eta` are azimuth time and it's increment.


    Parameters
    ----------
    Sr : numpy array
        SAR raw signal data :math:`N_a×N_r` or range compressed data.

    Returns
    -------
    1d array
        ACCC in each range cell.
    """

    Na, Nr = Sr.shape

    acccv = np.sum(Sr[1:, :] * np.conj(Sr[0:-1, :]), 0)

    if isplot:
        import matplotlib.pyplot as plt
        import torchlib
        plt.figure()
        plt.subplot(121)
        torchlib.cplot(acccv, '-b')
        plt.title('ACCC (all range cell)')
        plt.subplot(122)
        ccv = Sr[1:, 0] * np.conj(Sr[0:-1, 0])
        torchlib.cplot(ccv, '-b')
        torchlib.cplot([np.mean(ccv)], '-r')
        plt.title('CCC (0-th range cell)')
        plt.show()

    return acccv


if __name__ == '__main__':
    import pyailib as pl
    import torchlib as tl
    import torch as th

    shape = 'same'
    ftshift = False
    ftshift = True
    x_np = np.array([1, 2, 3, 4, 5])
    h_np = np.array([1 + 2j, 2, 3, 4, 5, 6, 7])

    x_th = th.tensor(x_np)
    h_th = th.tensor(h_np)
    x_th = th.stack([x_th, th.zeros(x_th.size())], dim=-1)
    h_th = th.stack([h_th.real, h_th.imag], dim=-1)

    y1 = pl.fftcorr1(x_np, h_np, axis=0, nfft=None, shape=shape, ftshift=ftshift)
    y2 = tl.fftcorr1(x_th, h_th, axis=0, nfft=None, shape=shape, ftshift=ftshift)

    y2 = th.view_as_complex(y2)
    y2 = y2.cpu().numpy()

    print(y1)
    print(y2)
    print(np.sum(np.abs(y1 - y2)), np.sum(np.angle(y1) - np.angle(y2)))
