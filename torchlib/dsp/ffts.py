#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2019-11-07 17:00:48
# @Author  : Zhi Liu (zhiliu.mind@gmail.com)
# @Link    : http://iridescent.ink
# @Version : $1.0$

import numpy as np
import torch as th


def fftshift(x, axis=None):
    r"""Shift the zero-frequency component to the center of the spectrum.

    This function swaps half-spaces for all axes listed (defaults to all).
    Note that ``y[0]`` is the Nyquist component only if ``len(x)`` is even.

    Parameters
    ----------
    x : {torch tensor}
        Input tensor.
    axis : {number}, optional
        Axes over which to shift. (Default is None, which shifts all axes.)

    Returns
    -------
    y : {torch tensor}
        The shifted tensor.

    See Also
    --------
    ifftshift : The inverse of `fftshift`.

    Examples
    --------
        import numpy as np
        import torchlib as tl
        import torch as th

        x = [1, 2, 3, 4, 5, 6]
        y = np.fft.fftshift(x)
        print(y)
        x = th.tensor(x)
        y = tl.fftshift(x)
        print(y)

        x = [1, 2, 3, 4, 5, 6, 7]
        y = np.fft.fftshift(x)
        print(y)
        x = th.tensor(x)
        y = tl.fftshift(x)
        print(y)

        axis = (0, 1)  # axis = 0, axis = 1
        x = [[1, 2, 3, 4, 5, 6], [0, 2, 3, 4, 5, 6]]
        y = np.fft.fftshift(x, axis)
        print(y)
        x = th.tensor(x)
        y = tl.fftshift(x, axis)
        print(y)


        x = [[1, 2, 3, 4, 5, 6, 7], [0, 2, 3, 4, 5, 6, 7]]
        y = np.fft.fftshift(x, axis)
        print(y)
        x = th.tensor(x)
        y = tl.fftshift(x, axis)
        print(y)

    """

    if axis is None:
        axis = tuple(range(x.dim()))
    elif type(axis) is int:
        axis = tuple([axis])
    for a in axis:
        n = x.size(a)
        p = int(n / 2.)
        x = th.roll(x, p, dims=a)
    return x


def ifftshift(x, axis=None):
    r"""Shift the zero-frequency component back.

    The inverse of `fftshift`. Although identical for even-length `x`, the
    functions differ by one sample for odd-length `x`.

    Parameters
    ----------
    x : {torch tensor}
        The input tensor.
    axis : {number}, optional
        Axes over which to shift. (Default is None, which shifts all axes.)

    Returns
    -------
    y : {torch tensor}
        The shifted tensor.

    See Also
    --------
    fftshift : The inverse of `ifftshift`.

    Examples
    --------
        import numpy as np
        import torchlib as tl
        import torch as th

        x = [1, 2, 3, 4, 5, 6]
        y = np.fft.fftshift(x)
        print(y)
        x = th.tensor(x)
        y = tl.fftshift(x)
        print(y)

        x = [1, 2, 3, 4, 5, 6, 7]
        y = np.fft.fftshift(x)
        print(y)
        x = th.tensor(x)
        y = tl.fftshift(x)
        print(y)

        axis = (0, 1)  # axis = 0, axis = 1
        x = [[1, 2, 3, 4, 5, 6], [0, 2, 3, 4, 5, 6]]
        y = np.fft.fftshift(x, axis)
        print(y)
        x = th.tensor(x)
        y = tl.fftshift(x, axis)
        print(y)


        x = [[1, 2, 3, 4, 5, 6, 7], [0, 2, 3, 4, 5, 6, 7]]
        y = np.fft.fftshift(x, axis)
        print(y)
        x = th.tensor(x)
        y = tl.fftshift(x, axis)
        print(y)

    """

    if axis is None:
        axis = tuple(range(x.dim()))
    elif type(axis) is int:
        axis = tuple([axis])
    for a in axis:
        n = x.size(a)
        p = int((n + 1) / 2.)
        x = th.roll(x, p, dims=a)
    return x


def padfft(X, nfft=None, axis=0, shift=False):
    r"""PADFT Pad array for doing FFT or IFFT

    PADFT Pad array for doing FFT or IFFT

    Parameters
    ----------
    X : {torch.ndarray}
        Data to be padded.
    nfft : {number or None}
        Padding size.
    axis : {number}, optional
        Padding dimension. (the default is 0)
    shift : {bool}, optional
        Whether to shift the frequency (the default is False)
    """

    if axis is None:
        axis = 0

    Nx = X.size(axis)

    if nfft < Nx:
        raise ValueError('Output size is smaller than input size!')

    pad = list(X.size())

    Np = int(np.uint(nfft - Nx))

    if shift:
        pad[axis] = int(np.fix((Np + 1) / 2.))
        Z = th.zeros(pad, dtype=X.dtype, device=X.device)
        X = th.cat((Z, X), dim=axis)
        pad[axis] = Np - pad[axis]
        Z = th.zeros(pad, dtype=X.dtype, device=X.device)
        X = th.cat((X, Z), dim=axis)
    else:
        pad[axis] = Np
        Z = th.zeros(pad, dtype=X.dtype, device=X.device)
        X = th.cat((X, Z), dim=axis)

    return X


def fftfreq(fs, n, norm=False, shift=False):
    r"""Return the Discrete Fourier Transform sample frequencies

    Return the Discrete Fourier Transform sample frequencies.

    The returned float array `f` contains the frequency bin centers in cycles
    per unit of the sample spacing (with zero at the start).  For instance, if
    the sample spacing is in seconds, then the frequency unit is cycles/second.

    Given a window length `n` and a sample spacing `d`, if shift is ``True`` and ``norm`` is True::

      f = [-n/2, ..., -1,     0, 1, ...,   n/2-1] / (d*n)   if n is even
      f = [-(n-1)/2, ..., -1, 0, 1, ..., (n-1)/2] / (d*n)   if n is odd

    Given a window length `n` and a sample spacing `d`, if shift is ``False``::

      f = [0, 1, ...,   n/2-1,     -n/2, ..., -1] / (d*n)   if n is even
      f = [0, 1, ..., (n-1)/2, -(n-1)/2, ..., -1] / (d*n)   if n is odd

    where :math:`d = 1/f_s`.

    Parameters
    ----------
    fs : {float}
        Sampling rate.
    n : {integer}
        Number of samples.
    norm : {bool}
        Normalize the frequencies.
    shift : {bool}
        Does shift the zero frequency to center (Default is ``False``).

    Returns
    -------
    torch tensor
        frequency array with size :math:`n×1`.
    """
    d = 1. / fs
    if n % 2 == 0:
        N = n
        N1 = int(n / 2.)
        N2 = int(n / 2.)
        endpoint = False
    else:
        N = n - 1
        N1 = int((n + 1) / 2.)
        N2 = int((n - 1) / 2.)
        endpoint = True

    if shift:
        f = np.linspace(-N / 2., N / 2., n, endpoint=endpoint)
    else:
        f = np.hstack((np.linspace(0, N / 2., N1, endpoint=endpoint),
                       np.linspace(-N / 2., 0, N2, endpoint=False)))
    if norm:
        return th.tensor(f / n, dtype=th.float32)
    else:
        return th.tensor(f / (d * n), dtype=th.float32)


def fft(x, nfft=None, axis=0, norm=False, shift=False):
    """FFT in torchlib

    IFFT in torchlib, since ifft in torch only supports complex-complex transformation,
    for real ifft, we insert imaginary part with zeros (torch.stack((x,torch.zeros_like(x), dim=-1))),
    also you can use torch's rifft.

    Parameters
    ----------
    x : {torch array}
        both complex and real representation are supported. Since torch does not
        support complex array, when :attr:`x` is complex, we will change the representation
        in real formation(last dimension is 2, real, imag), after FFT, it will be change back.
    nfft : {integer}, optional
        number of fft points (the default is None --> equals to signal dimension)
    axis : {number}, optional
        axis of fft (the default is 0, which the first dimension)
    norm : {bool}, optional
        normalization (the default is False)
    shift : {bool}, optional
        shift the zero frequency to center (the default is False)

    Returns
    -------
    y : {torch array}
        fft results torch array with the same type as :attr:`x`

    Raises
    ------
    ValueError
        nfft is small than signal dimension.
    """

    if th.is_complex(x):
        cplxflag = True
        x = th.view_as_real(x)
    else:
        cplxflag = False
        if x.size(-1) != 2:
            x = th.stack((x, th.zeros_like(x)), dim=-1)

    d = x.size(axis)
    if nfft is None:
        nfft = d
    if d < nfft:
        pad = list(x.size())
        pad[axis] = nfft - d
        z = th.zeros(pad, dtype=x.dtype, device=x.device)
        x = th.cat((x, z), dim=axis)
    elif d > nfft:
        raise ValueError('nfft is small than signal dimension!')

    y = th.fft(x.transpose(axis, -2), signal_ndim=1, normalized=norm).transpose(axis, -2)

    if cplxflag:
        y = th.view_as_complex(y)

    return y


def ifft(x, nfft=None, axis=0, norm=False, shift=False):
    """IFFT in torchlib

    IFFT in torchlib, since ifft in torch only supports complex-complex transformation,
    for real ifft, we insert imaginary part with zeros (torch.stack((x,torch.zeros_like(x), dim=-1))),
    also you can use torch's rifft.

    Parameters
    ----------
    x : {torch array}
        both complex and real representation are supported. Since torch does not
        support complex array, when :attr:`x` is complex, we will change the representation
        in real formation(last dimension is 2, real, imag), after IFFT, it will be change back.
    nfft : {integer}, optional
        number of ifft points (the default is None --> equals to signal dimension)
    axis : {number}, optional
        axis of ifft (the default is 0, which the first dimension)
    norm : {bool}, optional
        normalization (the default is False)
    shift : {bool}, optional
        shift the zero frequency to center (the default is False)

    Returns
    -------
    y : {torch array}
        ifft results torch array with the same type as :attr:`x`

    Raises
    ------
    ValueError
        nfft is small than signal dimension.
    """

    if th.is_complex(x):
        cplxflag = True
        x = th.view_as_real(x)
    else:
        cplxflag = False
        if x.size(-1) != 2:
            x = th.stack((x, th.zeros_like(x)), dim=-1)

    d = x.size(axis)
    if nfft is None:
        nfft = d
    if d < nfft:
        pad = list(x.size())
        pad[axis] = nfft - d
        z = th.zeros(pad, dtype=x.dtype, device=x.device)
        x = th.cat((x, z), dim=axis)
    elif d > nfft:
        raise ValueError('nfft is small than signal dimension!')

    y = th.ifft(x.transpose(axis, -2), signal_ndim=1, normalized=norm).transpose(axis, -2)

    if cplxflag:
        y = th.view_as_complex(y)
    return y


if __name__ == '__main__':

    import numpy as np

    nfft = 8
    x1 = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]])
    print(x1.shape)
    y1 = np.fft.fft(x1, n=nfft, axis=1, norm=None)
    print(y1, y1.shape)

    x2 = th.tensor(x1, dtype=th.float32)
    x2 = th.stack([x2, th.zeros(x2.size())], dim=-1)

    y2 = fft(x2, nfft=nfft, axis=1, norm=False)
    print(y2, y2.shape)
    x2 = ifft(y2, nfft=nfft, axis=1, norm=False)
    print(x2)
