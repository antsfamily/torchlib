#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2019-07-24 18:29:48
# @Author  : Zhi Liu (zhiliu.mind@gmail.com)
# @Link    : http://iridescent.ink
# @Version : $1.0$


import numpy as np
import torch as th
import torchlib as tl


def ebeo(a, b, op='+'):
    r"""element by element operation

    Element by element operation.

    Parameters
    ----------
    a : list, tuple, tensor or array
        The first list/tuple/nparray/tensor.
    b : list, tuple, tensor or array
        The second list/tuple/nparray/tensor.
    op : str, optional
        Supported operations are:
        - ``'+'`` or ``'add'`` for addition (default)
        - ``'-'`` or ``'sub'`` for substraction
        - ``'*'`` or ``'mul'`` for multiplication
        - ``'/'`` or ``'div'`` for division
        - ``'**'`` or ``pow`` for power
        - ``'<'``, or ``'lt'`` for less than
        - ``'<='``, or ``'le'`` for less than or equal to
        - ``'>'``, or ``'gt'`` for greater than
        - ``'>='``, or ``'ge'`` for greater than or equal to
        - ``'&'`` for bitwise and
        - ``'|'`` for bitwise or
        - ``'^'`` for bitwise xor
        - function for custom operation.

    Raises
    ------
    TypeError
        If the specified operator not in the above list, raise a TypeError.
    """
    if op in ['+', 'add']:
        return [i + j for i, j in zip(a, b)]
    if op in ['-', 'sub']:
        return [i - j for i, j in zip(a, b)]
    if op in ['*', 'mul']:
        return [i * j for i, j in zip(a, b)]
    if op in ['/', 'div']:
        return [i / j for i, j in zip(a, b)]
    if op in ['**', '^', 'pow']:
        return [i ** j for i, j in zip(a, b)]
    if isinstance(op, str):
        raise TypeError("Not supported operation: " + op + "!")
    else:
        return [op(i, j) for i, j in zip(a, b)]


def sinc(x):
    """Applies sinc function to a tensor

    Parameters
    ----------
    x : Tensor
        input tensor

    Returns
    -------
    Tensor
        after sinc transformation.
    """

    flag = False
    if type(x) is not th.Tensor:
        flag = True
        x = th.tensor(x)
    x = th.where(x.abs() < 1e-32, th.tensor(1., dtype=x.dtype, device=x.device), th.sin(tl.PI * x) / (tl.PI * x))

    if flag:
        return x.item()
    else:
        return x


def nextpow2(x):
    r"""get the next higher power of 2.

    Given an number :math:`x`, returns the first p such that :math:`2^p >=|x|`. 

    Args:
        x (int or float): an number.

    Returns:
        int: Next higher power of 2.

    Examples:

        ::

            print(prevpow2(-5), nextpow2(-5))
            print(prevpow2(5), nextpow2(5))
            print(prevpow2(0.3), nextpow2(0.3))
            print(prevpow2(7.3), nextpow2(7.3))
            print(prevpow2(-3.5), nextpow2(-3.5))

            # output
            2 3
            2 3
            -2 -1
            2 3
            1 2

    """

    return int(np.ceil(np.log2(np.abs(x) + 1e-32)))


def prevpow2(x):
    r"""get the previous lower power of 2.

    Given an number :math:`x`, returns the first p such that :math:`2^p <=|x|`. 

    Args:
        x (int or float): an number.

    Returns:
        int: Next higher power of 2.

    Examples:

        ::

            print(prevpow2(-5), nextpow2(-5))
            print(prevpow2(5), nextpow2(5))
            print(prevpow2(0.3), nextpow2(0.3))
            print(prevpow2(7.3), nextpow2(7.3))
            print(prevpow2(-3.5), nextpow2(-3.5))

            # output
            2 3
            2 3
            -2 -1
            2 3
            1 2

    """
    
    return int(np.floor(np.log2(np.abs(x) + 1e-32)))


def ematmul(A, B, cdim=None):
    r"""Element-by-element complex multiplication

    like A .* B in matlab

    Parameters
    ----------
    A : tensor
        any size torch tensor, both complex and real representation are supported.
        For real representation, the last dimension is 2 (the first --> real part, the second --> imaginary part).
    B : tensor
        any size torch tensor, both complex and real representation are supported.
        For real representation, the last dimension is 2 (the first --> real part, the second --> imaginary part).
        :attr:`B` has the same size as :attr:`A`.
    cdim : int or None
        if :attr:`A` and :attr:`B` are represented in real format, :attr:`cdim`
        should be specified (None for -1).

    Returns
    -------
    tensor
        result of element-by-element complex multiplication with the same repesentation as :attr:`A` and :attr:`B`.
    
    Examples
    ----------

    ::

        th.manual_seed(2020)
        Ar = th.randn((3, 3, 2))
        Br = th.randn((3, 3, 2))

        Ac = th.view_as_complex(Ar)
        Bc = th.view_as_complex(Br)

        Mr = th.view_as_real(Ac * Bc)
        print(th.sum(Mr - ematmul(Ar, Br)))
        print(th.sum(Ac * Bc - ematmul(Ac, Bc)))

        # output
        tensor(-1.1921e-07)
        tensor(-1.1921e-07+0.j)

    """

    if th.is_complex(A) and th.is_complex(B):
        return A.real * B.real - A.imag * B.imag + 1j * (A.real * B.imag + A.imag * B.real)
    else:
        cdim = -1 if cdim is None else cdim
        d = A.dim()
        return th.stack((A[tl.sl(d, cdim, [0])] * B[tl.sl(d, cdim, [0])] - A[tl.sl(d, cdim, [1])] * B[tl.sl(d, cdim, [1])], A[tl.sl(d, cdim, [0])] * B[tl.sl(d, cdim, [1])] + A[tl.sl(d, cdim, [1])] * B[tl.sl(d, cdim, [0])]), dim=cdim)


def matmul(A, B, cdim=None):
    r"""Complex matrix multiplication

    like A * B in matlab

    Parameters
    ----------
    A : tensor
        any size torch tensor, both complex and real representation are supported.
        For real representation, the last dimension is 2 (the first --> real part, the second --> imaginary part).
    B : tensor
        any size torch tensor, both complex and real representation are supported.
        For real representation, the last dimension is 2 (the first --> real part, the second --> imaginary part).
    cdim : int or None
        if :attr:`A` and :attr:`B` are represented in real format, :attr:`cdim`
        should be specified (None for -1).

    Returns
    -------
    tensor
        result of complex multiplication with the same repesentation as :attr:`A` and :attr:`B`.
    
    Examples
    ----------

    ::

        th.manual_seed(2020)
        Ar = th.randn((3, 3, 2))
        Br = th.randn((3, 3, 2))

        Ac = th.view_as_complex(Ar)
        Bc = th.view_as_complex(Br)

        print(th.sum(th.matmul(Ac, Bc) - matmul(Ac, Bc)))
        Mr = matmul(Ar, Br)
        Mc = th.view_as_real(th.matmul(Ac, Bc))
        print(th.sum(Mr - Mc))

        # output
        tensor(-4.7684e-07+5.9605e-08j)
        tensor(4.1723e-07)
    """

    if th.is_complex(A) and th.is_complex(B):
        return th.matmul(A.real, B.real) - th.matmul(A.imag, B.imag) + 1j * (th.matmul(A.real, B.imag) + th.matmul(A.imag, B.real))
    else:
        cdim = -1 if cdim is None else cdim
        d = A.dim()
        th.matmul(A[tl.sl(d, cdim, [0])], B[tl.sl(d, cdim, [0])])
        th.matmul(A[tl.sl(d, cdim, [1])], B[tl.sl(d, cdim, [1])])
        return th.stack((th.matmul(A[tl.sl(d, cdim, [0])], B[tl.sl(d, cdim, [0])]) - th.matmul(A[tl.sl(d, cdim, [1])], B[tl.sl(d, cdim, [1])]), th.matmul(A[tl.sl(d, cdim, [0])], B[tl.sl(d, cdim, [1])]) + th.matmul(A[tl.sl(d, cdim, [1])], B[tl.sl(d, cdim, [0])])), dim=cdim)


def c2r(X, cdim=-1):
    r"""complex representaion to real representaion

    Parameters
    ----------
    X : tensor
        input in complex representaion
    cdim : int, optional
        real and imag dimention in real format, by default -1

    Returns
    -------
    tensor
        output in real representaion

    see also :func:`r2c`

    Examples
    ----------

    ::

        th.manual_seed(2020)
        Xr = th.randint(0, 30, (3, 3, 2))
        Xc = Xr[..., 0] + 1j * Xr[..., 1]
        Yr = c2r(Xc, cdim=0)
        Yc = r2c(Yr, cdim=0)
        print(Xr, Xr.shape, 'Xr')
        print(Xc, Xc.shape, 'Xc')
        print(Yr, Yr.shape, 'Yr')
        print(Yc, Yc.shape, 'Yc')

        # ---output
        tensor([[[20,  6],
                [27, 12],
                [25, 21]],

                [[21, 19],
                [29, 24],
                [25, 10]],

                [[16, 14],
                [ 6,  9],
                [ 5, 29]]]) torch.Size([3, 3, 2]) Xr
        tensor([[20.+6.j, 27.+12.j, 25.+21.j],
                [21.+19.j, 29.+24.j, 25.+10.j],
                [16.+14.j,  6.+9.j,  5.+29.j]]) torch.Size([3, 3]) Xc
        tensor([[[20., 27., 25.],
                [21., 29., 25.],
                [16.,  6.,  5.]],

                [[ 6., 12., 21.],
                [19., 24., 10.],
                [14.,  9., 29.]]]) torch.Size([2, 3, 3]) Yr
        tensor([[20.+6.j, 27.+12.j, 25.+21.j],
                [21.+19.j, 29.+24.j, 25.+10.j],
                [16.+14.j,  6.+9.j,  5.+29.j]]) torch.Size([3, 3]) Yc

    """

    return th.stack((X.real, X.imag), dim=cdim)


def r2c(X, cdim=-1, keepcdim=False):
    r"""real representaion to complex representaion

    Parameters
    ----------
    X : tensor
        input in real representaion
    cdim : int, optional
        real and imag dimention in real format, by default -1
    keepcdim : bool, optional
        keep complex dimention?

    Returns
    -------
    tensor
        output in complex representaion

    see also :func:`c2r`

    Examples
    ----------

    ::

        th.manual_seed(2020)
        Xr = th.randint(0, 30, (3, 3, 2))
        Xc = Xr[..., 0] + 1j * Xr[..., 1]
        Yr = c2r(Xc, cdim=0)
        Yc = r2c(Yr, cdim=0)
        print(Xr, Xr.shape, 'Xr')
        print(Xc, Xc.shape, 'Xc')
        print(Yr, Yr.shape, 'Yr')
        print(Yc, Yc.shape, 'Yc')

        # ---output
        tensor([[[20,  6],
                [27, 12],
                [25, 21]],

                [[21, 19],
                [29, 24],
                [25, 10]],

                [[16, 14],
                [ 6,  9],
                [ 5, 29]]]) torch.Size([3, 3, 2]) Xr
        tensor([[20.+6.j, 27.+12.j, 25.+21.j],
                [21.+19.j, 29.+24.j, 25.+10.j],
                [16.+14.j,  6.+9.j,  5.+29.j]]) torch.Size([3, 3]) Xc
        tensor([[[20., 27., 25.],
                [21., 29., 25.],
                [16.,  6.,  5.]],

                [[ 6., 12., 21.],
                [19., 24., 10.],
                [14.,  9., 29.]]]) torch.Size([2, 3, 3]) Yr
        tensor([[20.+6.j, 27.+12.j, 25.+21.j],
                [21.+19.j, 29.+24.j, 25.+10.j],
                [16.+14.j,  6.+9.j,  5.+29.j]]) torch.Size([3, 3]) Yc

    """
    d = X.dim()
    if keepcdim:
        return X[tl.sl(d, cdim, [[0]])] + 1j * X[tl.sl(d, cdim, [[1]])]
    else:
        return X[tl.sl(d, cdim, [0])] + 1j * X[tl.sl(d, cdim, [1])]


def conj(X, cdim=None):
    r"""conjugates a tensor

    Both complex and real representation are supported.

    Parameters
    ----------
    X : tensor
        input
    cdim : int or None
        If :attr:`X` is complex-valued, :attr:`cdim` is ignored. If :attr:`X` is real-valued and :attr:`cdim` is integer
        then :attr:`X` will be treated as complex-valued, in this case, :attr:`cdim` specifies the complex axis;
        otherwise (None), :attr:`X` will be treated as real-valued

    Returns
    -------
    tensor
         the inputs's conjugate matrix.

    Examples
    ---------

    ::

        th.manual_seed(2020)
        X = th.rand((2, 3, 3))

        print('---conj')
        print(conj(X, cdim=0))
        print(conj(X[0] + 1j * X[1]))

        # ---output
        ---conj
        tensor([[[ 0.4869,  0.1052,  0.5883],
                [ 0.1161,  0.4949,  0.2824],
                [ 0.5899,  0.8105,  0.2512]],

                [[-0.6307, -0.5403, -0.8033],
                [-0.7781, -0.4966, -0.8888],
                [-0.5570, -0.7127, -0.0339]]])
        tensor([[0.4869-0.6307j, 0.1052-0.5403j, 0.5883-0.8033j],
                [0.1161-0.7781j, 0.4949-0.4966j, 0.2824-0.8888j],
                [0.5899-0.5570j, 0.8105-0.7127j, 0.2512-0.0339j]])

    """

    if th.is_complex(X):  # complex in complex
        return th.conj(X)
    else:
        if cdim is None:  # real
            return X
        else:  # complex in real
            d = X.dim()
            return th.cat((X[tl.sl(d, axis=cdim, idx=[[0]])], -X[tl.sl(d, axis=cdim, idx=[[1]])]), dim=cdim)


def real(X, cdim=None, keepcdim=False):
    r"""obtain real part of a tensor

    Both complex and real representation are supported.

    Parameters
    ----------
    X : tensor
        input
    cdim : int or None
        If :attr:`X` is complex-valued, :attr:`cdim` is ignored. If :attr:`X` is real-valued and :attr:`cdim` is integer
        then :attr:`X` will be treated as complex-valued, in this case, :attr:`cdim` specifies the complex axis;
        otherwise (None), :attr:`X` will be treated as real-valued
    keepcdim : bool, optional
        keep the complex dimension?

    Returns
    -------
    tensor
         the inputs's real part tensor.

    Examples
    ---------

    ::

        th.manual_seed(2020)
        X = th.rand((2, 3, 3))

        print('---real')
        print(real(X, cdim=0))
        print(real(X[0] + 1j * X[1]))

        # ---output
        ---real
        tensor([[0.4869, 0.1052, 0.5883],
                [0.1161, 0.4949, 0.2824],
                [0.5899, 0.8105, 0.2512]])
        tensor([[0.4869, 0.1052, 0.5883],
                [0.1161, 0.4949, 0.2824],
                [0.5899, 0.8105, 0.2512]])
    """

    if th.is_complex(X):  # complex in complex
        return X.real
    else:
        if cdim is None:  # real
            return X
        else:  # complex in real
            d = X.ndim
            idx = [[0]] if keepcdim else [0]
            return X[tl.sl(d, axis=cdim, idx=idx)]


def imag(X, cdim=None, keepcdim=False):
    r"""obtain imaginary part of a tensor

    Both complex and real representation are supported.

    Parameters
    ----------
    X : tensor
        input
    cdim : int or None
        If :attr:`X` is complex-valued, :attr:`cdim` is ignored. If :attr:`X` is real-valued and :attr:`cdim` is integer
        then :attr:`X` will be treated as complex-valued, in this case, :attr:`cdim` specifies the complex axis;
        otherwise (None), :attr:`X` will be treated as real-valued
    keepcdim : bool, optional
        keep the complex dimension?

    Returns
    -------
    tensor
         the inputs's imaginary part tensor.

    Examples
    ---------

    ::

        th.manual_seed(2020)
        X = th.rand((2, 3, 3))

        print('---imag')
        print(imag(X, cdim=0))
        print(imag(X[0] + 1j * X[1]))

        # ---output
        ---imag
        tensor([[0.6307, 0.5403, 0.8033],
                [0.7781, 0.4966, 0.8888],
                [0.5570, 0.7127, 0.0339]])
        tensor([[0.6307, 0.5403, 0.8033],
                [0.7781, 0.4966, 0.8888],
                [0.5570, 0.7127, 0.0339]])

    """

    if th.is_complex(X):  # complex in complex
        return X.imag
    else:
        if cdim is None:  # real
            return th.zeros_like(X)
        else:  # complex in real
            d = X.ndim
            idx = [[1]] if keepcdim else [1]
            return X[tl.sl(d, axis=cdim, idx=idx)]


def abs(X, cdim=None, keepcdim=False):
    r"""obtain amplitude of a tensor

    Both complex and real representation are supported.

    .. math::
       {\rm abs}({\bf X}) = |x| = \sqrt{u^2 + v^2}, x\in {\bf X}

    where, :math:`u, v` are the real and imaginary part of x, respectively.

    Parameters
    ----------
    X : tensor
        input
    cdim : int or None
        If :attr:`X` is complex-valued, :attr:`cdim` is ignored. If :attr:`X` is real-valued and :attr:`cdim` is integer
        then :attr:`X` will be treated as complex-valued, in this case, :attr:`cdim` specifies the complex axis;
        otherwise (None), :attr:`X` will be treated as real-valued
    keepdims : bool, optional
        keep the complex dimension?

    Returns
    -------
    tensor
         the inputs's amplitude.

    Examples
    ---------

    ::

        th.manual_seed(2020)
        X = th.rand((2, 3, 3))

        print('---abs')
        print(abs(X, cdim=0))
        print(abs(X[0] + 1j * X[1]))

        # ---output
        ---abs
        tensor([[0.7968, 0.5504, 0.9957],
                [0.7868, 0.7011, 0.9326],
                [0.8113, 1.0793, 0.2535]])
        tensor([[0.7968, 0.5504, 0.9957],
                [0.7868, 0.7011, 0.9326],
                [0.8113, 1.0793, 0.2535]])
    """

    if th.is_complex(X):  # complex in complex
        return X.abs()
    else:
        if cdim is None:  # real
            return X.abs()
        else:  # complex in real
            d = X.ndim
            idxreal = [[0]] if keepcdim else [0]
            idximag = [[1]] if keepcdim else [1]

            return (X[tl.sl(d, axis=cdim, idx=idxreal)]**2 + X[tl.sl(d, axis=cdim, idx=idximag)]**2).sqrt()


def pow(X, cdim=None, keepcdim=False):
    r"""obtain power of a tensor

    Both complex and real representation are supported.

    .. math::
       {\rm pow}({\bf X}) = |x|^2 = u^2 + v^2, x\in {\bf X}

    where, :math:`u, v` are the real and imaginary part of x, respectively.

    Parameters
    ----------
    X : tensor
        input
    cdim : int or None
        If :attr:`X` is complex-valued, :attr:`cdim` is ignored. If :attr:`X` is real-valued and :attr:`cdim` is integer
        then :attr:`X` will be treated as complex-valued, in this case, :attr:`cdim` specifies the complex axis;
        otherwise (None), :attr:`X` will be treated as real-valued
    keepcdim : bool, optional
        keep the complex dimension?

    Returns
    -------
    tensor
         the inputs's power.

    Examples
    ---------

    ::

        th.manual_seed(2020)
        X = th.rand((2, 3, 3))

        print('---pow')
        print(pow(X, cdim=0))
        print(pow(X[0] + 1j * X[1]))

        # ---output
        ---pow
        tensor([[0.6349, 0.3030, 0.9914],
                [0.6190, 0.4915, 0.8697],
                [0.6583, 1.1649, 0.0643]])
        tensor([[0.6349, 0.3030, 0.9914],
                [0.6190, 0.4915, 0.8697],
                [0.6583, 1.1649, 0.0643]])
    """

    if th.is_complex(X):  # complex in complex
        return (X.conj() * X).real
    else:
        if cdim is None:  # real
            return X**2
        else:  # complex in real
            d = X.ndim
            idxreal = [[0]] if keepcdim else [0]
            idximag = [[1]] if keepcdim else [1]

            return X[tl.sl(d, axis=cdim, idx=idxreal)]**2 + X[tl.sl(d, axis=cdim, idx=idximag)]**2


if __name__ == '__main__':

    print(prevpow2(-5), nextpow2(-5))
    print(prevpow2(5), nextpow2(5))
    print(prevpow2(0.3), nextpow2(0.3))
    print(prevpow2(7.3), nextpow2(7.3))
    print(prevpow2(-3.5), nextpow2(-3.5))

    th.manual_seed(2020)
    Ar = th.randn((3, 3, 2))
    Br = th.randn((3, 3, 2))

    Ac = th.view_as_complex(Ar)
    Bc = th.view_as_complex(Br)

    Mr = th.view_as_real(Ac * Bc)
    print(th.sum(Mr - ematmul(Ar, Br)))
    print(th.sum(Ac * Bc - ematmul(Ac, Bc)))

    print(th.sum(th.matmul(Ac, Bc) - matmul(Ac, Bc)))
    Mr = matmul(Ar, Br)
    Mc = th.view_as_real(th.matmul(Ac, Bc))
    print(th.sum(Mr - Mc))

    th.manual_seed(2020)
    Xr = th.randint(0, 30, (3, 3, 2))
    Xc = Xr[..., 0] + 1j * Xr[..., 1]
    Yr = c2r(Xc, cdim=0)
    Yc = r2c(Yr, cdim=0)
    print(Xr, Xr.shape, 'Xr')
    print(Xc, Xc.shape, 'Xc')
    print(Yr, Yr.shape, 'Yr')
    print(Yc, Yc.shape, 'Yc')

    th.manual_seed(2020)
    X = th.rand((2, 3, 3))

    print('---conj')
    print(conj(X, cdim=0))
    print(conj(X[0] + 1j * X[1]))

    print('---real')
    print(real(X, cdim=0))
    print(real(X[0] + 1j * X[1]))

    print('---imag')
    print(imag(X, cdim=0))
    print(imag(X[0] + 1j * X[1]))

    print('---abs')
    print(abs(X, cdim=0))
    print(abs(X[0] + 1j * X[1]))

    print('---pow')
    print(pow(X, cdim=0))
    print(pow(X[0] + 1j * X[1]))
