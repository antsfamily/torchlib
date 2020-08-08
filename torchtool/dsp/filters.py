#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2019-07-24 18:29:48
# @Author  : Zhi Liu (zhiliu.mind@gmail.com)
# @Link    : http://iridescent.ink
# @Version : $1.0$

import torch as th
from torchtool.utils.const import EPS
from torchtool.dsp.kernels import *


def sobelfilter(A, dtype=th.float32, gmod='absadd', full=False):
    r"""sobel filtering

    filter A with sobel operator

    .. math::
       G_v=\left[\begin{array}{ccc}{-1} & {-2} & {-1} \\ {0} & {0} & {0} \\ {+1} & {+2} & {+1}\end{array}\right] * A

    .. math::
       G_h=\left[\begin{array}{ccc}{-1} & {0} & {+1} \\ {-2} & {0} & {+2} \\ {-1} & {0} & {+1}\end{array}\right] * A


    Parameters
    ----------
    A : {2d or 3d array}
        image to be filtered :math:`H×W×C`.

    gmod: {str}

        - 'add' --> :math:`G=G_h+G_v`,
        - 'absadd' --> :math:`G=|G_h|+|G_v|`,
        - 'squaddsqrt' --> :math:`G=\sqrt{G_h^2 + G_v^2}`,
        - `hvfilt` --> apply sobel filter on hrizontal and vertical
        - `vhfilt` --> apply sobel filter on vertical and hrizontal
        - default `absadd`

    full: {bool}
        If True, then return Gh+Gv, Gh, Gv else return Gh+Gv

    Returns
    -------
    G : {2d or 3d array}
        2d-gradient

    Gh : {2d or 3d array}
        gradient in hrizontal

    Gv : {2d or 3d array}
        gradient in vertical

    """

    N, C, H, W = list(A.size())

    if A.is_cuda:
        WINh = HORIZONTAL_SOBEL_3x3.cuda()
        WINv = VERTICAL_SOBEL_3x3.cuda()
    else:
        WINh = HORIZONTAL_SOBEL_3x3
        WINv = VERTICAL_SOBEL_3x3

    WINh = WINh.repeat(C, 1).reshape(C, 1, 3, 3).type(dtype).requires_grad_(requires_grad=False)
    WINv = WINv.repeat(C, 1).reshape(C, 1, 3, 3).type(dtype).requires_grad_(requires_grad=False)

    Gh = th.nn.functional.conv2d(A, WINh, bias=None, stride=1, padding=1, dilation=1, groups=C)
    Gv = th.nn.functional.conv2d(A, WINv, bias=None, stride=1, padding=1, dilation=1, groups=C)

    if gmod is 'add':
        G = Gh + Gv
    if gmod is 'absadd':
        G = th.abs(Gh) + th.abs(Gv)
    if gmod is 'squaddsqrt':
        G = th.sqrt(Gh * Gh + Gv * Gv)
    if gmod is 'hvfilt':
        G = th.nn.functional.conv2d(Gh, WINv, bias=None, stride=1, padding=1, dilation=1, groups=C)
    if gmod is 'vhfilt':
        G = th.nn.functional.conv2d(Gv, WINh, bias=None, stride=1, padding=1, dilation=1, groups=C)

    if full:
        return G, Gh, Gv
    else:
        return G


if __name__ == '__main__':

    import matplotlib.pyplot as plt
    import matplotlib.image as img

    A = img.imread('../../data/images/LenaRGB.tif')
    print(A.shape)

    dtype = th.float32

    A = th.from_numpy(A.reshape([1] + list(A.shape)).transpose(0, 3, 1, 2))

    A = A.type(dtype)

    # A = A.cuda()

    _, Gh, Gv = sobelfilter(A, full=True)

    G = th.sqrt(Gh * Gh + Gv * Gv)
    Gh = th.abs(Gh)
    Gv = th.abs(Gv)
    G = Gh + Gv
    print(G.shape, G.min(), G.max())
    print(Gh.shape, Gh.min(), Gh.max())
    print(Gv.shape, Gv.min(), Gv.max())

    print(A.shape, G.shape, Gh.shape, Gv.shape)

    G_u8 = (255. * G / G.max()).type(th.uint8)
    Gh_u8 = (255. * Gh / Gh.max()).type(th.uint8)
    Gv_u8 = (255. * Gv / Gv.max()).type(th.uint8)

    G_mu_u8 = (255. * (G - G.mean()) / G.max()).type(th.uint8)
    Gh_mu_u8 = (255. * (Gh - Gh.mean()) / Gh.max()).type(th.uint8)
    Gv_mu_u8 = (255. * (Gv - Gv.mean()) / Gv.max()).type(th.uint8)

    A = A.permute(2, 3, 1, 0).squeeze().type(th.uint8)
    G = G.permute(2, 3, 1, 0).squeeze().type(th.uint8)
    Gh = Gh.permute(2, 3, 1, 0).squeeze().type(th.uint8)
    Gv = Gv.permute(2, 3, 1, 0).squeeze().type(th.uint8)

    G_u8 = G_u8.permute(2, 3, 1, 0).squeeze().type(th.uint8)
    Gh_u8 = Gh_u8.permute(2, 3, 1, 0).squeeze().type(th.uint8)
    Gv_u8 = Gv_u8.permute(2, 3, 1, 0).squeeze().type(th.uint8)

    G_mu_u8 = G_mu_u8.permute(2, 3, 1, 0).squeeze().type(th.uint8)
    Gh_mu_u8 = Gh_mu_u8.permute(2, 3, 1, 0).squeeze().type(th.uint8)
    Gv_mu_u8 = Gv_mu_u8.permute(2, 3, 1, 0).squeeze().type(th.uint8)

    plt.figure()
    plt.subplot(221)
    plt.imshow(A.data.cpu())
    plt.title('original')
    plt.subplot(222)
    plt.imshow(Gh.data.cpu())
    plt.title(r'$|G_h|$')
    plt.subplot(223)
    plt.imshow(Gv.data.cpu())
    plt.title(r'$|G_v|$')
    plt.subplot(224)
    plt.imshow(G.data.cpu())
    plt.title(r'$|G_h|+|G_v|$')
    plt.tight_layout()
    plt.show()

    plt.figure()
    plt.subplot(221)
    plt.imshow(A.data.cpu())
    plt.title('original')
    plt.subplot(222)
    plt.imshow(Gh_u8.data.cpu())
    plt.title(r'$|G_h|$, u8')
    plt.subplot(223)
    plt.imshow(Gv_u8.data.cpu())
    plt.title(r'$|G_v|$, u8')
    plt.subplot(224)
    plt.imshow(G_u8.data.cpu())
    plt.title(r'$|G_h|+|G_v|$, u8')
    plt.tight_layout()
    plt.show()

    plt.figure()
    plt.subplot(221)
    plt.imshow(A.data.cpu())
    plt.title('original')
    plt.subplot(222)
    plt.imshow(Gh_mu_u8.data.cpu())
    plt.title(r'$|G_h|-\mu_{G_h}$, u8')
    plt.subplot(223)
    plt.imshow(Gv_mu_u8.data.cpu())
    plt.title(r'$|G_v|-\mu_{G_v}$, u8')
    plt.subplot(224)
    plt.imshow(G_mu_u8.data.cpu())
    plt.title(r'$|G_h|+|G_v| - \mu_{|G_h|+|G_v|}$, u8')
    plt.tight_layout()
    plt.show()
