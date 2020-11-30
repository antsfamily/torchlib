#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2019-07-24 18:29:48
# @Author  : Zhi Liu (zhiliu.mind@gmail.com)
# @Link    : http://iridescent.ink
# @Version : $1.0$
# @Note    : https://github.com/Po-Hsun-Su/pytorch-ssim, https://github.com/jorge-pessoa/pytorch-msssim

import torch
import torch.nn.functional as F
from torch.autograd import Variable
from math import exp
from torchlib.utils.const import EPS


def gaussian(winsize, sigma):
    gauss = torch.Tensor([exp(-(x - winsize // 2)**2 / float(2 * sigma**2)) for x in range(winsize)])
    return gauss / gauss.sum()


def create_window(winsize, channel):
    _1D_window = gaussian(winsize, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, winsize, winsize).contiguous())
    return window


def _ssim(X, Y, window, winsize, channel, sizeavg=True, L=None, alpha=1, beta=1, gamma=1, full=False):
    r"""Structural similarity

    """

    if L is None:
        L = 255.0

    C1 = (0.01 * L)**2.0
    C2 = (0.03 * L)**2.0
    C3 = C2 / 2.0

    mu1 = F.conv2d(X, window, padding=winsize // 2, groups=channel)
    mu2 = F.conv2d(Y, window, padding=winsize // 2, groups=channel)

    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = torch.abs(F.conv2d(X * X, window, padding=winsize // 2, groups=channel) - mu1_sq)
    sigma2_sq = torch.abs(F.conv2d(Y * Y, window, padding=winsize // 2, groups=channel) - mu2_sq)
    sigma12 = torch.abs(F.conv2d(X * Y, window, padding=winsize // 2, groups=channel) - mu1_mu2)

    sigma1 = torch.sqrt(sigma1_sq)
    sigma2 = torch.sqrt(sigma2_sq)

    luminance = (2 * mu1_mu2 + C1) / (mu1_sq + mu2_sq + C1 + EPS)
    contrast = (2 * sigma1 * sigma2 + C2) / (sigma1_sq + sigma2_sq + C2 + EPS)
    structure = (sigma12 + C3) / (sigma1 * sigma2 + C3 + EPS)
    # print(sigma1.min(), sigma1.max())
    # print(sigma2_sq.min(), sigma2_sq.max(), "===")
    # print(sigma2.min(), sigma2.max(), "===")
    # print(luminance.min(), luminance.max())
    # print(contrast.min(), contrast.max())
    # print(structure.min(), structure.max())

    ssim_map = (luminance**alpha) * (contrast**beta) * (structure**gamma)

    if full:
        if sizeavg:
            ssim_map = ssim_map.mean()
            luminance = luminance.mean()
            contrast = contrast.mean()
            structure = structure.mean()
        else:
            ssim_map = ssim_map.mean(1).mean(1).mean(1)
            luminance = luminance.mean(1).mean(1).mean(1)
            contrast = contrast.mean(1).mean(1).mean(1)
            structure = structure.mean(1).mean(1).mean(1)
        return ssim_map, luminance, contrast, structure
    else:
        if sizeavg:
            ssim_map = ssim_map.mean()
        else:
            ssim_map = ssim_map.mean(1).mean(1).mean(1)
        return ssim_map


def ssim(X, Y, winsize=11, sizeavg=True, L=255, alpha=1, beta=1, gamma=1, full=False):
    r"""Structural similarity

    .. math::
       \begin{aligned} l(x, y) &=\frac{2 \mu_{x} \mu_{y}+c_{1}}{\mu_{x}^{2}+\mu_{y}^{2}+c_{1}} \\
       c(x, y) &=\frac{2 \sigma_{x} \sigma_{y}+c_{2}}{\sigma_{x}^{2}+\sigma_{y}^{2}+c_{2}} \\
       s(x, y) &=\frac{\sigma_{x y}+c_{3}}{\sigma_{x} \sigma_{y}+c_{3}} \end{aligned}

    where, :math:`c_1 = (k_1 L)^2, c_2 = (k_2 L)^2, c_3 = c_2 / 2`,
    :math:`L` is the dynamic range of the pixel-values (typically this is :math:`2 ^{\# \text { bits per pixel }}-1`.
    The structure similarity index is expressed as

    .. math::
       \operatorname{SSIM}(x, y)=\left[l(x, y)^{\alpha} \cdot c(x, y)^{\beta} \cdot s(x, y)^{\gamma}\right].

    When :math:`\alpha=\beta=\gamma=1`, SSIM is equal to

    .. math::
       \operatorname{SSIM}(x, y)=\frac{\left(2 \mu_{x} \mu_{y}+c_{1}\right)\left(2 \sigma_{x y}+
       c_{2}\right)}{\left(\mu_{x}^{2}+\mu_{y}^{2}+c_{1}\right)\left(\sigma_{x}^{2}+\sigma_{y}^{2}+c_{2}\right)}

    see http://wikipedia.moesalih.com/SSIM for more details.

    Parameters
    ----------
    X : {Torch Tensor}
        data 1
    Y : {Torch Tensor}
        data 2
    winsize : {int}
        window size
    sizeavg : {bool}, optional
        whether to average (the default is True, which average the result)
    L : {integer}, optional
        the dynamic range of the pixel-values (typically this is :math:`2 ^{\# \text { bits per pixel }}-1`. (the default is 255)
    alpha : {number}, optional
        luminance weight (the default is 1)
    beta : {number}, optional
        contrast weight (the default is 1)
    gamma : {number}, optional
        structure weight (the default is 1)
    full : {bool}, optional
        IF True, return SSIM, luminance, contrast and structure index (the default is False, which only return SSIM)

    Returns
    -------
    float
        IF ``full`` is True, returns SSIM, luminance, contrast and structure index, otherwise, only returns SSIM.
    """

    (_, channel, _, _) = X.size()
    window = create_window(winsize, channel)

    if X.is_cuda:
        window = window.cuda(X.get_device())
    window = window.type_as(X)

    return _ssim(X, Y, window, winsize, channel, sizeavg, L=L, full=full)


def msssim(X, Y, winsize=11, sizeavg=True, L=255, alpha=1, beta=1, gamma=1, normalize=False):
    device = X.device
    weights = torch.FloatTensor([0.0448, 0.2856, 0.3001, 0.2363, 0.1333]).to(device)
    levels = weights.size()[0]
    mssim = []
    mcontrast = []
    for _ in range(levels):
        sim, _, contrast, _ = ssim(X, Y, winsize=winsize, sizeavg=sizeavg, L=255, alpha=1, beta=1, gamma=1, full=True)
        mssim.append(sim)
        mcontrast.append(contrast)

        X = F.avg_pool2d(X, (2, 2))
        Y = F.avg_pool2d(Y, (2, 2))

    mssim = torch.stack(mssim)
    mcontrast = torch.stack(mcontrast)

    # Normalize (to avoid NaNs during training unstable models, not compliant with original definition)
    if normalize:
        mssim = (mssim + 1) / 2
        mcontrast = (mcontrast + 1) / 2

    pow1 = mcontrast ** weights
    pow2 = mssim ** weights
    # From Matlab implementation https://ece.uwaterloo.ca/~z70wang/research/iwssim/
    output = torch.prod(pow1[:-1] * pow2[-1])
    return output


if __name__ == '__main__':
    import torchlib as tl
    import cv2
    import numpy as np
    import matplotlib.pyplot as plt

    npImg1 = cv2.imread("../../data/images/einstein.png")
    npImg1 = cv2.imread("../../data/images/LenaRGB.tif")

    H, W, C = npImg1.shape
    print(npImg1.shape)

    L = 255.0
    L = 1.0

    img1 = torch.from_numpy(np.rollaxis(npImg1, 2)).float().unsqueeze(0) / (255.0 / L)
    print(img1.min(), img1.max())
    img2 = torch.rand(img1.size())
    img2 = img1.clone()
    # img2[:, :, 10:300, 10:300] = 1
    img2[:, :, 0:H, 0:W] = L / 2.0
    print(img2.min(), img2.max())

    if torch.cuda.is_available():
        img1 = img1.cuda()
        img2 = img2.cuda()

    img1 = Variable(img1, requires_grad=False)
    img2 = Variable(img2, requires_grad=True)

    # Functional: tl.ssim(img1, img2, window_size = 11, size_average = True)
    ssimo, luminanceo, contrasto, structureo = tl.ssim(img1, img2, L=L, full=True)
    ssimv, luminancev, contrastv, structurev = (ssimo.item(), luminanceo.item(), contrasto.item(), structureo.item())
    print("ssim, luminance, contrast, structure:", ssimv, luminancev, contrastv, structurev)

    msssimo = msssim(img1, img2, winsize=11, sizeavg=True, L=L, normalize=False)

    print("MSSIM: ", msssimo.item())

    img1 = img1.data.cpu()
    img2 = img2.data.cpu()
    print(img1.size())
    print(img2.size())
    img1 = img1.reshape(C, H, W)
    img1 = img1.permute(1, 2, 0)

    img2 = img2.reshape(C, H, W)
    img2 = img2.permute(1, 2, 0)

    plt.figure()
    plt.subplot(121)
    plt.imshow(img1)
    plt.subplot(122)
    plt.imshow(img2)
    plt.show()
