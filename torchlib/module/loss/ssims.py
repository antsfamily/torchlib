#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2019-07-24 18:29:48
# @Author  : Zhi Liu (zhiliu.mind@gmail.com)
# @Link    : http://iridescent.ink
# @Version : $1.0$
# @Note    : https://github.com/Po-Hsun-Su/pytorch-ssim, https://github.com/jorge-pessoa/pytorch-msssim

import torch
from torch.autograd import Variable
from torchlib import create_window, _ssim, msssim


class SSIMLoss(torch.nn.Module):
    r"""Structural similarity Object

    [description]
    """

    def __init__(self, winsize=11, sizeavg=True, L=None, alpha=1, beta=1, gamma=1, full=False):
        super(SSIMLoss, self).__init__()
        self.winsize = winsize
        self.sizeavg = sizeavg
        self.channel = 1
        self.window = create_window(winsize, self.channel)
        self.L = L
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.full = full

    def forward(self, X, Y):
        (_, channel, _, _) = X.size()

        if channel == self.channel and self.window.data.type() == X.data.type():
            window = self.window
        else:
            window = create_window(self.winsize, channel)

            if X.is_cuda:
                window = window.cuda(X.get_device())
            window = window.type_as(X)

            self.window = window
            self.channel = channel

        return 1.0 - _ssim(X, Y, window, self.winsize, channel, self.sizeavg, L=self.L,
                           alpha=self.alpha, beta=self.beta, gamma=self.gamma, full=self.full)


class MSSSIMLoss(torch.nn.Module):
    def __init__(self, winsize=11, sizeavg=True, L=None, alpha=1, beta=1, gamma=1):
        super(MSSSIMLoss, self).__init__()
        self.winsize = winsize
        self.sizeavg = sizeavg
        self.channel = 1
        self.L = L
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def forward(self, X, Y):
        # TODO: store window between calls if possible
        return 1.0 - msssim(X, Y, winsize=self.winsize, sizeavg=self.sizeavg, L=self.L,
                            alpha=self.alpha, beta=self.beta, gamma=self.gamma)


class StructureLoss(torch.nn.Module):
    def __init__(self, winsize=11, sizeavg=True, channel=3):
        super(StructureLoss, self).__init__()
        self.winsize = winsize
        self.sizeavg = sizeavg
        self.channel = channel

    def forward(self, X, Y):
        # TODO: store window between calls if possible
        return 1.0 - msssim(X, Y, winsize=self.winsize, sizeavg=self.sizeavg)


if __name__ == '__main__':
    import torchlib as tl
    import torch
    from torch import optim
    import psar as ps
    import numpy as np

    L = 255.0
    # L = 1.0

    img1 = torch.zeros(1, 3, 512, 512)
    img1[:, :, 10:500, 10:500] = L / 2.0
    img2 = torch.rand(img1.size())

    if torch.cuda.is_available():
        img1 = img1.cuda()
        img2 = img2.cuda()

    img1 = Variable(img1, requires_grad=False)
    img2 = Variable(img2, requires_grad=True)

    # Functional: tl.ssim(img1, img2, window_size = 11, size_average = True)
    ssim_value = tl.ssim(img1, img2).item()
    print("Initial ssim:", ssim_value)

    # Module: tl.SSIM(window_size = 11, size_average = True)
    ssim_loss = tl.SSIMLoss(L=1, alpha=1, beta=1, gamma=1, full=False)
    # ssim_loss = tl.MSSSIMLoss(L=1, alpha=1, beta=1, gamma=1)

    optimizer = optim.Adam([img2], lr=0.01)

    while ssim_value < 0.999:
        optimizer.zero_grad()
        ssim_out = ssim_loss(img1, img2)
        ssim_value = ssim_out.item()
        print(ssim_value)
        ssim_out.backward()
        optimizer.step()
