#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 220-07-24 18:29:48
# @Author  : Zhi Liu (zhiliu.mind@gmail.com)
# @Link    : http://iridescent.ink
# @Version : $1.0$

import math
import numpy as np
import torch as th


class DoubleGaussianKernelLR(th.optim.lr_scheduler._LRScheduler):
    r"""Set the learning rate of each parameter group using a double gaussian kernel
    schedule, where :math:`\eta_{max}` is set to the initial lr and
    :math:`T_{cur}` is the number of epochs since the last restart in SGDR:

    .. math::
        \eta_{t}=\left\{\begin{array}{l}
        \left(\eta_{\max }-\eta_{\text {start }}\right) \exp \left(\frac{-\left(t-t_{\eta_{\
        max }}\right)^{2}}{2 \sigma_{1}^{2}}\right)+\eta_{\text {start }}, \text { if } t<t
        _{\eta_{\max}} \\
        \left(\eta_{\max }-\eta_{\text {stop }}\right) \exp \left(\frac{-\left(t-t_{\eta_{\
        max }}\right)^{2}}{2 \sigma_{2}^{2}}\right)+\eta_{\text {stop }}, \text { otherwi
        se }
        \end{array}\right.


    When last_epoch=-1, sets initial lr as lr. Notice that because the schedule
    is defined recursively, the learning rate can be simultaneously modified
    outside this scheduler by other operators.


    Parameters
    ----------
    optimizer : {Optimizer}
        Wrapped optimizer.
    t_eta_max : {int}
        Iterations when the learning rate reach to the maximum value :math:`\eta_{\max}`.
    sigma1 : {int}
        Controls the shape of warming up phase.
    sigma2 : {int}
        Controls the shape of annealing phase.
    eta_start : {float}
        Starting learning rate. Default: 0.
    eta_stop : {float}
        Stopping learning rate. Default: 0.
    last_epoch : {int}
        The index of last epoch. Default: -1.

    .. note::
        The maximum learning rate are the base learning rate setted in Optimizer.


    """

    def __init__(self, optimizer, t_eta_max, sigma1, sigma2, eta_start=1e-6, eta_stop=1e-5, last_epoch=-1):
        self.sigma1 = sigma1
        self.sigma2 = sigma2
        self.t_eta_max = t_eta_max
        self.eta_start = eta_start
        self.eta_stop = eta_stop
        # self.eta_max = eta_max  # base lr
        super(DoubleGaussianKernelLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if not self._get_lr_called_within_step:
            warnings.warn("To get the last learning rate computed by the scheduler, "
                          "please use `get_last_lr()`.", UserWarning)
        if self.last_epoch == 0:
            return [self.eta_start for base_lr in self.base_lrs]
        elif self.last_epoch < self.t_eta_max:
            return [group['lr'] + (base_lr - self.eta_start) * (math.exp(-(self.last_epoch + 1 - self.t_eta_max)**2 / 2. / (self.sigma1**2)) - math.exp(-(self.last_epoch - self.t_eta_max)**2 / 2. / (self.sigma1**2)))
                    for base_lr, group in
                    zip(self.base_lrs, self.optimizer.param_groups)]
        return [group['lr'] + (base_lr - self.eta_stop) * (math.exp(-(self.last_epoch + 1 - self.t_eta_max)**2 / 2. / (self.sigma2**2)) - math.exp(-(self.last_epoch - self.t_eta_max)**2 / 2. / (self.sigma2**2)))
                for base_lr, group in
                zip(self.base_lrs, self.optimizer.param_groups)]

    def _get_closed_form_lr(self):
        if self.last_epoch < self.t_eta_max:
            return [(base_lr - self.eta_start) * math.exp(-(self.last_epoch - self.t_eta_max)**2 / 2. / (self.sigma1**2)) + self.eta_start
                    for base_lr in self.base_lrs]
        else:
            return [(base_lr - self.eta_stop) * math.exp(-(self.last_epoch - self.t_eta_max)**2 / 2. / (self.sigma2**2)) + self.eta_stop
                    for base_lr in self.base_lrs]



if __name__ == '__main__':

    import torch as th
    import torchlib as tl
    import matplotlib.pyplot as plt

    lr = 1e-1
    lr = 1e-2
    lr = 1e2

    num_epochs = 401
    batch_size = 1
    num_batch = 401

    params = {th.nn.parameter.Parameter(th.zeros(128), requires_grad=True),
              th.nn.parameter.Parameter(th.zeros(128), requires_grad=True),
              }

    optimizer = th.optim.Adam(params, lr=lr)
    # optimizer = th.optim.SGD(params, lr=lr, momentum=0.9)
    # scheduler = tl.optim.lr_scheduler.DoubleGaussianKernelLR(optimizer, t_eta_max=80, sigma1=15, sigma2=200, eta_start=1e-5, eta_stop=1e-5, last_epoch=-1)
    scheduler = tl.optim.lr_scheduler.DoubleGaussianKernelLR(optimizer, t_eta_max=50, sigma1=15, sigma2=100, eta_start=1e0, eta_stop=1e0, last_epoch=-1)

    print(optimizer)

    lrs = []
    for n in range(num_epochs):
        for b in range(num_batch):

            optimizer.step()

            # lrs.append(optimizer.param_groups[0]['lr'])

        scheduler.step()
        lrs.append(optimizer.param_groups[0]['lr'])

    plt.figure()
    plt.plot(lrs)
    plt.grid()
    plt.show()
