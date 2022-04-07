#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2019-02-18 10:14:12
# @Author  : Yan Liu & Zhi Liu (zhiliu.mind@gmail.com)
# @Link    : http://iridescent.ink
# @Version : $1.0$

import torch as th
import torchlib as tl
import matplotlib.pyplot as plt

device = 'cpu'
device = 'cuda:0'
axis = (0, 1)

X = tl.imread('../../../data/images/Einstein256.png')

X = X
X = X * 1e5

X = X + 1j * X

print(X.shape)

E = th.exp(1j * th.randn(X.shape))

Y = X.clone()
for a in axis:
    Y = th.fft.fft(Y, dim=a)
Y = Y * E
for a in axis:
    Y = th.fft.ifft(Y, dim=a)

loss_mse_fn = th.nn.MSELoss(reduction='mean')
loss_mae_fn = th.nn.L1Loss(reduction='mean')

loss_fda_mse_fn = tl.FourierDomainAmplitudeLoss(mode='mse', axis=axis, norm=False, reduction='mean')
loss_fda_mae_fn = tl.FourierDomainAmplitudeLoss(mode='mae', axis=axis, norm=False, reduction='mean')

loss_fda_mse_norm_fn = tl.FourierDomainAmplitudeLoss(mode='mse', axis=axis, norm=True, reduction='mean')
loss_fda_mae_norm_fn = tl.FourierDomainAmplitudeLoss(mode='mae', axis=axis, norm=True, reduction='mean')

print(loss_mse_fn(Y.abs(), X.abs()), "MSE")
print(loss_mae_fn(Y.abs(), X.abs()), "MAE")
print(loss_fda_mse_fn(Y, X), "FDA MSE")
print(loss_fda_mae_fn(Y, X), "FDA MAE")
print(loss_fda_mse_fn(Y / Y.abs().max(), X / X.abs().max()))
print(loss_fda_mae_fn(Y / Y.abs().max(), X / X.abs().max()))

maxv = max(Y.abs().max(), X.abs().max())

print(loss_fda_mse_fn(Y / maxv, X / maxv))
print(loss_fda_mae_fn(Y / maxv, X / maxv))

print(loss_fda_mse_norm_fn(Y, X), "FDA MSE norm")
print(loss_fda_mae_norm_fn(Y, X), "FDA MAE norm")

plt.figure()
plt.subplot(121)
plt.imshow(X.abs())
plt.subplot(122)
plt.imshow(Y.abs())
plt.show()
