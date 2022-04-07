#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2019-07-24 18:29:48
# @Author  : Zhi Liu (zhiliu.mind@gmail.com)
# @Link    : http://iridescent.ink
# @Version : $1.0$

import torch as th
import torchlib as tl


class FourierDomainLoss(th.nn.Module):
    r"""Fourier Domain Loss

    compute loss in fourier domain.

    """

    def __init__(self, cdim=None, ftdim=(-2, -1), iftdim=(-2, -1), ftn=None, iftn=None, ftnorm=None, iftnorm=None, err='mse', reduction='mean'):
        super(FourierDomainLoss, self).__init__()
        self.cdim = cdim
        self.ftdim = [ftdim] if (type(ftdim) is not list and type(ftdim) is not tuple) else ftdim
        self.iftdim = [iftdim] if (type(iftdim) is not list and type(iftdim) is not tuple) else iftdim
        self.ftn = [ftn] if (type(ftn) is not list and type(ftn) is not tuple) else ftn
        self.iftn = [iftn] if (type(iftn) is not list and type(iftn) is not tuple) else iftn
        self.ftnorm = [ftnorm] if (type(ftnorm) is not list and type(ftnorm) is not tuple) else ftnorm
        self.iftnorm = [iftnorm] if (type(iftnorm) is not list and type(iftnorm) is not tuple) else iftnorm
        self.reduction = reduction

        if err in ['mse', 'MSE', 'Mse']:
            self.err = th.nn.MSELoss(reduction=self.reduction)
        if err in ['mae', 'MAE', 'Mae']:
            self.err = th.nn.L1Loss(reduction=self.reduction)
        if str(type(err)).find('torch.nn.modules.loss') > 0:
            self.err = err

    def forward(self, P, G):
        d = P.dim()
        if self.cdim is not None:
            P = P[tl.sl(d, self.cdim, [[0]])] + 1j * P[tl.sl(d, self.cdim, [[1]])]
            G = G[tl.sl(d, self.cdim, [[0]])] + 1j * G[tl.sl(d, self.cdim, [[1]])]

        for dim, n, norm in zip(self.ftdim, self.ftn, self.ftnorm):
            if dim is None:
                pass
            else:
                P = th.fft.fft(P, n=n, dim=dim, norm=norm)
                G = th.fft.fft(G, n=n, dim=dim, norm=norm)

        for dim, n, norm in zip(self.iftdim, self.iftn, self.iftnorm):
            if dim is None:
                pass
            else:
                P = th.fft.ifft(P, n=n, dim=dim, norm=norm)
                G = th.fft.ifft(G, n=n, dim=dim, norm=norm)

        P = th.view_as_real(P)
        G = th.view_as_real(G)

        return self.err(P, G)


class FourierDomainAmplitudeLoss(th.nn.Module):
    r"""Fourier Domain Amplitude Loss

    compute amplitude loss in fourier domain.

    """

    def __init__(self, cdim=None, ftdim=(-2, -1), iftdim=(-2, -1), ftn=None, iftn=None, ftnorm=None, iftnorm=None, err='mse', reduction='mean'):
        super(FourierDomainAmplitudeLoss, self).__init__()
        self.cdim = cdim
        self.ftdim = [ftdim] if (type(ftdim) is not list and type(ftdim) is not tuple) else ftdim
        self.iftdim = [iftdim] if (type(iftdim) is not list and type(iftdim) is not tuple) else iftdim
        self.ftn = [ftn] if (type(ftn) is not list and type(ftn) is not tuple) else ftn
        self.iftn = [iftn] if (type(iftn) is not list and type(iftn) is not tuple) else iftn
        self.ftnorm = [ftnorm] if (type(ftnorm) is not list and type(ftnorm) is not tuple) else ftnorm
        self.iftnorm = [iftnorm] if (type(iftnorm) is not list and type(iftnorm) is not tuple) else iftnorm
        self.reduction = reduction

        if err in ['mse', 'MSE', 'Mse']:
            self.err = th.nn.MSELoss(reduction=self.reduction)
        if err in ['mae', 'MAE', 'Mae']:
            self.err = th.nn.L1Loss(reduction=self.reduction)
        if str(type(err)).find('th.nn.Loss'):
            self.err = err

    def forward(self, P, G):
        d = P.dim()
        if self.cdim is not None:
            P = P[tl.sl(d, self.cdim, [[0]])] + 1j * P[tl.sl(d, self.cdim, [[1]])]
            G = G[tl.sl(d, self.cdim, [[0]])] + 1j * G[tl.sl(d, self.cdim, [[1]])]

        for dim, n, norm in zip(self.ftdim, self.ftn, self.ftnorm):
            if dim is None:
                pass
            else:
                P = th.fft.fft(P, n=n, dim=dim, norm=norm)
                G = th.fft.fft(G, n=n, dim=dim, norm=norm)

        for dim, n, norm in zip(self.iftdim, self.iftn, self.iftnorm):
            if dim is None:
                pass
            else:
                P = th.fft.ifft(P, n=n, dim=dim, norm=norm)
                G = th.fft.ifft(G, n=n, dim=dim, norm=norm)

        P, G = P.abs(), G.abs()

        return self.err(P, G)


class FourierDomainPhaseLoss(th.nn.Module):
    r"""Fourier Domain Phase Loss

    compute phase loss in fourier domain.

    """

    def __init__(self, cdim=None, ftdim=(-2, -1), iftdim=(-2, -1), ftn=None, iftn=None, ftnorm=None, iftnorm=None, err='mse', reduction='mean'):
        super(FourierDomainPhaseLoss, self).__init__()
        self.cdim = cdim
        self.ftdim = [ftdim] if (type(ftdim) is not list and type(ftdim) is not tuple) else ftdim
        self.iftdim = [iftdim] if (type(iftdim) is not list and type(iftdim) is not tuple) else iftdim
        self.ftn = [ftn] if (type(ftn) is not list and type(ftn) is not tuple) else ftn
        self.iftn = [iftn] if (type(iftn) is not list and type(iftn) is not tuple) else iftn
        self.ftnorm = [ftnorm] if (type(ftnorm) is not list and type(ftnorm) is not tuple) else ftnorm
        self.iftnorm = [iftnorm] if (type(iftnorm) is not list and type(iftnorm) is not tuple) else iftnorm
        self.reduction = reduction

        if err in ['mse', 'MSE', 'Mse']:
            self.err = th.nn.MSELoss(reduction=self.reduction)
        if err in ['mae', 'MAE', 'Mae']:
            self.err = th.nn.L1Loss(reduction=self.reduction)
        if str(type(err)).find('th.nn.Loss'):
            self.err = err

    def forward(self, P, G):
        d = P.dim()
        if self.cdim is not None:
            P = P[tl.sl(d, self.cdim, [[0]])] + 1j * P[tl.sl(d, self.cdim, [[1]])]
            G = G[tl.sl(d, self.cdim, [[0]])] + 1j * G[tl.sl(d, self.cdim, [[1]])]

        for dim, n, norm in zip(self.ftdim, self.ftn, self.ftnorm):
            if dim is None:
                pass
            else:
                P = th.fft.fft(P, n=n, dim=dim, norm=norm)
                G = th.fft.fft(G, n=n, dim=dim, norm=norm)

        for dim, n, norm in zip(self.iftdim, self.iftn, self.iftnorm):
            if dim is None:
                pass
            else:
                P = th.fft.ifft(P, n=n, dim=dim, norm=norm)
                G = th.fft.ifft(G, n=n, dim=dim, norm=norm)

        P, G = P.angle(), G.angle()

        return self.err(P, G)


class FourierDomainNormLoss(th.nn.Module):
    r"""FourierDomainNormLoss

    .. math::
        C = \frac{{\rm E}(|I|^2)}{[E(|I|)]^2}

    see Fast Fourier domain optimization using hybrid

    """

    def __init__(self, reduction='mean', p=1.5):
        super(FourierDomainNormLoss, self).__init__()
        self.reduction = reduction
        self.p = p

    def forward(self, X, w=None):
        r"""[summary]

        [description]

        Parameters
        ----------
        X : {[type]}
            After fft in azimuth
        w : {[type]}, optional
            [description] (the default is None, which [default_description])

        Returns
        -------
        [type]
            [description]
        """

        if th.is_complex(X):
            X = X.abs()
        elif X.shape[-1] == 2:
            X = th.view_as_complex(X)
            X = X.abs()

        if w is None:
            wshape = [1] * (X.dim())
            wshape[-2] = X.size(-2)
            w = th.ones(wshape, device=X.device, dtype=X.dtype)
        fv = th.sum((th.sum(w * X, dim=-2)).pow(self.p), dim=-1)

        if self.reduction == 'mean':
            C = th.mean(fv)
        if self.reduction == 'sum':
            C = th.sum(fv)
        return C


if __name__ == '__main__':

    th.manual_seed(2020)
    xr = th.randn(10, 2, 4, 4) * 10000
    yr = th.randn(10, 2, 4, 4) * 10000
    xc = xr[:, [0], ...] + 1j * xr[:, [1], ...]
    yc = yr[:, [0], ...] + 1j * yr[:, [1], ...]

    flossr = FourierDomainLoss(cdim=1, ftdim=(-2, -1), iftdim=None, ftn=None, iftn=None, ftnorm='forward', iftnorm=None, err='mse', reduction='mean')
    flossc = FourierDomainLoss(cdim=None, ftdim=(-2, -1), iftdim=None, ftn=None, iftn=None, ftnorm='forward', iftnorm=None, err='mse', reduction='mean')
    print(flossr(xr, yr))
    print(flossc(xc, yc))

