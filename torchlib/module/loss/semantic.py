#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2019-07-24 18:29:48
# @Author  : Zhi Liu (zhiliu.mind@gmail.com)
# @Link    : http://iridescent.ink
# @Version : $1.0$
# @Note    :

import torch as th
from torchlib.dsp.kernels import VERTICAL_SOBEL_3x3, HORIZONTAL_SOBEL_3x3


class EdgeLoss(th.nn.Module):
    r"""Semantic Edge Loss Object

    Semantic Edge Loss
    """

    def __init__(self, window='normsobel', Ci=1, dtype=th.float32):
        super(EdgeLoss, self).__init__()

        self.Ci = Ci

        if window == 'sobel':
            WINh = HORIZONTAL_SOBEL_3x3
            WINv = VERTICAL_SOBEL_3x3
            self.WINh = WINh.repeat(Ci, 1).reshape(Ci, 1, 3, 3).type(dtype).requires_grad_(requires_grad=True)
            self.WINv = WINv.repeat(Ci, 1).reshape(Ci, 1, 3, 3).type(dtype).requires_grad_(requires_grad=True)
        if window == 'normsobel':
            WINh = HORIZONTAL_SOBEL_3x3 / 9.0
            WINv = VERTICAL_SOBEL_3x3 / 9.0
            self.WINh = WINh.repeat(Ci, 1).reshape(Ci, 1, 3, 3).type(dtype).requires_grad_(requires_grad=True)
            self.WINv = WINv.repeat(Ci, 1).reshape(Ci, 1, 3, 3).type(dtype).requires_grad_(requires_grad=True)

        if th.cuda.is_available():
            self.WINh = self.WINh.cuda()
            self.WINv = self.WINv.cuda()

    def forward(self, X, Y):

        Gh = th.nn.functional.conv2d(X, self.WINh, bias=None, stride=1, padding=1, dilation=1, groups=self.Ci)
        Gv = th.nn.functional.conv2d(X, self.WINv, bias=None, stride=1, padding=1, dilation=1, groups=self.Ci)

        GX = th.abs(Gh) + th.abs(Gv)

        Gh = th.nn.functional.conv2d(Y, self.WINh, bias=None, stride=1, padding=1, dilation=1, groups=self.Ci)
        Gv = th.nn.functional.conv2d(Y, self.WINv, bias=None, stride=1, padding=1, dilation=1, groups=self.Ci)

        GY = th.abs(Gh) + th.abs(Gv)

        return th.nn.L1Loss()(GX, GY)


class EdgeAwareLoss(th.nn.Module):
    def __init__(self):
        super(EdgeAwareLoss, self).__init__()
        self.device = th.device("cuda:0" if th.cuda.is_available() else "cpu")
        self.cudas = list(range(th.cuda.device_count()))
        self.features = EdgeFeatureExtractor()
        self.predictor = th.nn.Sequential()
        self.predictor.add_module('fc', th.nn.Conv2d(2, 1, 1, 1, 0, bias=False))
        self.predictor.add_module('sigmoid', th.nn.Sigmoid())
        self.features.to(self.device)
        self.predictor.to(self.device)
        self.optimizer = th.optim.Adam(self.parameters(), lr=LEARNING_RATE)
        self.edge_detector = EdgeDetector()
        self.soft_dice = SoftDiceLoss()
        self.lambdas = [float(1), float(0.5), float(0.25)]
        self.loss = None
        self.counter = int(0)
        self.best_loss = float(100500)
        self.current_loss = float(0)

    def evaluate(self, actual, desire):
        actual_features = th.nn.parallel.data_parallel(module=self.features, inputs=actual, device_ids=self.cudas)
        desire_features = th.nn.parallel.data_parallel(module=self.features, inputs=desire, device_ids=self.cudas)
        eloss = 0.0

        for i in range(len(desire_features)):
            eloss += F.l1_loss(actual_features[i], desire_features[i]) * self.lambdas[i]

        return desire_features, eloss

    def meta_optimize(self, lossD, length):
        self.current_loss += float(lossD.item()) / length

        if self.counter > ITERATION_LIMIT:
            self.current_loss = self.current_loss / float(ITERATION_LIMIT)
            if self.current_loss < self.best_loss:
                self.best_loss = self.current_loss
                print('! best_loss !', self.best_loss)
            else:
                for param_group in self.optimizer.param_groups:
                    lr = param_group['lr']
                    if lr >= LR_THRESHOLD:
                        param_group['lr'] = lr * 0.2
                        print('! Decrease LearningRate in Perceptual !', lr)
            self.counter = int(0)
            self.current_loss = float(0)

        self.counter += int(1)

    def fit(self, actual, desire):
        self.features.train()
        self.predictor.train()
        self.optimizer.zero_grad()
        desire_features, _ = self.evaluate(actual, desire)
        fake = th.nn.parallel.data_parallel(module=self.predictor, inputs=desire_features[-1].detach(), device_ids=self.cudas)
        real = self.edge_detector(desire)
        loss = F.binary_cross_entropy(fake.view(-1), real.view(-1))
        loss.backward(retain_graph=True)
        self.optimizer.step()
        self.meta_optimize(loss, float(actual.size(0)))

    def forward(self, actual, desire):
        self.predictor.eval()
        self.features.eval()
        _, eloss = self.evaluate(actual, desire)
        self.loss = eloss + F.binary_cross_entropy(actual.view(-1), desire.view(-1))
        self.fit(actual, desire)
        return self.loss

    def backward(self, retain_variables=True):
        return self.loss.backward(retain_variables=retain_variables)

