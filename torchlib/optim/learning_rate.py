#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2019-07-24 18:29:48
# @Author  : Zhi Liu (zhiliu.mind@gmail.com)
# @Link    : http://iridescent.ink
# @Version : $1.0$

import matplotlib.pyplot as plt


class FindLearningRate():

    def __init__(self, plotdir=None):
        self.plotdir = plotdir
        self.lrs = {'train': [], 'valid': [], 'test': []}

    def add(self, key, value):
        self.lrs[key].append(value)

    def get(self, idx=None):
        if idx is None:
            return self.lrs
        else:
            return self.lrs[idx]

    def plot(self):

        plt.figure()
        for lr in self.lrs:
            plt.plot(lr)
        plt.legend(legend)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.grid()

        if self.plotdir is None:
            plt.show()
        else:
            plt.savefig(self.plotdir + '/loss_lr.png')
            plt.close()


if __name__ == '__main__':

    fdlr = FindLearningRate(plotdir='./')

    for n in range(100):
        fdlr.add([n, n - 1])

    fdlr.plot()
