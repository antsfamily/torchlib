#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2019-07-24 18:29:48
# @Author  : Zhi Liu (zhiliu.mind@gmail.com)
# @Link    : http://iridescent.ink
# @Version : $1.0$

import math
import matplotlib.pyplot as plt


class LrFinder():

    def __init__(self, device='cpu', plotdir=None):
        r"""init

        Initialize LrFinder.

        Parameters
        ----------
        device : {string}, optional
            device string: ``'cpu'``(default), ``'cuda:0'``, ``cuda:1`` ...
        plotdir : {string}, optional
            If it is not None, plot the loss-lr curve and save the figure,
            otherwise plot and show but not save. (the default is None).
        """
        self.device = device
        self.plotdir = plotdir
        self.lrs = []
        self.losses = []
        self.avg_losses = []
        self.smt_losses = []

    def plot(self, lrmod='log', loss='smoothed'):
        r"""plot the loss-lr curve

        Plot the loss-learning rate curve.

        Parameters
        ----------
        lrmod : {str}, optional
            ``'log'`` --> use log scale, i.e. log10(lr) instead lr. (default)
            ``'linear'`` --> use original lr.
        loss : {str}, optional
            Specify which type of loss will be ploted. (the default is 'smoothed')
        """

        if lrmod in ['log', 'LOG', 'Log']:
            lrs = [math.log10(x) for x in self.lrs]
            lrmod = 'Log10'
            lrunitstr = '/Log10'
        if lrmod in ['linear', 'LINEAR', 'Linear']:
            lrs = self.lrs
            lrmod = ''
            lrunitstr = ''

        if loss in ['smoothed', 'SMOOTHED', 'Smoothed']:
            loss = 'Smoothed'
            losses = self.smt_losses
        if loss in ['average', 'AVERAGE', 'Average']:
            loss = 'Average'
            losses = self.avg_losses
        if loss in ['original', 'ORIGINAL', 'Original']:
            loss = 'Original'
            losses = self.losses

        plt.figure()
        plt.plot(lrs, losses)
        plt.xlabel('Learning rate' + lrunitstr)
        plt.ylabel('Loss')
        plt.grid()

        losslr_str = loss + 'Loss_' + lrmod + 'LR.png'

        if self.plotdir is None:
            plt.show()
        else:
            plt.savefig(self.plotdir + '/' + losslr_str)
            plt.close()

    def find(self, dataloader, model, optimizer, criterion, nin=1, nbgc=1, lr_init=1e-8, lr_final=1e2, beta=0.98, gamma=4.):
        r"""Find learning rate

        Find learning rate, see `How Do You Find A Good Learning Rate <https://sgugger.github.io/how-do-you-find-a-good-learning-rate.html>`_ .

        During traing, two types losses are computed

        The average loss is:

        .. math::
           \text{avg_loss}_i=\beta * \text{avg_loss}_{i-1}+(1-\beta) * \text{loss}_i

        The smoothed loss is:

        .. math::
            \text{smt_loss }_{i}=\frac{\text{avg_loss}_{i}}{1-\beta^{i+1}}

        If :math:`i > 1` and :math:`\text{smt_loss} > \gamma * \text{best_loss}`, stop.

        If :math:`\text{smt_loss} < \text{best_loss}` or :math:`i = 1`, let :math:`\text{best_loss} = \text{smt_loss}`.


        Parameters
        ----------
        dataloader : {torch.DataLoader}
            The dataloader that contains a dataset for training.
        model : {torch.Module}
            Your network module.
        optimizer : {torch.Optimizer}
            The optimizer such as SGD, Adam...
        criterion : {torch.Loss}
            The criterion/loss used for training model.
        nin : {number}, optional
            The number of inputs in dataloder, the first :attr:`nin` elements are inputs, the rest are targets(can be None). (the default is 1)
        nbgc : {number}, optional
            The number of batches for grad cumulation (the default is 1, which means no cumulation)
        lr_init : {number}, optional
            The initial learning rate (the default is 1e-8)
        lr_final : {number}, optional
            The final learning rate (the default is 1e-8)
        beta : {float}, optional
            [description] (the default is 0.98)
        gamma : {float}, optional
            The exploding factor :math:`\gamma`. (the default is 4.)


        Returns
        -------
        lrs : {list}
            Learning rates during training.
        smt_losses : {list}
            Smoothed losses during training.
        avg_losses : {list}
            Average losses during training.
        losses : {list}
            Original losses during training.

        Examples
        --------

        ::

            device = 'cuda:1'
            # device = 'cpu'

            num_epochs = 30
            X = th.randn(100, 2, 3, 4)
            Y = th.randn(100, 1, 3, 4)

            trainds = TensorDataset(X, Y)
            # trainds = TensorDataset(X)

            model = th.nn.Conv2d(2, 1, 1)
            model.to(device)

            trainld = DataLoader(trainds, batch_size=10, shuffle=False)

            criterion = th.nn.MSELoss(reduction='mean')

            optimizer = th.optim.SGD(model.parameters(), lr=1e-1)

            lrfinder = LrFinder(device)
            # lrfinder = LrFinder(device, plotdir='./')

            lrfinder.find(trainld, model, optimizer, criterion, nin=1,
                          nbgc=1, lr_init=1e-8, lr_final=10., beta=0.98)

            lrfinder.plot(lrmod='Linear')
            lrfinder.plot(lrmod='Log')
        """

        num = len(dataloader) - 1
        mult = (lr_final / lr_init) ** (1. / num)
        lr = lr_init
        optimizer.param_groups[0]['lr'] = lr
        avg_loss, best_loss = 0., 0.

        for b, data in enumerate(dataloader):
            data = [x.to(self.device) for x in data]

            if b % nbgc == 0:
                optimizer.zero_grad()

            outputs = model(*data[:nin])
            loss = criterion(outputs, *data[nin:])

            # Compute the smoothed loss
            avg_loss = beta * avg_loss + (1. - beta) * loss.item()
            smt_loss = avg_loss / (1. - beta**(b + 1))

            self.losses.append(loss.item())
            self.avg_losses.append(avg_loss)

            # Stop if the loss is exploding
            if b > 1 and smt_loss > gamma * best_loss:
                return self.lrs, self.smt_losses, self.avg_losses, self.losses

            # Record the best loss
            if smt_loss < best_loss or b == 1:
                best_loss = smt_loss

            # Store the values
            self.smt_losses.append(smt_loss)
            self.lrs.append(lr)

            # Do the SGD step
            loss.backward()
            optimizer.step()

            # Update the lr for the next step
            lr *= mult
            optimizer.param_groups[0]['lr'] = lr

        return self.lrs, self.smt_losses, self.avg_losses, self.losses


if __name__ == '__main__':

    import torch as th
    from torch.utils.data import DataLoader, TensorDataset

    device = 'cuda:1'
    # device = 'cpu'

    num_epochs = 30
    X = th.randn(100, 2, 3, 4)
    Y = th.randn(100, 1, 3, 4)

    trainds = TensorDataset(X, Y)
    # trainds = TensorDataset(X)

    model = th.nn.Conv2d(2, 1, 1)
    model.to(device)

    trainld = DataLoader(trainds, batch_size=10, shuffle=False)

    criterion = th.nn.MSELoss(reduction='mean')

    optimizer = th.optim.SGD(model.parameters(), lr=1e-1)

    lrfinder = LrFinder(device)
    # lrfinder = LrFinder(device, plotdir='./')

    lrfinder.find(trainld, model, optimizer, criterion, nin=1,
                  nbgc=1, lr_init=1e-8, lr_final=10., beta=0.98)

    lrfinder.plot(lrmod='Linear')
    lrfinder.plot(lrmod='Log')
