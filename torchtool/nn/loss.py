import torch as th
import torch.nn as nn
import torchtool as tht
from torchtool.utils.const import EPS
from torch.autograd import Variable


class DCLoss(nn.Module):
    def __init__(self):
        super(DCLoss, self).__init__()

    def forward(self, P, G):

        # TP = th.sum(((P == 1) + (G == 1)) == 2)
        # FP = th.sum(((P == 1) + (G == 0)) == 2)
        # FN = th.sum(((P == 0) + (G == 1)) == 2)
        # f = (FP + FN) / (2.0 * TP + EPS)

        TP = th.sum((P + G) == 2)
        FPFN = th.sum((P + G) == 1)
        return FPFN / (2.0 * TP + EPS)
        # return f


class CDLoss(nn.Module):
    def __init__(self):
        super(CDLoss, self).__init__()

    def forward(self, P, G):
        return th.mean(th.abs(P * (1 - G)))


class JaccardLoss(nn.Module):
    r"""Jaccard distance

    .. math::
       d_{J}({\mathbb A}, {\mathbb B})=1-J({\mathbb A}, {\mathbb B})=\frac{|{\mathbb A} \cup {\mathbb B}|-|{\mathbb A} \cap {\mathbb B}|}{|{\mathbb A} \cup {\mathbb B}|}

    """

    def __init__(self):
        super(JaccardLoss, self).__init__()

    def forward(self, P, G):
        capnum = th.sum(P * G)
        sumpg = th.sum(P) + th.sum(G)

        return (1.0 - capnum / (sumpg - capnum + EPS))


class IridescentLoss(nn.Module):
    r"""Iridescent Distance Loss

    .. math::
       d_{J}({\mathbb A}, {\mathbb B})=1-J({\mathbb A}, {\mathbb B})=\frac{|{\mathbb A} \cup {\mathbb B}|-|{\mathbb A} \cap {\mathbb B}|}{|{\mathbb A} \cup {\mathbb B}|}

    """

    def __init__(self):
        super(IridescentLoss, self).__init__()

    def forward(self, P, G):
        sumbothones = th.sum(P * G)
        sumbothzeros = th.sum((1 - P) * (1 - G))
        print(sumbothones, sumbothzeros)
        return 1.0 - (sumbothones + sumbothzeros) / (th.sum(P) + th.sum(G) - sumbothones + sumbothzeros + EPS)
        # return 1.0 - (th.sum(P * G) + th.sum((1 - P) * (1 - G))) / (th.sum(P) + th.sum(G) - th.sum(P * G) + EPS)


class F1Loss(nn.Module):
    r"""F1 distance Loss

    .. math::
       F_{\beta} = 1 -\frac{(1+\beta^2)鈰匬鈰匯}{\beta^2鈰匬 + R}
       :label: equ-F-measure

    where,

    .. math::
       {\rm PPV} = {P} = \frac{\rm TP}{{\rm TP} + {\rm FP}}
       :label: equ-Precision

    .. math::
       {\rm TPR} = {R} = \frac{\rm TP}{{\rm TP} + {\rm FN}}
       :label: equ-Recall

    """

    def __init__(self):
        super(F1Loss, self).__init__()

    def forward(self, P, G):

        TP = th.sum(P * G)
        FP = th.sum(P * (1. - G))
        FN = th.sum((1 - P) * G)
        P = TP / (TP + FP + EPS)
        R = TP / (TP + FN + EPS)

        return 1.0 - 2.0 * P * R / (P + R + EPS)


if __name__ == '__main__':
    import numpy as np

    # X = np.array([[1, 1, 1], [0, 1, 0]])
    # X = np.array([[0, 0, 1], [0, 0, 1]])
    X = np.array([[0.3, 0, 1], [0, 0.8, 1]])
    X = np.array([[0, 0, 1], [0, 0, 1]])
    G = np.array([[0, 0, 1], [0, 0, 1]])

    X = np.array([[0.1, 0.1, 0], [0, 0, 0]])
    G = np.array([[0, 0.5, 0], [0, 0, 0]])

    # X = np.array([[1, 1, 0], [0, 0, 0]])
    # G = np.array([[1, 1, 1], [1, 1, 1]])

    # X = np.array([[1, 1, 1], [1, 0, 0]])
    # G = np.array([[0, 0, 0], [0, 0, 0]])

    # X = np.array([[0, 1, 0], [0, 0, 0]])
    # G = np.array([[0, 0, 0], [0, 1, 0]])

    # X = np.array([[0, 0, 0], [0, 0, 0]])
    # G = np.array([[0, 0, 0], [0, 0, 0]])

    # X = np.array([[1, 0, 1], [1, 1, 1]])
    # G = np.array([[1, 1, 1], [1, 1, 1]])

    # X = np.array([[1, 1, 1], [1, 1, 1]])
    # G = np.array([[1, 1, 1], [1, 1, 1]])

    X = th.Tensor(X)
    G = th.Tensor(G)
    X = Variable(X, requires_grad=True)
    G = Variable(G, requires_grad=True)

    net = nn.ReLU()
    P = net(X)

    print(X)
    print(P)
    print(G)
    print(th.mean(th.abs(X - G)))

    # criterion = tht.DCLoss()
    criterion = tht.JaccardLoss()
    # criterion = tht.IridescentLoss()
    criterion = tht.F1Loss()

    loss = criterion(X, G)
    lossv = loss.item()
    print(lossv)
    loss.backward()

    print(lossv)
