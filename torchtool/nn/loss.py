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
        return 1.0 - capnum / (th.sum(P) + th.sum(G) - capnum)


if __name__ == '__main__':
    import numpy as np

    # X = np.array([[1, 1, 1], [0, 1, 0]])
    # X = np.array([[0, 0, 1], [0, 0, 1]])
    X = np.array([[0.3, 0, 1], [0, 0.8, 1]])
    G = np.array([[0, 0, 1], [0, 0, 1]])

    X = th.Tensor(X)
    G = th.Tensor(G)
    X = Variable(X, requires_grad=True)
    G = Variable(G, requires_grad=True)

    net = nn.ReLU()
    P = net(X)

    print(X)
    print(P)
    print(G)

    criterion = tht.DCLoss()

    loss = criterion(X, G)
    lossv = loss.item()
    loss.backward()

    print(lossv)
