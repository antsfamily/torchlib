import torch
import torch.nn.functional as F

from DeepSegmentor import LR_THRESHOLD, LEARNING_RATE, DIMENSION

ITERATION_LIMIT = int(1e6)


class SoftDiceLoss(torch.nn.Module):
    def __init__(self):
        super(SoftDiceLoss, self).__init__()

    def forward(self, actual, desire):
        smooth = 1e-6
        length = desire.size(0)
        m1 = actual.view(length, -1)
        m2 = desire.view(length, -1)
        intersection = (m1 * m2)
        score = 2. * (intersection.sum(1) + smooth) / (m1.sum(1) + m2.sum(1) + smooth)
        score = 1 - score.sum() / length
        return score


class FocalLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.gamma = 2.0

    def forward(self, actual, desire):
        length = desire.size(0)
        actual = actual.view(length, -1)
        desire = desire.view(length, -1)
        max_val = (-actual).clamp(min=0)
        loss = actual - actual * desire + max_val + ((-max_val).exp() + (-actual - max_val).exp()).log()
        invprobs = F.logsigmoid(-actual * (desire * 2 - 1))
        loss = (invprobs * self.gamma).exp() * loss
        return loss.mean()


class BinaryCrossEntropyLoss(torch.nn.Module):
    def __init__(self):
        super(BinaryCrossEntropyLoss, self).__init__()

    def forward(self, actual, desire):
        length = desire.size(0)
        actual = actual.view(length, -1)
        desire = desire.view(length, -1)
        bce = F.binary_cross_entropy(actual, desire)
        return bce


class ContourLoss(torch.nn.Module):
    def __init__(self):
        super(ContourLoss, self).__init__()
        self.edge_detector = EdgeDetector()
        self.dice = SoftDiceLoss()

    def forward(self, actual, desire):
        actualContours = self.edge_detector(actual)
        desireContours = self.edge_detector(desire)
        loss = self.dice(actual, desire) + F.l1_loss(actualContours, desireContours)
        return loss

class EdgeAwareLoss(torch.nn.Module):
    def __init__(self):
        super(EdgeAwareLoss, self).__init__()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.cudas = list(range(torch.cuda.device_count()))
        self.features = EdgeFeatureExtractor()
        self.predictor = torch.nn.Sequential()
        self.predictor.add_module('fc', torch.nn.Conv2d(2, 1, 1, 1, 0, bias=False))
        self.predictor.add_module('sigmoid', torch.nn.Sigmoid())
        self.features.to(self.device)
        self.predictor.to(self.device)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=LEARNING_RATE)
        self.edge_detector = EdgeDetector()
        self.soft_dice = SoftDiceLoss()
        self.lambdas = [float(1), float(0.5), float(0.25)]
        self.loss = None
        self.counter = int(0)
        self.best_loss = float(100500)
        self.current_loss = float(0)

    def evaluate(self, actual, desire):
        actual_features = torch.nn.parallel.data_parallel(module=self.features, inputs=actual, device_ids=self.cudas)
        desire_features = torch.nn.parallel.data_parallel(module=self.features, inputs=desire, device_ids=self.cudas)
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
        fake = torch.nn.parallel.data_parallel(module=self.predictor, inputs=desire_features[-1].detach(), device_ids=self.cudas)
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
