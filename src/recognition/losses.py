import torch.nn.functional as F
from typing import Tuple
from torch import Tensor
import torch
import torch.nn as nn
import torch.nn.functional as F


# nn.TripletMarginLoss()
class TripletLoss(nn.Module):
    def __init__(self, margin=1.0):
        super().__init__()
        self.margin = margin

    def calculate_euclidean(self, x1, x2):
        return (x1 - x2).pow(2).sum(1)

    def forward(self, anchor, positive, negative, size_average=True):
        distance_positive = self.calculate_euclidean(anchor, positive)
        distance_negative = self.calculate_euclidean(anchor, negative)
        losses = F.relu(distance_positive - distance_negative + self.margin)
        return losses.mean() if size_average else losses.sum()


def convert_label_to_similarity(normed_feature: Tensor, label: Tensor) -> Tuple[Tensor, Tensor]:
    similarity_matrix = normed_feature @ normed_feature.transpose(1, 0)
    label_matrix = label.unsqueeze(1) == label.unsqueeze(0)

    positive_matrix = label_matrix.triu(diagonal=1)
    negative_matrix = label_matrix.logical_not().triu(diagonal=1)

    similarity_matrix = similarity_matrix.view(-1)
    positive_matrix = positive_matrix.view(-1)
    negative_matrix = negative_matrix.view(-1)
    return similarity_matrix[positive_matrix], similarity_matrix[negative_matrix]


class CircleLoss(nn.Module):
    def __init__(self, m: float, gamma: float) -> None:
        super(CircleLoss, self).__init__()
        self.m = m
        self.gamma = gamma
        self.soft_plus = nn.Softplus()

    def forward(self, sp: Tensor, sn: Tensor) -> Tensor:
        ap = torch.clamp_min(- sp.detach() + 1 + self.m, min=0.)
        an = torch.clamp_min(sn.detach() + self.m, min=0.)

        delta_p = 1 - self.m
        delta_n = self.m

        logit_p = - ap * (sp - delta_p) * self.gamma
        logit_n = an * (sn - delta_n) * self.gamma

        loss = self.soft_plus(torch.logsumexp(logit_n, dim=0) + torch.logsumexp(logit_p, dim=0))

        return loss


# triplet loss with pos feature enhance
class MyLoss(nn.Module):
    def __init__(self, margin=1.0, gamma=2.0):
        super().__init__()
        self.margin = margin
        self.gamma = gamma

    def calculate_euclidean(self, x1, x2):
        return (x1 - x2).pow(2).sum(1)

    def calculate_pos_anchor(self,anchor,pos):
        return (1-anchor).pow(self.gamma) * pos.pow(self.gamma) + \
               anchor.pow(self.gamma) *(1-pos).pow(self.gamma)

    def forward(self, anchor, positive, negative, size_average=True):
        distance_positive = self.calculate_euclidean(anchor, positive)
        distance_negative = self.calculate_euclidean(anchor, negative)
        losses = F.relu(distance_positive - distance_negative + self.margin)
        return losses.mean() if size_average else losses.sum()


if __name__ == "__main__":
    feat = nn.functional.normalize(torch.rand(2,3,10, requires_grad=True))
    # lbl = torch.randint(high=10, size=(24,))
    # print(lbl.numpy())
    # #lbl = [['a','s','a']]
    # inp_sp, inp_sn = convert_label_to_similarity(feat, lbl)

    # criterion = CircleLoss(m=0.25, gamma=256)
    # circle_loss = criterion(inp_sp, inp_sn)
    #print(circle_loss)
    print(feat)
    a = feat[:,0,:]
    print(a.size())
    p = feat[:,1,:]
    n = feat[:,2,:]

    criterion = MyLoss()
    loss = criterion(a,p,n)
    print(loss)

    # a = feat[:, 0, :]
    # b = feat[:, 1, :]
    # loss = (1-a).pow(2.0) * b
    # print(loss.sum(1))




