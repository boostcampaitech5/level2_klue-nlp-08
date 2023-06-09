import torch
from torch import nn


def get_loss(loss_type):
    if loss_type == "cross_entropy":
        return nn.CrossEntropyLoss
    elif loss_type == "focal_loss":
        return FocalLoss
    else:
        raise ValueError("정의되지 않은 loss type입니다.")

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=5, logits=False, reduce=False):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.logits = logits
        self.reduce = reduce

    def forward(self, inputs, targets):
        nn.CrossEntropyLoss()
        ce_loss = nn.CrossEntropyLoss()(inputs, targets)
        pt = torch.exp(-ce_loss)
        F_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        if self.reduce:
            return torch.mean(F_loss)
        else:
            return F_loss
