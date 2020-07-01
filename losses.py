import torch
import torch.nn as nn
import torch.nn.functional as F

# BCE Dice Loss
class BCEDiceLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, y_pred, y_true):
        bce = F.binary_cross_entropy_with_logits(y_pred, y_true)
        smooth = 1e-6
        y_pred = torch.sigmoid(y_pred)
        y_pred = (y_pred > 0.5).float()
        # 展平
        y_pred = y_pred.view(1,-1)
        y_true = y_true.view(1,-1)
        intersection = (y_pred * y_true)
        dice = (2. * intersection.sum(1) + smooth) / (y_pred.sum(1) + y_true.sum(1) + smooth)
        dice = 1 - dice.sum()
        return  0.5*bce + dice

# Focal Dice Loss
class FocalDiceLoss(nn.Module):
    def __init__(self, gamma, alpha=None):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, y_pred, y_true):
        # 展平
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        y_pred = y_pred.view(1,-1)
        y_true = y_true.view(1,-1)
        pt = torch.sigmoid(y_pred)
        pt = torch.cat([1-pt,pt], dim=0)
        pt = torch.gather(pt, 0, y_true.long())
        log_pt = torch.log(pt)
        # 根据y的标签决定logpt
        # log_pt = torch.gather(log_pt, 1, torch.LongTensor(y_true))
        # pt = torch.exp(log_pt)
        if self.alpha:
            at = self.alpha*torch.ones(y_pred.shape[-1])
            at = at.view(1,-1)
            at = at.to(device)
            at = torch.cat([1-at, at], dim=0)
            at = torch.gather(at, 0, y_true.long())
            log_pt = log_pt*at
        focal_loss = -(1-pt)**self.gamma * log_pt

        smooth = 1e-6
        p = torch.sigmoid(y_pred)
        pred = (p > 0.5).float()
        # 展平
        # p = p.view(1,-1)
        # y_true = y_true.view(1,-1)
        intersection = (pred * y_true)
        dice = (2. * intersection.sum(1) + smooth) / (pred.sum(1) + y_true.sum(1) + smooth)
        loss = torch.mean(focal_loss)
        return loss
