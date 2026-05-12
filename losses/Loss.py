import torch
import torch.nn as nn
import torch.nn.functional as F


# ===== 基础 loss =====

def mse_loss(pred, target, weight=None):
    loss_map = (pred - target) ** 2

    if weight is not None:
        loss_map = loss_map * weight
        return loss_map.sum() / weight.sum()

    return loss_map.mean()


def bce_loss(pred, target, weight=None):
    loss_map = F.binary_cross_entropy_with_logits(
        pred,
        target,
        reduction="none",
    )

    if weight is not None:
        loss_map = loss_map * weight
        return loss_map.sum() / weight.sum()

    return loss_map.mean()

def dice_loss(pred, target, weight=None, eps=1e-6):
    pred = torch.sigmoid(pred)  # 因为你用 logits
    pred = pred.view(pred.size(0), -1)
    target = target.view(target.size(0), -1)

    intersection = (pred * target).sum(dim=1)
    union = pred.sum(dim=1) + target.sum(dim=1)

    dice = (2 * intersection + eps) / (union + eps)
    return 1 - dice.mean()


def sobel_loss(pred, target, weight=None):
    sobel_x = torch.tensor([[1,0,-1],[2,0,-2],[1,0,-1]], dtype=torch.float32).view(1,1,3,3).to(pred.device)
    sobel_y = torch.tensor([[1,2,1],[0,0,0],[-1,-2,-1]], dtype=torch.float32).view(1,1,3,3).to(pred.device)

    pred_x = F.conv2d(pred, sobel_x, padding=1)
    pred_y = F.conv2d(pred, sobel_y, padding=1)

    tgt_x = F.conv2d(target, sobel_x, padding=1)
    tgt_y = F.conv2d(target, sobel_y, padding=1)

    return F.l1_loss(pred_x, tgt_x) + F.l1_loss(pred_y, tgt_y)


# ===== Loss Registry（关键） =====

LOSS_REGISTRY = {
    "mse": mse_loss,
    "bce": bce_loss,
    "sobel": sobel_loss,
    "dice": dice_loss
}


# ===== 组合 Loss 类 =====

class CombinedLoss(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.loss_items = []

        weights = config.get("weights", {})

        for name, weight in weights.items():
            if name not in LOSS_REGISTRY:
                raise ValueError(f"Unknown loss: {name}")
            self.loss_items.append((LOSS_REGISTRY[name], weight, name))

    def forward(self, pred, target, weight=None):
        total_loss = 0
        loss_dict = {}

        for fn, w, name in self.loss_items:
            l = fn(pred, target, weight)
            total_loss += w * l
            loss_dict[name] = l.item()

        return total_loss, loss_dict