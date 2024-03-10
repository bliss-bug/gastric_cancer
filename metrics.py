import torch
import torch.nn.functional as F


def IoU(pred: torch, label: torch):
    # [H*W]
    intersection = (pred * label).sum()
    union = pred.sum() + label.sum() - intersection

    iou = intersection / union
    return iou


def mIoU(pred: torch, label: torch, num_classes=3):
    N = pred.shape[0]
    pred, label = pred.view(N, -1), label.view(N, -1)  # [N, H*W], [N, H*W]
    pred = F.one_hot(pred, num_classes).transpose(1, 2)  # [N, C, H*W]
    label = F.one_hot(label, num_classes).transpose(1, 2)  # [N, C, H*W]

    miou = torch.tensor([[IoU(pred[i,j,:], label[i,j,:]) for j in range(num_classes)] for i in range(N)], dtype=float)

    return miou.nanmean(dim=1).mean()



def BinaryDice(pred: torch, label: torch):
    # [H*W]
    intersection = 2. * (pred * label).sum()
    sum = pred.sum() + label.sum()

    b_dice = intersection / sum
    return b_dice



def Dice(pred: torch, label: torch, num_classes=3):
    N = pred.shape[0]
    pred, label = pred.view(N, -1), label.view(N, -1)  # [N, H*W], [N, H*W]
    pred = F.one_hot(pred, num_classes).transpose(1, 2)  # [N, C, H*W]
    label = F.one_hot(label, num_classes).transpose(1, 2)  # [N, C, H*W]

    dice = torch.tensor([[BinaryDice(pred[i,j,:], label[i,j,:]) for j in range(num_classes)] for i in range(N)], dtype=float)

    return dice.nanmean(dim=1).mean()



if __name__ == '__main__':
    pred = torch.randint(0,20,(4,5,5))
    label = torch.randint(0,20,(4,5,5))
    print(mIoU(pred, label, 20).item(), Dice(pred, label, 20).item())