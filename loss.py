import torch
import torch.nn as nn
import torch.nn.functional as F


class BinaryDiceLoss(nn.Module):
    def __init__(self, smooth = 1.0):
        super(BinaryDiceLoss, self).__init__()
        self.smooth = smooth
    
    def forward(self, pred: torch, target: torch):
        N = pred.shape[0]
        pred, target = pred.view(N, -1), target.view(N, -1)  # [N, H*W]

        intersection = 2. * (pred * target).sum() + self.smooth
        sum = pred.sum() + target.sum() + self.smooth

        loss = 1. - intersection / sum

        return loss



class DiceLoss(nn.Module):
    def __init__(self, smooth = 1.0, weight = None):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        self.weight = weight

    def forward(self, pred: torch, target: torch):
        '''
        pred: N * C * H * W
        target: N * H * W
        '''
        N, C = pred.shape[:2]
        if self.weight is not None:
            assert len(self.weight) == C
        pred = pred.softmax(dim=1)

        pred, target = pred.view(N, C, -1), target.view(N, -1)  # [N, C, H*W], [N, H*W]
        target = F.one_hot(target, C).transpose(1, 2)  # [N, C, H*W]

        total_loss = 0
        b_dice = BinaryDiceLoss(self.smooth)

        for i in range(C):
            loss = b_dice(pred[:,i,:], target[:,i,:])
            total_loss += loss if self.weight is None else loss * self.weight[i]

        return total_loss / C
    


class FocalLoss(nn.Module):
    def __init__(self, gamma = 2.0, alpha = None):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, input: torch, target: torch):
        '''
        input: N * C * H * W
        target: N * H * W
        '''
        ce_loss = F.cross_entropy(input, target, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss
        
        if self.alpha is not None:
            focal_loss = self.alpha * focal_loss
            
        return torch.mean(focal_loss)



class DFLoss(nn.Module):
    def __init__(self, lamda = [1, 1], smooth = 1.0, weight = None, gamma = 2.0, alpha = None):
        super(DFLoss, self).__init__()
        self.lamda = lamda
        self.dice_loss = DiceLoss(smooth, weight)
        self.focal_loss = FocalLoss(gamma, alpha)

    def forward(self, input: torch, target: torch):
        loss1 = self.dice_loss(input, target)
        loss2 = self.focal_loss(input, target)

        return self.lamda[0] * loss1 + self.lamda[1] * loss2



class DiceLoss2(nn.Module):
    def __init__(self, smooth=1.0):
        super(DiceLoss2, self).__init__()
        self.smooth = smooth

    def forward(self, pred, mask):
        """
        pred: [B, 1, H, W]
        mask: [B, 1, H, W]
        """
        assert pred.shape == mask.shape, "pred and mask should have the same shape."
        p = torch.sigmoid(pred)
        intersection = torch.sum(p * mask)
        union = torch.sum(p) + torch.sum(mask)
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)

        return 1 - dice
    

class FocalLoss2(nn.Module):
    def __init__(self, gamma=2.0, alpha=0.5):
        super(FocalLoss2, self).__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, pred, mask):
        """
        pred: [B, 1, H, W]
        mask: [B, 1, H, W]
        """
        assert pred.shape == mask.shape, "pred and mask should have the same shape."
        p = torch.sigmoid(pred)
        num_pos = torch.sum(mask)
        num_neg = mask.numel() - num_pos
        w_pos = (1 - p) ** self.gamma
        w_neg = p ** self.gamma

        loss_pos = -self.alpha * mask * w_pos * torch.log(p + 1e-12)
        loss_neg = -(1 - self.alpha) * (1 - mask) * w_neg * torch.log(1 - p + 1e-12)

        loss = (torch.sum(loss_pos) + torch.sum(loss_neg)) / (num_pos + num_neg + 1e-12)

        return loss



class DFLoss2(nn.Module):
    def __init__(self, lamda = [1, 1], smooth = 1.0, gamma = 2.0):
        super(DFLoss2, self).__init__()
        self.lamda = lamda
        self.dice_loss = DiceLoss2(smooth)
        self.focal_loss = FocalLoss2(gamma)

    def forward(self, input: torch, target: torch):
        """
        input: [B, 1, H, W]
        target: [B, H, W]
        """
        target = torch.unsqueeze(target, dim=1)
        loss1 = self.dice_loss(input, target)
        loss2 = self.focal_loss(input, target)

        return self.lamda[0] * loss1 + self.lamda[1] * loss2


if __name__ == '__main__':
    pred = torch.rand(4,3,512,512)
    target = torch.randint(0,3,(4,512,512)).long()

    f_loss = DFLoss()
    print(f_loss(pred, target))