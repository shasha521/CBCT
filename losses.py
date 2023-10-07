import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2 as cv


def tv_loss(x, beta = 0.5, reg_coeff = 5):
    '''Calculates TV loss for an image `x`.
        
    Args:
        x: image, torch.Variable of torch.Tensor
        beta: See https://arxiv.org/abs/1412.0035 (fig. 2) to see effect of `beta` 
    '''
    dh = torch.pow(x[:,:,:,1:] - x[:,:,:,:-1], 2)
    dw = torch.pow(x[:,:,1:,:] - x[:,:,:-1,:], 2)
    a,b,c,d=x.shape
    return reg_coeff*(torch.sum(torch.pow(dh[:, :, :-1] + dw[:, :, :, :-1], beta))/(a*b*c*d))

class TVLoss(nn.Module):
    def __init__(self, tv_loss_weight=1):
        super(TVLoss, self).__init__()
        self.tv_loss_weight = tv_loss_weight

    def forward(self, x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = self.tensor_size(x[:, :, 1:, :])
        count_w = self.tensor_size(x[:, :, :, 1:])
        h_tv = torch.pow((x[:, :, 1:, :] - x[:, :, :h_x - 1, :]), 2).sum()

        print(x)
        print(x[:, :, 1:, :], x[:, :, :h_x - 1, :])
        print(h_tv)

        w_tv = torch.pow((x[:, :, :, 1:] - x[:, :, :, :w_x - 1]), 2).sum()
        return self.tv_loss_weight * 2 * (h_tv / count_h + w_tv / count_w) / batch_size

    @staticmethod
    def tensor_size(t):
        return t.size()[1] * t.size()[2] * t.size()[3]

class TV1Loss(nn.Module):
    def __init__(self, tv_loss_weight=1):
        super(TV1Loss, self).__init__()
        self.tv_loss_weight = tv_loss_weight

    def forward(self, x, y):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = self.tensor_size(x[:, :, 1:, :])
        grd=0
        for i in range(batch_size):
            high=cv.warpPolar(x[i].permute((1,2,0)), (4, 4), (4//2, 4//2), 4//2, cv.INTER_LINEAR +cv.WARP_POLAR_LINEAR)
      
            low=cv.warpPolar(torch.clamp(y[i],0,1).permute((1,2,0)), (4, 4), (4//2, 4//2), 4//2, cv.INTER_LINEAR +cv.WARP_POLAR_LINEAR)
    
            high = torch.from_numpy(high)
            low = torch.from_numpy(low)
            high=high.permute(2,0,1).unsqueeze(0)
            low=low.permute(2,0,1).unsqueeze(0)
            high_mat= (high[:, :, 1:, :] - high[:, :, :h_x - 1, :])
            low_mat=(low[:, :, 1:, :] - low[:, :, :h_x - 1, :])
            grd+=torch.pow((high_mat-low_mat), 2).sum()/count_h
        return self.tv_loss_weight * 2 * grd / batch_size

    @staticmethod
    def tensor_size(t):
        return t.size()[1] * t.size()[2] * t.size()[3]

class CharbonnierLoss(nn.Module):
    """Charbonnier Loss (L1)"""

    def __init__(self, eps=1e-3):
        super(CharbonnierLoss, self).__init__()
        self.eps = eps

    def forward(self, x, y):
        diff = x - y
        # loss = torch.sum(torch.sqrt(diff * diff + self.eps))
        loss = torch.mean(torch.sqrt((diff * diff) + (self.eps*self.eps)))
        return loss

