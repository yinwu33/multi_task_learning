import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from utils.utils import gather_feature


class DetLoss(nn.Module):
    '''
    total loss for the objection detection task

    :param pred: predicted dict --> {"cls": [B,C,H,W], "wh": [B,2,H,W], "reg": [B,2,H,W]}
    :param target: groundtruth dict
    :param weights: weights for balancing cls_loss, wh_loss and reg_loss.
    :param alpha: We set alpha as 2 according to original implementation
    :param beta: We set beta as 4 according to original implementation
    '''
    def __init__(self, weights):
        super(DetLoss, self).__init__()
        self.loss = total_loss
        self.weights = weights

    def forward(self, pred, target):
        return self.loss(pred, target, self.weights)


def total_loss(pred, target, weights=(1,0.1,1), alpha=2, beta=4):
    device = pred["cls"].device
    cls_loss = heatmap_variant_focal_loss(pred["cls"],
                                          target["heat_map"].to(device),
                                          alpha=alpha, beta=beta)
    wh_loss = reg_l1_loss(pred["wh"], target["wh"].to(device),
                          target["index"].to(device), target["mask"].to(device))
    reg_loss = reg_l1_loss(pred["reg"], target["reg"].to(device),
                           target["index"].to(device), target["mask"].to(device))
    loss = cls_loss * weights[0] + wh_loss * weights[1] + reg_loss * weights[2]
    # print(loss)
    return loss


def heatmap_variant_focal_loss(pred, target, alpha=2, beta=4):
    pred = torch.clamp(pred, 1e-12)

    pos_indices = target.eq(1).float()
    neg_indices = target.lt(1).float()

    pos_loss = torch.pow(1 - pred, alpha) \
               * torch.log(pred) * pos_indices
    neg_loss = torch.pow(1 - target, beta) \
               * torch.pow(pred, alpha) \
               * torch.log(1 - pred) * neg_indices

    num_pos = pos_indices.float().sum()

    if num_pos == 0:
        loss = - neg_loss.sum()

    else:
        loss = - (pos_loss.sum() + neg_loss.sum()) / num_pos

    return loss


def reg_l1_loss(pred, target, index, mask):
    # [B,2,W,H] -> [B, HxW, 2]
    pred = pred.view(pred.shape[0], pred.shape[1], -1).permute((0, 2, 1)).contiguous()
    # [B, max_num_bbox, 2]
    pred = gather_feature(pred, index)
    # [B, max_num_bbox] -> [B, max_num_box, 2]
    mask = mask.unsqueeze(-1).expand_as(pred).float()
    pred = pred * mask
    # [B, max_num_bbox, 2]
    target = target * mask
    loss = F.l1_loss(pred, target, reduction='sum')
    loss = loss / (mask.sum() + 1e-4)

    return loss


# class ordLoss(nn.Module):
#     def __init__(self):
#         super(ordLoss, self).__init__()
#         self.loss = 0.0
#
#     def forward(self, ord_labels, target):
#         B, C, H, W = ord_labels.size()
#         ord_num = C
#         if torch.cuda.is_available():
#             K = torch.zeros((B, C, H, W), dtype=torch.float32).cuda()
#             for i in range(ord_num):
#                 K[:, i, :, :] = K[:, i, :, :] + i * torch.ones((B, H, W), dtype=torch.float32).cuda()
#         else:
#             K = torch.zeros((B, C, H, W), dtype=torch.float32)
#             for i in range(ord_num):
#                 K[:, i, :, :] = K[:, i, :, :] + i * torch.ones((B, H, W), dtype=torch.float32)
#
#         mask_0 = (K <= target).detach()
#         mask_1 = (K > target).detach()
#
#         one = torch.ones(ord_labels[mask_1].size())
#         if torch.cuda.is_available():
#             one = one.cuda()
#
#         self.loss = torch.sum(torch.log(torch.clamp(ord_labels[mask_0], min=1e-7, max=1e7))) \
#                     + torch.sum(torch.log(torch.clamp(one - ord_labels[mask_1], min=1e-7, max=1e7)))
#
#         B = B * H * W
#         self.loss /= (-B)
#         return self.loss


class OrdinalRegressionLoss(nn.Module):
    def __init__(self, beta=95, ord_num=94):
        super(OrdinalRegressionLoss, self).__init__()
        self.ord_num = ord_num
        self.beta = beta

    def forward(self, prob, gt):
        """
        :param prob: ordinal regression probability, B x 2*Ord Num x H x W, torch.Tensor
        :param gt: depth ground truth, BXHxW, torch.Tensor
        :return: loss: loss value, torch.float
        """
        # B, C, H, W = prob.shape
        valid_mask = torch.logical_and(gt > 0.0, gt<95.0)
        gt = torch.unsqueeze(gt, dim=1)
        ord_label, mask = create_ord_label(gt, ord_num=self.ord_num, beta=self.beta)
        entropy = -prob * ord_label
        loss = torch.sum(entropy, dim=1)[valid_mask]
        return loss.mean()


def create_ord_label(gt, ord_num, beta, discretization="UD"):
    B, C, H, W = gt.shape
    device = gt.device
    ord_c0 = torch.ones(B, ord_num, H, W).to(device)
    # [B, H, W]
    if discretization == "SID":
        label = ord_num * torch.log(gt) / np.log(beta)
    else:
        label = ord_num * (gt - 1.0) / (beta - 1.0)
    label = label.long()
    # start, end, steps
    mask = torch.linspace(0, ord_num - 1, ord_num,
                          requires_grad=False).view(1, ord_num, 1, 1).to(gt.device)
    # mask [B, ord_num, H, W]
    mask = mask.repeat(B, 1, H, W).contiguous().long()
    mask = (mask > label)
    ord_c0[mask] = 0
    ord_c1 = 1 - ord_c0
    # ord_label [B, ord_num * 2, H, W]
    ord_label = torch.cat((ord_c0, ord_c1), dim=1)
    return ord_label, mask


class ScaleInvariantLogLoss(nn.Module):
    # from the paper :
    # “Predicting Depth, Surface Normals and Semantic Labels
    # with a Common Multi-Scale Convolutional Architecture by David Eigen and Rob Fergus"
    def __init__(self, variance_focus=0.85):
        super(ScaleInvariantLogLoss, self).__init__()
        self.variance_focus = variance_focus

    def forward(self, depth_pred, depth_gt):
        depth_pred = depth_pred.squeeze(1)
        valid_mask = torch.logical_and(depth_gt > 0.0, depth_gt < 1.0)
        d = torch.log(depth_pred[valid_mask]) - torch.log(depth_gt)[valid_mask]
        return torch.sqrt((d ** 2).mean() - self.variance_focus * (d.mean() ** 2)) * 10


class L1MaskedLoss(nn.Module):
    def __init__(self):
        super(L1MaskedLoss, self).__init__()
        self.l1_loss = nn.L1Loss()

    def forward(self, depth_pred, depth_gt):
        depth_pred = depth_pred.squeeze(1)
        valid_mask = torch.logical_and(depth_gt > 0.0, depth_gt < 100.0)
        loss = self.l1_loss(depth_pred[valid_mask], depth_gt[valid_mask])
        return loss


class MaskedMSELoss(nn.Module):
    def __init__(self):
        super(MaskedMSELoss, self).__init__()

    def forward(self, depth_pred, depth_gt):
        depth_pred = depth_pred.squeeze(1)
        valid_mask = torch.logical_and(depth_gt > 0.0, depth_gt<95.0)
        return nn.MSELoss()(depth_pred[valid_mask], depth_gt[valid_mask])


def dynamic_weight_average(act_losses, prev_losses, num_task):
    # implemented from the paper available under
    # https: // openaccess.thecvf.com / content_CVPR_2019 / papers / Liu_End - To - End_Multi - Task_Learning_With_Attention_CVPR_2019_paper.pdf
    # taken from equation (7)

    weights = []
    for task in range(num_task):
        try:
            weights.append(act_losses[task] / prev_losses[task])
        except:
            weights.append(0)

    return weights


def dynamic_weight_average_loss_weights(weights, num_task, epochs):

    loss_weights = []
    for task in range(num_task):
        try:
            loss_weights.append((num_task*torch.exp(weights[task]/epochs)/(sum(weights)/epochs)).item())
        except:
            loss_weights.append(0)

    return loss_weights

