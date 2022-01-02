from __future__ import division
import cv2
import torch
import torch.nn.functional as F
from utils.utils import gather_feature


def peaks_extraction(heat_map):
    '''
    extract the peaks in heatmap,
    the peak extraction serves as a sufficient NMS alternative as mentioned in paper.
    '''
    heat_map_maxpooling = F.max_pool2d(heat_map, 3, 1, 1)
    mask = (heat_map_maxpooling == heat_map)
    return heat_map * mask


def topk_peaks(heat_map, K=40):
    b, c, h, w = heat_map.shape
    # top k peaks in every single fearure map. idx are index of top k peaks
    topk_scores, topk_idx = torch.topk(heat_map.view(b, -1), K)
    # [B, K]
    topk_cls = (topk_idx / (h * w)).int()
    # index of top k peaks in their classes
    topk_idx = topk_idx % (h * w)

    topk_ys = (topk_idx / w).int().float()
    topk_xs = (topk_idx % w).int().float()

    return topk_scores, topk_idx, topk_cls, topk_xs, topk_ys


def decode(heat_map, wh, reg, K=60):
    '''
    decode predicted centers into bboxes
    :param heat_map: [B, C, H, W]
    :param wh: [B, 2, H, W]
    :param reg: [B, 2, H, W]
    :param K: number of the centers that will be kept
    :return:
    bboxes: [B, K, 4]
    cls: [B, K, 1]
    scores: [B, K, 1]
    '''
    b, c, h, w = heat_map.shape

    heat_map = peaks_extraction(heat_map)
    topk_scores, topk_idx, topk_cls, topk_xs, topk_ys = topk_peaks(heat_map, K)
    # [B, 2, H, W] -> [B, HxW, 2]
    wh = wh.view(b, wh.shape[1], -1).permute((0, 2, 1)).contiguous()
    reg = reg.view(b, reg.shape[1], -1).permute((0, 2, 1)).contiguous()

    wh = gather_feature(wh, topk_idx).view(reg.shape[0], K, 2)
    reg = gather_feature(reg, topk_idx).view(reg.shape[0], K, 2)

    xs = topk_xs + reg[:, :, 0]
    ys = topk_ys + reg[:, :, 1]

    cls = topk_cls.view(b, K, 1).float()
    scores = topk_scores.view(b, K, 1)

    w = wh[:, :, 0]
    h = wh[:, :, 1]

    half_w, half_h = w / 2, h / 2

    # #for opencv
    left_x = xs - half_w
    left_y = ys - half_h
    right_x = xs + half_w
    right_y = ys + half_h
    left_x = left_x.unsqueeze(-1)
    left_y = left_y.unsqueeze(-1)
    right_x = right_x.unsqueeze(-1)
    right_y = right_y.unsqueeze(-1)
    bboxes = torch.cat([left_x, left_y, right_x, right_y], dim=2)

    # #for matplotlib
    # left_x = xs - half_w
    # left_y = ys - half_h
    # left_x = left_x.unsqueeze(-1)
    # left_y = left_y.unsqueeze(-1)
    # w = w.unsqueeze(-1)
    # h = h.unsqueeze(-1)
    # bboxes = torch.cat([left_x, left_y, w, h], dim=2)

    return bboxes, cls, scores
