import numpy as np
import torch


def discretization_decode(label, discretization="UD", beta=94.0, gamma=0.0, ord_num=94):
    '''
    :param label: ord_labels is ordinal outputs for each spatial locations,
                  size is B x H X W X C (C = 2num_ord, num_ord is interval of SID)
    :param discretization: uniform discretization (UD) or spacing-increasing discretization (SID)
    :param ord_num: is interval of discretization
    :return: depth (disparity)  value
    '''
    if discretization == "SID":
        t0 = torch.exp(np.log(beta) * label.float() / ord_num)
        t1 = torch.exp(np.log(beta) * (label.float() + 1) / ord_num)
    else:
        pass
        # t0 = 1.0 + (beta - 1.0) * label.float() / ord_num
        # t1 = 1.0 + (beta - 1.0) * (label.float() + 1) / ord_num

    # depth = (t0 + t1) / 2 - gamma
    depth = label.float()

    return depth


