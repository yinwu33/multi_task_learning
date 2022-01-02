import os
import cv2
import glob
import datetime
import json
import torch
import shutil
import random
import numpy as np
import pandas as pd
import torch.nn as nn
import matplotlib.pyplot as plt
import torchvision.transforms as T
from torch.utils.data import random_split

try:
    from mean_average_precision import MetricBuilder
except ImportError as e:
    import os
    os.system('pip install --upgrade git+https://github.com/bes-dev/mean_average_precision.git')
    print('mean average precision was upgraded')
    from mean_average_precision import MetricBuilder


def iou(pred, target, num_classes=4):
    ious = []
    # background class (o) is ignored
    for cls in range(1, num_classes):
        pred_inds = pred == cls
        target_inds = target == cls
        intersection = (pred_inds[target_inds]).long().sum().data.cpu()[0]
        union = pred_inds.long().sum().data.cpu()[0] + target_inds.long().sum().data.cpu()[0] - intersection
        if union == 0:
            ious.append(float('nan'))
        else:
            ious.append(float(intersection) / float(max(union, 1)))
        return np.array(ious)


def save_checkpoint(state_dict, is_best, save_path, filename):
    filename = os.path.join(save_path, filename)
    torch.save(state_dict, filename)
    if is_best:
        bestname = os.path.join(save_path, 'model_best.pth.tar')
        shutil.copyfile(filename, bestname)


def class_color_mapping(pred):
    # map semantic segmentation class [0,3] into color
    class_color_map = np.array([(0, 0, 0), (0, 255, 0), (255, 0, 0), (0, 0, 255)])
    return class_color_map[pred]


def hex_to_rgb(hex):
    hex = hex.lstrip('#')
    red, green, blue = bytes.fromhex(hex)
    return red, green, blue


def multi_acc(pred, label):
    pred = torch.argmax(pred, dim=1)
    corrects = (pred == label).float()
    acc = corrects.sum() / corrects.numel()
    acc = acc * 100
    return acc


def train_val_split(dataset, val_split=0.15):
    length = len(dataset)
    train_set, val_set = random_split(dataset,
                                      [int(round(length * (1 - val_split))),
                                       int(round(length * val_split))])

    return train_set, val_set


def gather_feature(fmap, index):
    # for topk:
    # index is of the shape [B, K]
    # fmap: [B, HxW, 2]
    # dim: 2
    dim = fmap.size(-1)
    # index now if of the shape [B, K, 2]
    index = index.unsqueeze(-1).expand(*index.shape, dim)
    fmap = fmap.gather(dim=1, index=index)

    return fmap


def get_centers(bbox, xywh=False):
    '''
    :param bbox: bboxes coordinates [x_left_top, y_left_top, x_right_bottom, y_right_bottom]
    :param xywh: if true, centers coordinates will have the form : [left_x, left_y, w, h]
    '''
    if xywh:
        x = bbox[:, 0] + bbox[:, 2] / 2
        y = bbox[:, 1] + bbox[:, 3] / 2
        x = x.unsqueeze(-1)
        y = y.unsqueeze(-1)

        return torch.cat((x, y), dim=1)

    else: return (bbox[:, :2] + bbox[:, 2:]) / 2  # x1y1 x2y2


# functions cornernet_gaussian_radius, gaussian_radius, gaussian2D, draw_umich_gaussian
# are from https://github.com/xingyizhou/CenterNet
def cornernet_gaussian_radius(det_size, min_overlap=0.7):
    # calculation of radius, original implementation according to cornernet.
    width, height = det_size[:, 0], det_size[:, 1]

    a1 = 1
    b1 = (height + width)
    c1 = width * height * (1 - min_overlap) / (1 + min_overlap)
    sq1 = np.sqrt(b1 ** 2 - 4 * a1 * c1)
    r1 = (b1 + sq1) / 2

    a2 = 4
    b2 = 2 * (height + width)
    c2 = (1 - min_overlap) * width * height
    sq2 = np.sqrt(b2 ** 2 - 4 * a2 * c2)
    r2 = (b2 + sq2) / 2

    a3 = 4 * min_overlap
    b3 = -2 * min_overlap * (height + width)
    c3 = (min_overlap - 1) * width * height
    sq3 = np.sqrt(b3 ** 2 - 4 * a3 * c3)
    r3 = (b3 + sq3) / 2
    return torch.min(r1, torch.min(r2, r3))


def gaussian_radius(det_size, min_overlap=0.7):
    width, height = det_size[:, 0], det_size[:, 1]

    return torch.sqrt(width * height * min_overlap / 4)


def gaussian2D(radius, sigma=1):
    y, x = np.ogrid[-radius:radius + 1, -radius:radius + 1]
    h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
    # eps, the minimum of the type of data h.dtype
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    return h


def draw_umich_gaussian(heat_map, center, radius, k=1):
    diameter = 2 * radius + 1
    gaussian = gaussian2D(radius, sigma=diameter / 6)
    gaussian = torch.Tensor(gaussian)

    x, y = int(center[0]), int(center[1])
    height, width = heat_map.shape[0:2]

    left, right = min(x, radius), min(width - x, radius + 1)
    top, bottom = min(y, radius), min(height - y, radius + 1)

    masked_heat_map = heat_map[y - top:y + bottom, x - left:x + right]
    masked_gaussian = gaussian[radius - top:radius + bottom, radius - left:radius + right]

    if min(masked_gaussian.shape) > 0 and min(masked_heat_map.shape) > 0:
        masked_heat_map = torch.max(masked_heat_map, masked_gaussian * k)
        heat_map[y - top:y + bottom, x - left:x + right] = masked_heat_map


def read_cityscapes_properties(img_dir, gt_dir):
    tmp = []
    cities = os.listdir(img_dir)
    for city in cities:
        city_img_dir = os.path.join(img_dir, city)
        city_gt_dir = os.path.join(gt_dir, city)
        for basename in os.listdir(city_img_dir):
            prefix = basename.split('_')[:3]
            # only for leftImg. For rightImg, modify the suffix.
            img_name = prefix[0] + '_' + prefix[1] + '_' + prefix[2] + '_' + "leftImg8bit.png"
            lbl_name = prefix[0] + '_' + prefix[1] + '_' + prefix[2] + '_' + "gtFine_labelIdsnew.png"
            json_name = prefix[0] + '_' + prefix[1] + '_' + prefix[2] + '_' + "gtFine_polygonsnew.json"

            img_file = os.path.join(city_img_dir, img_name)
            lbl_file = os.path.join(city_gt_dir, lbl_name)
            json_file = os.path.join(city_gt_dir, json_name)

            img_shape = cv2.imread(img_file, 1).shape
            tmp.append([img_shape[0], img_shape[1],
                        img_shape[0] / img_shape[1],
                        img_shape[2], img_file,
                        lbl_file, json_file])

    df = pd.DataFrame(tmp, columns=['img_height', 'img_width',
                                    'img_ratio', 'num_channels',
                                    'image_path', 'sem_gt_path',
                                    'det_gt_path'])
    return df


def read_a2d2_properties(root_path):
    tmp = []
    img_files = sorted(glob.glob(os.path.join(root_path, '*/camera/*/*.png')))
    lbl_files = sorted(glob.glob(os.path.join(root_path, '*/label/*/*new.png')))
    files = list(zip(img_files, lbl_files))

    for img_file, lbl_file in files:
        img_shape = cv2.imread(img_file, 1).shape
        tmp.append([img_shape[0], img_shape[1],
                    img_shape[0] / img_shape[1],
                    img_shape[2], img_file,
                    lbl_file])

    df = pd.DataFrame(tmp, columns=['img_height', 'img_width',
                                    'img_ratio', 'num_channels',
                                    'image_path', 'sem_gt_path'])
    return df


def data_visualization(df):
    fig, axs = plt.subplots(4, 4, figsize=(20, 20))
    for i in range(4):
        for j in range(2):
            n = np.random.randint(0, len(df))
            axs[i, j * 2].imshow(cv2.imread(df['image_path'].loc[n]), 1)
            axs[i, j * 2].set_title('{}. image'.format(n))
            axs[i, j * 2 + 1].imshow(cv2.imread(df['sem_gt_path'].loc[n]), 1, cmap='gray')
            axs[i, j * 2 + 1].set_title('{}. mask'.format(n))


def fixed_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def denormalize(img, mean, std):
    # denormalize the normalized images using mean value and standard deviation of dataset.
    transforms = T.Compose([T.Normalize(mean=[0., 0., 0.],
                                        std=[1 / std[0], 1 / std[1], 1 / std[2]]),
                           T.Normalize(mean=[-mean[0], -mean[1], -mean[2]],
                                       std=[1., 1., 1.])])
    return transforms(img)


def draw_pred_bboxes(img, bboxes, cls, scores, thres=0.2, dataset='a2d2'):
    '''
    draw bboxes and their classes and scores on image

    :param img: [C, H ,W]
    :param bboxes: [num_bboxes]
    :param cls: [num_bboxes]
    :param scores: [num_bboxes]
    :param thres: threshold, bbox with score lower
     than this value will not be drawed
    '''
    img = np.array(img.permute(1,2,0).cpu().detach() * 255, dtype=np.uint8)
    img = np.ascontiguousarray(img)

    if dataset=='cityscapes':
        cls_color_map = [(120, 176, 103), (120, 176, 103), (211, 80, 248), (211, 80, 248),
                         (187, 111, 173), (187, 111, 173), (159, 98, 92), (159, 98, 92),
                         (209, 59, 74), (209, 59, 74), (125, 86, 253), (236, 124, 23),
                         (236, 124, 23), (165, 175, 116), (165, 175, 116), (218, 241, 158)]

        cls_name_map = ["person", "person group",  "rider", "rider group",
                        "car", "car group", "truck", "truck group", "bus",
                        "bus group", "on rails", "motorcycle", "motorcycle group",
                        "bicycle", "bicycle group", "traffic light"]

    else:
        cls_color_map = [(120, 176, 103), (211, 80, 248), (187, 111, 173), (159, 98, 92),
                         (209, 59, 74), (125, 86, 253), (236, 124, 23), (99, 197, 21), (76, 203, 180)]

        cls_name_map = ["car", "pedestrian", "truck", "small vehicles",
                        "utility vehicle", "bicycle",  "tractor",
                        "traffic signal", "traffic sign"]

    # heat map size is 1 /4 of input size
    bboxes = bboxes * 4

    for i, bbox in enumerate(bboxes):
        score = scores[i].item()
        if score < thres:
            continue
        color = cls_color_map[int(cls[i].item())]
        label_name = cls_name_map[int(cls[i].item())]
        score_str = str(float('%.2f' % score))
        label = label_name + ' ' + score_str

        img = cv2.rectangle(img, (bbox[0], bbox[1]),
                            (bbox[2], bbox[3]), color, 1)
        # traffic signs are small objects and there is a large number of them on image,
        # to better visualize other obejects, we label them as box with text.
        if int(cls[i].item()) != 8:
            img = add_label_to_rectangle(img, label, bbox, color)

    return img


def draw_gt_bboxes(img, bboxes, lbls, num_bboxes, dataset='a2d2'):
# draw gt bboxes on image.
    img = np.array(img.permute(1,2,0).cpu().detach() * 255, dtype=np.uint8)
    img = np.ascontiguousarray(img)

    if dataset=='cityscapes':
        cls_color_map = [(120, 176, 103), (120, 176, 103), (211, 80, 248), (211, 80, 248),
                         (187, 111, 173), (187, 111, 173), (159, 98, 92), (159, 98, 92),
                         (209, 59, 74), (209, 59, 74), (125, 86, 253), (236, 124, 23),
                         (236, 124, 23), (165, 175, 116), (165, 175, 116), (218, 241, 158)]

    else:
        cls_color_map = [(120, 176, 103), (211, 80, 248), (187, 111, 173), (159, 98, 92),
                         (209, 59, 74), (125, 86, 253), (236, 124, 23), (99, 197, 21), (76, 203, 180)]

    bboxes = bboxes.resize_(num_bboxes, 4)
    lbls = lbls.resize_(num_bboxes)

    for i, bbox in enumerate(bboxes):
        color = cls_color_map[int(lbls[i].item())]
        img = cv2.rectangle(img, (bbox[0], bbox[1]),
                            (bbox[2], bbox[3]), color, 1)

    return img


def add_label_to_rectangle(img, label, bbox, bg_color, text_color=(0, 0, 0)):
    '''
    adds label, inside or outside the rectangle

    :param img: the image on which the label is to be written
    :param label: contains both class and score
    :param bbox: a single bounding box
    :param bg_color: backgound color of the text field
    '''

    text_width = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.3, 1)[0][0]
    label_bg = [bbox[0], bbox[1], bbox[0] + text_width, bbox[1] - 10]
    cv2.rectangle(img, (label_bg[0], label_bg[1]),
                  (label_bg[2] + 5, label_bg[3]), bg_color, -1)

    cv2.putText(img, label, (bbox[0] + 2, bbox[1] - 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.3, text_color, 1)

    return img


def mean_average_precision(batched_preds, batched_gt, num_classes=9):
    metric_fn = MetricBuilder.build_evaluation_metric("map_2d",
                                                      async_mode=True,
                                                      num_classes=num_classes)
    bboxes, cls, scores = batched_preds
    batched_preds = torch.cat((bboxes * 4, cls, scores), dim=2).detach().cpu().numpy()

    for i, gt in enumerate(batched_gt['bboxes']):
        gt_bboxes = gt.resize_(batched_gt['num_bboxes'][i], 4)
        gt_label = batched_gt['lbls'][i].unsqueeze(-1).resize_(batched_gt['num_bboxes'][i], 1)
        gt_diffcult = torch.zeros_like(gt_label)
        gt_crowd = torch.zeros_like(gt_label)
        gt = torch.cat([gt_bboxes, gt_label, gt_diffcult, gt_crowd], dim=1)
        gt = gt.numpy()

        preds = batched_preds[i]

        metric_fn.add(preds, gt)
    # iou_thresholds=np.arange(0.5, 1.0, 0.05), recall_thresholds=np.arange(0., 1.01, 0.01), mpolicy='soft'
    # print(f"COCO mAP: {metric_fn.value([0.5, 1.0, 0.05], np.arange(0., 1.01, 0.01), 'soft')['mAP']}")
    map = metric_fn.value([0.5, 1.0, 0.05], np.arange(0., 1.01, 0.01), 'soft')['mAP']
    return map


def rmse(pred, label):
    pred = pred.squeeze(1)
    valid_mask = torch.logical_and(label > 0.0, label < 1.0)
    criterion = nn.MSELoss()
    loss = torch.sqrt(criterion(pred[valid_mask], label[valid_mask]))
    return loss


def apply_color_map(depth_map, color_map=cv2.COLORMAP_MAGMA):
    depth_map = depth_map.detach().squeeze().float().cpu().numpy()
    depth_map = 255 - depth_map * 255
    depth_map = cv2.applyColorMap(depth_map.astype(np.uint8), color_map)
    depth_map = cv2.cvtColor(depth_map, cv2.COLOR_BGR2RGB)
    return depth_map


def mkdir(save_path):
    if not os.path.exists(save_path):
        raise Exception(f"The folder or path: \n{save_path}\n is not exist, please check the path or create the folder!")

    time = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    path = os.path.join(save_path, time)
    if not os.path.exists(path):
        os.mkdir(path)
        print('create new folder' + str(path))
    else:
        print('folder exist')
    
    return path


