import os
import cv2
import json
import glob
import numpy as np
from utils.utils import hex_to_rgb
import torch
import torch.utils.data as data
from datasets.cityscapes import Cityscapes
from datasets.a2d2 import A2D2
from tqdm import tqdm


def cal_mean_std(dataset):
    # calculation of mean and standard variance for normalization.
    print("calculating...")

    if dataset == "cityscapes":
        dataset = Cityscapes(mode='train', task='depth', transform=False, train_extra=True)
    else:
        dataset = A2D2(task='sem_seg', transform=False)

    loader = data.DataLoader(dataset,
                             batch_size=32,
                             num_workers=8,
                             shuffle=False)
    with tqdm(total=len(loader)) as bar1:
        mean = 0.0
        for images, _ in loader:
            batch_samples = images.size(0)
            images = images.view(batch_samples, images.size(1), -1)
            mean += images.mean(2).sum(0)
            bar1.update(1)

        mean = mean / len(loader.dataset)

    with tqdm(total=len(loader)) as bar2:
        var = 0.0
        for images, _ in loader:
            batch_samples = images.size(0)
            images = images.view(batch_samples, images.size(1), -1)
            var += ((images - mean.unsqueeze(1)) ** 2).sum([0, 2])
            bar2.update(1)

        std = torch.sqrt(var / (len(loader.dataset) * 512 * 256))

    print(mean)
    print(std)

    return mean, std


def cal_alpha_beta(train_extra=False):
    # calculate the maximum and minimum depth label of Cityscape dataset
    # result: alpha=1, beta=126, gamma=0.0
    alpha = 10000.
    beta = 0.
    #TODO train_extra
    if train_extra:
        pass
    for mode in ['train', 'val', 'test']:
        print("calculating, current: '{}' split".format(mode))
        datasets = Cityscapes(mode=mode, train_extra=train_extra)
        files = datasets.file_list
        for file in files:
            depth_label_file = file['depth_reg_lbl_file']
            assert os.path.exists(depth_label_file), "file not found: {}".format(depth_label_file)
            depth_label = cv2.imread(depth_label_file, 0)
            # values greater than 96 are noise, not actual depth label
            mask = np.logical_and(depth_label>0., depth_label<96.0)
            valid_depth = depth_label[mask]
            tmp_max = np.max(valid_depth)
            tmp_min = np.min(valid_depth)
            alpha = min(tmp_min, alpha)
            beta = max(tmp_max, beta)

    print("alpha={}, beta={}, gamma={}".format(alpha, beta, 1.0 - alpha))


def convert(gt_path, dataset):
    assert (dataset == 'cityscapes' or dataset == 'a2d2'), 'dataset should be either cityscapes or a2d2'
    # 0 for background, 1 for drivable, 2 for non-drivable, 3 for lane-marking.
    convert_dict_cs = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 1, 8: 2, 9: 1, 10: 2, 11: 0, 12: 0,
                       13: 0, 14: 0, 15: 0, 16: 0, 17: 0, 18: 0, 19: 0, 20: 0, 21: 0, 22: 0, 23: 0,
                       24: 0, 25: 0, 26: 0, 27: 0, 28: 0, 29: 0, 30: 0, 31: 0, 32: 0, 33: 0, -1: 0}

    convert_dict_a2d2 = {"#ff0000": 0, "#c80000": 0, "#960000": 0, "#800000": 0, "#b65906": 0, "#963204": 0,
                         "#5a1e01": 0, "#5a1e1e": 0, "#cc99ff": 0, "#bd499b": 0, "#ef59bf": 0, "#ff8000": 0,
                         "#c88000": 0, "#968000": 0, "#00ff00": 0, "#00c800": 0, "#009600": 0, "#0080ff": 0,
                         "#1e1c9e": 0, "#3c1c64": 0, "#00ffff": 0, "#1edcdc": 0, "#3c9dc7": 0, "#ffff00": 0,
                         "#ffffc8": 0, "#e96400": 0, "#6e6e00": 0, "#808000": 2, "#ffc125": 3, "#400040": 0,
                         "#b97a57": 0, "#000064": 0, "#8b636c": 2, "#d23273": 0, "#ff0080": 0, "#fff68f": 0,
                         "#960096": 0, "#ccff99": 0, "#eea2ad": 0, "#212cb1": 0, "#b432b4": 1, "#ff46b9": 0,
                         "#eee9bf": 2, "#93fdc2": 0, "#9696c8": 1, "#b496c8": 2, "#48d1cc": 0, "#c87dd2": 0,
                         "#9f79ee": 0, "#8000ff": 3, "#ff00ff": 1, "#87ceff": 0, "#f1e6ff": 0, "#60458f": 0,
                         "#352e52": 0}

    if dataset == 'cityscapes':
        gt = cv2.imread(gt_path, 0)
        converted_gt = np.zeros_like(gt)

        for key in convert_dict_cs.keys():
            index = np.where(gt == key)
            converted_gt[index] = convert_dict_cs[key]

    else:
        gt = cv2.imread(gt_path, 1)
        gt = cv2.cvtColor(gt, cv2.COLOR_BGR2RGB)
        converted_gt = np.zeros((gt.shape[0], gt.shape[1]), dtype=np.uint8)

        for key in convert_dict_a2d2.keys():
            index = np.where(np.all(gt == hex_to_rgb(key), axis=-1))
            converted_gt[index] = convert_dict_a2d2[key]

    converted_gt_path = gt_path.split('.')[0] + 'new' + '.' + gt_path.split('.')[1]
    cv2.imwrite(converted_gt_path, converted_gt)


def label_conversion(gt_dir, dataset):
    """
    example of gt_dir for cityscapes dataset:
    your_path/leftImg8bit_trainvaltest/leftImg8bit/train
    for a2d2 dataset: your/path/camera_lidar_semantic
    """
    assert (dataset == 'cityscapes' or dataset == 'a2d2'), 'dataset should be either cityscapes or a2d2'
    print("converting...")
    if dataset == 'cityscapes':
        cities = os.listdir(gt_dir)
        for city in cities:
            city_gt_dir = os.path.join(gt_dir, city)
            for basename in os.listdir(city_gt_dir):
                prefix = basename.split('_')[:3]
                gt_name = prefix[0] + '_' + prefix[1] + '_' + prefix[2] + '_' + "gtFine_labelIds.png"
                gt_path = os.path.join(city_gt_dir, gt_name)
                convert(gt_path, dataset)
    else:
        gt_files = sorted(glob.glob(os.path.join(gt_dir, '*/label/*/*.png')))
        for gt_path in gt_files:
            convert(gt_path, dataset)
    print("conversion finished")


def lbl_to_id(label):
    id_dict ={
        "person": 0, "rider": 1, "car": 2, "truck": 3, "bus": 4,
        "on rails": 5, "motorcycle": 6, "bicycle": 7, "caravan": 8,
        "trailer": 9, "car group": 10, "traffic light": 11
    }

    return id_dict[label]


def polygon_to_bbox(gt_path):
    classes = ["person", "rider", "car", "truck", "bus",
                "on rails", "motorcycle", "bicycle",
                "caravan", "trailer", "car group", "traffic light"]
    with open(gt_path) as f:
        gt = json.load(f)

    bboxes = []
    lbls = []
    new_dict = {"imgHeight": gt["imgHeight"], "imgWidth": gt["imgWidth"]}

    for object in gt["objects"]:
        lbl = object["label"]
        if lbl not in classes:
            continue

        lbl = lbl_to_id(lbl)

        vertex = object["polygon"]
        vertex_x, vertex_y = list(zip(*vertex))

        left_x = int(min(vertex_x))
        left_y = int(min(vertex_y))
        right_x = int(max(vertex_x))
        right_y = int(max(vertex_y))

        bbox = [left_x, left_y, right_x, right_y]

        bboxes.append(bbox)
        lbls.append(lbl)

    new_dict["bboxes"] = bboxes
    new_dict["lbls"] = lbls
    converted_gt_path = gt_path.split('.')[0] + 'new' + '.' + gt_path.split('.')[1]
    with open(converted_gt_path, 'w') as f:
        json.dump(new_dict, f)


# Note that it converts only one specific split of datasets, for example traindataset.
def polygon_conversion(gt_path):
    print("converting...")
    gt_files = sorted(glob.glob(os.path.join(gt_path, '*/*polygons.json')))
    for gt in gt_files:
        polygon_to_bbox(gt)
    print("conversion finished")


# # compute mean and std values of a dataset for normalization.
# if __name__ == '__main__':
#     root_path = "/disk/users/hb662/camera_lidar_semantic/"
#     files = A2D2.load_a2d2_sem_seg(root_path)
#     mean, std = cal_mean_std(files, 'a2d2')
#     print(mean, std)

if __name__ == '__main__':
    cal_mean_std('a2d2')
