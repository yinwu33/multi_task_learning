import torch
import cv2
import numpy as np
from utils.utils import get_centers, gaussian_radius, draw_umich_gaussian, draw_gt_bboxes, hex_to_rgb
from configs.cfg import args
import matplotlib.pyplot as plt
import torchvision.transforms as T


def centers_gt_gen_cs(gt, num_classes=16, max_objs=130):
    '''
    generate points from bboxes

    :param gt: groundtruth bounding box label
    :param num_classes: 12
    :param max_objs: maximal objects number
    :return: gt_dict, dictionary of groudtruth
    heat_map: generated heat_map of the shape [num_classes, H, W]
    wh: generated width and height of the shape [2, H, W]
    reg: generated regression of the shape [2, H, W]
    '''
    # [H ,W]
    gt_heat_map = torch.zeros(num_classes, args['input_size'][1] // 4, args['input_size'][0] // 4)
    gt_wh = torch.zeros(max_objs, 2)
    gt_reg = torch.zeros(max_objs, 2)
    mask = torch.zeros(max_objs)
    index = torch.zeros(max_objs, dtype=torch.int64)
    # absolute ratio
    h_ratio = args['input_size'][1] // 4 / 950
    w_ratio = args['input_size'][0] // 4 / 1900
    # [num_bboxes, 4]
    bboxes = gt["bboxes"]
    cls = gt["lbls"]
    num_bboxes = len(bboxes)
    if num_bboxes != 0:
        bboxes = torch.tensor(bboxes, dtype=torch.float32)
        cls = torch.tensor(cls, dtype=torch.float32)
        bboxes[:, 0::2] *= w_ratio
        bboxes[:, 1::2] *= h_ratio

        centers = get_centers(bboxes)
        centers_int = centers.int()
        gt_reg[:num_bboxes] = centers - centers_int
        # the index-th pixel of the image
        index[:num_bboxes] = centers_int[:, 1] * args['input_size'][0] // 4 + centers_int[:, 0]
        mask[:num_bboxes] = 1

        wh = torch.zeros_like(centers)
        wh[:, 0] = bboxes[:, 2] - bboxes[:, 0]
        wh[:, 1] = bboxes[:, 3] - bboxes[:, 1]
        gt_wh[:num_bboxes] = wh

        radius = gaussian_radius(gt_wh)
        radius = torch.clamp_min(radius, 0)
        radius = radius.type(torch.int).numpy()

        for i in range(num_bboxes):
            # id of i-th class
            id = cls[i]
            draw_umich_gaussian(gt_heat_map[id.type(torch.long)],
                                centers_int[i], radius[i])

    gt_dict = {
        "heat_map": gt_heat_map,
        "wh": gt_wh,
        "reg": gt_reg,
        "mask": mask,
        "index": index
    }

    return gt_dict


def centers_gt_gen_a2d2(gt, num_classes=9, max_objs=80):
    '''
    generate points from bboxes

    :param gt: groundtruth bounding box label
    :param num_classes: 11
    :param max_objs: maximal objects number
    :return: gt_dict, dictionary of groudtruth
    score_map: generated heat_map of the shape [num_classes, H, W]
    wh: generated width and height of the shape [2, H, W]
    reg: generated regression of the shape [2, H, W]
    '''
    gt_heat_map = torch.zeros(num_classes, args['input_size'][1] // 4, args['input_size'][0] // 4)
    gt_wh = torch.zeros(max_objs, 2)
    gt_reg = torch.zeros(max_objs, 2)
    mask = torch.zeros(max_objs)
    index = torch.zeros(max_objs, dtype=torch.int64)
    # absolute ratio
    h_ratio = args['input_size'][1] // 4 / 960
    w_ratio = args['input_size'][0] // 4 / 1920
    # [num_bboxes, 4]
    bboxes = gt["bboxes"]
    cls = gt["lbls"]
    num_bboxes = len(bboxes)
    if num_bboxes != 0:
        bboxes = torch.tensor(bboxes, dtype=torch.float32)
        cls = torch.tensor(cls, dtype=torch.float32)
        bboxes[:, 0::2] *= w_ratio
        bboxes[:, 1::2] *= h_ratio

        centers = get_centers(bboxes, xywh=True)
        centers_int = centers.int()
        gt_reg[:num_bboxes] = centers - centers_int # Runtimeerror
        # the index-th pixel of the image
        index[:num_bboxes] = centers_int[:, 1] * args['input_size'][0] // 4 + centers_int[:, 0]
        mask[:num_bboxes] = 1

        wh = torch.zeros_like(centers)
        wh[:, 0] = bboxes[:, 2]
        wh[:, 1] = bboxes[:, 3]
        gt_wh[:num_bboxes] = wh

        radius = gaussian_radius(gt_wh)
        radius = torch.clamp_min(radius, 0)
        radius = radius.type(torch.int).numpy()

        for i in range(num_bboxes):
            # id of i-th class
            id = cls[i]
            draw_umich_gaussian(gt_heat_map[id.type(torch.long)],
                                centers_int[i], radius[i])

    gt_dict = {
        "heat_map": gt_heat_map,
        "wh": gt_wh,
        "reg": gt_reg,
        "mask": mask,
        "index": index
    }

    return gt_dict


def gt_gen_cs(gt, max_objs=130):
    '''
    process groundtruch label and generate bounding boxes for visualization
    :param gt: groundtruth label
    '''
    gt_bboxes = torch.zeros(max_objs, 4)
    gt_lbls = torch.zeros(max_objs)
    index = torch.zeros(max_objs, dtype=torch.int64)
    # absolute ratio
    h_ratio = args['input_size'][1] / 950
    w_ratio = args['input_size'][0] / 1900
    # [num_bboxes, 4]
    bboxes = gt["bboxes"]
    cls = gt["lbls"]
    num_bboxes = len(bboxes)
    if num_bboxes != 0:
        bboxes = torch.tensor(bboxes, dtype=torch.float32)
        cls = torch.tensor(cls, dtype=torch.float32)
        bboxes[:, 0::2] *= w_ratio
        bboxes[:, 1::2] *= h_ratio
        gt_bboxes[:num_bboxes] = bboxes
        gt_lbls[:num_bboxes] = cls

    gt_dict = {
        "bboxes": gt_bboxes,
        "lbls": gt_lbls,
        "num_bboxes" : num_bboxes
    }

    return gt_dict


# def empty_gt_gen(num_classes=9, max_objs=80):
#     gt_heat_map = torch.zeros(num_classes, args['input_size'][0] // 4, args['input_size'][1] // 4)
#     gt_wh = torch.zeros(max_objs, 2)
#     gt_reg = torch.zeros(max_objs, 2)
#     mask = torch.zeros(max_objs)
#     index = torch.zeros(max_objs, dtype=torch.int64)
#
#     gt_dict = {
#         "heat_map": gt_heat_map,
#         "wh": gt_wh,
#         "reg": gt_reg,
#         "mask": mask,
#         "index": index
#     }
#
#     return gt_dict


def gt_gen_a2d2(gt, max_objs=80):
    gt_bboxes = torch.zeros(max_objs, 4)
    gt_lbls = torch.zeros(max_objs)
    index = torch.zeros(max_objs, dtype=torch.int64)
    # absolute ratio
    h_ratio = args['input_size'][1] / 960
    w_ratio = args['input_size'][0] / 1920
    # [num_bboxes, 4]
    bboxes = gt["bboxes"]
    cls = gt["lbls"]
    num_bboxes = len(bboxes)
    if num_bboxes != 0:
        bboxes = torch.tensor(bboxes, dtype=torch.float32)
        cls = torch.tensor(cls, dtype=torch.float32)
        bboxes[:, 0::2] *= w_ratio
        bboxes[:, 1::2] *= h_ratio
        bboxes[:, 2] = bboxes[:, 0] + bboxes[:, 2]
        bboxes[:, 3] = bboxes[:, 1] + bboxes[:, 3]

        gt_bboxes[:num_bboxes] = bboxes
        gt_lbls[:num_bboxes] = cls

    gt_dict = {
        "bboxes": gt_bboxes,
        "lbls": gt_lbls,
        "num_bboxes" : num_bboxes
    }

    return gt_dict


def test():
    img_file = '/media/Data/dataset/a2d2/camera_lidar_semantic_bboxes/20180925_101535/camera/cam_front_center/20180925101535_camera_frontcenter_000027489.png'
    lbl_file = '/media/Data/dataset/a2d2/camera_lidar_semantic_bboxes/20180925_101535/label/cam_front_center/20180925101535_label_frontcenter_000027489.png'
    label = {
        'img_file': img_file,
        'lbl_file': lbl_file
    }
    img = cv2.cvtColor(cv2.imread(label['img_file']),
                       cv2.COLOR_BGR2RGB)

    label = cv2.cvtColor(cv2.imread(label['lbl_file']),
                         cv2.COLOR_BGR2RGB)
    img = T.Compose([T.ToTensor()])(img)

    a2d2_color_det = {
        "#ff0000": 0, "#c80000": 0, "#960000": 0, "#800000": 0,
        "#b65906": 1, "#963104": 1, "#5a1e01": 1, "#5a1e1e": 1,
        "#cc99ff": 2, "#bd499b": 2, "#ef59bf": 2, "#ff8000": 3,
        "#c88000": 3, "#968000": 3, "#00ff00": 4, "#00c800": 4,
        "#009600": 4, "#0080ff": 5, "#1e1c9e": 5, "#3c1c64": 5,
        "#00ffff": 6, "#1edcdc": 6, "#3c9dc7": 6, "#ffff00": 7,
        "#ffffc8": 7, "#e96400": 10, "#6e6e00": 10, "#808000": 10,
        "#ffc125": 10, "#400040": 10, "#b97a57": 10, "#000064": 8,
        "#8b636c": 10, "#d23273": 10, "#ff0080": 10, "#fff68f": 10,
        "#960096": 10, "#ccff99": 10, "#eea2ad": 10, "#212cb1": 10,
        "#b432b4": 10, "#ff46b9": 9, "#eee9bf": 10, "#93fdc2": 10,
        "#9696c8": 10, "#b496c8": 10, "#48d1cc": 10, "#c87dd2": 10,
        "#9f79ee": 10, "#8000ff": 10, "#ff00ff": 10, "#87ceff": 10,
        "#f1e6ff": 10, "#60458f": 10, "#352e52": 10
    }

    label = np.array(label)
    gt_dict = {"bboxes": [], "lbls": []}
    for key, value in a2d2_color_det.items():
        # only the first 9 valid classes
        if value != 10:
            binary_img = np.zeros(label.shape[0:2], dtype=np.uint8)
            index = np.where(np.all(label == hex_to_rgb(key), axis=-1))
            binary_img[index] = 1
            # stats: statistics output for each label, including the background label(first row)
            # [x, y, w, h, area]
            # TODO, threshold for area of components
            _, _, stats, _ = cv2.connectedComponentsWithStats(binary_img,
                                                              connectivity=8,
                                                              ltype=None)
            # first row is background stats
            for i in range(1, len(stats)):
                if value == 0 or value == 3:
                    # if boxes are of car or truck class, we filter out those who are unlikely detected
                    if stats[i][4] > 40 and stats[i][2] / stats[i][3] < 2 and stats[i][3] / stats[i][2] < 2:
                        # print(stats[i][:4])
                        gt_dict["bboxes"].append(stats[i][:4])  # *stats[1:, :5])
                        gt_dict["lbls"].append(value)
                else:
                    gt_dict["bboxes"].append(stats[i][:4])  # *stats[1:, :5])
                    gt_dict["lbls"].append(value)
    gt = gt_gen_a2d2(gt_dict, max_objs=80)
    img = draw_gt_bboxes(img, gt['bboxes'], gt['lbls'], gt['num_bboxes'])
    plt.imshow(img)
    plt.show()


if __name__ == '__main__':
    test()


