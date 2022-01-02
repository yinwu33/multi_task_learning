import os
import cv2
import torch
import json
import random
import numpy as np
import torchvision.transforms as T
from glob import glob
from torch.utils.data import Dataset
from configs.cfg import args
from utils.utils import hex_to_rgb
from datasets.gt_processing import *
from tqdm import tqdm


class A2D2(Dataset):
    def __init__(self, transform=True, task='multi_task',
                 partial=None, max_objs=120):
        """A2D2 dataset

        Args:
            mode (str, optional): 'train', 'val', 'test'. Defaults to 'train'.
            transform (bool, optional): [description]. Defaults to True.
        """
        self.task = task
        self.max_objs = max_objs
        self.transform = transform
        self.transforms = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=[0.4499, 0.4365, 0.4364],
                        std=[0.1902, 0.1972, 0.2085])
        ])
        self.data = A2D2.load_files()
        # random.seed(40)
        random.shuffle(self.data)
        self._lst = self.data[:30000]
        self._lst_test = self.data[30000:]

        if task == 'multi_task' or task == 'obj_det' or task == 'test':
            # get the path of label file and read bboxes from it.
            label_path = args['a2d2_label_path']
            with open(label_path, 'r') as f:
                gt = json.load(f)
                self.det_label = gt

        if partial is not None:
            random.shuffle(self._lst)
            self._lst = self._lst[1000:1000+partial]

    def __getitem__(self, index):

        if self.task == "test":
            img = cv2.cvtColor(cv2.imread(self._lst_test[index]['img_file']),
                               cv2.COLOR_BGR2RGB)
            # crop so that images from a2d2 are more similar with those from cityscapes
            img = img[248:1208, :]
            img = cv2.resize(img, args['input_size'], cv2.INTER_LINEAR)
            img = self.transforms(img)
            label = cv2.cvtColor(cv2.imread(self._lst[index]['lbl_file']),
                                 cv2.COLOR_BGR2RGB)
            label = cv2.resize(label, args['input_size'], interpolation=cv2.INTER_NEAREST)
            sem_label = A2D2.lbl_convert(label)
            sem_label = torch.from_numpy(sem_label).squeeze().long()

            img_name = self._lst[index]['img_file'].split('/')[-1]
            bboxes = self.det_label[img_name]
            label_map = gt_gen_a2d2(bboxes, max_objs=self.max_objs)
            det_label = centers_gt_gen_a2d2(bboxes, max_objs=self.max_objs)

            return img, sem_label, det_label, label_map

        img = cv2.cvtColor(cv2.imread(self._lst[index]['img_file']),
                           cv2.COLOR_BGR2RGB)
        # crop so that images from a2d2 are more similar with those from cityscapes
        img = img[248:1208, :]
        img = cv2.resize(img, args['input_size'], cv2.INTER_LINEAR)

        if self.transform:
            img = self.transforms(img)
        else:
            img = T.Compose([T.ToTensor()])(img)

        if self.task == 'sem_seg':
            label = cv2.cvtColor(cv2.imread(self._lst[index]['lbl_file']),
                                 cv2.COLOR_BGR2RGB)
            label = label[248:1208, :]
            label = cv2.resize(label, args['input_size'], interpolation=cv2.INTER_NEAREST)
            label = A2D2.lbl_convert(label)
            label = torch.from_numpy(label).squeeze().long()

            return img, label

        if self.task == 'obj_det':
            img_name = self._lst[index]['img_file'].split('/')[-1]
            bboxes = self.det_label[img_name]
            # label for mean average precision calculation
            label_map = gt_gen_a2d2(bboxes, max_objs=self.max_objs)
            det_label = centers_gt_gen_a2d2(bboxes, max_objs=self.max_objs)
            return img, det_label, label_map

        if self.task == 'multi_task':
            label = cv2.cvtColor(cv2.imread(self._lst[index]['lbl_file']),
                                 cv2.COLOR_BGR2RGB)
            label = cv2.resize(label, args['input_size'], interpolation=cv2.INTER_NEAREST)
            sem_label = A2D2.lbl_convert(label)
            sem_label = torch.from_numpy(sem_label).squeeze().long()

            img_name = self._lst[index]['img_file'].split('/')[-1]
            bboxes = self.det_label[img_name]
            det_label = centers_gt_gen_a2d2(bboxes, max_objs=self.max_objs)
            label_map = gt_gen_a2d2(bboxes, max_objs=self.max_objs)

            # depth_label = torch.zeros_like(sem_label)

            return img, sem_label, det_label, label_map

    def __len__(self):
        if self.task == 'test':
            return len(self._lst_test)
        return len(self._lst)

    @staticmethod
    def load_files():
        files = []
        for img_file, lbl_file in A2D2.get_files(args['a2d2_dir']):
            files.append(
                {
                    'img_file': img_file,
                    'lbl_file': lbl_file
                }
            )
        return files

    @staticmethod
    def get_files(root_path):
        img_files = sorted(glob(os.path.join(root_path, '*/camera/cam_front_center/*.png')))
        lbl_files = sorted(glob(os.path.join(root_path, '*/label/cam_front_center/*.png')))
        files = list(zip(img_files, lbl_files))
        return files

    @staticmethod
    def lbl_convert(label):
        a2d2_color_seg = {
            "#ff0000": 0, "#c80000": 0, "#960000": 0, "#800000": 0, "#b65906": 0, "#963204": 0,
            "#5a1e01": 0, "#5a1e1e": 0, "#cc99ff": 0, "#bd499b": 0, "#ef59bf": 0, "#ff8000": 0,
            "#c88000": 0, "#968000": 0, "#00ff00": 0, "#00c800": 0, "#009600": 0, "#0080ff": 0,
            "#1e1c9e": 0, "#3c1c64": 0, "#00ffff": 0, "#1edcdc": 0, "#3c9dc7": 0, "#ffff00": 0,
            "#ffffc8": 0, "#e96400": 0, "#6e6e00": 0, "#808000": 2, "#ffc125": 2, "#400040": 0,
            "#b97a57": 0, "#000064": 0, "#8b636c": 2, "#d23273": 0, "#ff0080": 0, "#fff68f": 0,
            "#960096": 0, "#ccff99": 0, "#eea2ad": 0, "#212cb1": 0, "#b432b4": 1, "#ff46b9": 0,
            "#eee9bf": 2, "#93fdc2": 0, "#9696c8": 1, "#b496c8": 2, "#48d1cc": 1, "#c87dd2": 2,
            "#9f79ee": 0, "#8000ff": 2, "#ff00ff": 1, "#87ceff": 0, "#f1e6ff": 0, "#60458f": 0,
            "#352e52": 0
        }
        label = np.array(label)
        label_convert = np.zeros_like(label, dtype=np.uint8)

        for key, value in a2d2_color_seg.items():
            index = np.where(np.all(label == hex_to_rgb(key), axis=-1))
            label_convert[index] = value
        
        return cv2.cvtColor(label_convert, cv2.COLOR_RGB2GRAY)

    @staticmethod
    def lbl_to_bbox(label, label_path):
        # convert instance masks to bounding boxes if bboxes.json where bboxes info are stored, is not available.

        gt_dict = {"bboxes": [], "lbls": []}

        # "Car":0, "Pedestrian":1, "Truck":2, "Small vehicles":3,
        # "Utility vehicle":4, "Bicycle":5,  "Tractor": 6,
        # "Traffic signal":7, "Traffic sign": 8
        a2d2_color_det = {"#0080ff": 7, "#1e1c9e": 7, "#3c1c64": 7,
                          "#00ffff": 8, "#1edcdc": 8, "#3c9dc7": 8}
        label = np.array(label)
        # additional instance masks label, corresponding semantic segmentation images which do not
        # contain dynamic objects are not included here.
        # instance label files should be placed in camera_lidar_semantic folder.
        label_instance_file = os.path.join(label_path.split('camera_lidar_semantic')[0] +
                                      'camera_lidar_semantic' +
                                      label_path.split('camera_lidar_semantic')[1].replace("label", "instance"))

        if os.path.isfile(label_instance_file):
            label_instance = cv2.imread(label_instance_file, cv2.IMREAD_ANYDEPTH)
            label_instance = label_instance[248:1208, :]
            class_idx = {
                "cars": 1,
                "pedestrians": 2,
                "trucks": 3,
                "smallVehicle": 4,
                "utilityVehicle": 5,
                "bicycle": 6,
                "tractor": 7
            }

            for cls, idx in class_idx.items():
                num_instance = 0
                index = np.where((label_instance // 1024) == idx)
                # in png, instance_id starts at 0, here at 1
                instance_id = label_instance[index] % (idx * 1024)
                if instance_id.size:  # if array not empty
                    num_instance = np.max(instance_id)
                    for id in range(1, num_instance + 1):
                        # pixel which belong to id-th instance of class 'idx'.
                        instance_pts = np.where(label_instance == (idx * 1024 + id))
                        if instance_pts[0].size:  # if not empty
                            left_x = np.min(instance_pts[1])
                            left_y = np.min(instance_pts[0])
                            right_x = np.max(instance_pts[1])
                            right_y = np.max(instance_pts[0])
                            w = right_x - left_x
                            h = right_y - left_y
                            # if boxes are not of pedestrain or bicycle class,
                            # we filter out those who are unlikely detected.
                            if idx == 6 or idx == 2:
                                if w > 4 and h > 5:
                                    # Object of type int64 is not JSON serializable
                                    gt_dict["bboxes"].append([int(left_x), int(left_y), int(w), int(h)])
                                    gt_dict["lbls"].append(int(idx - 1))
                            else:
                                if w * h > 300:
                                    # Object of type int64 is not JSON serializable
                                    gt_dict["bboxes"].append([int(left_x), int(left_y), int(w), int(h)])
                                    gt_dict["lbls"].append(int(idx - 1))

        for key, value in a2d2_color_det.items():
                binary_img = np.zeros(label.shape[0:2], dtype=np.uint8)
                index = np.where(np.all(label == hex_to_rgb(key), axis=-1))
                binary_img[index] = 1
                # stats: statistics output for each label, including the background label(first row)
                # [x, y, w, h, area]
                _, _, stats, _ = cv2.connectedComponentsWithStats(binary_img,
                                                                  connectivity=8)
                # first row is background stats
                for i in range(1, len(stats)):
                    if stats[i][4] > 20:
                        # *stats[1:, :5])
                        gt_dict["bboxes"].append([int(stats[i][0]), int(stats[i][1]),
                                                  int(stats[i][2]), int(stats[i][3])])
                        gt_dict["lbls"].append(int(value))

        return gt_dict


def lbl_to_bbox_json(root_path):
    '''
    convert grundtruth label to bounding boxes and store them in bboxes.json
    :param root_path: path to camera_lidar_semantic label, 'YOURPATH/camera_lidar_semantic'
    :return: None
    '''
    gt = {}

    class_idx = {
        "cars": 1,
        "pedestrians": 2,
        "trucks": 3,
        "smallVehicle": 4,
        "utilityVehicle": 5,
        "bicycle": 6,
        "tractor": 7
    }

    # "Car":0, "Pedestrian":1, "Truck":2, "Small vehicles":3,
    # "Utility vehicle":4, "Bicycle":5,  "Tractor": 6,
    # "Traffic signal":7, "Traffic sign": 8
    a2d2_color_det = {"#0080ff": 7, "#1e1c9e": 7, "#3c1c64": 7,
                      "#00ffff": 8, "#1edcdc": 8, "#3c9dc7": 8}

    files = A2D2.get_files(root_path)
    with tqdm(total=len(files)) as bar:

        for file in files:
            bar.update(1)
            gt_dict = {"bboxes": [], "lbls": []}

            label = cv2.cvtColor(cv2.imread(file[1]),
                                 cv2.COLOR_BGR2RGB)
            label = label[248:1208, :]

            # additional instance masks label, corresponding semantic segmentation images which do not
            # contain dynamic objects are not included here.
            # instance label files should be placed in camera_lidar_semantic folder.
            label_instance_file = os.path.join(file[1].split('camera_lidar_semantic')[0] + 'camera_lidar_semantic' +
                                               file[1].split('camera_lidar_semantic')[1].replace("label", "instance"))

            if os.path.isfile(label_instance_file):
                label_instance = cv2.imread(label_instance_file, cv2.IMREAD_ANYDEPTH)
                label_instance = label_instance[248:1208, :]

                for cls, idx in class_idx.items():
                    num_instance = 0
                    index = np.where((label_instance // 1024) == idx)
                    # in png, instance_id starts at 0, here at 1
                    instance_id = label_instance[index] % (idx * 1024)
                    if instance_id.size:  # if array not empty
                        num_instance = np.max(instance_id)
                        for id in range(1, num_instance + 1):
                            # pixel which belong to id-th instance of class 'idx'.
                            instance_pts = np.where(label_instance == (idx * 1024 + id))
                            if instance_pts[0].size:  # if not empty
                                left_x = np.min(instance_pts[1])
                                left_y = np.min(instance_pts[0])
                                right_x = np.max(instance_pts[1])
                                right_y = np.max(instance_pts[0])
                                w = right_x - left_x
                                h = right_y - left_y
                                # if boxes are not of pedestrain or bicycle class,
                                # we filter out those who are unlikely detected.
                                if idx == 6 or idx == 2:
                                    if w > 4 and h > 5:
                                        # Object of type int64 is not JSON serializable
                                        gt_dict["bboxes"].append([int(left_x), int(left_y), int(w), int(h)])
                                        gt_dict["lbls"].append(int(idx - 1))
                                else:
                                    if w * h > 300:
                                        # Object of type int64 is not JSON serializable
                                        gt_dict["bboxes"].append([int(left_x), int(left_y), int(w), int(h)])
                                        gt_dict["lbls"].append(int(idx - 1))

            for key, value in a2d2_color_det.items():
                binary_img = np.zeros(label.shape[0:2], dtype=np.uint8)
                index = np.where(np.all(label == hex_to_rgb(key), axis=-1))
                binary_img[index] = 1
                # stats: statistics output for each label, including the background label(first row)
                # [x, y, w, h, area]
                _, _, stats, _ = cv2.connectedComponentsWithStats(binary_img,
                                                                  connectivity=8)
                # first row is background stats
                for i in range(1, len(stats)):
                    if stats[i][4] > 20:
                        # *stats[1:, :5])
                        gt_dict["bboxes"].append([int(stats[i][0]), int(stats[i][1]),
                                                  int(stats[i][2]), int(stats[i][3])])
                        gt_dict["lbls"].append(int(value))

            img_name = file[1].split('/')[-1].replace("label", "camera")
            gt[img_name] = gt_dict

            json_path = os.path.join(root_path, 'bboxes.json')
            with open(json_path, 'w') as f:
                f.write(json.dumps(gt))


    # @staticmethod
    # def lbl_to_bbox(label):
    #     # "Car":0, "Bicycle":1, "Pedestrian":2, "Truck":3, "Small vehicles":4, "Traffic signal":5,
    #     # "Traffic sign": 6, "Utility vehicle":7, "Tractor": 8, "Electronic traffic":9
    #     a2d2_color_det = {
    #         "#ff0000": 0, "#c80000": 0, "#960000": 0, "#800000": 0,
    #         "#b65906": 1, "#963104": 1, "#5a1e01": 1, "#5a1e1e": 1,
    #         "#cc99ff": 2, "#bd499b": 2, "#ef59bf": 2, "#ff8000": 3,
    #         "#c88000": 3, "#968000": 3, "#00ff00": 4, "#00c800": 4,
    #         "#009600": 4, "#0080ff": 5, "#1e1c9e": 5, "#3c1c64": 5,
    #         "#00ffff": 6, "#1edcdc": 6, "#3c9dc7": 6, "#ffff00": 7,
    #         "#ffffc8": 7, "#e96400": 10, "#6e6e00": 10, "#808000": 10,
    #         "#ffc125": 10, "#400040": 10, "#b97a57": 10, "#000064": 8,
    #         "#8b636c": 10, "#d23273": 10, "#ff0080": 10, "#fff68f": 10,
    #         "#960096": 10, "#ccff99": 10, "#eea2ad": 10, "#212cb1": 10,
    #         "#b432b4": 10, "#ff46b9": 9, "#eee9bf": 10, "#93fdc2": 10,
    #         "#9696c8": 10, "#b496c8": 10, "#48d1cc": 10, "#c87dd2": 10,
    #         "#9f79ee": 10, "#8000ff": 10, "#ff00ff": 10, "#87ceff": 10,
    #         "#f1e6ff": 10, "#60458f": 10, "#352e52": 10
    #     }
    #     label = np.array(label)
    #     gt_dict = {"bboxes" : [],"lbls" : []}
    #
    #     for key, value in a2d2_color_det.items():
    #         # only the first 9 valid classes
    #         if value != 10:
    #             binary_img = np.zeros(label.shape[0:2], dtype=np.uint8)
    #             index = np.where(np.all(label == hex_to_rgb(key), axis=-1))
    #             binary_img[index] = 1
    #             # stats: statistics output for each label, including the background label(first row)
    #             # [x, y, w, h, area]
    #             _, _, stats, _ = cv2.connectedComponentsWithStats(binary_img,
    #                                                               connectivity=8)
    #             # first row is background stats
    #             for i in range(1, len(stats)):
    #                 if value ==0 or value ==3:
    #                     # if boxes are of car or truck class, we filter out those who are unlikely detected
    #                     if stats[i][4] > 40 and stats[i][2] / stats[i][3] < 2 and stats[i][3] / stats[i][2] < 2:
    #                         # print(stats[i][:4])
    #                         gt_dict["bboxes"].append(stats[i][:4])  # *stats[1:, :5])
    #                         gt_dict["lbls"].append(value)
    #                 else:
    #                     gt_dict["bboxes"].append(stats[i][:4])  # *stats[1:, :5])
    #                     gt_dict["lbls"].append(value)
    #
    #     return gt_dict


if __name__ == '__main__':
    lbl_to_bbox_json('/home/wuyin/data/dataset/a2d2/camera_lidar_semantic')
    # with open('/home/wuyin/data/dataset/a2d2/camera_lidar_semantic/bboxes.json') as f:
    #     gt = json.load(f)
    #     print(len(gt))
