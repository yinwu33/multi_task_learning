import os
import cv2
import json
import numpy as np
import random
from torch.utils import data
from configs.cfg import args
from torchvision import transforms as T
from datasets.gt_processing import *
from torch.utils.data import DataLoader


class Cityscapes(data.Dataset):
    def __init__(self, mode='train', task='sem_seg', train_extra=False, transform=True, max_objs=130, partial=args['partial']):
        """CityScapes Dataset

        Args:
            mode (str, optional): train, validation or test. Defaults to 'train'.
        """
        self.task = task
        self.mode = mode
        self.max_objs = max_objs
        self.file_list = Cityscapes.load_files(self.mode, train_extra)

        if partial is not None:
            random.shuffle(self.file_list)
            self.file_list = self.file_list[:partial]

        self.transform = transform
        self.transforms = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=[0.3160, 0.3553, 0.3110],
                        std=[0.1927, 0.1964, 0.1965]),
        ])

        self.transform_label = T.Compose([T.ToTensor()])

    def __getitem__(self, index):

        files = self.file_list[index]
        img_name = files['img_name']
        img_path = files['img_file']
        seg_label_path = files['sem_seg_lbl_file']
        det_label_path = files['obj_det_lbl_file']
        depth_label_path = files['depth_reg_lbl_file']

        # imread image and label, then convert them into rgb and gray value
        img = cv2.cvtColor(cv2.imread(img_path),
                           cv2.COLOR_BGR2RGB)  # convert to RGB
        # crop to reduce the unused pixels in depth label
        img = img[0:950, 148:2048]
        randomfloat = random.uniform(0, 1)
        if randomfloat > 0.5:
            img = cv2.flip(img, 1)
        img = cv2.resize(img, args['input_size'], cv2.INTER_LINEAR)
        if self.transform:
            img = self.transforms(img)
        else:
            img = T.Compose([T.ToTensor()])(img)

        if self.task == "test":
            return img_name

        if self.task == 'sem_seg':
            label = cv2.imread(seg_label_path, 0),
            label = label[0:950, 148:2048]
            label = cv2.resize(label, args['input_size'], interpolation=cv2.INTER_NEAREST)
            label = Cityscapes.seg_lbl_convert(label)
            label = torch.from_numpy(label).squeeze().long()
            return img, label, img_name

        if self.task == 'obj_det':
            label = Cityscapes.polygon_to_bbox(det_label_path)
            label_hm = centers_gt_gen_cs(label, max_objs=self.max_objs)
            label_map = gt_gen_cs(label, max_objs=self.max_objs)
            return img, label_hm, label_map, img_name

        if self.task == 'depth':
            depth_label = cv2.imread(depth_label_path, 0)
            depth_label = depth_label[0:950, 148:2048]
            if randomfloat > 0.5:
                depth_label = cv2.flip(depth_label, 1)
            depth_label = cv2.resize(
                depth_label, args['input_size'], interpolation=cv2.INTER_NEAREST)
            # depth value in original label in a descending order,
            # for example value 95 (max value in cityscapes dataset) means depth 0,
            # which causes problem when we use ordinal regression which assumes that pixel
            # with lager value is deeper.
            depth_label = 1 - depth_label / 100
            # depth_label = self.random_flip(depth_label)
            depth_label = torch.from_numpy(depth_label)
            return img, depth_label

        if self.task == 'multi_task':
            sem_label = cv2.imread(seg_label_path, 0)
            sem_label = sem_label[0:950, 148:2048]
            sem_label = cv2.resize(sem_label, args['input_size'], interpolation=cv2.INTER_NEAREST)
            sem_label = Cityscapes.seg_lbl_convert(sem_label)
            sem_label = torch.from_numpy(sem_label).squeeze().long()

            label = Cityscapes.polygon_to_bbox(det_label_path)
            det_label = centers_gt_gen_cs(label, max_objs=self.max_objs)
            depth_label = cv2.imread(depth_label_path, 0)
            depth_label = depth_label[0:950, 148:2048]
            depth_label = cv2.resize(
                depth_label, args['input_size'], interpolation=cv2.INTER_NEAREST)
            # depth value in original label in a descending order,
            # for example value 95 (max value in cityscapes dataset) means depth 0,
            # which causes problem when we use ordinal regression which assumes that pixel
            # with lager value is deeper.
            # depth_label = 95.0 - depth_label
            depth_label = 1 - depth_label / 100

            depth_label = torch.from_numpy(depth_label)

            # we creat those zero (empty) labels so that the data from both datasets
            # are of the same shape, and can be loaded batch-wise.
            # sem_label = torch.zeros_like(depth_label)
            return img, sem_label, det_label, depth_label

    def __len__(self):
        return len(self.file_list)

    @staticmethod
    def load_files(mode, train_extra):
        """
        NOTE THAT, YOU HAVE TO SPECIFY WHICH SPILT (TRAIN/VAL/TEST) YOU WANT
        :return: list of dict of img names, image file paths and label files paths
        """

        paths = Cityscapes.get_files(mode, train_extra)
        img_path = paths['img_path']
        label_path = paths['label_path']
        disparity_path = paths['disparity_path']

        cities = os.listdir(img_path)

        files = []

        for c in cities:
            c_items = [
                name.split('_leftImg8bit.png')[0]
                for name in os.listdir(os.path.join(img_path, c))
            ]
            for it in c_items:
                files.append(
                    {
                        'img_name': it,
                        'img_file': os.path.join(img_path, c, it + '_leftImg8bit.png'),
                        'sem_seg_lbl_file': os.path.join(label_path, c,
                                                         it + '_gtFine_labelIds.png'),
                        'obj_det_lbl_file': os.path.join(label_path, c,
                                                         it + '_gtFine_polygons.json'),
                        'depth_reg_lbl_file': os.path.join(disparity_path, c,
                                                           it + '_disparity.png'),
                    }
                )
        # TODO, the extra dataset of disparity
        if train_extra:
            img_extra_path = paths['img_extra_path']
            disparity_extra_path = paths['disparity_extra_path']
            label_extra_path = paths['label_extra_path']
            cities = os.listdir(img_extra_path)

            for c in cities:
                c_items = [
                    name.split('_leftImg8bit.png')[0]
                    for name in os.listdir(os.path.join(img_extra_path, c))
                ]
                for it in c_items:
                    files.append(
                        {
                            'img_name': it,
                            'img_file': os.path.join(img_extra_path, c, it + '_leftImg8bit.png'),
                            'sem_seg_lbl_file': os.path.join(label_extra_path, c,
                                                             it + '_gtCoarse_labelIds.png'),
                            'obj_det_lbl_file': os.path.join(label_extra_path, c,
                                                             it + '_gtCoarse_polygons.json'),
                            'depth_reg_lbl_file': os.path.join(disparity_extra_path, c,
                                                               it + '_disparity.png'),
                            'depth_reg_lbl_file': os.path.join(disparity_extra_path, c,
                                                             it + '_disparity.png'),
                        }
                    )
        return files

    @staticmethod
    def get_files(mode, train_extra):
        assert (mode == 'train' or mode == 'test' or mode == 'val')

        paths = {}

        root = args['cityscapes_dir']
        paths['img_path'] = os.path.join(root, 'leftImg8bit', mode)
        paths['label_path'] = os.path.join(root, 'gtFine', mode)
        paths['disparity_path'] = os.path.join(root, 'disparity', mode)

        if train_extra:
            paths['img_extra_path'] = os.path.join(root, 'leftImg8bit', 'train_extra')
            paths['label_extra_path'] = os.path.join(root, 'gtCoarse', 'train_extra')
            paths['disparity_extra_path'] = os.path.join(root, 'disparity', 'train_extra')

        return paths

    @staticmethod
    def seg_lbl_convert(label):
        # 0 for background, 1 for drivable, 2 for non-drivable, 3 for lane-marking.
        convert_dict = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 1, 8: 2, 9: 1, 10: 2, 11: 0, 12: 0,
                        13: 0, 14: 0, 15: 0, 16: 0, 17: 0, 18: 0, 19: 0, 20: 0, 21: 0, 22: 0, 23: 0,
                        24: 0, 25: 0, 26: 0, 27: 0, 28: 0, 29: 0, 30: 0, 31: 0, 32: 0, 33: 0, -1: 0}

        label = np.array(label)
        converted_label = np.zeros_like(label)

        for key, value in convert_dict.items():
            idx = np.where(label == key)
            converted_label[idx] = value

        return converted_label

    @staticmethod
    def lbl_to_id(label):
        id_dict = {"person": 0, "person group": 1, "rider": 2, "rider group": 3,
                   "car": 4, "car group": 5, "truck": 6, "truck group": 7, "bus": 8,
                   "bus group": 9, "on rails": 10, "motorcycle": 11, "motorcycle group": 12,
                   "bicycle": 13, "bicycle group": 14, "traffic light": 15,
                   }

        return id_dict[label]

    @staticmethod
    def polygon_to_bbox(gt_path):
        classes = ["person", "person group", "rider", "rider group",
                   "car", "car group", "truck", "truck group", "bus",
                   "bus group", "on rails", "motorcycle", "motorcycle group",
                   "bicycle", "bicycle group", "traffic light"]
        with open(gt_path) as f:
            gt = json.load(f)

        bboxes = []
        lbls = []
        converted_dict = {}

        for object in gt["objects"]:
            lbl = object["label"]
            if lbl not in classes:
                continue

            lbl = Cityscapes.lbl_to_id(lbl)

            vertex = object["polygon"]
            vertex_x, vertex_y = list(zip(*vertex))

            left_x = int(min(vertex_x))
            left_y = int(min(vertex_y))
            right_x = int(max(vertex_x))
            right_y = int(max(vertex_y))

            bbox = [left_x, left_y, right_x, right_y]

            bboxes.append(bbox)
            lbls.append(lbl)

        converted_dict["bboxes"] = bboxes
        converted_dict["lbls"] = lbls

        return converted_dict


if __name__ == '__main__':
    cs = Cityscapes(mode='train', task='depth', train_extra=False)
    train_loader = DataLoader(cs,
                              batch_size=1,
                              num_workers=8,
                              shuffle=True)
    while True:
        next(iter(train_loader))

