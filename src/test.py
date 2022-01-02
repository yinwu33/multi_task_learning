import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from src.datasets.cityscapes import Cityscapes
from src.datasets.a2d2 import A2D2
from src.models.model import *
from models.heatmap_decode import decode
from src.utils.utils import multi_acc, mean_average_precision, rmse
from src.configs.cfg import args


# functions for evaluation of networks.
# test_multi_task for evaluation of multi-task learning network in terms of mIoU, mAP, RMSE.
# test_sem, test_det, test_depth for evaluation of three baseline single task models respectively.
def test_multi_task(net_path):
    # set network and load parameters
    net = ShareNet(args['net'],
                     sem_num_classes=args['sem_num_classes'],
                     det_num_classes=args['det_num_classes'],
                     pretrained=args['pretrained'])
    net.load_state_dict(torch.load(net_path)['model_state_dict'])

    # set device
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    net.to(device).eval()

    test_set_cs = Cityscapes(mode='test', task='depth', train_extra=False)
    test_loader_cs = DataLoader(test_set_cs, batch_size=1, shuffle=False)

    test_set_a2d2 = A2D2(task='test')

    test_loader_a2d2 = DataLoader(test_set_a2d2, batch_size=1, shuffle=False)

    mIoU = 0.0
    mAP = 0.0
    RMSE = 0.0

    # start to test
    with tqdm(total=len(test_loader_a2d2)) as epoch_bar:
        for i, data2 in enumerate(iter(test_loader_a2d2)):
            # two small-batches at a time, so that each batch contains data from both datasets
            data1 = next(iter(test_loader_cs))
            with torch.no_grad():
                inputs1 = data1[0].to(device)
                inputs2 = data2[0].to(device)

                labels_sem = data2[1].to(device)
                labels_depth = data1[1].to(device)

                labels_map = data2[3]

                outputs_sem_a2d2 = net(inputs2)[0]
                outputs_det_a2d2 = net(inputs2)[1]
                outputs_depth_cs = net(inputs1)[2]

                bboxes, cls, scores = decode(outputs_det_a2d2['cls'],
                                             outputs_det_a2d2['wh'],
                                             outputs_det_a2d2['reg'], 40)

                mIoU += multi_acc(outputs_sem_a2d2, labels_sem)
                mAP += mean_average_precision((bboxes, cls, scores), labels_map)
                RMSE += rmse(outputs_depth_cs, labels_depth)

                # demonstrate epoch bar
                epoch_bar.update(1)

    mIoU = mIoU / len(test_loader_a2d2)
    mAP = mAP / len(test_loader_a2d2)
    RMSE = RMSE / len(test_loader_a2d2)

    print(mIoU)
    print(mAP)
    print(RMSE)


def test_sem(net_path):
    # set network and load parameters
    net = SemNet(args['net'],
                sem_num_classes=args['sem_num_classes'])
    net.load_state_dict(torch.load(net_path))

    # set device
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    net.to(device).eval()

    test_set_a2d2 = A2D2(task='test')

    test_loader_a2d2 = DataLoader(test_set_a2d2, batch_size=1, shuffle=False)

    mIoU = 0.0

    # start to test
    with tqdm(total=len(test_loader_a2d2)) as epoch_bar:
        for i, data2 in enumerate(iter(test_loader_a2d2)):
            with torch.no_grad():
                inputs2 = data2[0].to(device)

                labels_sem = data2[1].to(device)

                outputs_sem_a2d2 = net(inputs2)[0]

                mIoU += multi_acc(outputs_sem_a2d2, labels_sem)

                # demonstrate epoch bar
                epoch_bar.update(1)

    mIoU = mIoU / len(test_loader_a2d2)

    print(mIoU)


def test_det(net_path):
    # set network and load parameters
    net = DetNet(args['net'],
                     det_num_classes=args['det_num_classes'])
    net.load_state_dict(torch.load(net_path))

    # set device
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    net.to(device).eval()

    test_set_a2d2 = A2D2(task='test')

    test_loader_a2d2 = DataLoader(test_set_a2d2, batch_size=1, shuffle=False)

    mAP = 0.0

    # start to test
    with tqdm(total=len(test_loader_a2d2)) as epoch_bar:
        for i, data2 in enumerate(iter(test_loader_a2d2)):
            # two small-batches at a time, so that each batch contains data from both datasets
            with torch.no_grad():
                inputs2 = data2[0].to(device)

                labels_map = data2[3]

                outputs_det_a2d2 = net(inputs2)[0]

                bboxes, cls, scores = decode(outputs_det_a2d2['cls'],
                                             outputs_det_a2d2['wh'],
                                             outputs_det_a2d2['reg'], 40)

                mAP += mean_average_precision((bboxes, cls, scores), labels_map)

                # demonstrate epoch bar
                epoch_bar.update(1)

    mAP = mAP / len(test_loader_a2d2)

    print(mAP)


def test_depth(net_path):
    # set network and load parameters
    net = DepthUNet('resnet50')
    net.load_state_dict(torch.load(net_path))

    # set device
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    net.to(device).eval()

    test_set_cs = Cityscapes(mode='test', task='depth', train_extra=False)
    test_loader_cs = DataLoader(test_set_cs, batch_size=1, shuffle=False)

    test_set_a2d2 = A2D2(task='test')

    test_loader_a2d2 = DataLoader(test_set_a2d2, batch_size=1, shuffle=False)

    RMSE = 0.0

    # start to test
    with tqdm(total=len(test_loader_a2d2)) as epoch_bar:
        for i, data2 in enumerate(iter(test_loader_a2d2)):
            # two small-batches at a time, so that each batch contains data from both datasets
            data1 = next(iter(test_loader_cs))
            with torch.no_grad():
                inputs1 = data1[0].to(device)

                labels_depth = data1[1].to(device)

                outputs_depth_cs = net(inputs1)[0]

                RMSE += rmse(outputs_depth_cs, labels_depth)

                # demonstrate epoch bar
                epoch_bar.update(1)

    RMSE = RMSE / len(test_loader_a2d2)

    print(RMSE)


if __name__ == '__main__':
    test_det('/home/wuyin/Documents/ZhuBangyu/save/2020-09-06-01-16-51/det/net_parameters.pth')
